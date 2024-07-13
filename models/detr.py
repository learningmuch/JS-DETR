# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
损失函数部分：
先来看看与loss函数相关的一些参数: matcher就是将预测结果与GT进行匹配的匈牙利算法。weight_ dict是 为各部分loss设置的权重，主要包括分类与回归损失，
分类使用的是交叉熵损失,而回归损失包括bbox 的L1Loss (计算x、 y. w、h的绝对值误差)与GloU Loss。
若设置了masks参数,则代表分割任务，那么还需加入对应 的loss类型。
另外，若设置了aux_ loss,即代表需要计算解码器中间层预测结果对应的loss,那么也要设置对应的loss权重。
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # 分类
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 回归
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.query_embed 类似于传统目标检测里面的anchor 这里设置了100个  [100,256]
        # 这个就是  output pos. enc. 在decoder中使用。 类似与传统的anchor，是可以学习的。
        # 经过不停的迭代之后，模型可以学习到(建模到)这个100(num_queries)个物体之间的相互关系,大概是个什么关系：例如其中一个例子。比如狗基本上只会出现在地面上，不会出现在天空上，建模大概就是建模的这种关系，
        # 比如桌子椅子一般在房间里面，这些都是可以迭代出来的。nn.Embedding 等价于 nn.Parameter,都是可以更新学习的。
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # out: list{0: tensor=[bs,2048,19,26] + mask=[bs,19,26]}  经过backbone resnet50 block5输出的结果
        # pos: list{0: [bs,256,19,26]}  位置编码
        features, pos = self.backbone(samples)
        # src: Tensor [bs,2048,19,26]
        # mask: Tensor [bs,19,26]
        src, mask = features[-1].decompose()
        assert mask is not None

        # 数据输入transformer进行前向传播
        # self.input_proj(src) [bs,2048,19,26]->[bs,256,19,26]
        # mask: False的区域是不需要进行注意力计算的
        # self.query_embed.weight  类似于传统目标检测里面的anchor 这里设置了100个 shape: [100,256]
        # pos[-1]  位置编码  [bs, 256, 19, 26]
        # hs: [6, bs, 100, 256]
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # 分类 [6个decoder, bs, 100, 256] -> [6, bs, 100, 92(类别)]
        outputs_class = self.class_embed(hs)
        # 回归 [6个decoder, bs, 100, 256] -> [6, bs, 100, 4]
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        # dict: 3
        # 0 pred_logits 分类头输出[bs, 100, 92(类别数)]
        # 1 pred_boxes 回归头输出[bs, 100, 4]
        # 3 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
        此类计算 DETR 的损失。
    该过程分两步进行：
        1） 我们计算真实方框和模型输出之间的匈牙利赋值
        2）我们监督每对匹配的地面真相/预测（监督类和框）
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            在DETR模型中，我们需要定义一个损失函数（criterion）来指导模型的训练。这个损失函数具有一些参数，包括：

            num_classes：目标类别的数量，不包括特殊的无目标类别。
            matcher：用于计算目标与预测框之间匹配关系的模块。
            weight_dict：一个字典，包含损失名称作为键，相对权重作为值。
            eos_coef：应用于无目标类别的相对分类权重。
            losses：要应用的所有损失的列表。
            通过定义这个损失函数，我们可以结合匹配模块和各种损失来计算模型的整体损失。不同的损失函数可以帮助模型优化不同的方面，比如类别分类、边界框回归等。在这段描述中，提到了可以使用get_loss函数来获取可用的损失函数列表。
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # If the categories in the dataset are unbalanced, the classification loss is replaced by focal loss

    # def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    #     """Classification loss (Focal Loss)
    #         If the categories in the dataset are unbalanced, the classification loss is replaced by focal loss
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #     gamma = 2 # Focal Loss 的超参数 gamma
    #     alpha = None  # 可选，Focal Loss 的超参数 alpha
    #     assert 'pred_logits' in outputs
    #     src_logits = outputs['pred_logits']
    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    #     target_classes = torch.full(src_logits.shape[:2], self.num_classes,
    #                                 dtype=torch.int64, device=src_logits.device)
    #     target_classes[idx] = target_classes_o
    #     # 计算 Focal Loss
    #     loss_focal = nn.CrossEntropyLoss(reduction='mean')
    #     loss_focal = loss_focal(src_logits.transpose(1, 2), target_classes)
    #     # 计算预测概率
    #     pt = torch.exp(-loss_focal)
    #     # 应用 Focal Loss 的权重修正
    #     if gamma != 0:
    #         loss_focal = (1 - pt) ** gamma * loss_focal
    #     # 可选：应用 Focal Loss 的类别权重修正
    #     if alpha is not None:
    #         loss_focal = alpha * loss_focal
    #     # 将 Focal Loss 添加到损失字典中
    #     losses = {'loss_focal': loss_focal}
    #     if log:
    #         # TODO this should probably be a separate loss, not hacked in this one here
    #         losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    #     return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        仅用做log，不涉及反向传播梯度。
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            计算与边界框相关的损失，即L1回归损失和GIoU损失。
            目标字典(targets)中必须包含键"boxes"，它包含一个维度为[nb_target_boxes, 4]的张量。
            目标边界框的格式应为(center_x, center_y, w, h)，并且已经通过图像大小进行了归一化。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        # 修改了L1回归损失为smoothL1回归损失
        # loss_bbox = F.smooth_l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # 此次修改了损失函数 把giou换成ciou了
        # from torchvision.ops import box_iou, box_area
        # import math
        # from .loss import box_cxcywh_to_xyxy
        # iou = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        # # print(f"target is {target_boxes}")
        # # print(f"iou is {iou}")
        # # print(f"box_opsis {box_ops.generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))}")
        # cious = iou - box_ops.generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        # v = (4 / (math.pi ** 2)) * torch.pow(
        #     (torch.atan(src_boxes[:, 2] / src_boxes[:, 3]) - torch.atan(target_boxes[:, 2] / target_boxes[:, 3])),
        #     2).unsqueeze(1)
        # # print(f"cious is {cious}")
        # # print(f"v is {v}")
        # alpha = v / (1 - iou + v)
        # # print(f"alpha is {alpha}")
        # loss_ciou = 1 - cious + alpha * v
        # # print(f"loss_ciou is {loss_ciou}")
        # losses['loss_ciou'] = loss_ciou.sum() / num_boxes
        # # print(f"losses is {losses} ,num_boxes is{num_boxes}")

        # # ##11
        # # # 此次修改了损失函数 把giou换成ciou了
        # from torchvision.ops import box_iou, box_area
        # import math
        # from .loss import box_cxcywh_to_xyxy
        # from .loss import getCenterPoint
        # # from .loss import getDistance_2
        # from .loss import getDistance
        # from .loss import getConvexShape
        # # 从下面开始都是对ciou损失函数的修改呢。注释写在下面，但是要注意，这个只针对batchsize等于2的时候生效。
        # iou = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        # # 计算边界框的IoU
        # center_point_gt = getCenterPoint(target_boxes)
        # center_point_pd = getCenterPoint(src_boxes)
        # # 计算真实框和预测框的中心点坐标
        # first_gt = center_point_gt[0]
        # second_gt = center_point_gt[1]
        # second_pd = center_point_pd[1]
        # first_pd = center_point_pd[0]
        # # 提取中心点坐标中的第一个真实框和第一个预测框的坐标
        # rho_0 = getDistance(first_gt, first_pd) ** 2
        # rho_1 = getDistance(second_gt, second_pd) ** 2
        # # 计算中心点之间的距离的平方
        # rho_2 = (rho_0 + rho_1) / 2
        # # 计算中心点距离的平均值
        # bbox_cvx = getConvexShape(target_boxes, src_boxes)
        # # 计算两个边界框的最小闭包框
        # c_2 = bbox_cvx
        # # 最小闭包框的对角线距离
        # cious = iou - (rho_2 / (c_2 + 1e-7))
        # # 计算CIOU值
        # v = (4 / (math.pi ** 2)) * torch.pow(
        #     (torch.atan(src_boxes[:, 2] / src_boxes[:, 3]) - torch.atan(target_boxes[:, 2] / target_boxes[:, 3])),
        #     2).unsqueeze(1)
        # # 计算v值，其中包括边界框宽高比的差异
        # alpha = v / (1 - iou + v)
        # # 计算alpha值
        # loss_ciou = 1 - cious + alpha * v
        # # 计算CIoU损失函数
        # # 此giou非彼giou ，这个是ciou
        # losses['loss_giou'] = loss_ciou.sum() / num_boxes


        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
           这段代码用于计算实例分割任务中的损失函数，包括了sigmoid focal loss和dice loss。
           通过计算这些损失函数，可以衡量模型在实例分割方面的预测准确性和相似度。
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) 非常简单的多层感知器（也称为FFN）"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    # 搭建backbone resnet + PositionEmbeddingSine
    backbone = build_backbone(args)

    # 搭建transformer
    transformer = build_transformer(args)

    # 搭建整个DETR模型
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    # 是否需要额外的分割任务
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # HungarianMatcher()  二分图匹配
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from models import build_model
from PIL import Image
import os
import torchvision
from torchvision.ops.boxes import batched_nms
import cv2


def get_args_parser():

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # 检测的图像路径
    parser.add_argument('--source_dir', default=r'E:\DETR\detr.6.21\data\coco\test2017',
                        help='path where to save, empty for no saving')

    # parser.add_argument('--source_dir_1', default='data/val',
    #                     help='path where to save, empty for no saving')
    # 检测结果保存路径
    parser.add_argument('--output_dir', default=r'E:\各种版本的detr修改\output\res50',
                        help='path where to save, empty for no saving')
    # 存放模型的位置
    parser.add_argument('--resume', default=r'weight_all/res50/test/checkpoint_best_epoch77.pth',
                        help='resume from checkpoint')

    # parser.add_argument('--device', default='cpu',
    #                     help='device to use for training / testing')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    # 如果设置为resnet101，后面的权重文件路径也需要修改一下
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', default='False',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default="coco")
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')



    # parser.add_argument('--device', default='cuda',
    #                     help='device to use for training / testing')

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default="True")
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# 这段代码是一个用于将中心坐标宽高表示的边界框转换为左上角坐标和右下角坐标表示的边界框的函数。
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 这段代码用于将预测得到的边界框重新缩放到原始图像的尺寸。这个函数的作用是将预测得到的边界框根据原始图像的尺寸进行缩放，使其与原始图像相对应。这样可以在原始图像上正确显示和处理边界框。
# 将归一化坐标转换为实际图像坐标
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# 这段代码用于过滤边界框，根据置信度阈值和非最大值抑制(NMS)进行筛选。
def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.3):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    return scores, boxes


# COCO classes
CLASSES = [
    'N/A', 'esophagealcarcinoma',
]

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # 计算线条和字体的粗细
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    # 生成随机颜色或使用指定的颜色
    # color = color or [random.randint(0, 255) for _ in range(3)]
    color = [0, 0, 255]
    # 提取框的坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 调整框的大小是为了适应绘制标记框的需求
    c2 = (c1[0] + int(x[2] - x[0])), (c1[1] + int(x[3] - x[1]))
    # 在图像上绘制矩形边界框
    cv2.rectangle(img, c1, c2, color=color, thickness=tl, lineType=cv2.LINE_AA)
    # 添加标签
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  # 获取文本尺寸
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 绘制填充矩形框
        cv2.rectangle(img, c1, c2, color=color, thickness=-1, lineType=cv2.LINE_AA)
        # 在图像上添加标签文本
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def main(args):
    print(args)
    time_first = time.time()
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], False)
    model.to(device)
    print(args.device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", n_parameters)

    image_Totensor = torchvision.transforms.ToTensor()
    image_file_path = os.listdir(args.source_dir)  # E:\detr-main\demo\input

    for image_item in image_file_path:
        print("推理图片编号:", image_item)
        image_path = os.path.join(args.source_dir, image_item)
        image = Image.open(image_path)
        image_tensor = image_Totensor(image)
        image_tensor = torch.reshape(image_tensor,
                                     [-1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]])
        image_tensor = image_tensor.to(device)
        time1 = time.time()
        # 对输入的图像数据进行推理（或称为前向传播）的操作 将输出结果赋值给inference_result变量，可以在后续的代码中使用这个结果进行进一步的处理、分析或展示。
        inference_result = model(image_tensor)
        time2 = time.time()
        print("推理时间为:", time2 - time1)
        # 获取推理结果中的类别概率（probabilities）在inference_result中，pred_logits是模型输出的原始预测结果，它包含了各个类别的得分（logits）
        # 索引操作[0, :, :-1]选择的是第一个样本的所有类别概率
        probas = inference_result['pred_logits'].softmax(-1)[0, :, :-1].cpu()
        # 这段代码用于将推理结果中的边界框（bounding boxes）进行尺度调整。具体来说，它使用rescale_bboxes函数对推理结果中的边界框进行调整，使其适应原始图像的尺寸。
        # inference_result['pred_boxes'][0,]表示推理结果中第一个样本的所有边界框的坐标信息。这个框得到的比例和金标准的不太一样。
        # 左上角的 x 坐标、左上角的 y 坐标、右下角的 x 坐标和右下角的 y 坐标。（使用了归一化之后就方便缩放）
        # 批次大小（batch size）、通道数（channel）、图像高度（height）和图像宽度（width）。所以分别是宽高

        bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0,].cpu(),
                                       (image_tensor.shape[3], image_tensor.shape[2]))
        # 得到最终的分数，和框的最终坐标。
        scores, boxes = filter_boxes(probas, bboxes_scaled)
        # 将保留的置信度分数转换为NumPy数组形式，即scores.data.numpy()。这样可以方便后续的处理和可视化
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
        # 遍历每个保留的边界框，获取其类别和置信度信息，并在图像上绘制相应的边界框和标签文本。
        for i in range(boxes.shape[0]):
            class_id = scores[i].argmax()
            label = CLASSES[class_id]
            confidence = scores[i].max()
            text = f"{label} {confidence:.3f}"
            print(text)
            image = np.array(image)
            plot_one_box(boxes[i], image, label=text)
        # OpenCV默认使用BGR格式表示图像
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # cv2.imshow("images", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        image = Image.fromarray(image)
        image.save(os.path.join(args.output_dir, image_item))
    time_final= time.time()
    total_time = (time_final-time_first) /60
    print("一共花的时间是",total_time)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
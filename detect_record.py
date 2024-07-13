'''使用这个计算AP、precision、recall'''
import os
import util.misc as utils
import random
import numpy as np
from models.detr import build
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET

torch.set_grad_enabled(False)
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
transform_input = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device="cuda")
    return b


def save_images(pil_img, prob, boxes, img_save_path):
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cv2.rectangle(pil_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), thickness=2)
        fontScale = 0.75
        cl = p.argmax()
        text = f'{CLASSES[cl]} {p[cl]:0.2f}'
        retval, baseLine = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=1)
        topleft = (int(xmin), int(ymin) - retval[1])
        bottomright = (topleft[0] + retval[0], topleft[1] + retval[1])
        cv2.rectangle(pil_img, (topleft[0], topleft[1] - baseLine), bottomright, thickness=-1, color=(0, 0, 255))

        cv2.putText(pil_img, text, (int(xmin), int(ymin)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(0,0,0), thickness=1)
    cv2.imwrite(img_save_path, pil_img)

def calIoU(mat1, mat2):
    x1 = max(min(mat1[0], mat1[2]), min(mat2[0], mat2[2]))
    x2 = min(max(mat1[0], mat1[2]), max(mat2[0], mat2[2]))
    y1 = max(min(mat1[1], mat1[3]), min(mat2[1], mat2[3]))
    y2 = min(max(mat1[1], mat1[3]), max(mat2[1], mat2[3]))
    area = (mat1[2]-mat1[0])*(mat1[3]-mat1[1]) + (mat2[2]-mat2[0])*(mat2[3]-mat2[1])
    if x1<x2 and y1<y2:
        intersection = (x2 - x1) * (y2 - y1)
    else:
        intersection = 0
    iou = intersection / (area - intersection)
    return iou

def main(chenkpoint_path, img_root, save_dir, save_img=False):
    args = torch.load(chenkpoint_path)['args']
    model = build(args)[0]
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # 加载模型参数
    model_data = torch.load(chenkpoint_path)['model']
    model.load_state_dict(model_data)

    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pos = 0
    tp = 0
    fp = 0

    imgnames = list(set([x[:-4] for x in os.listdir(img_root)]))
    for name in imgnames:
        img_path = os.path.join(img_root,name+".jpg")
        img = Image.open(img_path).convert('RGB')
        size = img.size

        if size[0] > size[1]:
            new_width = 640
            new_height = int(640 * size[1] / size[0])
        else:
            new_height = 640
            new_width = int(640 * size[0] / size[1])
        img = img.resize((new_width, new_height))
        inputs = transform_input(img).unsqueeze(0)
        outputs = model(inputs.to(device))
        # 这类最后[0, :, :-1]索引其实是把背景类筛选掉了
        probs = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # 可修改阈值,只输出概率大于0.5的物体
        keep = probs.max(-1).values > 0.5
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], size)

        bboxs = bboxes_scaled.tolist()
        gt_boxs = []
        tree = ET.parse(os.path.join(img_root,name+".xml"))
        root = tree.getroot()
        for object in root.findall('object'):
            pos += 1
            Xmin = int(object.find('bndbox').find('xmin').text)
            Ymin = int(object.find('bndbox').find('ymin').text)
            Xmax = int(object.find('bndbox').find('xmax').text)
            Ymax = int(object.find('bndbox').find('ymax').text)
            gt_boxs.append(list((Xmin,Ymin,Xmax,Ymax)))
        for bbox in bboxs:
            for gt_box in gt_boxs:
                if calIoU(bbox,gt_box)>=0.5:
                    tp += 1
                    break
                if gt_box==gt_boxs[-1]:
                    fp += 1

        # 保存输出结果
        ori_img = np.array(img)
        if save_img:
            save_img_path = os.path.join(save_dir,"img")
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            save_images(ori_img, probs[keep], bboxes_scaled, os.path.join(save_img_path,name+".jpg"))
    rec = tp/pos
    prec = tp/(tp+fp)
    with open(os.path.join(save_dir,"result.txt"),"w") as f:
        f.write(f'tp:{tp},fp:{fp},pos:{pos}'+'\n')
        f.write('recall:{0:.4f},precision:{1:.4f}'.format(rec, prec))
        # f.write(f'recall:{rec},precision:{prec}')
    print(f'tp:{tp},fp:{fp},pos:{pos}')
    print('recall:{0:.4f},precision:{1:.4f}'.format(rec, prec))

if __name__ == "__main__":
    CLASSES = ["bg", "cancer"]
    main(chenkpoint_path=r"weight_all/DA_ciou/test/checkpoint_best_epoch61.pth", img_root=r"data/val",
         save_dir="output", save_img=False)

    # main(chenkpoint_path=r"weight_all/DA/test/checkpoint_best_epoch39.pth", img_root=r"data/val",
    #      save_dir="output", save_img=True)


import glob
import math
import numpy as np

from PIL import Image
import cv2
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import torchvision.models as models
torch.set_grad_enabled(False)

import os

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, save_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(save_path)
    # plt.show()


def png2jpg(img_path):
    img = cv2.imread(img_path, 0)
    # w, h = img.shape[::-1]
    infile = img_path
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    # img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS) # 修改原图大小
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=100)
        else:
            img.convert('RGB').save(outfile, quality=100)
        # os.remove(img_path) # 覆盖原文件
        return outfile
    except Exception as e:
        print("PNG to JPG error!", e)


# Step1: 加载模型，会加载到电脑的".cache"文件中
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval();
# 从本地加载，还是会从网上下
# model = torch.hub.load(r'./weight_all/weight_res50','checkpoint0099.pth',pretrained=True, source='local')
# model = torch.load('./weight_all/weight_res50/checkpoint0099.pth', map_location='cpu')
# model.eval();


# Step2: 循环读取文件中的图片,文件位置为'./data/images',并将文件保存
# golb.golb会返回匹配路径下所有符合的patten,以列表的形式返回
paths = glob.glob(os.path.join(r'./demo/input', '*.*'))
print(paths)

for path in paths:
    # 问题1：无法读取png图像
    if os.path.splitext(path)[1] == ".png":
    # 问题1解1：用imread读取png
        im = cv2.imread(path)
        im = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    # 问题1解2：将png转换为jpg,但感觉可能解1会更快一点,且该方法画质有损明显
    #     png2jpg(path)
    #     im = Image.open(os.path.splitext(path)[0] + '.jpg')
    else:
        im = Image.open(path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    img_save_path = r'./demo/output/' + os.path.splitext(os.path.split(path)[1])[0] + '.jpg'
    plot_results(im, probas[keep], bboxes_scaled, img_save_path)


"""通过标注给原图或者检测后的图画框"""
# import os
# import os.path
# import xml.etree.cElementTree as ET
# import cv2
# def draw(image_path, root_saved_path):
#     """
#     图片根据标注画框
#     """
#     src_img_path = image_path
#     for file in os.listdir(src_img_path):
#         print(file)
#         file_name, suffix = os.path.splitext(file)
#         if suffix == '.xml':
#             # print(file)
#             xml_path = os.path.join(src_img_path, file)
#             image_path = os.path.join(src_img_path, file_name+'.jpg')
#             img = cv2.imread(image_path)
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             for obj in root.iter('object'):
#                 name = obj.find('name').text
#                 xml_box = obj.find('bndbox')
#                 x1 = int(xml_box.find('xmin').text)
#                 x2 = int(xml_box.find('xmax').text)
#                 y1 = int(xml_box.find('ymin').text)
#                 y2 = int(xml_box.find('ymax').text)
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
#                 # 字为绿色
#                 cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)
#             cv2.imwrite(os.path.join(root_saved_path, file_name+'.jpg'), img)
#
#
# if __name__ == '__main__':
#     image_path = r"D:\zcs\dataset\demo"
#     root_saved_path = r"D:\zcs\dataset\demo_xml"
#     draw(image_path, root_saved_path)


import os
import xml.dom.minidom
import cv2 as cv
from tqdm import tqdm

# ImgPath = r'D:\zcs\dataset\demo'
# ImgPath = r'D:\zcs\picture\out\res'
ImgPath = r'D:\zcs\picture\compare\fastrcnn'
AnnoPath = r'D:\zcs\dataset\demo_xml'
FinalPath =r'D:\zcs\picture\1'

imagelist = os.listdir(ImgPath)

for image in tqdm(imagelist):

    image_pre, ext = os.path.splitext(image)

    imgfile = ImgPath +"/"+ image
    xmlfile = AnnoPath +"/"+ image_pre + '.xml'
    finalfile = FinalPath +"/"+ image


    # 打开xml文档
    DOMTree = xml.dom.minidom.parse(xmlfile)
    # 得到文档元素对象
    collection = DOMTree.documentElement
    # 读取图片
    img = cv.imread(imgfile)

    filenamelist = collection.getElementsByTagName("filename")
    filename = filenamelist[0].childNodes[0].data
    # print(filename)
    # 得到标签名为object的信息
    objectlist = collection.getElementsByTagName("object")

    for objects in objectlist:
        # 每个object中得到子标签名为name的信息
        namelist = objects.getElementsByTagName('name')
        # 通过此语句得到具体的某个name的值
        objectname = namelist[0].childNodes[0].data

        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(float(x1_list[0].childNodes[0].data))
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(float(y1_list[0].childNodes[0].data))
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(float(x2_list[0].childNodes[0].data))
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(float(y2_list[0].childNodes[0].data))
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            # cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
            #            thickness=1)
            cv.imwrite('%s' % finalfile, img)

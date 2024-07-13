# import os
# import cv2
# import xml.etree.ElementTree as ET


# def draw_bounding_boxes(image_path, xml_path, output_folder):
#     image = cv2.imread(image_path)
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     for obj in root.findall('object'):
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         ymin = int(bbox.find('ymin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymax = int(bbox.find('ymax').text)
#
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#
#     # 保存绘制了边界框的图像
#     output_path = os.path.join(output_folder, os.path.basename(image_path))
#     cv2.imwrite(output_path, image)

import cv2
import os
import xml.etree.ElementTree as ET


def draw_bounding_boxes(image_path, xml_path, output_folder):
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # 读取图像
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return
    except Exception as e:
        print(f"Error occurred while reading image: {e}")
        return

    # 解析 XML 文件
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error occurred while parsing XML: {e}")
        return

    # 绘制边界框
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # 保存绘制了边界框的图像
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    try:
        cv2.imwrite(output_path, image)
        print(f"Saved image with bounding boxes: {output_path}")
    except Exception as e:
        print(f"Error occurred while saving image: {e}")

# 别出现中文地址，不然读取不了
# image_folder = r'E:\各种版本的detr修改\output\DA'
image_folder = r'E:\demo\output1'
xml_folder = r'E:\demo\demo_xml'
output_folder = r'E:\demo\final_output\DA_CIOU11.26'
# output_folder = r'E:\各种版本的detr修改\output\final_output\DA'

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

for image_file in image_files:
    image_name = os.path.splitext(image_file)[0]
    xml_file = f'{image_name}.xml'

    if xml_file in os.listdir(xml_folder):
        image_path = os.path.join(image_folder, image_file)
        xml_path = os.path.join(xml_folder, xml_file)

        draw_bounding_boxes(image_path, xml_path, output_folder)
    else:
        print(f'XML file not found for image: {image_file}')

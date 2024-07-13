""" 给医学图像做脱敏处理"""
import cv2

def apply_mosaic(image, block_size=15):
    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 将图像划分为块并进行块状平均值
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            average_color = block.mean(axis=(0, 1), dtype=int)
            image[y:y+block_size, x:x+block_size] = average_color

    return image

def apply_mosaic_corners(input_image_path, output_image_path, block_size=15):
    # 读取图像
    image = cv2.imread(input_image_path)

    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 计算每个角的大小
    # 左上角角度的长宽
    corner_sizetl1 = int(min(height, width) / 5)  # 十分之一
    corner_sizetl2 = int(min(height, width) / 5.5)  # 十分之一
    # 右上角角度
    corner_sizetr1 = int(min(height, width) / 4.5)  # 十分之一
    corner_sizetr2 = int(min(height, width) / 5.5)  # 十分之一
    # 左下角角度
    corner_sizebl1 = int(min(height, width) / 9)  # 十分之一
    corner_sizebl2 = int(min(height, width) / 3.2)  # 十分之一
    # 右下角角度
    corner_sizebr1 = int(min(height, width) / 9)  # 十分之一
    corner_sizebr2 = int(min(height, width) / 4.5)  # 十分之一

    # 定义四个角的区域
    top_left_corner = image[0:corner_sizetl1, 0:corner_sizetl2]
    top_right_corner = image[0:corner_sizetr1, width - corner_sizetr2:width]
    bottom_left_corner = image[height - corner_sizebl1:height, 0:corner_sizebl2]
    bottom_right_corner = image[height - corner_sizebr1:height, width - corner_sizebr2:width]

    # 分别应用马赛克效果到四个角
    mosaic_top_left = apply_mosaic(top_left_corner, block_size)
    mosaic_top_right = apply_mosaic(top_right_corner, block_size)
    mosaic_bottom_left = apply_mosaic(bottom_left_corner, block_size)
    mosaic_bottom_right = apply_mosaic(bottom_right_corner, block_size)

    # 将马赛克效果应用回原始图像
    image[0:corner_sizetl1, 0:corner_sizetl2] = mosaic_top_left
    image[0:corner_sizetr1, width - corner_sizetr2:width] = mosaic_top_right
    image[height - corner_sizebl1:height, 0:corner_sizebl2] = mosaic_bottom_left
    image[height - corner_sizebr1:height, width - corner_sizebr2:width] = mosaic_bottom_right

    # 保存处理后的图像
    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    input_image_path = r"H:\total\fig1\008228.jpg"  # 替换为你的输入图像路径
    output_image_path = r"H:\total\total_1\fig1\008228_mosaic.jpg"

    # 可选：调整块状平均值的块大小
    block_size = 15

    apply_mosaic_corners(input_image_path, output_image_path, block_size)

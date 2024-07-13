"""高斯模糊来对图片做模糊处理"""
import cv2

def apply_corner_blur(input_image_path, output_image_path):
    # 读取图像
    image = cv2.imread(input_image_path)

    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # # 计算每个角的大小
    # 1045
    # 左上角角度的长宽
    corner_sizetl1 = int(min(height, width) / 8)  # 十分之一
    corner_sizetl2 = int(min(height, width) / 6)  # 十分之一
    # 右上角角度
    corner_sizetr1 = int(min(height, width) / 8)  # 十分之一
    corner_sizetr2 = int(min(height, width) / 4)  # 十分之一
    # # 左下角角度
    corner_sizebl1 = int(min(height, width) / 9)  # 十分之一
    corner_sizebl2 = int(min(height, width) / 4)  # 十分之一
    # 右下角角度
    corner_sizebr1 = int(min(height, width) / 9)  # 十分之一
    corner_sizebr2 = int(min(height, width) / 4.5)  # 十分之一

    # 1090
    # 计算每个角的大小
    # # 左上角角度的长宽
    # corner_sizetl1 = int(min(height, width) / 5)  # 十分之一
    # corner_sizetl2 = int(min(height, width) / 5.5)  # 十分之一
    # # # 右上角角度
    #
    # corner_sizetr1 = int(min(height, width) / 4.5)  # 十分之一
    # corner_sizetr2 = int(min(height, width) / 4)  # 十分之一
    # # # 左下角角度
    #
    # corner_sizebl1 = int(min(height, width) / 10)  # 十分之一
    # corner_sizebl2 = int(min(height, width) / 4)  # 十分之一
    # # 右下角角度
    # corner_sizebr1 = int(min(height, width) / 10)  # 十分之一
    # corner_sizebr2 = int(min(height, width) / 4)  # 十分之一

    # 8549
    # 左上角角度的长宽
    # corner_sizetl1 = int(min(height, width) / 5)  # 十分之一
    # corner_sizetl2 = int(min(height, width) / 5.5)  # 十分之一
    # # 右上角角度
    # corner_sizetr1 = int(min(height, width) / 4.5)  # 十分之一宽
    # corner_sizetr2 = int(min(height, width) / 5)  # 十分之一
    # # # 左下角角度
    # corner_sizebl1 = int(min(height, width) / 9)  # 十分之一
    # corner_sizebl2 = int(min(height, width) / 3.2)  # 十分之一
    # # 右下角角度
    # corner_sizebr1 = int(min(height, width) / 10)  # 十分之一
    # corner_sizebr2 = int(min(height, width) / 4)  # 十分之一

    # 定义四个角的区域
    top_left_corner = image[0:corner_sizetl1, 0:corner_sizetl2]
    top_right_corner = image[0:corner_sizetr1, width - corner_sizetr2:width]
    bottom_left_corner = image[height - corner_sizebl1:height, 0:corner_sizebl2]
    bottom_right_corner = image[height - corner_sizebr1:height, width - corner_sizebr2:width]


    # 对每个角进行高斯模糊
    blurred_top_left = cv2.GaussianBlur(top_left_corner, (55, 55), 0)
    blurred_top_right = cv2.GaussianBlur(top_right_corner, (55, 55), 0)
    blurred_bottom_left = cv2.GaussianBlur(bottom_left_corner, (55, 55), 0)
    blurred_bottom_right = cv2.GaussianBlur(bottom_right_corner, (55, 55), 0)

    # 将模糊后的角放回原始图像
    image[0:corner_sizetl1, 0:corner_sizetl2] = blurred_top_left
    image[0:corner_sizetr1, width - corner_sizetr2:width] = blurred_top_right
    image[height - corner_sizebl1:height, 0:corner_sizebl2] = blurred_bottom_left
    image[height - corner_sizebr1:height, width - corner_sizebr2:width] = blurred_bottom_right

    # 保存处理后的图像
    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    # input_image_path = r"H:\total_1\fig10\001123.jpg"  # 替换为你的输入图像路径
    # output_image_path = r"H:\total_1\total_1\fig10_lack\001123.jpg"
    input_image_path = r"H:\total_1\1206\fig2\001054.jpg"  # 替换为你的输入图像路径
    output_image_path = r"H:\total_2\1206\fig2\001054.jpg"
    # input_image_path = r"H:\total_1\1206\fig7\deformr\009194.jpg"  # 替换为你的输入图像路径
    # output_image_path = r"H:\total_2\1206\fig7\deformr\008427.jpg"

    apply_corner_blur(input_image_path, output_image_path)

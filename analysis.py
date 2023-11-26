from PIL import Image
import numpy as np

def calculate_gray_mean_std(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 将图像转换为灰度图像
    gray_image = image.convert('L')

    # 将图像转换为NumPy数组
    gray_array = np.array(gray_image)

    # 计算平均灰度和标准差
    mean_gray = np.mean(gray_array)
    std_gray = np.std(gray_array)

    return mean_gray, std_gray


def calculate_red_mean_std(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 提取红通道数据
    red_channel = image_array[:, :, 0]

    # 计算红通道平均值和标准差
    mean_red = np.mean(red_channel)
    std_red = np.std(red_channel)

    return mean_red, std_red


def image_analysis(image_path):
    mean_gray, std_gray = calculate_gray_mean_std(image_path)
    mean_red, std_red = calculate_red_mean_std(image_path)

    if mean_gray > 40 and std_gray > 50 and mean_red > 40 and std_red > 50:
        return "偏白且偏红"
    elif mean_gray > 40 and std_gray > 50:
        return "偏白"
    elif mean_red > 40 and std_red > 50:
        return "偏红"
    else:
        return "不偏白且不偏红"










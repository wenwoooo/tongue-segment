import torch
from PIL import Image
import os
import cv2
import numpy as np
from torchvision.utils import save_image


# 加载和预处理图像
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # 调整图像大小以适应模型输入要求
    image = cv2.resize(image, (512, 512))
    image_tougue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将图像转换为HSV颜色空间

    return image_tougue, image


# 预测图像中的舌体
def predict_tongue(image_path):
    tougue, _ = preprocess_image(image_path)
    lower_hsv = np.array([0, 10, 40])  # 最小HSV阈值
    upper_hsv = np.array([100, 200, 200])  # 最大HSV阈值
    mask = cv2.inRange(tougue, lower_hsv, upper_hsv)
    mask = cv2.bitwise_not(mask)
    # 执行膨胀操作
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    # 执行腐蚀操作
    erosion = cv2.erode(dilation, kernel, iterations=1)
    # 对掩膜进行开运算
    opened_mask = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)  # 掩膜进行开运算的目的是去除图像中的小的噪点和细小的非理想区域，以得到更干净、更准确的舌体分割结果。

    return opened_mask


# 舌体分割结果可视化
def visualize_segmentation(image, mask,i):
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    # segmented_image是一个numpy数组uint8
    cv2.imwrite(f"C:/Users/wen/Desktop/tiaoshibanben/tongue/notedata/sheti{i + 1}.png", segmented_image)
    # cv2.imshow("Segmentation Result", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

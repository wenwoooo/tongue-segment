import time

from torch.utils.data import dataset

from analysis import image_analysis
from data import *
from net import *
import train as tr
import numpy as np


net = UNet().cpu()  # 放在cuda上
weights = 'params/unet.pth'  # 导入网络

a = tr.MyDataset("C:/Users/wen/Desktop/tiaoshibanben/tongue/").__len__()
imgarr = np.arange(a)  # 计算标签文件中文件名的数量
for i in range(a):

    image_path = f'C:/Users/wen/Desktop/tiaoshibanben/tongue/ordata/sheti{i + 1}.JPG'  # 导入测试图片

    _, image = preprocess_image(image_path)  # 预处理了图像数据
    mask = predict_tongue(image_path)  # 预处理了舌体数据

    # 掩码合并图像
    visualize_segmentation(image, mask,i)

    _input = f'C:/Users/wen/Desktop/tiaoshibanben/tongue/notedata/sheti{i + 1}.png'
    img = keep_image_size_open(_input, size=(256, 256))
    img_data = transform(img)
    # print(img_data.shape)
    img_data = torch.unsqueeze(img_data, dim=0)
    # print(img_data)
    out = net(img_data)
    # print(out)
    save_image(img_data, f'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/orininal{i+1}.jpg')
    save_image(out, f'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/result{i+1}.jpg')

    img_before = Image.open(f'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/result{i+1}.jpg')
    img_after = Image.open(f'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/orininal{i+1}.jpg')

    # 把图像转成数组格式img = np.array(image)
    img_after_array = np.array(img_after)
    img_before_array = np.array(img_before)

    image_analysis(f'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/result{i+1}.jpg')
    print( image_analysis(f'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/result{i+1}.jpg'))

    # 转化为图片计算
    shape_after = img_after_array.shape
    shape_before = img_before_array.shape

    # print(shape_after, shape_before)

    # 将分隔好的图片进行对应像素点还原,即将黑白分隔图转化为有颜色的提取图
    if shape_after == shape_before:
        height = shape_after[0]
        width = shape_after[1]
        dst = np.zeros((height, width, 3))
        for h in range(0, height):
            for w in range(0, width):
                (b1, g1, r1) = img_after_array[h, w]
                (b2, g2, r2) = img_before_array[h, w]

                if (b1, g1, r1) <= (90, 90, 90):
                    img_before_array[h, w] = (144, 238, 144)
                dst[h, w] = img_before_array[h, w]
        img2 = Image.fromarray(np.uint8(dst))
        img2.save(rf"result\blend{i + 1}.png", "png")

    else:
        print("失败！")
    time.sleep(3)

print("successed")
print(dataset)









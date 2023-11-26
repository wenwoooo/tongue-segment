from net import *
from data import transform
from utils import *

net = UNet().cpu()  # 或者放在cuda上

weights = 'params/unet.pth'  # 导入网络

if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('success')
else:
    print('no loading')
i=4
image_path = 'C:/Users/wen/Desktop/tiaoshibanben/tongue/ordata/sheti5.JPG'  # 导入测试图片

_, image = preprocess_image(image_path)  # 预处理了图像数据
mask = predict_tongue(image_path)  # 预处理了舌体数据

# 掩码合并图像
visualize_segmentation(image, mask,i)

_input = f'C:/Users/wen/Desktop/tiaoshibanben/tongue/notedata/sheti{i+1}.png'  # 导入测试图片 开始模型训练
img = keep_image_size_open(_input)
img_data = transform(img)
img_data = torch.unsqueeze(img_data, dim=0)

out = net(img_data)

save_image(img_data, 'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/orininal5.jpg')
save_image(out, 'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/result5.jpg')

img_before = Image.open(r'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/result5.jpg')
img_after = Image.open(r'C:/Users/wen/Desktop/tiaoshibanben/tongue/result/orininal5.jpg')

img_before_array = np.array(img_before)
img_after_array = np.array(img_after)  # 把图像转成数组格式img = np.asarray(image)

shape_before = img_before_array.shape
shape_after = img_after_array.shape
print(shape_after, shape_before)

if shape_after == shape_before:  # 拼接测试图片
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
    img2.save(r"result\blend5.png", "png")

else:
    print("失败！")

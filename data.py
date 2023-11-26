import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):  # 拿到标签文件夹中图片的名字
        self.path = path
        self.name = os.listdir(os.path.join(path, 'notedata'))

    def __len__(self):  # 计算标签文件中文件名的数量
        return len(self.name)

    def __getitem__(self, index):  # 将标签文件夹中的文件名在原图文件夹中进行匹配（由于标签是png的格式而原图是jpg所以需要进行一个转化）
        segment_name = self.name[index]  # XX.png
        segment_path = os.path.join(self.path, 'notedata', segment_name)
        image_path = os.path.join(self.path, 'ordata', segment_name.replace('png', 'jpg'))  # png与jpg进行转化

        segment_image = keep_image_size_open(segment_path)  # 等比例缩放
        image = keep_image_size_open(image_path)  # 等比例缩放

        return transform(image), transform(segment_image)


if __name__ == "__main__":
    data = MyDataset("C:/Users/wen/Desktop/tiaoshibanben/tongue/")
    # print(len(data))   27
    print(data[0][0].shape)
    print(data[0][1].shape)


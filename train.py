from h5py._hl import dataset
from torch import optim
from data import *
from net import *
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data import MyDataset
from net import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = 'C:/Users/wen/Desktop/tiaoshibanben/tongue'
save_path = 'train_image'

if __name__ == "__main__":

    dic = []  ###

    data_loader = DataLoader(MyDataset(data_path), batch_size=1)  # batch_size用3/4 都可以看电脑性能 每个图片分为一个批次
    # print(MyDataset(data_path).__len__())  # 手动传递实例作为参数
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print('success load weight')
    else:
        print('not success load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1
    while True:
        avg = []  ###
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()


            if i % 5 == 0:
                print('{}-{}-train_loss===>>{}'.format(epoch, i, train_loss.item()))

            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)

            # 为方便看效果将原图、标签图、训练图进行拼接
            _image = image[0]
            _out_image = out_image[0]
            _segment_image = segment_image[0]

            img = torch.stack([_image, _out_image, _segment_image], dim=0)
            save_image(img, f'{save_path}/sheti{i+1}.jpg')

            avg.append(float(train_loss.item()))  ###

        loss_avg = sum(avg) / len(avg)

        dic.append(loss_avg)

        epoch += 1
    print(dic)

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, models
import os
import torch.hub


def rightness(output, target):
    preds = output.data.max(dim=1, keepdim=True)[1]
    return preds.eq(target.data.view_as(preds)).cpu().sum(), len(target)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1]),0))
            imgs.append((words[0], int(words[1]),1))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        # self.rot_ornot = rot_ornot

    def __getitem__(self, index):
        fn, label, rot_ornot = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        if rot_ornot==1:
            img = img.rotate(90, resample=False, expand=False, center=None)
        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

def main():
# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    os.environ['CUDA_VISIBLE_DEVICES']='5,3'
    root = "/home/lsm/BalancedSamples/"
    train_data_normal = MyDataset(root, 'train.txt',
                           transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ]))
    # train_data_rot90 = MyDataset(root, 'train.txt',
    #                        transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                            transforms.ToTensor(),
    #                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                            ]))
    test_data = MyDataset(root, 'test.txt',
                          transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ]))

# print(len(test_data))
##num_classes = len(train_data.classes)
# print(test_data.imgs[1])

    train_loader_normal = torch.utils.data.DataLoader(train_data_normal, batch_size=4, shuffle=True, num_workers=2)
    # train_loader_rot90 = torch.utils.data.DataLoader(train_data_rot90, batch_size=4, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)

    num_classes = 8
    net = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)
    # net = models.resnet50(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    record = []
    num_epoch = 40
    net=net.cuda()
    net.train(True)
    perf_max = 0
    train_max = 0
    for epoch in range(num_epoch):
        train_rights = []
        train_losses = []
        print("epoch " + str(epoch) + " running...")
        for batch_idx, (data, target) in enumerate(train_loader_normal):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(),target.cuda()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.cpu()
            right = rightness(output, target)
            train_rights.append(right)
            train_losses.append(loss.data.numpy())

        #    if batch_idx%100==0:
        net.eval()
        val_rights = []
        for (data, target) in test_loader:
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            output = net(data)
            right = rightness(output, target)
            val_rights.append(right)
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
        record.append((100. * train_r[0] / train_r[1], 100. * val_r[0] / val_r[1]))
        print("epoch " + str(epoch) + ": train_perf " + str(100. * float(train_r[0]) / train_r[1]) + "%; test_perf " + str(100. * float(val_r[0]) / val_r[1]) + "%")

        if val_r[0] == perf_max:
            if train_r[0] > train_max:
                state = {
                    'state_dict': net.state_dict()
                }
                torch.save(state, 'checkpoint_file_name')
        if val_r[0] > perf_max:
            perf_max = val_r[0]
            if train_r[0] > train_max:
                train_max = train_r[0]
            state = {
                'state_dict':net.state_dict()
            }
            torch.save(state, 'checkpoint_file_name')


if __name__ == '__main__':
    main()

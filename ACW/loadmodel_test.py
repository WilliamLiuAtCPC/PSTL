from PIL import Image
import torch
import torch.nn as nn
#import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms,models
import matplotlib.pyplot as plt
import numpy as np
import torch.hub
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def rightness(output, target):
    preds = output.data.max(dim=1, keepdim=True)[1]
    return preds.eq(target.data.view_as(preds)).cpu().sum(), len(target)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root, datatxt, transform=None, target_transform=None): #初始化一些需要传入的参数
        super(MyDataset,self).__init__()
        fh = open(root + datatxt, 'r') #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs=imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        fn, label = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB') #按照path读入图片from PIL import Image # 按照路径读取图片
 
        if self.transform is not None:
            img = self.transform(img) #是否进行transform
        return img,label
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
 
def showimg(data):
    img = np.asarray(data)#, dtype=float
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.pause(0.01)

if __name__ == '__main__':

    root = "/home/lsm/BalancedSamples/"
    # train_data=MyDataset(root,'train.txt', transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                                            transforms.ToTensor(),
    #                                                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    #                                                            ]))
    test_data=MyDataset(root,'test.txt', transforms.Compose([transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                               ]))

    # train_loader = torch.utils.data.DataLoader(train_data,batch_size = 4, shuffle=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size = 4, shuffle=True,num_workers=4)
    # test_loader = torch.load('test_loader')
    checkpoint = torch.load('checkpoint_file_name')
    num_classes = 8
    # net = models.resnet50(pretrained=True)
    net = torch.hub.load(
        'moskomule/senet.pytorch',
        'se_resnet50',
        pretrained=True, )
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    net = net.cuda()
    val_rights=[]
    print("net loaded")
    
    for (data,target) in test_loader:
        data,target=Variable(data),Variable(target)
        data, target = data.cuda(), target.cuda()
        output=net(data)
        right=rightness(output,target)
        val_rights.append(right)
        
    val_r=(sum([tup[0] for tup in val_rights]),sum([tup[1] for tup in val_rights]))
    print(" test_perf "+str(val_r[0])+" "+str(val_r[1]))
    print(float(val_r[0])/val_r[1])
from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
# from PIL import Image
from .util import read_image


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class MyDataset(t.utils.data.Dataset):#change to our data type
    def __init__(self, root, datatxt,opt):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:
            line = line.rstrip('\n')
            words = line.split()
            imgs.append(words)
        self.imgs = imgs
        self.tsf = Transform(opt.min_size, opt.max_size)
        self.root = root
        fh.close()

    def __getitem__(self, index):
        words = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        fn = words[0]
        ori_img = read_image(self.root+fn, color=True)
        # img = Image.open(self.root+fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        words = words[1:]
        it = int(len(words)/5)
        bbox = []
        label = []
        for i in range(it):
            toolname = words[i*5]
            bbox.append([int(words[i*5+2]),int(words[i*5+1]),int(words[i*5+4]),int(words[i*5+3])])
            label.append(TOOL_LABEL_NAMES.index(toolname))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # print(bbox)
        # print(label)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


class TestDataset(t.utils.data.Dataset):
    def __init__(self, root, datatxt, opt):  # 初始化一些需要传入的参数
        super(TestDataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:
            line = line.rstrip('\n')
            words = line.split()
            imgs.append(words)
        self.imgs = imgs
        self.tsf = Transform(opt.min_size, opt.max_size)
        self.root = root
        fh.close()

    def __getitem__(self, index):
        words = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        fn = words[0]
        ori_img = read_image(self.root+fn, color=True)
        # img = Image.open(self.root+fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        words = words[1:]
        it = int(len(words)/5)
        bbox = []
        label = []
        for i in range(it):
            toolname = words[i*5]
            bbox.append([int(words[i*5+2]),int(words[i*5+1]),int(words[i*5+4]),int(words[i*5+3])])
            label.append(TOOL_LABEL_NAMES.index(toolname))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # print(bbox)
        # print(label)
        # img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        difficult = [0]
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.imgs)

class Test700Dataset(t.utils.data.Dataset):
    def __init__(self, root, datatxt, opt):  # 初始化一些需要传入的参数
        super(Test700Dataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:
            line = line.rstrip('\n')
            words = line.split()
            imgs.append(words)
        self.imgs = imgs
        self.tsf = Transform(opt.min_size, opt.max_size)
        self.root = root
        fh.close()

    def __getitem__(self, index):
        words = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        fn = words[0]
        ori_img = read_image(self.root+fn, color=True)
        bbox = []
        bbox.append([int(words[2]),int(words[1]),int(words[4]),int(words[3])])
        bbox = np.stack(bbox).astype(np.float32)
        img = preprocess(ori_img)
        return fn, img, ori_img.shape[1:], bbox

    def __len__(self):
        return len(self.imgs)

class RawDataset(t.utils.data.Dataset):
    def __init__(self, root, datatxt, opt):  # 初始化一些需要传入的参数
        super(RawDataset, self).__init__()
        fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:
            line = line.rstrip('\n')
            words = line.split()
            imgs.append(words)
        self.imgs = imgs
        self.tsf = Transform(opt.min_size, opt.max_size)
        self.root = root
        fh.close()

    def __getitem__(self, index):
        words = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        fn = words[0]
        ori_img = read_image(self.root+fn, color=True)
        # img = Image.open(self.root+fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        # words = words[1:]
        # it = int(len(words)/5)

        img = preprocess(ori_img)
        return fn, img, ori_img.shape[1:]

    def __len__(self):
        return len(self.imgs)

TOOL_LABEL_NAMES = (
    'Grasper',
    'Bipolar',
    'Hook',
    'Scissors',
    'Clipper',
    'Irrigator',
    'SpecimenBag',
    )

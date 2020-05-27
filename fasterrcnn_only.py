from torch.utils import data as data_
import os
import torch as t
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data.mydataset import Test700Dataset
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]='5'

toolNameList = ["Background","Grasper","Bipolar","Hook","Scissors","Clipper","Irrigator","SpecimenBag"]

def weighedMean(box_list):
    new_box = []
    score_list = []
    score_sum = 0
    #get weight
    for i in range(len(box_list)):
        score_list.append(box_list[i][0])
        score_sum += box_list[i][0]
    for i in range(len(score_list)):
        score_list[i]/=score_sum
    #weighed sum
    for i in range(len(box_list[0])):
        coordinate=0
        for j in range(len(box_list)):
            coordinate+=box_list[j][i]*score_list[j]
        new_box.append(coordinate)
#    print(new_box)
    return new_box

def fuseBoxes(words,num_box):
    new_boxes = []
    label_list = []
    for i in range(num_box):
        label_list.append(words[i*6])
    label_list = list(set(label_list))
    for tool in label_list:
        box_list = []
        for i in range(num_box):
            if words[i*6]==tool:
                box = []
                for j in range(5):
                    box.append(float(words[i*6+j+1]))
                box_list.append(box)
        new_box = weighedMean(box_list)
#        new_box = directMean(box_list)
        new_box[0] = tool
#        print(new_box)
        new_boxes.append(new_box)
    return new_boxes

def writeResult(result_file,new_boxes,currentAxis):
    for box in new_boxes:
        corrdinates = []
        for content in box:
            corrdinates.append(content)
            result_file.write(" "+str(content))
        x1, y1, x2, y2 = corrdinates[1], corrdinates[2], corrdinates[3], corrdinates[4]
        plt.text(x1, y1, corrdinates[0], size=15, color='r')
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=2)
        currentAxis.add_patch(rect)
    result_file.write("\n")

def isTool(img,net):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])
    img = transform(img)
    imgs = [img,]
    test_loader = t.utils.data.DataLoader(imgs, batch_size=1, shuffle=False, num_workers=1)
    for data in test_loader:
        data = Variable(data)
        data = data.cuda()
        output = net(data)
    return int(output.argmax())

def vote(tool_freq,net,img):
    first_look = int(tool_freq.argmax())
    if first_look != 0:
        return first_look
    else:
        flag = 0
        for i in range(len(tool_freq)-1):
            if tool_freq[i+1] != 0:
                flag = 1
                break
        if flag == 0:
            return first_look
        else:
            # img = img.resize((224,224))
            # decision = isTool(img,net)
            tool_freq[first_look]=0
            first_look = int(tool_freq.argmax())
            return first_look

def decide2(net,img):
    img = img.resize((224, 224))
    return isTool(img,net)


def decide(net,img,step):
    tl_corner_position = [0,0] #(x,y)
    w = img.size[0]
    h = img.size[1]

    if w<224:
        img = img.resize((224, round(h / w * 224)))
        w = img.size[0]
        h = img.size[1]
    if h<224:
        img = img.resize((round(w / h * 224),224 ))
        w = img.size[0]
        h = img.size[1]

    tool_freq = np.zeros(8)
    while (tl_corner_position[1]+224<h):
        while (tl_corner_position[0]+224<w):
            region = img.crop((tl_corner_position[0],tl_corner_position[1],tl_corner_position[0]+224,tl_corner_position[1]+224))
            flag = isTool(region,net)
            tool_freq[flag]+=1
            tl_corner_position[0] = tl_corner_position[0]+step[0] #右移slider
        region = img.crop((w-224,tl_corner_position[1],w,tl_corner_position[1]+224))
        flag = isTool(region,net)
        tool_freq[flag]+=1
        tl_corner_position[1] = tl_corner_position[1]+step[1] #下移slider
        tl_corner_position[0] = 0
    tl_corner_position[1] = h - 224
    while (tl_corner_position[0]+224<w):
        region = img.crop((tl_corner_position[0],tl_corner_position[1],tl_corner_position[0]+224,tl_corner_position[1]+224))
        flag = isTool(region,net)
        tool_freq[flag]+=1
        tl_corner_position[0] = tl_corner_position[0]+step[0] #右移slider
    region = img.crop((w-224,tl_corner_position[1],w,tl_corner_position[1]+224))
    flag = isTool(region,net)
    tool_freq[flag]+=1
    result = vote(tool_freq,net,img)
    return result

def makeDir():
    # if os.path.exists('./result/'):
    #     os.removedirs('./result/')
    # else:
        os.mkdir('./result/')
        os.mkdir('./result/bbox/')
        os.mkdir('./result/bbox/1.Grasper')
        os.mkdir('./result/bbox/2.Bipolar')
        os.mkdir('./result/bbox/3.Hook')
        os.mkdir('./result/bbox/4.Scissors')
        os.mkdir('./result/bbox/5.Clipper')
        os.mkdir('./result/bbox/6.Irrigator')
        os.mkdir('./result/bbox/7.SpecimenBag')
        os.mkdir('./result/mask/')
        os.mkdir('./result/mask/1.Grasper')
        os.mkdir('./result/mask/2.Bipolar')
        os.mkdir('./result/mask/3.Hook')
        os.mkdir('./result/mask/4.Scissors')
        os.mkdir('./result/mask/5.Clipper')
        os.mkdir('./result/mask/6.Irrigator')
        os.mkdir('./result/mask/7.SpecimenBag')

def drawBbox(data_root,fn,bboxes,save_root):
    img = Image.open(data_root+fn).convert("RGB")
    plt.imshow(img)
    currentAxis = plt.gca()
    for bbox in bboxes[0]:
        # print(bbox)
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], fill=False,
                                 edgecolor='r', linewidth=2)
        currentAxis.add_patch(rect)
    plt.savefig(save_root + fn)
    plt.close()

def main(**kwargs):
    opt._parse(kwargs)
    # checkpoint = t.load('se_0314_all')
    # classifier = t.hub.load(
    #         'moskomule/senet.pytorch',
    #         'se_resnet50',
    #         pretrained=True, )
    checkpoint = t.load('res50_0314_all')
    classifier = models.resnet50()
    num_classes = 8
    step = [112, 112]

    num_ftrs = classifier.fc.in_features
    classifier.fc = nn.Linear(num_ftrs, num_classes)
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.eval()
    classifier = classifier.cuda()
    result_file = open('result0520_fasterrcnn.txt', 'w')
    save_root = './result/bbox/'
    makeDir()

    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('checkpoints/fasterrcnn_04081709_0.6626689194895079')

    data_root = '/home/lsm/testSamples700_new/'
    test_file = 'GT707.txt'
    test700 = Test700Dataset(data_root,test_file,opt)
    test_dataloader = data_.DataLoader(test700,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    print('data loaded!')
    for ii, (fn, imgs, sizes, gt_bboxes_) in tqdm(enumerate(test_dataloader)):
        # print(gt_bboxes_)
        gt_x1 = int(gt_bboxes_[0][0][1])
        gt_y1 = int(gt_bboxes_[0][0][0])
        gt_x2 = int(gt_bboxes_[0][0][3])
        gt_y2 = int(gt_bboxes_[0][0][2])
        # print([gt_x1,gt_y1,gt_x2,gt_y2])
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        result_file.write(fn[0])
        # drawBbox(data_root,fn[0],pred_bboxes_,save_root)
        img = Image.open(data_root + fn[0]).convert("RGB")
        plt.imshow(img)
        currentAxis = plt.gca()
        line = fn[0]
        for i in range(len(pred_bboxes_[0])):
            bbox = pred_bboxes_[0][i]
            score = pred_scores_[0][i]
            label = pred_labels_[0][i]
            x1, y1, x2, y2 = bbox[1], bbox[0], bbox[3], bbox[2]
                # plt.text(x1, y1, toolNameList[decision]+" "+str(score), size=15, color='r')
            line = line + ' ' + toolNameList[label+1]+ ' ' + str(score) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2)
        words = line.split()
        # result_file.write(words[0])
        num_box = int((len(words) - 1) / 6)
        if num_box > 0:
            new_boxes = fuseBoxes(words[1:], num_box)
            writeResult(result_file, new_boxes,currentAxis)

        rect = patches.Rectangle((gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1, fill=False, edgecolor='g', linewidth=2)
        currentAxis.add_patch(rect)
        plt.savefig(save_root + fn[0])
        plt.close()
    result_file.close()




if __name__ == '__main__':
    main()

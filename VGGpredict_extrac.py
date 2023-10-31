import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import json
import scipy.io as io
import random


# # # #
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG = models.vgg19_bn(pretrained=True)
        num_ftrs = VGG.classifier[6].in_features
        VGG.classifier[6] = nn.Linear(num_ftrs, 10)
        self.feature = VGG.features
        # self.feature.requires_grad =False
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.avgpool.requires_grad = False
        self.classifier = nn.Sequential(*list(VGG.classifier.children()))
        # pretrained_dict=torch.load('./data/new_feature/model_data/2020-09-07 15:15:10.006644/480_0.48')
        # model_dict = self.classifier.state_dict()
        # class_dict = {k[len('classifier.'):]: v for k, v in pretrained_dict.items() if k[:len('classifier.')]=='classifier.' and k[len('classifier.'):] in model_dict}
        # model_dict.update(class_dict)
        # self.classifier.load_state_dict(model_dict)
        # model_dict = self.feature.state_dict()
        # feature_dict = {k[len('feature.'):]: v for k, v in pretrained_dict.items() if
        #               k[:len('feature.')] == 'feature.' and k[len('feature.'):] in model_dict}
        # model_dict.update(feature_dict)
        # self.feature.load_state_dict(model_dict)


    def forward(self, x):
        # output = self.feature(x)
        # output = self.avgpool(output)
        # output = torch.flatten(output, 1)
        output = self.classifier(x)
        return output


# # #
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         VGG = models.vgg19_bn(pretrained=False)
#         num_ftrs = VGG.classifier[6].in_features
#         VGG.classifier[6] = nn.Linear(num_ftrs, 10)
#         self.feature = VGG.features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-2])
#         pretrained_dict=torch.load('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/2021-03-26 14:38:35.206539/7_0.8507826086956523')
#         model_dict = self.classifier.state_dict()
#         class_dict = {k[len('classifier.'):]: v for k, v in pretrained_dict.items() if k[:len('classifier.')]=='classifier.' and k[len('classifier.'):] in model_dict}
#         model_dict.update(class_dict)
#         self.classifier.load_state_dict(model_dict)
#         model_dict = self.feature.state_dict()
#         feature_dict = {k[len('feature.'):]: v for k, v in pretrained_dict.items() if
#                       k[:len('feature.')] == 'feature.' and k[len('feature.'):] in model_dict}
#         model_dict.update(feature_dict)
#         self.feature.load_state_dict(model_dict)
#
#
#     def forward(self, x):
#         output = self.feature(x)
#         output = self.avgpool(output)
#         output = torch.flatten(output, 1)
#         output = self.classifier(output)
#         return output



#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         resnet = models.resnet50(pretrained=False)
#         self.conv1=resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4
#         self.avgpool = resnet.avgpool
#         pretrained_dict = torch.load('/home/xutianyuan/papers/DSCMR/data/model/2020-11-19 20:58:12.156848/14_0.512')
#
#         model_dict = self.conv1.state_dict()
#         conv1_dict = {k[len('conv1.'):]: v for k, v in pretrained_dict.items() if
#                       k[:len('conv1.')] == 'conv1.' and k[len('conv1.'):] in model_dict}
#         model_dict.update(conv1_dict)
#         self.conv1.load_state_dict(model_dict)
#
#         model_dict = self.bn1.state_dict()
#         bn1_dict = {k[len('bn1.'):]: v for k, v in pretrained_dict.items() if
#                       k[:len('bn1.')] == 'bn1.' and k[len('bn1.'):] in model_dict}
#         model_dict.update(bn1_dict)
#         self.bn1.load_state_dict(model_dict)
#
#         model_dict = self.layer1.state_dict()
#         layer1_dict = {k[len('layer1.'):]: v for k, v in pretrained_dict.items() if
#                       k[:len('layer1.')] == 'layer1.' and k[len('layer1.'):] in model_dict}
#         model_dict.update(layer1_dict)
#         self.layer1.load_state_dict(model_dict)
#
#         model_dict = self.layer2.state_dict()
#         layer2_dict = {k[len('layer2.'):]: v for k, v in pretrained_dict.items() if
#                        k[:len('layer2.')] == 'layer2.' and k[len('layer2.'):] in model_dict}
#         model_dict.update(layer2_dict)
#         self.layer2.load_state_dict(model_dict)
#
#         model_dict = self.layer3.state_dict()
#         layer3_dict = {k[len('layer3.'):]: v for k, v in pretrained_dict.items() if
#                        k[:len('layer3.')] == 'layer3.' and k[len('layer3.'):] in model_dict}
#         model_dict.update(layer3_dict)
#         self.layer3.load_state_dict(model_dict)
#
#         model_dict = self.layer4.state_dict()
#         layer4_dict = {k[len('layer4.'):]: v for k, v in pretrained_dict.items() if
#                        k[:len('layer4.')] == 'layer4.' and k[len('layer4.'):] in model_dict}
#         model_dict.update(layer4_dict)
#         self.layer4.load_state_dict(model_dict)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return x

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         self.conv1=resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4
#         self.avgpool = resnet.avgpool
#         self.fc = nn.Linear(2048, 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
# #
#         return x
#

def extractor(img_path, net, use_gpu):
    net.eval()
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path)
    if img.mode!='RGB':
        img=img.convert("RGB")
    img = transform(img)
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    return y



if __name__ == '__main__':

    # img_dir = '/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/images'
    # data_dir = '/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/feature'
    # f = open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/sentence_label_train.txt', 'r')
    # sentences_train = json.load(f)
    # f.close()
    # f = open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/sentence_label_test.txt', 'r')
    # sentences_test = json.load(f)
    # f.close()
    # pic_fea = []
    # model = Encoder()
    # for item in sentences_train:
    #     img_name = os.path.join(img_dir,item['img'])
    #     print(img_name)
    #     feature = extractor(img_name, model, True)
    #     pic_fea.append(feature)
    # io.savemat(os.path.join(data_dir, 'train_img_resnet.mat'), {'name': pic_fea})
    # pic_fea = []
    # for item in sentences_test:
    #     img_name = os.path.join(img_dir,item['img'])
    #     print(img_name)
    #     feature = extractor(img_name, model, True)
    #     pic_fea.append(feature)
    # io.savemat(os.path.join(data_dir, 'test_img_resnet.mat'), {'name': pic_fea})
    # img_dir = '/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia9/vaild_img'
    img_dir = '/home/xutianyuan/papers/DSCMR/data/nus-wide/image'
    data_dir = '/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result'
    f = open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_train.txt', 'r')
    sentences_train = json.load(f)
    f.close()
    f = open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_test.txt', 'r')
    sentences_test = json.load(f)
    f.close()
    f = open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_valid.txt', 'r')
    sentences_valid = json.load(f)
    f.close()
    pic_fea = []
    model = Encoder()
    model.eval()
    cccc=0
    for item in sentences_train:
        cccc+=1
        print(cccc)
        img_name = os.path.join(img_dir, item['img'])
        feature = extractor(img_name, model, True)
        pic_fea.append(feature)
    x=np.asarray(pic_fea)
    x=np.resize(x, (len(pic_fea), 4096))

    train_x = io.loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/train_img.mat')
    io.savemat(os.path.join(data_dir, 'train_img1.mat'), {'name': np.concatenate((train_x['name'], x))})
    pic_fea = []
    for item in sentences_test:
        cccc+=1
        print(cccc)
        img_name = os.path.join(img_dir, item['img'])
        feature = extractor(img_name, model, True)
        pic_fea.append(feature)
    x=np.asarray(pic_fea)
    x=np.resize(x, (len(pic_fea), 4096))

    test_x = io.loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/test_img.mat')
    io.savemat(os.path.join(data_dir, 'test_img1.mat'), {'name': np.concatenate((test_x['name'], x))})

    pic_fea = []
    for item in sentences_valid:
        cccc+=1
        print(cccc)
        img_name = os.path.join(img_dir, item['img'])
        feature = extractor(img_name, model, True)
        pic_fea.append(feature)
    x = np.asarray(pic_fea)
    x = np.resize(x, (len(pic_fea),4096))

    io.savemat(os.path.join(data_dir, 'valid_img1.mat'), {'name': x})

    #
    # img_dir = 'data/nus-wide/image'
    # data_dir = '/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus/feature'
    # f = open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus/data/train_data.txt', 'r')
    # sentences_train = json.load(f)
    # f.close()
    # f = open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus/data/test_data.txt', 'r')
    # sentences_test = json.load(f)
    # f.close()
    # pic_fea = []
    # model = Encoder()
    # for item in sentences_train:
    #     img_name = os.path.join(img_dir,item['img'])
    #     print(img_name)
    #     feature = extractor(img_name, model, True)
    #     pic_fea.append(feature)
    # io.savemat(os.path.join(data_dir, 'train_img_resnet1.mat'), {'name': pic_fea})
    # pic_fea = []
    # for item in sentences_test:
    #     img_name = os.path.join(img_dir,item['img'])
    #     print(img_name)
    #     feature = extractor(img_name, model, True)
    #     pic_fea.append(feature)
    # io.savemat(os.path.join(data_dir, 'test_img_resnet1.mat'), {'name': pic_fea})


    # data_dir = 'data/mirflickr25k/feature'
    # f = open('/home/xutianyuan/papers/DSCMR/data/mirflickr25k/train_data.txt', 'r')
    # sentences_train = json.load(f)
    # f.close()
    # f = open('/home/xutianyuan/papers/DSCMR/data/mirflickr25k/test_data.txt', 'r')
    # sentences_test = json.load(f)
    # f.close()
    # pic_fea = []
    # model = Encoder()
    # for item in sentences_train:
    #     img_name = item['img']
    #     print(img_name)
    #     feature = extractor(img_name, model, True)
    #     pic_fea.append(feature)
    # io.savemat(os.path.join(data_dir, 'train_img_resnet_20.mat'), {'name': pic_fea})
    # pic_fea = []
    # for item in sentences_test:
    #     img_name =item['img']
    #     print(img_name)
    #     feature = extractor(img_name, model, True)
    #     pic_fea.append(feature)
    # io.savemat(os.path.join(data_dir, 'test_img_resnet_20.mat'), {'name': pic_fea})
    #

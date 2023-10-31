# from os.path import join,exists
# from os import listdir, makedirs
# import numpy as np
# import json
# import random
# P = "data/mirflickr25k"
# BASE = join(P, 'mirflickr')
# IMG_P = BASE  # image 路径
# TXT_P = join(BASE, 'meta/tags')  #  text 路径
# LAB_P = join(P, 'mirflickr25k_annotations_v080')
# COM_TAG_F = join(BASE, 'doc/common_tags.txt')  # common tags
# N_DATA = 25000
#
# # label 文件列表
# key_lab = lambda s: s.split('.txt')[0]  # 按类名字典序升序
# fs_lab = [s for s in listdir(LAB_P) if "README" not in s and "_r1" not in s]
# fs_lab = sorted(fs_lab, key=key_lab)
# label = {value.split('.txt')[0]: key for key, value in enumerate(fs_lab)}
# fs_lab = [join(LAB_P, s) for s in fs_lab]
# N_CLASS = len(fs_lab)
#
#
# def sample_of_lab(lab_f):
#     """读 annotation 文件，获取属于该类的 samples 标号"""
#     samples = []
#     with open(lab_f, 'r') as f:
#         for line in f:
#             sid = int(line)
#             samples.append(sid)
#     return samples
# # 处理 label
# all_lab = np.zeros((N_DATA, N_CLASS))
# for i in range(len(fs_lab)):
#     samp_ls = sample_of_lab(fs_lab[i])
#     for s in samp_ls:
#         all_lab[s - 1][i] = 1  # s-th 样本属于 i-th 类
#
#
# # image 文件列表
# fs_img = [f for f in listdir(IMG_P) if '.jpg' in f]
# fs_img = {int(f.split('.jpg')[0].split('im')[-1])-1:join(IMG_P, f) for f in fs_img}
#
# # 处理 common tags
# tag_idx, idx_tag = {}, {}
# cnt = 0
# with open(COM_TAG_F, 'r') as f:
#     for line in f:
#         line = line.split()
#         tag_idx[line[0]] = cnt
#         idx_tag[cnt] = line[0]
#         cnt += 1
#
# # text 文件列表
# def get_tags(tag_f):
#     f=open(tag_f, 'r')
#     tg=f.read().split()
#     tg=[ item for item in tg if tag_idx.get(item)]
#     return ' '.join(tg)
# fs_tags={int(f.split('.txt')[0].split('tags')[-1])-1:get_tags(join(TXT_P, f)) for f in listdir(TXT_P)}
#
# all_data=[]
# for id, item in enumerate(all_lab):
#     lab_child=[ index for index,value in enumerate(item) if value]
#     if len(lab_child) and exists(fs_img[id]) and len(fs_tags[id].split())>0:
#         lab_sing=lab_child[random.randint(0,len(lab_child)-1)]
#         all_data.append({'id':id,'img':fs_img[id],'text':fs_tags[id],'label':lab_sing})
# f=open('data/mirflickr25k/all_data.txt','w')
# f.write(json.dumps(all_data))
# f.close()
# f=open('data/mirflickr25k/label.txt','w')
# f.write(json.dumps(label))
# f.close()
# import json,random
# import collections
# f=open('data/mirflickr25k/all_data.txt')
# all_data=json.load(f)
# dict_label=collections.defaultdict(list)
# [dict_label[item['label']].append(item) for item in all_data]
# final_data=[]
# remain_data=[]
# for key,value in dict_label.items():
#     if len(value)<100:
#         final_data.extend(value)
#     elif len(value)<500:
#         final_data.extend([item for item in value if len(item['text'].split()) > 4])
#     else:
#         final_data.extend([item for item in value if len(item['text'].split()) > 5])
#
# dict_label=collections.defaultdict(list)
# [dict_label[item['label']].append(item) for item in final_data]
# f=open('data/mirflickr25k/final_data.txt','w')
# f.write(json.dumps(final_data))
# f.close()
# # # pass
#
# import os,random,json,collections
# img_dir='data/final_all_data_info/pascal/data/images'
# cp_dir='data/final_all_data_info/pascal/data/img_all'
# label_dir='./data/final_all_data_info/pascal/data/label.txt'
# f=open('data/final_all_data_info/pascal/data/sentence_label_train.txt')
# all_data=json.load(f)
# f.close()
# f=open('data/final_all_data_info/pascal/data/sentence_label_test.txt')
# all_data.extend(json.load(f))
# random.shuffle(all_data)
# f.close()
# f=open(label_dir)
# label_number=json.load(f)
# label_number = {value:key for key,value in label_number.items()}
# f.close()
# for item in all_data[:800]:
#     train_label ='train/' +  label_number[item['label']]
#     train_label_dir = os.path.join(cp_dir, train_label)
#     test_label = 'val/' +label_number[item['label']]
#     test_label_dir = os.path.join(cp_dir, test_label)
#     if not os.path.exists(train_label_dir):
#         os.mkdir(train_label_dir)
#     if not os.path.exists(test_label_dir):
#         os.mkdir(test_label_dir)
#     origin_path=os.path.join(img_dir,item['img'])
#     os.system("cp {} {}".format(origin_path, train_label_dir))
# for item in all_data[800:]:
#     train_label ='train/' +  label_number[item['label']]
#     train_label_dir = os.path.join(cp_dir, train_label)
#     test_label = 'val/' +label_number[item['label']]
#     test_label_dir = os.path.join(cp_dir, test_label)
#     if not os.path.exists(train_label_dir):
#         os.mkdir(train_label_dir)
#     if not os.path.exists(test_label_dir):
#         os.mkdir(test_label_dir)
#     origin_path=os.path.join(img_dir,item['img'])
#     os.system("cp {} {}".format(origin_path, test_label_dir))
# import os,random,json
# f=open('data/mirflickr25k/final_data.txt')
# all_data=json.load(f)
# random.shuffle(all_data)
# f.close()
# label_total={6: 40, 8: 20, 21: 30, 9: 80, 4: 50, 23: 40, 18: 80, 13: 80, 20: 30, 19: 84, 7: 30, 0: 40, 14: 90, 10: 15, 11: 50, 3: 10, 22: 45, 15: 30, 12: 40, 5: 10, 1: 6, 17: 15, 2: 10, 16: 10}
#
# train_val=[]
# test_val=[]
# for item in all_data:
#     if label_total[item['label']]:
#         label_total[item['label']]-=1
#         test_val.append(item)
#     else:
#         train_val.append(item)
# f=open('data/mirflickr25k/train_data.txt','w')
# f.write(json.dumps(train_val))
# f.close()
# f=open('data/mirflickr25k/test_data.txt','w')
# f.write(json.dumps(test_val))
# f.close()
#

import os,random,json,collections
img_dir='/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/valid_img'
cp_dir='/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/img_train_val1'
label_dir='./data/final_all_data_info/pascal/data/label.txt'
f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/train_data.txt')
all_data=json.load(f)
f.close()
f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/test_data.txt')
all_data.extend(json.load(f))
f.close()
f=open(label_dir)
label_number=json.load(f)
label_number = {value:key for key,value in label_number.items()}
f.close()
for item in all_data[:800]:
    train_label ='train/' +  label_number[item['label']]
    train_label_dir = os.path.join(cp_dir, train_label)
    test_label = 'val/' +label_number[item['label']]
    test_label_dir = os.path.join(cp_dir, test_label)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    if not os.path.exists(test_label_dir):
        os.mkdir(test_label_dir)
    origin_path=os.path.join(img_dir,item['img'])
    os.system("cp {} {}".format(origin_path, train_label_dir))
for item in all_data[800:]:
    train_label ='train/' +  label_number[item['label']]
    train_label_dir = os.path.join(cp_dir, train_label)
    test_label = 'val/' +label_number[item['label']]
    test_label_dir = os.path.join(cp_dir, test_label)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    if not os.path.exists(test_label_dir):
        os.mkdir(test_label_dir)
    origin_path=os.path.join(img_dir,item['img'])
    os.system("cp {} {}".format(origin_path, test_label_dir))




import numpy as np
import json
import scipy.io as io
f=open('data/final_all_data_info/pascal/data/sentence_label_train.txt')
train_data=json.load(f)
f.close()
f=open('data/final_all_data_info/pascal/data/sentence_label_test.txt')
test_data=json.load(f)
f.close()
io.savemat('data/final_all_data_info/pascal/feature/train_label_resnet.mat',{'name':
np.resize(np.array([item['label'] for item in train_data]),(800,1))})
io.savemat('data/final_all_data_info/pascal/feature/test_label_resnet.mat',{'name':
np.resize(np.array([item['label'] for item in test_data]),(100,1))})

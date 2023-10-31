import os
import os,random,json,collections
import random,json

import json
import numpy as np
import scipy.io as io


#
# #
# # dir='./data/nus-wide/Groundtruth/'
# # valid_label=dir+'AllLabels'
# # txt_dir='./data/nus-wide/NUS_WID_Tags/All_Tags.txt'
# # label_dir='./data/nus-wide/label.txt'
# # tar1k_dir='./data/nus-wide/NUS_WID_Tags/AllTags1k.txt'
# #
# # def get_nus_tag_map():
# #     with open('./data/nus-wide/NUS_WID_Tags/Final_Tag_List.txt') as f:
# #         tag_list = f.readlines()[:1000]
# #     tag_list = list(map(lambda x: x.strip(), tag_list))
# #     id2tag = {n: tag for n, tag in enumerate(tag_list)}
# #
# #     return  id2tag
# # tag_dict=get_nus_tag_map()
# # f=open(label_dir)
# # label_number=json.load(f)
# # f.close()
# # f=open(txt_dir)
# # txt_list=f.read().split('\n')[:-1]
# # f.close()
# # f=open(tar1k_dir)
# # tar1k_list=f.read().split('\n')[:-1]
# # f.close()
# # f=open('/home/xutianyuan/papers/DSCMR/data/nus-wide/ImageList/Imagelist.txt','r')
# # img_list=f.read().split('\n')[:-1]
# # f.close()
# # label_dict={}
# # person_list=[]
# # animal_list=[]
# # sky_list=[]
# # window_list=[]
# # water_list=[]
# # flowers_list=[]
# # food_list=[]
# # clouds_list=[]
# # grass_list=[]
# # toy_list=[]
# # list_label=['person','animal','sky','window','water','flowers','food','clouds','grass','toy']
# # for item in os.listdir(valid_label):
# #     label_file = os.path.join(valid_label, item)
# #     with open(label_file, 'r') as f:
# #         context = f.read().split()
# #         label_dict[item.split('_')[1][:-4]]=context
# # count=0
# # for img_id,txt_id,bow in  zip(img_list,txt_list,tar1k_list):
# #     label_sum=0
# #     for key,value in label_dict.items():
# #         if  key in list_label:
# #             label_sum+=int(value[count])
# #
# #     count += 1
# #     if label_sum!=1:
# #         continue
# #
# #     print(count)
# #     text_1k=[tag_dict[key] for key,item in enumerate(bow.split('\t')) if item=='1']
# #     # if len(text_1k)<1:
# #     #     continue
# #     txt_info=txt_id.split(' ',1)
# #     if len(txt_info)<1:
# #         print(txt_info[0]+'文本问题')
# #         continue
# #     id=int(txt_info[0].strip())
# #     text=''
# #     try:
# #         text=txt_info[1].strip()
# #     except:
# #         continue
# #     # if len(text.split())<1:
# #     #     # print(txt_info[0]+'文本问题')
# #     #     continue
# #     bow_txt=' '.join(text_1k)
# #     img_info=img_id.split('_')
# #     if int(img_info[-1][:-4])!=id:
# #         print('id'+str(id)+'有问题')
# #         continue
# #     img=img_id.split('\\')[0]+'/'+img_info[-1]
# #     if not os.path.exists('data/nus-wide/image/'+img):
# #         continue
# #     if label_dict['person'][count-1]=='1':
# #         person_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['person']})
# #         continue
# #     if label_dict['animal'][count-1]=='1':
# #         animal_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['animal']})
# #         continue
# #     if label_dict['sky'][count-1]=='1':
# #         sky_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['sky']})
# #         continue
# #     if label_dict['window'][count-1]=='1':
# #         window_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['window']})
# #         continue
# #     if label_dict['water'][count-1]=='1':
# #         water_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['water']})
# #         continue
# #     if label_dict['flowers'][count-1]=='1':
# #         flowers_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['flowers']})
# #         continue
# #     if label_dict['food'][count-1]=='1':
# #         food_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['food']})
# #         continue
# #     if label_dict['clouds'][count-1]=='1':
# #         clouds_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['clouds']})
# #         continue
# #     if label_dict['grass'][count-1]=='1':
# #         grass_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['grass']})
# #         continue
# #     if label_dict['toy'][count-1]=='1':
# #         toy_list.append({'id': id, 'img': img, 'text': text, 'bow': bow_txt, 'label': label_number['toy']})
# #         continue
# #
# # all_data_list=[]
# # random.shuffle(person_list)
# # all_data_list.extend(person_list)
# # random.shuffle(animal_list)
# # all_data_list.extend(animal_list)
# # random.shuffle(sky_list)
# # all_data_list.extend(sky_list)
# # random.shuffle(window_list)
# # all_data_list.extend(window_list)
# # random.shuffle(water_list)
# # all_data_list.extend(water_list)
# # random.shuffle(flowers_list)
# # all_data_list.extend(flowers_list)
# # random.shuffle(food_list)
# # all_data_list.extend(food_list)
# # random.shuffle(clouds_list)
# # all_data_list.extend(clouds_list)
# # random.shuffle(grass_list)
# # all_data_list.extend(grass_list)
# # random.shuffle(toy_list)
# # all_data_list.extend(toy_list)
# # random.shuffle(all_data_list)
# # f=open('./data/nus-wide/all_datatotal.txt','w')
# # f.write(json.dumps(all_data_list))
# # f.close()
#
# #
# f=open('./data/nus-wide/validdata/all_datatotal.txt')
# remain_data=json.load(f)
# f.close()
# remain_data=[item for item in remain_data if len(item['bow'].split())>3]
# random.shuffle(remain_data)
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/test_data.txt')
# test=json.load(f)
# test=[item['id'] for item in test]
# f.close()
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/train_data.txt')
# train=json.load(f)
# train=[item['id'] for item in train]
# f.close()
# remain_train=[]
# re_train=[]
# remain_test=[]
# re_test=[]
# remain_valid=[]
# re_valid=[]
# count=0
# for item in remain_data:
#     if count<5000 and item['id'] not in test and item['id'] not in train:
#         count+=1
#         remain_valid.append(item)
#         re_valid.append(item['id'])
# count1=0
# for item in remain_data:
#     if count1<661 and  item['id'] not in test and item['id'] not in train and item['id'] not in re_valid:
#         count1+=1
#         remain_test.append(item)
#         re_test.append(item['id'])
# count2=0
# for item in remain_data:
#     if count2<941 and item['id'] not in test and item['id'] not in train and item['id'] not in re_valid and item['id'] not in re_test:
#         count2+=1
#         remain_train.append(item)
#         re_train.append(item['id'])
#
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_train.txt','w')
# f.write(json.dumps(remain_train))
# f.close()
#
#
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_test.txt','w')
# f.write(json.dumps(remain_test))
# f.close()
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_valid.txt','w')
# f.write(json.dumps(remain_valid))
# f.close()
#
#












#
# remain_data=remain_data[:17684]
# # #
# # # # #
# f=open("/home/xutianyuan/papers/DSCMR/data/nus-wide/all_data_three.txt")
# all_data=json.load(f)
# all_data.extend(remain_data)
# f.close()
# train_info=[]
# test_info=[]
# random.shuffle(all_data)
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide8/train_data.txt','w')
# f.write(json.dumps(all_data[:-23000]))
# f.close()
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide8/test_data.txt','w')
# f.write(json.dumps(all_data[-23000:]))
# f.close()
# # #
# # #
# # #
# # #
# # #
f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_train.txt','r')
sentences_train=json.load(f)
f.close()
f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_test.txt','r')
sentences_test=json.load(f)
f.close()
f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/data/remain_valid.txt','r')
sentences_valid=json.load(f)
f.close()
train_x = io.loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/train_label.mat')
test_x = io.loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/test_label.mat')

train_label=[item['label'] for item in sentences_train]
x = np.asarray(train_label)
x=np.resize(x,(len(train_label),1))
io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/train_label1.mat',{'name':np.concatenate((train_x['name'],x))})
test_label=[item['label'] for item in sentences_test]
x = np.asarray(test_label)
x=np.resize(x,(len(test_label),1))
io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/test_label1.mat',{'name':np.concatenate((test_x['name'],x))})
valid_label=[item['label'] for item in sentences_valid]
x = np.asarray(valid_label)
x=np.resize(x,(len(valid_label),1))
io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide_result/valid_label1.mat',{'name':x})

#
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide8/test_data.txt','r')
# sentences_test=json.load(f)
# f.close()
#
# train_label=[item['label'] for item in sentences_train]
# x = np.asarray(train_label)
# x=np.resize(x,(len(train_label),1))
# io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide8/train_label.mat',{'name':x})
# test_label=[item['label'] for item in sentences_test]
# x = np.asarray(test_label)
# x=np.resize(x,(len(test_label),1))
# io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide8/test_label.mat',{'name':x})



# img_dir='data/nus-wide/image'
# cp_dir='data/final_all_data_info/nus-wide8/img_train_val'
# label_dir='./data/nus-wide/label.txt'
# f=open('data/final_all_data_info/nus-wide8/train_data.txt')
# train_data=json.load(f)
# f.close()
# f=open('data/final_all_data_info/nus-wide8/test_data.txt')
# test_data=json.load(f)
# f.close()
# f=open(label_dir)
# label_number=json.load(f)
# label_number = {value:key for key,value in label_number.items()}
# f.close()
# i=0
# for item in train_data:
#     i+=1
#     print(i)
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
# for item in test_data:
#     i+=1
#     print(i)
#     train_label = 'train/' + label_number[item['label']]
#     train_label_dir = os.path.join(cp_dir, train_label)
#     test_label = 'val/' + label_number[item['label']]
#     test_label_dir = os.path.join(cp_dir, test_label)
#     if not os.path.exists(train_label_dir):
#         os.mkdir(train_label_dir)
#     if not os.path.exists(test_label_dir):
#         os.mkdir(test_label_dir)
#     origin_path = os.path.join(img_dir, item['img'])
#     os.system("cp {} {}".format(origin_path, test_label_dir))
# #
#
# #
# #
# # from collections import defaultdict
# #
# # f=open("/home/xutianyuan/papers/DSCMR/data/nus-wide/all_data.txt")
# # all_data=json.load(f)
# # f.close()
# # train_info=[]
# # test_info=[]
# # random.shuffle(all_data)
# # dict_all=defaultdict(list)
# # f=open('./data/nus-wide/label.txt')
# # label_number=json.load(f)
# # label_number = {value:key for key,value in label_number.items()}
# # for item in all_data:
# #     dict_all[item['label']].append(item)
# # all_data_1k=[]
# # for key,value in dict_all.items():
# #     random.shuffle(value)
# #     all_data_1k.extend(value[:1000])
# #
# #
# # random.shuffle(all_data_1k)
# # f=open('./data/nus-wide/all_data_1K.txt','w')
# # f.write(json.dumps(all_data_1k))
# # f.close()
# # #
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide3/test_data.txt')
# test_data=json.load(f)
# f.close()
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide2/train_data.txt')
# test_data.extend(json.load(f))
# f.close()
# from collections import defaultdict
# dict_all=defaultdict(list)
# for item in test_data:
#     dict_all[item['label']].append(item)
# x=1
# x=set()

# # j=0
# # test_data=[str(item['id'])+'.jpg' for item in test_data]
# # for item in os.listdir('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide1/img_train_val/val'):
# #     for i in os.listdir(os.path.join('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide1/img_train_val/val',item)):
# #         if i  in test_data:
# #             j += 1
# #             print(j)
# from PIL import Image
#
# def process_image_channels(image, image_path):
#     # process the 4 channels .png
#     try:
#         if image.mode == 'RGBA':
#             r, g, b, a = image.split()
#             image = Image.merge("RGB", (r, g, b))    # process the 1 channel image
#         elif image.mode != 'RGB':
#             image = image.convert("RGB")
#             os.remove(image_path)
#             image.save(image_path)
#         elif image.mode == 'RGB':
#             X=1
#         else:
#             xx=1
#     except:
#         x=1
#     return image
#
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/train_data.txt')
# test_data=json.load(f)
# test_data=[item['img'] for item in test_data]
# f.close()
# count=0
# for item in os.listdir('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/img_train_val/train'):
#     for i in os.listdir(os.path.join('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/img_train_val/train',item)):
#         if i in test_data:
#             count=count+1
#             print(count)
#         # print(i)
#         # img =  Image.open(os.path.join('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia/data/img_train_val/train', item)+'/'+i)
#         # process_image_channels(img,os.path.join('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia/data/img_train_val/train', item)+'/'+i)
# # from  PIL import  Image
# # for item in os.listdir('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia/data/valid_img'):
# #     img = Image.open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia/data/valid_img/'+item)
# #     print(item,end=' ')
# #     x=1
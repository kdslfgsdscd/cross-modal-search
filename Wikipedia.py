import os
from scipy import  io
import numpy as np
from os.path import join
import  json
import random
P = "data/wikipedia_dataset"
IMG_P = "images"
TXT_P = "texts"
TRAIN_LIST = "trainset_txt_img_cat.list"
TEST_LIST = "testset_txt_img_cat.list"

os.chdir(P)  # 切去解压目录
print(os.getcwd())
all_info=[]
with open(TRAIN_LIST, "r") as f:
    for line in f:
        txt_f, img_f, lab = line.split()
        if img_f=='09fb94dd44b50a2f8b2f5d8b10ecca13' or img_f=='191f73325ae3cbc7c7633d26b4ddaa67' or img_f=='a7fdb243026b5090f551f17a6500db96':
            continue
        all_info.append({'img':img_f+'.jpg','label':int(lab)-1,'text_id':txt_f})
with open(TEST_LIST, "r") as f:
    for line in f:
        txt_f, img_f, lab = line.split()
        if img_f == '09fb94dd44b50a2f8b2f5d8b10ecca13' or img_f == '191f73325ae3cbc7c7633d26b4ddaa67' or img_f=='a7fdb243026b5090f551f17a6500db96':
            continue
        all_info.append({'img':img_f+'.jpg','label':int(lab)-1,'text_id':txt_f})
# labels = np.asarray(train_ls_lab).reshape(-1,1)
def parse(fn):
    res =[]
    flag = False
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "</text>":
                break
            if flag:
                res.append(line)
            if line == "<text>":
                flag = True
    return res


"""解析 xml"""
sentences = []
for item in all_info:
    txt_f = join(TXT_P, "{}.xml".format(item['text_id']))
    doc = parse(txt_f)
    doc=" ".join(doc)
    item['text']=doc

# random.shuffle(all_info)
train_info=all_info[:2173]
test_info=all_info[2173:]
random.shuffle(test_info)
test_info=test_info[-462:]
train_labels = np.asarray([item['label'] for item in train_info]).reshape(-1,1)
test_labels = np.asarray([item['label'] for item in test_info]).reshape(-1,1)
with open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/train_data.txt','w') as f:
    f.write(json.dumps(train_info))
with open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/test_data.txt','w') as f:
    f.write(json.dumps(test_info))
io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/train_label.mat',{'name':train_labels})
io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/test_label.mat',{'name':test_labels})


import os,random,json,collections
old_img_dir='/home/xutianyuan/papers/DSCMR/data/wikipedia_dataset/images'
total_dir='/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia6/vaild_img'
cp_dir='./data/final_all_data_info/wikipedia6/img_train_val'
label_dir='./data/final_all_data_info/wikipedia6/label.txt'
f=open('./data/final_all_data_info/wikipedia6/train_data.txt')
train_data=json.load(f)
f.close()
f=open('./data/final_all_data_info/wikipedia6/test_data.txt')
test_data=json.load(f)
f.close()
f=open(label_dir)
label_number=json.load(f)
label_number = {value:key for key,value in label_number.items()}
f.close()
for item in train_data:
    train_label ='train/' +  label_number[item['label']]
    img_dir=old_img_dir+'/'+label_number[item['label']]
    train_label_dir = os.path.join(cp_dir, train_label)
    test_label = 'val/' +label_number[item['label']]
    test_label_dir = os.path.join(cp_dir, test_label)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    if not os.path.exists(test_label_dir):
        os.mkdir(test_label_dir)
    origin_path=os.path.join(img_dir,item['img'])
    os.system("cp {} {}".format(origin_path, train_label_dir))
    os.system("cp {} {}".format(origin_path, total_dir))
for item in test_data:
    train_label ='train/' +  label_number[item['label']]
    img_dir=old_img_dir+'/'+label_number[item['label']]
    train_label_dir = os.path.join(cp_dir, train_label)
    test_label = 'val/' +label_number[item['label']]
    test_label_dir = os.path.join(cp_dir, test_label)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    if not os.path.exists(test_label_dir):
        os.mkdir(test_label_dir)
    origin_path=os.path.join(img_dir,item['img'])
    os.system("cp {} {}".format(origin_path, test_label_dir))
    os.system("cp {} {}".format(origin_path, total_dir))
import _pickle as cPickle

# with open('/home/xutianyuan/papers/ACMR-master/data/nuswide/img_train_id_feats.pkl', 'rb') as f:
#     train_img_feats = cPickle.load(f)
# with open('/home/xutianyuan/papers/ACMR-master/data/nuswide/train_id_bow.pkl', 'rb') as f:
#     train_txt_vecs = cPickle.load(f)
# with open('/home/xutianyuan/papers/ACMR-master/data/nuswide/train_id_label_map.pkl', 'rb') as f:
#     train_labels = cPickle.load(f)
# with open('/home/xutianyuan/papers/ACMR-master/data/nuswide/img_test_id_feats.pkl', 'rb') as f:
#     test_img_feats = cPickle.load(f)
# with open('./data/nuswide/test_id_bow.pkl', 'rb') as f:
#     test_txt_vecs = cPickle.load(f)
# with open('./data/nuswide/test_id_label_map.pkl', 'rb') as f:
#     test_labels = cPickle.load(f)
# with open('data/nuswide/train_ids.pkl', 'rb') as f:
#     train_ids = cPickle.load(f)
# with open('data/nuswide/test_ids.pkl', 'rb') as f:
#     test_ids = cPickle.load(f)
# with open('data/nuswide/train_id_label_single.pkl', 'rb') as f:
#     train_labels_single = cPickle.load(f)
# with open('data/nuswide/test_id_label_single.pkl', 'rb') as f:
#     test_labels_single = cPickle.load(f)
#
#
def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta,criteria,mid1_feature, mid2_feature,imgs,txts):


    batch_size = len(labels_1)
    visual_norm = F.normalize(view1_feature, p=2, dim=1)
    textual_norm = F.normalize(view2_feature, p=2, dim=1)
    similarity = torch.matmul(visual_norm, textual_norm.t())
    labels_z=labels_1.argmax(dim=1)
    labels_ = labels_z.expand(batch_size, batch_size).eq(
        labels_z.expand(batch_size, batch_size).t())
    loss = 0
    for i in range(batch_size):
        pred = similarity[i]
        label = labels_[i].float()
        pos_inds = torch.nonzero(label == 1).squeeze(1)
        xp=len(pos_inds)
        neg_inds = torch.nonzero(label == 0).squeeze(1)
        yp=len(neg_inds)
        loss_pos = torch.log(1 + torch.exp(-1 * (pred[pos_inds.expand(yp, xp)] - pred[neg_inds.expand(xp, yp).t()])))
        loss += loss_pos.sum()

        pred = similarity[:, i]
        label = labels_[:, i].float()
        pos_inds = torch.nonzero(label == 1).squeeze(1)
        xp=len(pos_inds)
        neg_inds = torch.nonzero(label == 0).squeeze(1)
        yp=len(neg_inds)
        loss_pos = torch.log(1 + torch.exp(-1 * (pred[pos_inds.expand(yp, xp)] - pred[neg_inds.expand(xp, yp).t()])))
        loss += loss_pos.sum()

    loss /= batch_size




    term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()
    label_len = labels_1.shape[1]
    batch_size = len(labels_1)
    delt = 1
    alx=0.01
    pw=2.0
    cos_pos = Cos_similarity(view1_feature, view2_feature)
    pos_samples = cos_pos.unsqueeze(dim=1).repeat(1, batch_size)
    repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
    repert_input2 = view2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
    all_value =alx- pos_samples + all_samples-torch.eye(batch_size).cuda()*alx
    loss1 = torch.sum((torch.log((1+torch.pow(torch.exp(all_value),pw)))*1/pw)*(torch.ones(batch_size)-torch.eye(batch_size)).cuda())
    # loss1 = torch.sum(torch.clamp(all_value, min=0))
    # loss1 = torch.sum((torch.log((1 + torch.exp(all_value)))) * (torch.ones(batch_size) - torch.eye(batch_size)).cuda())
    # loss1 = torch.sum((torch.exp(all_value)) * (torch.ones(batch_size) - torch.eye(batch_size)).cuda())

    mask = torch.ones(batch_size, label_len).cuda()
    standard_labels = torch.sum(torch.mul(view1_predict, labels_1), dim=-1)
    repert_stand = torch.mul(mask, standard_labels.view(-1, 1))
    all_value = delt - repert_stand + view1_predict-labels_1*delt
    loss2 =  torch.sum((torch.log((1+torch.pow(torch.exp(all_value),pw)))*1/pw)*(torch.ones(labels_1.shape).cuda()-labels_1))
    # loss2 = torch.sum(torch.clamp(all_value, min=0))
    # loss2 = torch.sum((torch.log((1 + torch.exp(all_value)))) *(torch.ones(labels_1.shape).cuda()-labels_1))
    # loss2 = torch.sum((torch.exp(all_value)) *(torch.ones(labels_1.shape).cuda()-labels_1))


    mask = torch.ones(batch_size, label_len).cuda()
    standard_labels = torch.sum(torch.mul(view2_predict, labels_1), dim=-1)
    repert_stand = torch.mul(mask, standard_labels.view(-1, 1))
    all_value = delt - repert_stand + view2_predict-labels_1*delt
    loss2_1 = torch.sum((torch.log(1+torch.pow(torch.exp(all_value),pw))*1/pw)*(torch.ones(labels_1.shape).cuda()-labels_1))
    # loss2_1 = torch.sum(torch.clamp(all_value, min=0))
    # loss2_1 = torch.sum((torch.log((1 + torch.exp(all_value)))) *(torch.ones(labels_1.shape).cuda()-labels_1))
    # loss2_1 = torch.sum((torch.exp(all_value)) *(torch.ones(labels_1.shape).cuda()-labels_1))


    return  term1 +  0.1*alpha * loss1 +alpha*1e-3*loss

dict_num={}
f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result/pascal_data_result/pascal1.log')
for item in f.readlines():
    if item[:5]=='Epoch':
        key=int(item.split()[1].split('/')[0])
    if item[:4]=='test':
        value=float(item.split()[2])
        dict_num[key]=value

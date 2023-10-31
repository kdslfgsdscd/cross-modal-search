# import  os,json,random
# root='/home/xutianyuan/papers/PascalSentenceDataset'
# all_data={}
# test_data=[]
# label_dir='/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/label.txt'
# f=open(label_dir)
# label_number=json.load(f)
# f.close()
# for item in os.listdir(root+'/dataset'):
#     temp_data=[]
#     for img in os.listdir(root+'/dataset/'+item):
#         id=img[:-4]
#         f=open(root+'/sentence/'+item+'/'+id+'txt')
#         text=f.read()
#         text=" ".join(text.split('\n'))
#         f.close()
#         label=label_number[item]
#         all_data[img]=text
# f = open('./data/final_all_data_info/pascal/data/train_data.txt')
# train_data = json.load(f)
# f.close()
# f = open('./data/final_all_data_info/pascal/data/test_data.txt')
# test_data = json.load(f)
# f.close()
# for item in train_data:
#     item['text']=all_data[item['img']]
# for item in test_data:
#     item['text']=all_data[item['img']]
#     # random.shuffle(temp_data)
# #     test_data.extend(temp_data[-5:])
# #     train_data.extend(temp_data[:40])
# #
# with open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/train_data1.txt','w') as f:
#     f.write(json.dumps(train_data))
# with open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal/data/test_data.txt1','w') as f:
#     f.write(json.dumps(test_data))
# # img_dir='./data/final_all_data_info/pascal/data/valid_img'
# # cp_dir='./data/final_all_data_info/pascal1/img_train_val'
# # label_dir='./data/final_all_data_info/pascal1/label.txt'
# # f=open('./data/final_all_data_info/pascal1/train_data.txt')
# # train_data=json.load(f)
# # f.close()
# # f=open('./data/final_all_data_info/pascal1/test_data.txt')
# # test_data=json.load(f)
# # f.close()
# #
# # import json
# # import numpy as np
# # import scipy.io as io
# # train_label=[item['label'] for item in train_data]
# # x = np.asarray(train_label)
# # x=np.resize(x,(len(train_label),1))
# # io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal1/train_label.mat',{'name':x})
# # test_label=[item['label'] for item in test_data]
# # x = np.asarray(test_label)
# # x=np.resize(x,(len(test_label),1))
# # io.savemat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal1/test_label.mat',{'name':x})
# # f=open(label_dir)
# # label_number=json.load(f)
# # label_number = {value:key for key,value in label_number.items()}
# # f.close()
# # for item in train_data:
# #     train_label ='train/' +  label_number[item['label']]
# #     train_label_dir = os.path.join(cp_dir, train_label)
# #     test_label = 'val/' +label_number[item['label']]
# #     test_label_dir = os.path.join(cp_dir, test_label)
# #     if not os.path.exists(train_label_dir):
# #         os.mkdir(train_label_dir)
# #     if not os.path.exists(test_label_dir):
# #         os.mkdir(test_label_dir)
# #     origin_path=os.path.join(img_dir,item['img'])
# #     os.system("cp {} {}".format(origin_path, train_label_dir))
# # for item in test_data:
# #     train_label ='train/' +  label_number[item['label']]
# #     train_label_dir = os.path.join(cp_dir, train_label)
# #     test_label = 'val/' +label_number[item['label']]
# #     test_label_dir = os.path.join(cp_dir, test_label)
# #     if not os.path.exists(train_label_dir):
# #         os.mkdir(train_label_dir)
# #     if not os.path.exists(test_label_dir):
# #         os.mkdir(test_label_dir)
# #     origin_path=os.path.join(img_dir,item['img'])
# #     os.system("cp {} {}".format(origin_path, test_label_dir))


import json

f=open("/home/xutianyuan/papers/DSCMR/data/nus-wide/validdata/all_datatotal.txt")
all_total_data=json.load(f)
f.close()
all_total_data=[item['id'] for item in all_total_data]

f=open("/home/xutianyuan/papers/DSCMR/data/nus-wide/validdata/all_data5.txt")
remain_data=json.load(f)
f.close()
remain_data=[item['id'] for item in remain_data]

f=open("/home/xutianyuan/papers/DSCMR/data/nus-wide/validdata/all_data_three.txt")
single_data=json.load(f)
f.close()
single_data=[item['id'] for item in single_data]
total_data=remain_data+single_data

f=open("/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide8/train_data.txt")
train_data=json.load(f)
f.close()
train_data=[item['id'] for item in train_data]

f=open("/home/xutianyuan/papers/DSCMR/data/final_all_data_info/nus-wide8/test_data.txt")
test_data=json.load(f)
f.close()
test_data=[item['id'] for item in test_data]

train_test_data=train_data+test_data
x=1

    #
    # repert_input1 = mid1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
    # repert_input2 = mid1_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    # all_samples1 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
    # low1_distance=torch.sign(all_samples1)
    # repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
    # repert_input2 = view1_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    # all_samples2 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
    # high1_distance=torch.sign(all_samples2)
    # subtr=high1_distance - low1_distance
    # mul=torch.mul(subtr,-1*all_samples2)
    # loss1=torch.sum(mul)/(100*99)
    #
    #
    # repert_input1 = mid2_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
    # repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    # all_samples1 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
    # low1_distance=torch.sign(all_samples1)
    # repert_input1 = view2_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
    # repert_input2 = view2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    # all_samples2 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
    # high1_distance=torch.sign(all_samples2)
    # subtr=high1_distance - low1_distance
    # mul=torch.mul(subtr,-1*all_samples2)
    # loss2=torch.sum(mul)/(100*99)
    #
    # features1 = Cos_similarity(mid1_feature, mid2_feature)
    # num_id = 4
    # feat_dims = 1
    # anneal = 1e-5
    # labels_list = torch.argmax(labels_1, -1).tolist()
    # label_need = [key for (key, value) in Counter(labels_list).items() if value >= num_id]
    # dict_need = defaultdict(list)
    # for index, item in enumerate(labels_list):
    #     if item in label_need:
    #         dict_need[item].append(features1[index].view(1, -1))
    # each_class_list = [torch.cat(value[:num_id]) for (key, value) in dict_need.items()]
    # sort_tensor = torch.cat(each_class_list).view(-1, feat_dims)
    # batch_size = len(label_need) * num_id
    # mask = 1.0 - torch.eye(batch_size)
    # mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    # sim_all = compute_aff(sort_tensor)
    # sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, batch_size, 1)
    # # compute the difference matrix
    # sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
    # # pass through the sigmoid
    # sim_sg = sigmoid(sim_diff, temp=anneal) * mask.cuda()
    # # compute the rankings
    # sim_all_rk = torch.sum(sim_sg, dim=-1) + 1
    #
    # # ------ differentiable ranking of only positive set in retrieval set ------
    # # compute the mask which only gives non-zero weights to the positive set
    # xs = sort_tensor.view(num_id, int(batch_size / num_id), feat_dims)
    # pos_mask = 1.0 - torch.eye(int(batch_size / num_id))
    # pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(num_id, int(batch_size / num_id), 1, 1)
    # # compute the relevance scores
    # sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
    # sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(batch_size / num_id), 1)
    # # compute the difference matrix
    # sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
    # # pass through the sigmoid
    # sim_pos_sg = sigmoid(sim_pos_diff, temp=anneal) * pos_mask.cuda()
    # # compute the rankings of the positive set
    # sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1
    #
    # # sum the values of the Smooth-AP for all instances in the mini-batch
    # ap2 = torch.zeros(1).cuda()
    # group = int(batch_size / num_id)
    # for ind in range(num_id):
    #     pos_divide = torch.sum(
    #         sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
    #     ap2 = ap2 + ((pos_divide / group) / batch_size)


    # mask = 1.0 - torch.eye(batch_size)
    # sim_pos_samples = Cos_similarity(mid1_feature, mid2_feature).unsqueeze(dim=1).repeat(1, batch_size)
    # sim_repert_input1 = mid1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
    # sim_repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    # sim_all_samples = Cos_similarity(sim_repert_input1.permute(0, 2, 1), sim_repert_input2.permute(0, 2, 1))
    # sim_all_value = sim_pos_samples - sim_all_samples
    # samples_all_sg = sigmoid(sim_all_value, 1e-9) * mask.cuda()
    # samples_all_rank = torch.sum(samples_all_sg, dim=-1) + 1
    # ap = torch.zeros(1).cuda()
    # for ind in range(batch_size):
    #     pos_divide = torch.sum(1 / samples_all_rank[ind])
    #     ap = ap + pos_divide / batch_size
# from __future__ import print_function
# from __future__ import division
# import torchvision
# import time
# import copy
# import torch
# from evaluate import fx_calc_map_label
# import numpy as np
# from collections import Counter,defaultdict
# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)
#
#
# def sigmoid(tensor, temp=1.0):
#     exponent = -tensor / temp
#     exponent = torch.clamp(exponent, min=-50, max=50)
#     y = 1.0 / (1.0 + torch.exp(exponent))
#     return y
#
#
# def compute_aff(x):
#     return torch.mm(x, x.t())
#
#
# def calc_label_sim(label_1, label_2):
#     Sim = label_1.float().mm(label_2.float().t())
#     return Sim
#
# def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta,mid1_feature, mid2_feature):
#     Cos_similarity = torch.nn.CosineSimilarity(dim=1)
#     term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
#             (view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()
#     batch_size = 100
#     delt = 0.0001
#     pos_samples = Cos_similarity(mid1_feature, mid2_feature).unsqueeze(dim=1).repeat(1, batch_size)
#     repert_input1 = mid1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
#     all_value = delt - pos_samples + all_samples
#     all_value = sigmoid(all_value,1e-9)*all_value
#     loss1 = torch.sum(all_value) - delt * batch_size
#
#     low_fea1=(-1)*compute_aff(view1_feature)
#     emd_feat1=(-1)*compute_aff(mid1_feature)
#     #
#     # batch_size = 100
#     # delt = 0.0001
#     # pos_samples = Cos_similarity(view1_feature, mid1_feature).unsqueeze(dim=1).repeat(1, batch_size)
#     # repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # repert_input2 = mid1_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
#     # all_value = delt - pos_samples + all_samples
#     # all_value = sigmoid(all_value, 1e-9) * all_value
#     # loss2 = torch.sum(all_value) - delt * batch_size
#     #
#     # batch_size = 100
#     # delt = 0.0001
#     # pos_samples = Cos_similarity(view2_feature, mid2_feature).unsqueeze(dim=1).repeat(1, batch_size)
#     # repert_input1 = view2_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
#     # all_value = delt - pos_samples + all_samples
#     # all_value = sigmoid(all_value, 1e-9) * all_value
#     # loss3 = torch.sum(all_value) - delt * batch_size
#
#     return loss1+term1
#
# def train_model(model, data_loaders, optimizer, alpha, beta,device="cpu", num_epochs=500):
#     since = time.time()
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     test_img_acc_history = []
#     test_txt_acc_history = []
#     epoch_loss_history =[]
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch+1, num_epochs))
#         print('-' * 20)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'test']:
#             if phase == 'train':
#                 # Set model to training mode
#                 model.train()
#             else:
#                 # Set model to evaluate mode
#                 model.eval()
#
#             running_loss = 0.0
#             # Iterate over data.
#             for imgs, txts, labels in data_loaders[phase]:
#                 # imgs = imgs.to(device)
#                 # txts = txts.to(device)
#                 # labels = labels.to(device)
#                 if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
#                     print("Data contains Nan.")
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     # Get model outputs and calculate loss
#                     # Special case for inception because in training it has an auxiliary output. In train
#                     #   mode we calculate the loss by summing the final output and the auxiliary output
#                     #   but in testing we only consider the final output.
#                     if torch.cuda.is_available():
#                         imgs = imgs.cuda()
#                         txts = txts.cuda()
#                         labels = labels.cuda()
#
#
#                     # zero the parameter gradients
#                     optimizer.zero_grad()
#
#                     # Forward
#                     view1_feature, view2_feature, view1_predict, view2_predict ,mid1_feature,mid2_feature= model(imgs, txts)
#
#                     loss = calc_loss(view1_feature, view2_feature, view1_predict,
#                                      view2_predict, labels, labels, alpha, beta,mid1_feature, mid2_feature)
#
#                     img_preds = view1_predict
#                     txt_preds = view2_predict
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # statistics
#                 running_loss += loss.item()
#             epoch_loss = running_loss / len(data_loaders[phase].dataset)
#             # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
#             # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
#             t_imgs, t_txts, t_labels = [], [], []
#             with torch.no_grad():
#                 for imgs, txts, labels in data_loaders['test']:
#                     if torch.cuda.is_available():
#                             imgs = imgs.cuda()
#                             txts = txts.cuda()
#                             labels = labels.cuda()
#                     t_view1_feature, t_view2_feature, _, _ , _, _ = model(imgs, txts)
#                     t_imgs.append(t_view1_feature.cpu().numpy())
#                     t_txts.append(t_view2_feature.cpu().numpy())
#                     t_labels.append(labels.cpu().numpy())
#             t_imgs = np.concatenate(t_imgs)
#             t_txts = np.concatenate(t_txts)
#             t_labels = np.concatenate(t_labels).argmax(1)
#             img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
#             txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
#
#             print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
#
#             # deep copy the model
#             if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
#                 best_acc = (img2text + txt2img) / 2.
#                 best_model_wts = copy.deepcopy(model.state_dict())
#             if phase == 'test':
#                 test_img_acc_history.append(img2text)
#                 test_txt_acc_history.append(txt2img)
#                 epoch_loss_history.append(epoch_loss)
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best average ACC: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history
# from __future__ import print_function
# from __future__ import division
# import torchvision
# import time
# import copy
# import torch
# from evaluate import fx_calc_map_label
# import numpy as np
# from collections import Counter,defaultdict
# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)
#
#
# def sigmoid(tensor, temp=1.0):
#     exponent = -tensor / temp
#     exponent = torch.clamp(exponent, min=-50, max=50)
#     y = 1.0 / (1.0 + torch.exp(exponent))
#     return y
#
# def others(tensor, temp=1.0):
#     X=torch.sum(torch.exp(tensor))-100
#     y = torch.log(1.0 +X )
#     return y
#
# def sigmoid1(tensor, temp=1.0):
#     exponent = -tensor / temp
#     exponent = torch.clamp(exponent, min=-50, max=50)
#     y = 1.0 / (1.0 + torch.exp(exponent))-0.5
#     return y
#
#
# def hingeLoss(tensor):
#     y = torch.log(1.0 + torch.sum(torch.exp(tensor),dim=-1))
#     return torch.sum(y)
#
# def compute_aff(x):
#     return torch.mm(x, x.t())
#
#
# def calc_label_sim(label_1, label_2):
#     Sim = label_1.float().mm(label_2.float().t())
#     return Sim
#
# def l2_norm(input, axit=1):
#     norm = torch.norm(input,2,axit,True)
#     output = torch.div(input, norm)
#     return output
#
# def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta,criteria,mid1_feature, mid2_feature,imgs,txts):
#
#     # labels=torch.argmax(labels_1, -1)
#     # term1=criteria(view1_predict,labels)+criteria(view2_predict,labels)
#
#     gama=1e-9
#     label_len=20
#     Cos_similarity = torch.nn.CosineSimilarity(dim=1)
#     batch_size = len(labels_1)
#     delt = 1
#     cos_pos=Cos_similarity(view1_feature, view2_feature)
#     pos_samples =cos_pos.unsqueeze(dim=1).repeat(1, batch_size)
#     repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     repert_input2 = view2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
#     all_value = delt - pos_samples + all_samples
#     all_value = sigmoid(all_value,gama)*all_value
#     loss1 =others(- pos_samples + all_samples)
#
#
#
#     delt=1
#     mask=torch.ones(batch_size,label_len).cuda()
#     standard_labels=torch.sum(torch.mul(view1_predict, labels_1), dim=-1)
#     repert_stand=torch.mul(mask,standard_labels.view(-1,1))
#     all_value=delt-repert_stand+view1_predict
#     all_value = sigmoid(all_value, gama) * all_value
#     loss2 = torch.sum(all_value) - delt * batch_size
#
#
#
#     mask=torch.ones(batch_size,label_len).cuda()
#     standard_labels=torch.sum(torch.mul(view2_predict, labels_1), dim=-1)
#     repert_stand=torch.mul(mask,standard_labels.view(-1,1))
#     all_value=delt-repert_stand+view2_predict
#     all_value = sigmoid(all_value, gama) * all_value
#     loss2_1 = torch.sum(all_value) - delt * batch_size
#
#
#     all_samples=compute_aff(imgs)
#     mid_elem=torch.mm(torch.mul(torch.eye(batch_size).cuda(),all_samples),torch.ones(batch_size,batch_size).cuda())
#     all_samp_sub=mid_elem-all_samples
#     low1_distance = sigmoid1(all_samp_sub,gama)*2
#
#     all_samples=compute_aff(view1_feature)
#     mid_elem=torch.mm(torch.mul(torch.eye(batch_size).cuda(),all_samples),torch.ones(batch_size,batch_size).cuda())
#     all_samp_sub=mid_elem-all_samples
#     high1_distance = sigmoid1(all_samp_sub,gama)*2
#
#     subtr=low1_distance-high1_distance
#     mul=torch.mul(subtr,-1*all_samp_sub)
#     loss3=torch.sum(mul)/(batch_size*(batch_size-1))
#
#
#
#     all_samples=compute_aff(txts)
#     mid_elem=torch.mm(torch.mul(torch.eye(batch_size).cuda(),all_samples),torch.ones(batch_size,batch_size).cuda())
#     all_samp_sub=mid_elem-all_samples
#     low2_distance = sigmoid1(all_samp_sub,gama)*2
#
#     all_samples=compute_aff(view2_feature)
#     mid_elem=torch.mm(torch.mul(torch.eye(batch_size).cuda(),all_samples),torch.ones(batch_size,batch_size).cuda())
#     all_samp_sub=mid_elem-all_samples
#     high2_distance = sigmoid1(all_samp_sub,gama)*2
#
#     subtr= low2_distance-high2_distance
#     mul=torch.mul(subtr,-1*all_samp_sub)
#     loss3_1=torch.sum(mul)/(batch_size*(batch_size-1))
#
#
#     return beta*loss1+alpha*loss2+alpha*loss2_1+beta*loss3+beta*loss3_1
#
#
# def train_model(model, data_loaders, optimizer, alpha, beta,criteria,device="cpu", num_epochs=500):
#     since = time.time()
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")
#     test_img_acc_history = []
#     test_txt_acc_history = []
#     epoch_loss_history =[]
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch+1, num_epochs))
#         print('-' * 20)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'test']:
#             if phase == 'train':
#                 # Set model to training mode
#                 model.train()
#             else:
#                 # Set model to evaluate mode
#                 model.eval()
#
#             running_loss = 0.0
#             # Iterate over data.
#             for imgs, txts, labels in data_loaders[phase]:
#                 # imgs = imgs.to(device)
#                 # txts = txts.to(device)
#                 # labels = labels.to(device)
#                 if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
#                     print("Data contains Nan.")
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     # Get model outputs and calculate loss
#                     # Special case for inception because in training it has an auxiliary output. In train
#                     #   mode we calculate the loss by summing the final output and the auxiliary output
#                     #   but in testing we only consider the final output.
#                     if torch.cuda.is_available():
#                         imgs = imgs.cuda()
#                         txts = txts.cuda()
#                         labels = labels.cuda()
#
#
#                     # zero the parameter gradients
#                     optimizer.zero_grad()
#
#                     # Forward
#                     view1_feature, view2_feature, view1_predict, view2_predict ,mid1_feature,mid2_feature= model(imgs, txts)
#
#                     loss = calc_loss(view1_feature, view2_feature, view1_predict,
#                                      view2_predict, labels, labels, alpha, beta,criteria,mid1_feature, mid2_feature,imgs,txts)
#
#                     img_preds = view1_predict
#                     txt_preds = view2_predict
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # statistics
#                 running_loss += loss.item()
#             epoch_loss = running_loss / len(data_loaders[phase].dataset)
#             # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
#             # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
#             t_imgs, t_txts, t_labels = [], [], []
#             with torch.no_grad():
#                 for imgs, txts, labels in data_loaders['test']:
#                     if torch.cuda.is_available():
#                             imgs = imgs.cuda()
#                             txts = txts.cuda()
#                             labels = labels.cuda()
#                     t_view1_feature, t_view2_feature, _, _ , _, _ = model(imgs, txts)
#                     t_imgs.append(t_view1_feature.cpu().numpy())
#                     t_txts.append(t_view2_feature.cpu().numpy())
#                     t_labels.append(labels.cpu().numpy())
#             t_imgs = np.concatenate(t_imgs)
#             t_txts = np.concatenate(t_txts)
#             t_labels = np.concatenate(t_labels).argmax(1)
#             img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
#             txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
#
#             print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
#
#             # deep copy the model
#             if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
#                 best_acc = (img2text + txt2img) / 2.
#                 best_model_wts = copy.deepcopy(model.state_dict())
#             if phase == 'test':
#                 test_img_acc_history.append(img2text)
#                 test_txt_acc_history.append(txt2img)
#                 epoch_loss_history.append(epoch_loss)
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best average ACC: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history
#
#
#     #
#     # repert_input1 = mid1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # repert_input2 = mid1_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # all_samples1 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
#     # low1_distance=torch.sign(all_samples1)
#     # repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # repert_input2 = view1_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # all_samples2 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
#     # high1_distance=torch.sign(all_samples2)
#     # subtr=high1_distance - low1_distance
#     # mul=torch.mul(subtr,-1*all_samples2)
#     # loss1=torch.sum(mul)/(100*99)
#     #
#     #
#     # repert_input1 = mid2_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # all_samples1 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
#     # low1_distance=torch.sign(all_samples1)
#     # repert_input1 = view2_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # repert_input2 = view2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # all_samples2 = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))-1
#     # high1_distance=torch.sign(all_samples2)
#     # subtr=high1_distance - low1_distance
#     # mul=torch.mul(subtr,-1*all_samples2)
#     # loss2=torch.sum(mul)/(100*99)
#     #
#     # features1 = Cos_similarity(mid1_feature, mid2_feature)
#     # num_id = 4
#     # feat_dims = 1
#     # anneal = 1e-5
#     # labels_list = torch.argmax(labels_1, -1).tolist()
#     # label_need = [key for (key, value) in Counter(labels_list).items() if value >= num_id]
#     # dict_need = defaultdict(list)
#     # for index, item in enumerate(labels_list):
#     #     if item in label_need:
#     #         dict_need[item].append(features1[index].view(1, -1))
#     # each_class_list = [torch.cat(value[:num_id]) for (key, value) in dict_need.items()]
#     # sort_tensor = torch.cat(each_class_list).view(-1, feat_dims)
#     # batch_size = len(label_need) * num_id
#     # mask = 1.0 - torch.eye(batch_size)
#     # mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # sim_all = compute_aff(sort_tensor)
#     # sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # # compute the difference matrix
#     # sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
#     # # pass through the sigmoid
#     # sim_sg = sigmoid(sim_diff, temp=anneal) * mask.cuda()
#     # # compute the rankings
#     # sim_all_rk = torch.sum(sim_sg, dim=-1) + 1
#     #
#     # # ------ differentiable ranking of only positive set in retrieval set ------
#     # # compute the mask which only gives non-zero weights to the positive set
#     # xs = sort_tensor.view(num_id, int(batch_size / num_id), feat_dims)
#     # pos_mask = 1.0 - torch.eye(int(batch_size / num_id))
#     # pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(num_id, int(batch_size / num_id), 1, 1)
#     # # compute the relevance scores
#     # sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
#     # sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(batch_size / num_id), 1)
#     # # compute the difference matrix
#     # sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
#     # # pass through the sigmoid
#     # sim_pos_sg = sigmoid(sim_pos_diff, temp=anneal) * pos_mask.cuda()
#     # # compute the rankings of the positive set
#     # sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1
#     #
#     # # sum the values of the Smooth-AP for all instances in the mini-batch
#     # ap2 = torch.zeros(1).cuda()
#     # group = int(batch_size / num_id)
#     # for ind in range(num_id):
#     #     pos_divide = torch.sum(
#     #         sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
#     #     ap2 = ap2 + ((pos_divide / group) / batch_size)
#
#
#     # mask = 1.0 - torch.eye(batch_size)
#     # sim_pos_samples = Cos_similarity(mid1_feature, mid2_feature).unsqueeze(dim=1).repeat(1, batch_size)
#     # sim_repert_input1 = mid1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     # sim_repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     # sim_all_samples = Cos_similarity(sim_repert_input1.permute(0, 2, 1), sim_repert_input2.permute(0, 2, 1))
#     # sim_all_value = sim_pos_samples - sim_all_samples
#     # samples_all_sg = sigmoid(sim_all_value, 1e-9) * mask.cuda()
#     # samples_all_rank = torch.sum(samples_all_sg, dim=-1) + 1
#     # ap = torch.zeros(1).cuda()
#     # for ind in range(batch_size):
#     #     pos_divide = torch.sum(1 / samples_all_rank[ind])
#     #     ap = ap + pos_divide / batch_size
# # from __future__ import print_function
# # from __future__ import division
# # import torchvision
# # import time
# # import copy
# # import torch
# # from evaluate import fx_calc_map_label
# # import numpy as np
# # from collections import Counter,defaultdict
# # print("PyTorch Version: ", torch.__version__)
# # print("Torchvision Version: ", torchvision.__version__)
# #
# #
# # def sigmoid(tensor, temp=1.0):
# #     exponent = -tensor / temp
# #     exponent = torch.clamp(exponent, min=-50, max=50)
# #     y = 1.0 / (1.0 + torch.exp(exponent))
# #     return y
# #
# #
# # def compute_aff(x):
# #     return torch.mm(x, x.t())
# #
# #
# # def calc_label_sim(label_1, label_2):
# #     Sim = label_1.float().mm(label_2.float().t())
# #     return Sim
# #
# # def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta,mid1_feature, mid2_feature):
# #     Cos_similarity = torch.nn.CosineSimilarity(dim=1)
# #     term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
# #             (view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()
# #     batch_size = 100
# #     delt = 0.0001
# #     pos_samples = Cos_similarity(mid1_feature, mid2_feature).unsqueeze(dim=1).repeat(1, batch_size)
# #     repert_input1 = mid1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
# #     repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
# #     all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
# #     all_value = delt - pos_samples + all_samples
# #     all_value = sigmoid(all_value,1e-9)*all_value
# #     loss1 = torch.sum(all_value) - delt * batch_size
# #
# #     low_fea1=(-1)*compute_aff(view1_feature)
# #     emd_feat1=(-1)*compute_aff(mid1_feature)
# #     #
# #     # batch_size = 100
# #     # delt = 0.0001
# #     # pos_samples = Cos_similarity(view1_feature, mid1_feature).unsqueeze(dim=1).repeat(1, batch_size)
# #     # repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
# #     # repert_input2 = mid1_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
# #     # all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
# #     # all_value = delt - pos_samples + all_samples
# #     # all_value = sigmoid(all_value, 1e-9) * all_value
# #     # loss2 = torch.sum(all_value) - delt * batch_size
# #     #
# #     # batch_size = 100
# #     # delt = 0.0001
# #     # pos_samples = Cos_similarity(view2_feature, mid2_feature).unsqueeze(dim=1).repeat(1, batch_size)
# #     # repert_input1 = view2_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
# #     # repert_input2 = mid2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
# #     # all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
# #     # all_value = delt - pos_samples + all_samples
# #     # all_value = sigmoid(all_value, 1e-9) * all_value
# #     # loss3 = torch.sum(all_value) - delt * batch_size
# #
# #     return loss1+term1
# #
# # def train_model(model, data_loaders, optimizer, alpha, beta,device="cpu", num_epochs=500):
# #     since = time.time()
# #     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     test_img_acc_history = []
# #     test_txt_acc_history = []
# #     epoch_loss_history =[]
# #
# #     best_model_wts = copy.deepcopy(model.state_dict())
# #     best_acc = 0.0
# #
# #     for epoch in range(num_epochs):
# #         print('Epoch {}/{}'.format(epoch+1, num_epochs))
# #         print('-' * 20)
# #
# #         # Each epoch has a training and validation phase
# #         for phase in ['train', 'test']:
# #             if phase == 'train':
# #                 # Set model to training mode
# #                 model.train()
# #             else:
# #                 # Set model to evaluate mode
# #                 model.eval()
# #
# #             running_loss = 0.0
# #             # Iterate over data.
# #             for imgs, txts, labels in data_loaders[phase]:
# #                 # imgs = imgs.to(device)
# #                 # txts = txts.to(device)
# #                 # labels = labels.to(device)
# #                 if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
# #                     print("Data contains Nan.")
# #
# #                 # zero the parameter gradients
# #                 optimizer.zero_grad()
# #
# #                 # forward
# #                 # track history if only in train
# #                 with torch.set_grad_enabled(phase == 'train'):
# #                     # Get model outputs and calculate loss
# #                     # Special case for inception because in training it has an auxiliary output. In train
# #                     #   mode we calculate the loss by summing the final output and the auxiliary output
# #                     #   but in testing we only consider the final output.
# #                     if torch.cuda.is_available():
# #                         imgs = imgs.cuda()
# #                         txts = txts.cuda()
# #                         labels = labels.cuda()
# #
# #
# #                     # zero the parameter gradients
# #                     optimizer.zero_grad()
# #
# #                     # Forward
# #                     view1_feature, view2_feature, view1_predict, view2_predict ,mid1_feature,mid2_feature= model(imgs, txts)
# #
# #                     loss = calc_loss(view1_feature, view2_feature, view1_predict,
# #                                      view2_predict, labels, labels, alpha, beta,mid1_feature, mid2_feature)
# #
# #                     img_preds = view1_predict
# #                     txt_preds = view2_predict
# #
# #                     # backward + optimize only if in training phase
# #                     if phase == 'train':
# #                         loss.backward()
# #                         optimizer.step()
# #
# #                 # statistics
# #                 running_loss += loss.item()
# #             epoch_loss = running_loss / len(data_loaders[phase].dataset)
# #             # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
# #             # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
# #             t_imgs, t_txts, t_labels = [], [], []
# #             with torch.no_grad():
# #                 for imgs, txts, labels in data_loaders['test']:
# #                     if torch.cuda.is_available():
# #                             imgs = imgs.cuda()
# #                             txts = txts.cuda()
# #                             labels = labels.cuda()
# #                     t_view1_feature, t_view2_feature, _, _ , _, _ = model(imgs, txts)
# #                     t_imgs.append(t_view1_feature.cpu().numpy())
# #                     t_txts.append(t_view2_feature.cpu().numpy())
# #                     t_labels.append(labels.cpu().numpy())
# #             t_imgs = np.concatenate(t_imgs)
# #             t_txts = np.concatenate(t_txts)
# #             t_labels = np.concatenate(t_labels).argmax(1)
# #             img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
# #             txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
# #
# #             print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
# #
# #             # deep copy the model
# #             if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
# #                 best_acc = (img2text + txt2img) / 2.
# #                 best_model_wts = copy.deepcopy(model.state_dict())
# #             if phase == 'test':
# #                 test_img_acc_history.append(img2text)
# #                 test_txt_acc_history.append(txt2img)
# #                 epoch_loss_history.append(epoch_loss)
# #
# #         print()
# #
# #     time_elapsed = time.time() - since
# #     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# #     print('Best average ACC: {:4f}'.format(best_acc))
# #
# #     # load best model weights
# #     model.load_state_dict(best_model_wts)
# #     return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history



# def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta,criteria,mid1_feature, mid2_feature,imgs,txts):
#
#     term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + (
#             (view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()
#     label_len = 10
#     Cos_similarity = torch.nn.CosineSimilarity(dim=1)
#     batch_size = len(labels_1)
#     delt = 1
#     cos_pos = Cos_similarity(view1_feature, view2_feature)
#     pos_samples = cos_pos.unsqueeze(dim=1).repeat(1, batch_size)
#     repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
#     repert_input2 = view2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
#     all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
#     all_value = 0.01 - pos_samples + all_samples-torch.eye(batch_size).cuda()*0.01
#     loss1 = torch.sum(torch.pow(torch.log((1+torch.pow(torch.exp(all_value),3))),1/3)-(torch.eye(batch_size)*pow(math.log(1+pow(math.exp(0),3)),1/3)).cuda())
#     # loss1 = torch.sum(torch.clamp(all_value, min=0))
#
#     mask = torch.ones(batch_size, label_len).cuda()
#     x=labels_1.cuda()
#     standard_labels = torch.sum(torch.mul(view1_predict, labels_1), dim=-1)
#     repert_stand = torch.mul(mask, standard_labels.view(-1, 1))
#     all_value = delt - repert_stand + view1_predict-x*delt
#     loss2 =  torch.sum(torch.pow(torch.log((1+torch.pow(torch.exp(all_value),3))),1/3))
#     # loss2 = torch.sum(torch.clamp(all_value, min=0))
#
#     mask = torch.ones(batch_size, label_len).cuda()
#     standard_labels = torch.sum(torch.mul(view2_predict, labels_1), dim=-1)
#     repert_stand = torch.mul(mask, standard_labels.view(-1, 1))
#     all_value = delt - repert_stand + view2_predict-x*delt
#     loss2_1 = torch.sum(torch.pow(torch.log(1+torch.pow(torch.exp(all_value),3)),1/3))
#     # loss2_1 = torch.sum(torch.clamp(all_value, min=0))
#
#     return term1 + 0.1*alpha * loss1 + alpha*(loss2_1 + loss2)
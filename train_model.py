from __future__ import print_function
from __future__ import division
import torchvision
import time
import copy
import torch
import math
from evaluate import fx_calc_map_label
import numpy as np
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
import torch.nn.functional as F

#
def sigmoid(tensor, temp=1.0):
    exponent = -tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

def compute_aff(x):
    return torch.mm(x, x.t())

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

Cos_similarity = torch.nn.CosineSimilarity(dim=1)
def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta,criteria,mid1_feature, mid2_feature,imgs,txts):

    r=0.1
    rr=0.01
    batch_size = len(labels_1)
    visual_norm = F.normalize(view1_feature, p=2, dim=1)
    textual_norm = F.normalize(view2_feature, p=2, dim=1)
    similarity = torch.matmul(visual_norm, textual_norm.t())
    similarity1=torch.matmul(visual_norm, visual_norm.t())
    similarity2=torch.matmul(textual_norm, textual_norm.t())
    labels_z = labels_1.argmax(dim=1)
    labels_ = labels_z.expand(batch_size, batch_size).eq(
        labels_z.expand(batch_size, batch_size).t())
    loss = 0
    pw=2.0
    sum_x=0
    for i in range(batch_size):
        pred = similarity[i]
        pred1 = similarity1[i]
        pred2 = similarity2[i]
        label = labels_[i].float()
        pos_inds = torch.nonzero(label == 1).squeeze(1)
        xp = len(pos_inds)
        neg_inds = torch.nonzero(label == 0).squeeze(1)
        yp = len(neg_inds)
        x=r*((pred[pos_inds.expand(yp, xp)] - pred[neg_inds.expand(xp, yp).t()]) ** 2).sum(1).sqrt().mean()+ \
          rr*((pred1[pos_inds.expand(yp, xp)] - pred1[neg_inds.expand(xp, yp).t()]) ** 2).sum(1).sqrt().mean()+\
          rr*((pred2[pos_inds.expand(yp, xp)] - pred2[neg_inds.expand(xp, yp).t()]) ** 2).sum(1).sqrt().mean()
        loss_pos = torch.log(1 + torch.pow(torch.exp(x -1 * (pred[pos_inds.expand(yp, xp)] - pred[neg_inds.expand(xp, yp).t()])),pw))*1/pw
        loss += loss_pos.sum()
        pred = similarity[:, i]
        pred1 = similarity1[:,i]
        pred2 = similarity2[:,i]
        label = labels_[:, i].float()
        pos_inds = torch.nonzero(label == 1).squeeze(1)
        xp = len(pos_inds)
        neg_inds = torch.nonzero(label == 0).squeeze(1)
        yp = len(neg_inds)
        x=r*((pred[pos_inds.expand(yp, xp)] - pred[neg_inds.expand(xp, yp).t()]) ** 2).sum(1).sqrt().mean()+\
          rr*((pred1[pos_inds.expand(yp, xp)] - pred1[neg_inds.expand(xp, yp).t()]) ** 2).sum(1).sqrt().mean()+\
          rr*((pred2[pos_inds.expand(yp, xp)] - pred2[neg_inds.expand(xp, yp).t()]) ** 2).sum(1).sqrt().mean()
        loss_pos = torch.log(1 + torch.pow(torch.exp(x -1 * (pred[pos_inds.expand(yp, xp)] - pred[neg_inds.expand(xp, yp).t()])),pw))*1/pw
        loss += loss_pos.sum()
    loss /= batch_size
    term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()
    label_len = labels_1.shape[1]
    batch_size = len(labels_1)
    delt = 1
    alx=0.01
    cos_pos = Cos_similarity(view1_feature, view2_feature)
    pos_samples = cos_pos.unsqueeze(dim=1).repeat(1, batch_size)
    repert_input1 = view1_feature.unsqueeze(dim=1).repeat(1, batch_size, 1)
    repert_input2 = view2_feature.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    all_samples = Cos_similarity(repert_input1.permute(0, 2, 1), repert_input2.permute(0, 2, 1))
    all_value =alx- pos_samples + all_samples-torch.eye(batch_size).cuda()*alx
    loss1 = torch.sum((torch.log(1+torch.pow(torch.exp(all_value),pw))*1/pw)*(torch.ones(batch_size)-torch.eye(batch_size)).cuda())

    mask = torch.ones(batch_size, label_len).cuda()
    standard_labels = torch.sum(torch.mul(view1_predict, labels_1), dim=-1)
    repert_stand = torch.mul(mask, standard_labels.view(-1, 1))
    all_value = delt - repert_stand + view1_predict-labels_1*delt
    loss2 =  torch.sum((torch.log(1+torch.pow(torch.exp(all_value),pw))*1/pw)*(torch.ones(labels_1.shape).cuda()-labels_1))

    mask = torch.ones(batch_size, label_len).cuda()
    standard_labels = torch.sum(torch.mul(view2_predict, labels_1), dim=-1)
    repert_stand = torch.mul(mask, standard_labels.view(-1, 1))
    all_value = delt - repert_stand + view2_predict-labels_1*delt
    loss2_1 = torch.sum((torch.log(1+torch.pow(torch.exp(all_value),pw))*1/pw)*(torch.ones(labels_1.shape).cuda()-labels_1))

    return 0.01 *alpha * loss+term1 + 0.1*alpha * loss1 + alpha*(loss2_1 + loss2)


def train_model(model, data_loaders, optimizer, alpha, beta,criteria,device="cpu", num_epochs=200):
    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    count=0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:
                # imgs = imgs.to(device)
                # txts = txts.to(device)
                # labels = labels.to(device)
                if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()


                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict ,mid1_feature,mid2_feature= model(imgs, txts)

                    loss = calc_loss(view1_feature, view2_feature, view1_predict,
                                     view2_predict, labels, labels, alpha, beta,criteria,mid1_feature, mid2_feature,imgs,txts)

                    img_preds = view1_predict
                    txt_preds = view2_predict

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            t_imgs, t_txts, t_labels = [], [], []
            if phase=='test':
                with torch.no_grad():
                    for imgs, txts, labels in data_loaders['test']:
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.cuda()
                        t_view1_feature, t_view2_feature, _, _, _, _ = model(imgs, txts)
                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels).argmax(1)
                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

                print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
                # deep copy the model
                if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                    best_acc = (img2text + txt2img) / 2.
                    count=0
                    best_model_wts = copy.deepcopy(model.state_dict())
                print('best_acc: {:.4f}'.format(best_acc))
                if phase == 'test':
                    test_img_acc_history.append(img2text)
                    test_txt_acc_history.append(txt2img)
                    epoch_loss_history.append(epoch_loss)

            print()
        count+=1
        if count>10:
            print('Training stop early')
        if count>30:
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history



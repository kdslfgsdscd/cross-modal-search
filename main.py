import torch
import torch.optim as optim
from model import DRSC_NN
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label
from smoothAP import SmoothAP
######################################################################
# Start running
from scipy.io import savemat
import os

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    loss_fun = SmoothAP(0.01, 200, 2, 2048)
    DATA_DIR = '/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result/'
    alpha = 1e-3
    beta = 1e-3
    MAX_EPOCH = 300
    batch_size = 300
    # batch_size = 512
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 1e-3
    criteria = torch.nn.CrossEntropyLoss()
    print('...Data loading is beginning...')

    data_loader, input_data_par,img_testx,txt_testx,lanx = get_loader(DATA_DIR, batch_size)

    print('...Data loading is completed...')

    model_ft = DRSC_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], output_dim=input_data_par['num_class']).to(device)
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas,weight_decay=weight_decay)

    print('...Training is beginning...')
    # Train and evaluate
    model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta,criteria, loss_fun,MAX_EPOCH)
    print('...Training is completed...')

    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict,_,_ = model_ft(torch.tensor(img_testx).to(device), torch.tensor(txt_testx).to(device))
    label = torch.argmax(torch.tensor(input_data_par['label_test']), dim=1)
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    savemat(os.path.join("/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result", 'vis_img.mat'), {'name': view1_feature})
    savemat(os.path.join("/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result", 'vis_txt.mat'), {'name': view2_feature})
    savemat(os.path.join("/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result", 'vis_lab.mat'), {'name': lanx})
    view1_predict = view1_predict.detach().cpu().numpy()
    view2_predict = view2_predict.detach().cpu().numpy()
    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))

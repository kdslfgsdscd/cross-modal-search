from time import time

import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from scipy.io import loadmat

import h5py
import argparse
# Training settings
parser = argparse.ArgumentParser(description='dorefa-net implementation')

#########################
#### data parameters ####
#########################
parser.add_argument("--epcho", type=str, default="1", # wiki xmedianet2views nus inria
                    help="data name")
parser.add_argument('--rate', type=str, default='0.4')
args = parser.parse_args()
print(args)

# --max_epochs 100 --log_name noisylabel_se --loss CE  --lr 0.05 --train_batch_size 50 --beta 1 parser.add_argument('--zeta', type=float, default=1.)
# --max_epochs 50 --log_name noisylabel_mce --loss MCE  --lr 0.05 --train_batch_size 50 --beta 0.7 --noisy_ratio 0.2 --data_name wiki
# --max_epochs 50 --log_name noisylabel_mce --loss MCE  --lr 0.05 --train_batch_size 50 --beta 0.4 --noisy_ratio 0.6 --data_name wiki
#
# def get_data():
#     label_train = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_label.mat")['name']
#     txt_train = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_txt.mat")['name']
#     label_test = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_label.mat")['name']
#     img_test = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_img.mat")['name']
#     # label_train = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_cross_label1-4.mat")['name']
#     # txt_train = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_cross_txt1-4.mat")['name']
#     # label_test = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_cross_label1-4.mat")['name']
#     # img_test = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/' + "test_cross_img1-4.mat")['name']
#
#     data_train, label_train = txt_train, label_train
#     data_test, label_test = img_test, label_test
#     return data_train, label_train,data_test, label_test


def plot_embedding(result_train, label_train, result_test, label_test, title):
    ax = plt.axes()
    ax.tick_params("both", which='major', direction='in')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0.01, 1)
    x_min, x_max = np.min(result_train, 0), np.max(result_train, 0)
    result_train = (result_train - x_min) / (x_max - x_min)
    for i in range(result_train.shape[0]):
        plt.text(result_train[i, 0], result_train[i, 1], str(label_train[i]),
                 color=plt.cm.Set1(label_train[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 10})
    plt.title('')
    plt.savefig('img_test'+title+'.svg',dpi=600, format='svg')
    #
    plt.cla()

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0.01, 1)
    x_min, x_max = np.min(result_test, 0), np.max(result_test, 0)
    result_test = (result_test - x_min) / (x_max - x_min)
    for i in range(result_test.shape[0]):
        plt.text(result_test[i, 0], result_test[i, 1], str(label_test[i]),
                 color=plt.cm.Set1(label_test[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 10})
    plt.title('')
    plt.savefig('txt_test'+title+'.svg',dpi=600, format='svg')

    plt.cla()

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0.01, 1)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for i in range(result_train.shape[0]):
        if label_train[i]==1 or label_train[i]==2:
            plt.text(result_train[i, 0], result_train[i, 1], '▲',
                     color=plt.cm.Set1(label_train[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 10})

        if label_train[i]==8:
            plt.text(result_train[i, 0], result_train[i, 1], '▲',
                     color=plt.cm.Set1(3 / 10.),
                     fontdict={'weight': 'bold', 'size': 10})
    for i in range(result_test.shape[0]):
        if label_train[i]==1 or label_train[i]==2 :
            plt.text(result_test[i, 0], result_test[i, 1], '●',
                     color=plt.cm.Set1(label_test[i]/ 10.),
                     fontdict={'weight': 'bold', 'size': 10})

        if label_train[i]==8:
            plt.text(result_test[i, 0], result_test[i, 1], '●',
                     color=plt.cm.Set1(3 / 10.),
                     fontdict={'weight': 'bold', 'size': 10})

    plt.title('')
    plt.savefig('txt_img_test'+title+'.svg',dpi=600, format='svg')

if __name__ == '__main__':
    # data_train, label_train, data_test, label_test = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x = str(args.rate)+'_'+str(args.epcho)
    fea=loadmat('/home/xutianyuan/papers/MRL-main/final_result_data/iniriacevis/' + "70_0.80_fea.mat")['name']
    lab=loadmat('/home/xutianyuan/papers/MRL-main/final_result_data/iniriacevis/' + "70_0.80_lab.mat")['name']
    lab1=lab[1]
    lab=lab[0]
    fea1=fea[1]
    fea=fea[0]
    # h = h5py.File('/home/xutianyuan/papers/MRL-main/data/wiki/wiki_deep_doc2vec_data_corr_ae.h5py', 'r')
    # try:
    #     test_texts_idx = h['test_text'][()].astype('float32')
    # except Exception as e:
    #     test_texts_idx = h['test_texts'][()].astype('float32')
    # test_texts_labels = h['test_texts_labels'][()]
    # test_texts_labels -= np.min(test_texts_labels)
    #
    # test_imgs_deep = h['test_imgs_deep'][()].astype('float32')
    # test_imgs_labels = h['test_imgs_labels'][()]
    lab -= np.min(lab)
    lab1 -= np.min(lab1)
    result_train = tsne.fit_transform(fea)
    result_test = tsne.fit_transform(fea1)
    plot_embedding(result_train, lab, result_test, lab1,'70')

# dict_num=[]
# f=open('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/wikipedia_result/wiki_data_result/wiki_5.log')
# for item in f.readlines():
#     # if item[:5]=='Epoch':
#     #     key=int(item.split()[1].split('/')[0])
#     if item[:4]=='test':
#         value=float(item.split()[2])
#         dict_num.append(value)
#
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
# x_values=list(range(300))
# y_values=dict_num
# plt.plot(x_values,y_values, label="total_loss")
# plt.xlabel("Learning Epoch")
# plt.ylabel("total_loss")
# y_major_locator=MultipleLocator(0.001)
# #把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()
# #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
# # plt.xlim(-0.5,11)
# # #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# plt.ylim(0.026,0.032)
# #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
# plt.savefig('func1.jpg')
# # plt.cla()
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# # x_value=range(300)
# # plt.plot(x_value, dict_num, label="total_loss")
# # plt.xlabel("x_value")
# # plt.ylabel("y_value")
# # plt.legend()
# # plt.grid()
# # plt.savefig('func2.jpg')
# # # plt.cla()
# # # plt.cla()
# x_value = np.arange(-5, 5, 0.01)
# y_value = []
# for item in x_value:
#     y_value.append(math.log(1+math.pow(math.exp(item),1))/1)
# plt.plot(x_value, y_value, label="lger(x;1)")
# plt.xlabel("x_value")
# plt.ylabel("y_value")
# plt.legend()
# plt.grid()
# plt.savefig('lger(x;1).jpg')
# # # # plt.cla()
# # # # x_value = np.arange(-5, 5, 0.01)
# # # # y_value = []
# # # # for item in x_value:
# # # #     y_value.append(math.log(1+math.pow(math.exp(item),2))/2)
# # # # plt.plot(x_value, y_value, label="lger(x;2)")
# # # # plt.xlabel("x_value")
# # # # plt.ylabel("y_value")
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.savefig('lger(x;2).jpg')
# # # # plt.cla()
# # # # x_value = np.arange(-5, 5, 0.01)
# # # # y_value = []
# # # # for item in x_value:
# # # #     y_value.append(math.log(1+math.pow(math.exp(item),3))/3)
# # # # plt.plot(x_value, y_value, label="lger(x;3)")
# # # # plt.xlabel("x_value")
# # # # plt.ylabel("y_value")
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.savefig('lger(x;3).jpg')
# # # # plt.cla()
# # # # x_value = np.arange(-5, 5, 0.01)
# # # # y_value = []
# # # # for item in x_value:
# # # #     y_value.append(math.log(1+math.pow(math.exp(item),5))/5)
# # # # plt.plot(x_value, y_value, label="lger(x;5)")
# # # # plt.xlabel("x_value")
# # # # plt.ylabel("y_value")
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.savefig('lger(x;5).jpg')
# # # # plt.cla()
# # # # x_value = np.arange(-5, 5, 0.01)
# # # # y_value = []
# # # # for item in x_value:
# # # #     y_value.append(math.log(1+math.pow(math.exp(item),7))/7)
# # # # plt.plot(x_value, y_value, label="lger(x;7)")
# # # # plt.xlabel("x_value")
# # # # plt.ylabel("y_value")
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.savefig('lger(x;7).jpg')
# # # # plt.cla()
# # # # x_value = np.arange(-5, 5, 0.01)
# # # # y_value = []
# # # # for item in x_value:
# # # #     y_value.append(math.log(1+math.pow(math.exp(item),15))/15)
# # # # plt.plot(x_value, y_value, label="lger(x;15)")
# # # # plt.xlabel("x_value")
# # # # plt.ylabel("y_value")
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.savefig('lger(x;15).jpg')
# # # # plt.cla()
# # # # x_value = np.arange(-5, 5, 0.01)
# # # # y_value = []
# # # # for item in x_value:
# # # #     if item<=0:
# # # #         y_value.append(0)
# # # #     else:
# # # #         y_value.append(item)
# # # # plt.plot(x_value, y_value, label="rank(x)")
# # # # plt.xlabel("x_value")
# # # # plt.ylabel("y_value")
# # # # plt.legend()
# # # # plt.grid()
# # # # plt.savefig('rank.jpg')
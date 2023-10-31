import numpy as np
from scipy.spatial import distance
def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
  if dist_method == 'L2':
    dist = distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)



def fx_calc_map_label1(image, text, label, k = 5, dist_method='COS'):
  x=json.load(open("/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result/data/test_data.txt",'r'))
  if dist_method == 'L2':
    dist = distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)


# #
from scipy.io import loadmat
import json
label_train = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result/' + "vis_lab.mat")['name']
txt_train = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result/' + "vis_txt.mat")['name']
img_test = loadmat('/home/xutianyuan/papers/DSCMR/data/final_all_data_info/pascal_result/' + "vis_img.mat")['name']
fx_calc_map_label1(img_test,txt_train,label_train)
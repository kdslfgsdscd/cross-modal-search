import torch.nn as nn
import torch.nn.functional as F
import torch

class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class DRSC_NN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, img_input_dim=4096, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10):
        super(DRSC_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, img_output_dim)
        self.text_net = TextNN(text_input_dim, text_output_dim)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer_m = nn.Linear(minus_one_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

    def forward(self, img, text):
        mid1_feature = self.img_net(img)
        mid2_feature = self.text_net(text)
        mid1_feature = self.linearLayer(mid1_feature)
        mid2_feature =self.linearLayer(mid2_feature)

        view1_feature =self.linearLayer_m(mid1_feature)
        view2_feature = self.linearLayer_m(mid2_feature)

        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)
        return view1_feature, view2_feature, view1_predict, view2_predict,mid1_feature,mid2_feature

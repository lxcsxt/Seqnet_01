from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn


class Backbone(nn.Module):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        self.resnet = resnet
        self.out_channels = 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        res2 = self.resnet.layer1(x)
        res3 = self.resnet.layer2(res2)
        res4 = self.resnet.layer3(res3)

        return OrderedDict([["feat_res2", res2], ["feat_res3", res3], ["feat_res4", res4]])

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()
        self.add_module("layer4", resnet.layer4)
        self.add_module("se_layer", SELayer(2048))  # 添加通道注意力模块，针对layer4输出的2048通道特征
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])
# class Res5Head(nn.Sequential):
#     def __init__(self, resnet):
#         super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
#         self.out_channels = [1024, 2048]
#
#     def forward(self, x):
#         feat = super(Res5Head, self).forward(x)
#         x = F.adaptive_max_pool2d(x, 1)
#         feat = F.adaptive_max_pool2d(feat, 1)
#         return OrderedDict([["feat_res4", x], ["feat_res5", feat]])



def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)

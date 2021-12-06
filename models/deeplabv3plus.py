# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.base_models.xceptionNet import xception
from models.sync_batchnorm import SynchronizedBatchNorm2d
from models.base_models.resnet_atrous import resnet50_atrous, resnet101_atrous, resnet152_atrous


class deeplabv3plus(nn.Module):
    def __init__(self, num_class, backbone='resnet50', pretrained_base=False, **kwargs):
        super(deeplabv3plus, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16//cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        # self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        # self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL,
                      1, padding=cfg.MODEL_SHORTCUT_KERNEL//2, bias=True),
            SynchronizedBatchNorm2d(
                cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM,
                      cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(
                cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM,
                      3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(
                cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(
            cfg.MODEL_ASPP_OUTDIM, num_class, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(
            backbone, os=cfg.MODEL_OUTPUT_STRIDE, pretrained=pretrained_base, **kwargs)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        b = x.size()[0]
        c = x.size()[1]
        h = x.size()[2]
        w = x.size()[3]
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        # feature_aspp = self.upsample_sub(feature_aspp)
        upsample_sub = nn.UpsamplingBilinear2d(
            size=(layers[0].size()[2], layers[0].size()[3]))
        feature_aspp = upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        # result = self.upsample4(result)
        upsamplele4 = nn.UpsamplingBilinear2d(size=(h, w))
        result = upsamplele4(result)
        return result


def build_backbone(backbone_name, pretrained=False, os=16, **kwargs):
    if backbone_name == 'resnet50':
        net = resnet50_atrous(pretrained=pretrained, os=os, **kwargs)
        return net
    elif backbone_name == 'resnet101':
        net = resnet101_atrous(pretrained=pretrained, os=os, **kwargs)
        return net
    elif backbone_name == 'resnet152':
        net = resnet152_atrous(pretrained=pretrained, os=os, **kwargs)
        return net
    elif backbone_name == 'xception' or backbone_name == 'Xception':
        net = xception(pretrained=False, os=os, **kwargs)
        return net
    else:
        raise ValueError(
            'backbone.py: The backbone named %s is not supported yet.' % backbone_name)


class ASPP(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0,
                      dilation=rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 *
                      rate, dilation=6*rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 *
                      rate, dilation=12*rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 *
                      rate, dilation=18*rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = SynchronizedBatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
#		self.conv_cat = nn.Sequential(
#				nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
#				SynchronizedBatchNorm2d(dim_out),
#				nn.ReLU(inplace=True),
#		)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(
            global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result


# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------


class Configuration():
    def __init__(self):
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        # self.MODEL_NUM_CLASSES = 2
        self.TRAIN_BN_MOM = 0.0003


cfg = Configuration()

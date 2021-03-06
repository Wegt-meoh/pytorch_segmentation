"""Bilateral Segmentation Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_models.resnet_atrous import resnet50_atrous, resnet101_atrous, resnet152_atrous
from models.base_models.resnet import resnet34
# H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
#                         \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor


class BiSeNetVVV(nn.Module):
    def __init__(self, num_class, backbone='resnet34', pretrained_base=False, aux=False, **kwargs):
        super(BiSeNetVVV, self).__init__()
        self.spatial_path = SpatialPath(3, 128, **kwargs)
        self.context_path = ContextPath(backbone, pretrained_base, **kwargs)
        self.ffm = FeatureFusion(256, 256, 4, **kwargs)
        self.head = _BiSeHead(256, 64, num_class, **kwargs)
        self.aux = aux

        if aux:
            self.auxlayer1 = _BiSeHead(128, 256, num_class, **kwargs)
            self.auxlayer2 = _BiSeHead(128, 256, num_class, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        spatial_out = self.spatial_path(x)
        context_out = self.context_path(x)
        fusion_out = self.ffm(spatial_out, context_out[-1])

        x = self.head(fusion_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs = []

        outputs.append(x)

        if self.aux:
            auxout1 = self.auxlayer1(context_out[0])
            auxout1 = F.interpolate(
                auxout1, size, mode='bilinear', align_corners=True)
            outputs.append(auxout1)
            auxout2 = self.auxlayer2(context_out[1])
            auxout2 = F.interpolate(
                auxout2, size, mode='bilinear', align_corners=True)
            outputs.append(auxout2)

        return tuple(outputs)


class _BiSeHead(nn.Module):
    def __init__(self, in_channels, inter_channels, num_class, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_BiSeHead, self).__init__()
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, inter_channels, 3,
                        1, 1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            Depthwise_Separable_Conv(inter_channels, num_class, 1)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class SpatialPath(nn.Module):
    """Spatial path"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, rate=1, **kwargs):
        super(SpatialPath, self).__init__()

        self.conv7x7 = _ConvBNReLU(
            in_channels, 64, 7, 2, 3, norm_layer=norm_layer)
        self.conv3x3_1 = _ConvBNReLU(
            64, 64, 3, 2, 1, norm_layer=norm_layer)
        self.conv3x3_2 = _ConvBNReLU(
            64, 64, 3, 2, 1, norm_layer=norm_layer)

        self.branch1 = nn.Sequential(
            nn.Conv2d(64, out_channels, 1,
                      1, 0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, 1,
                      6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, 1, 12 *
                      rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, 1, 18 *
                      rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        # self.branch5_conv = nn.Conv2d(
        #     in_channels, out_channels, 1, 1, 0, bias=True)
        # self.branch5_bn = nn.BatchNorm2d(out_channels)
        # self.branch5_relu = nn.ReLU(inplace=True)
        # self.conv_cat = nn.Sequential(
        #     nn.Conv2d(out_channels*5, out_channels,
        #               1, 1, padding=0, bias=True),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        # )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # global_feature = torch.mean(x, 2, True)
        # global_feature = torch.mean(global_feature, 3, True)
        # global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature)
        # global_feature = self.branch5_relu(global_feature)
        # global_feature = F.interpolate(
        #     global_feature, (row, col), None, 'bilinear', True)

        # feature_cat = torch.cat(
        #     [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class _GlobalAvgPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_GlobalAvgPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Depthwise_Separable_Conv(in_channels, out_channels, 1),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class AttentionRefinmentModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(AttentionRefinmentModule, self).__init__()
        self.conv3x3 = _ConvBNReLU(
            in_channels, out_channels, 3, 1, 1, norm_layer=norm_layer)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels, 1,
                        1, 0, norm_layer=norm_layer),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv3x3(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x


class ContextPath(nn.Module):
    def __init__(self, backbone='resnet18', pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ContextPath, self).__init__()
        if backbone == 'resnet18':
            pretrained = resnet18(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet34':
            pretrained = resnet34(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet50':
            pretrained = resnet50(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet101':
            pretrained = resnet101(pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

        inter_channels = 128
        basic_input_channels = 256
        if backbone not in ('resnet18', 'resnet34'):
            basic_input_channels *= 4
        self.global_context = _GlobalAvgPooling(
            basic_input_channels*2, inter_channels, norm_layer)

        self.arms = nn.ModuleList(
            [AttentionRefinmentModule(basic_input_channels*2, inter_channels, norm_layer, **kwargs),
             AttentionRefinmentModule(basic_input_channels, inter_channels, norm_layer, **kwargs)]
        )
        self.refines = nn.ModuleList(
            [_ConvBNReLU(inter_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer),
             _ConvBNReLU(inter_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer)]
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        context_blocks = []
        context_blocks.append(x)
        x = self.layer2(x)
        context_blocks.append(x)
        c3 = self.layer3(x)
        context_blocks.append(c3)
        c4 = self.layer4(c3)
        context_blocks.append(c4)
        context_blocks.reverse()

        global_context = self.global_context(c4)
        last_feature = global_context
        context_outputs = []
        for i, (feature, arm, refine) in enumerate(zip(context_blocks[:2], self.arms, self.refines)):
            feature = arm(feature)
            feature += last_feature
            last_feature = F.interpolate(feature, size=context_blocks[i + 1].size()[2:],
                                         mode='bilinear', align_corners=True)
            last_feature = refine(last_feature)
            context_outputs.append(last_feature)

        return context_outputs


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = _ConvBNReLU(
            in_channels, out_channels, 1, 1, 0, norm_layer=norm_layer, **kwargs)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels // reduction,
                        1, 1, 0, norm_layer=norm_layer),
            _ConvBNReLU(out_channels // reduction, out_channels,
                        1, 1, 0, norm_layer=norm_layer),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        pass

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

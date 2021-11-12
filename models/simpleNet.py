import torch
import torch.nn.functional as F
import torch.nn as nn

from models.base_models.resnet import *

#(h+2*padding-dilation*(kernel_size-1))-1
# ------------------------------------------+1 下取整
#stride
class SimpleNet(nn.Module):
    def __init__(self,num_class,pretrained_base=True,**kwargs):
        super().__init__()
        self.spatial_path=SpatialPath()
        self.context_path=ContextPath(pretrained_base,**kwargs)
        self.decoder=nn.Sequential(
            nn.Conv2d(512,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.cls_conv=nn.Conv2d(128,num_class,1,1)

    def forward(self,x):
        h,w=x.size()[2:]

        spatial_out=self.spatial_path(x)
        context_out=self.context_path(x)

        x=torch.cat(spatial_out,context_out)

        x=self.decoder(x)

        x=self.cls_conv(x)

        x=F.interpolate(size=(h,w),mode='bileaner',align_corners=True)

        return x

class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,  3,2,1)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(64,128,3,2,1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,256,3,2,1)
        self.bn3=nn.BatchNorm2d(256)
        self.conv3=nn.Conv2d(128,512,3,2,1)
        self.bn3=nn.BatchNorm2d(512)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)

        return x

class ContextPath(nn.Module):
    def __init__(self,pretrained_base,**kwargs):
        super().__init__()
        self.backbone=resnet34(pretrained=pretrained_base,**kwargs)
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        return x

if __name__=='__main__':
    pass
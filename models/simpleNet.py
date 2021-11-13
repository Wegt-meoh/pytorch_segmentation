import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.activation import ReLU

from models.base_models.resnet import *

#(h+2*padding-dilation*(kernel_size-1))-1
# ------------------------------------------+1 下取整
#stride
class SimpleNet(nn.Module):
    def __init__(self,num_class,pretrained_base=True,**kwargs):
        super().__init__()
        self.spatial_path=SpatialPath()        
        self.decoder=Decoder()
        self.cls_conv=nn.Conv2d(256,num_class,1,1,padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.context_path=ContextPath(pretrained_base,**kwargs)

    def forward(self,x):
        h,w=x.size()[2:]

        spatial_out=self.spatial_path(x)
        context_out=self.context_path(x)

        x=torch.cat((spatial_out,context_out),dim=1)

        x=self.decoder(x)#channels 1280

        x=self.cls_conv(x)

        x=F.interpolate(x,size=(h,w),mode='bilinear',align_corners=True)

        return x

class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(
            nn.Conv2d(3,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128,256,3,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,512,3,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),            
            nn.Dropout(0.1)
        )

    def forward(self,x):
        return self.head(x)

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
        if pretrained_base == False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        c3=self.layer3(x)
        c4=self.layer4(c3)
        c4=F.interpolate(c4,c3.size()[2:],mode='bilinear',align_corners=True)

        res=torch.cat((c3,c4),dim=1)

        return res

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1280,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )        
        self.upsample=nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2=nn.Sequential(
            nn.Conv2d(512,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(256,256,1,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(256,256,1,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(256,256,1,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(256,256,1,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
    def forward(self,x):
        x=self.conv1(x)        
        x=self.conv2(x)        
        x=self.conv3(x)
        x=self.upsample(x)                
        x=self.conv4(x)   
        x=self.upsample(x)     
        x=self.conv5(x)   
        x=self.upsample(x)     
        x=self.conv6(x)        

        return x


if __name__=='__main__':
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
import cv2


class bottle_neck(nn.Module):
    def __init__(self,in_channels,mid_channels,stride=1,downsample=False):
        super().__init__()
        #assert in_channels==4*mid_channels
        self.conv1=nn.Conv2d(kernel_size=1,in_channels=in_channels,out_channels=mid_channels,stride=stride)
        self.bn1=nn.BatchNorm2d(num_features=mid_channels)
        self.conv2=nn.Conv2d(kernel_size=3,in_channels=mid_channels,out_channels=mid_channels,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=mid_channels)
        self.conv3=nn.Conv2d(kernel_size=1,in_channels=mid_channels,out_channels=mid_channels*4)
        self.bn3=nn.BatchNorm2d(mid_channels*4)
        self.relu=nn.ReLU(inplace=True)
        self.stride=stride
        self.downsample=downsample

        if self.downsample:
            self.conv4=nn.Conv2d(kernel_size=1,in_channels=in_channels,out_channels=4*mid_channels,stride=stride)
            self.bn4=nn.BatchNorm2d(num_features=4*mid_channels)

    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample:
            residual=self.conv4(residual)
            residual=self.bn4(residual)
        out+=residual
        out=self.relu(out)
        return out

class block1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            bottle_neck(in_channels=64, mid_channels=64, stride=1, downsample=True),
            bottle_neck(in_channels=256, mid_channels=64),
            bottle_neck(in_channels=256, mid_channels=64)
        )
    def forward(self,x):
        return self.layer(x)


class block2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            bottle_neck(in_channels=256, mid_channels=128, stride=2, downsample=True),
            bottle_neck(in_channels=512, mid_channels=128),
            bottle_neck(in_channels=512, mid_channels=128),
            bottle_neck(in_channels=512, mid_channels=128)
        )
    def forward(self,x):
        return self.layer(x)


class block3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            bottle_neck(in_channels=512,mid_channels=256,stride=2,downsample=True),
            bottle_neck(in_channels=1024,mid_channels=256),
            bottle_neck(in_channels=1024, mid_channels=256),
            bottle_neck(in_channels=1024, mid_channels=256),
            bottle_neck(in_channels=1024, mid_channels=256),
            bottle_neck(in_channels=1024, mid_channels=256)
        )

    def forward(self,x):
        return self.layer(x)

class block4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            bottle_neck(in_channels=1024,mid_channels=512,stride=2,downsample=True),
            bottle_neck(in_channels=2048,mid_channels=512),
            bottle_neck(in_channels=2048, mid_channels=512),
        )

    def forward(self,x):
        return self.layer(x)

class resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0=nn.Conv2d(kernel_size=7,in_channels=3,out_channels=64,stride=2,padding=3)
        self.bn0=nn.BatchNorm2d(num_features=64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.b1=block1()
        self.b2=block2()
        self.b3=block3()
        self.b4=block4()
        self.averagepooling=nn.AvgPool2d(kernel_size=7)
        self.fc=nn.Linear(2048,45)
        #self.layer=nn.Sequencial

    def forward(self,x):
        '''
        :param x: [batchsize,channels,H,W]
        :return:
        '''
        out=self.conv0(x)
        out=self.bn0(out)
        out=self.relu(out)
        out=self.maxpooling(out)
        out=self.b1(out)
        out=self.b2(out)
        out=self.b3(out)
        out=self.b4(out) #[batchsize,2048,7,7]
        out=self.averagepooling(out) #[batchsize,2048,1,1]
        out=out.squeeze(dim=2)
        out =out.squeeze(dim=2)
        out=self.fc(out) #maybe [batchsize,45]
        return out

class pretrained_resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=torchvision.models.resnet50(pretrained=True)
        self.droputs = [nn.Dropout(0.5) for _ in range(5)]
        self.regressor = nn.Linear(1000, 45)

    def forward(self, x):
        '''
        :param x: [batchsize,channels,H,W]
        :return: out [batchsize,45]
        '''
        out = self.resnet(x)
        for idx in range(5):
            if idx == 0:
                logits = self.regressor(out)
            else:
                logits += self.regressor(out)
        logits /= 5
        return logits


class pretrained_resnet152(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=torchvision.models.resnet152(pretrained=True)
        self.droputs=[nn.Dropout(0.5) for _ in range(5)]
        self.regressor=nn.Linear(1000,45)

    def forward(self,x):
        '''
        :param x: [batchsize,channels,H,W]
        :return: out [batchsize,45]
        '''
        out=self.resnet(x)
        for idx in range(5):
            if idx==0:
                logits=self.regressor(out)
            else:
                logits+=self.regressor(out)
        logits/=5
        return logits


class pretrained_resnet101_32x8d(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=torchvision.models.resnext101_32x8d(pretrained=True)
        self.fc=nn.Linear(1000,45)

    def forward(self,x):
        '''
        :param x: [batchsize,channels,H,W]
        :return: out [batchsize,45]
        '''
        out=self.resnet(x)
        out=self.fc(out)
        return out

class pretrained_resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=torchvision.models.resnet34(pretrained=True)
        self.droputs = [nn.Dropout(0.5) for _ in range(5)]
        self.regressor = nn.Linear(1000, 45)

    def forward(self, x):
        '''
        :param x: [batchsize,channels,H,W]
        :return: out [batchsize,45]
        '''
        out = self.resnet(x)
        for idx in range(5):
            if idx == 0:
                logits = self.regressor(out)
            else:
                logits += self.regressor(out)
        logits /= 5
        return logits

class pretrained_resnet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=torchvision.models.resnet101(pretrained=True)
        self.droputs = [nn.Dropout(0.5) for _ in range(5)]
        self.regressor = nn.Linear(1000, 45)

    def forward(self, x):
        '''
        :param x: [batchsize,channels,H,W]
        :return: out [batchsize,45]
        '''
        out = self.resnet(x)
        for idx in range(5):
            if idx == 0:
                logits = self.regressor(out)
            else:
                logits += self.regressor(out)
        logits /= 5
        return logits
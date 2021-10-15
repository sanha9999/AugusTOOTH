import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Identity
from torch.nn.modules.pooling import MaxPool2d

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()

        # model (ResNet)
        

    def forward(self, x):

        return x

def conv_1(): # model start
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )

def bottleneck_block(in_channels, mid_channels, out_channels, down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0))
    layers.extend([
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
    ])
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, down:bool = False, starting:bool=False):
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_channels, mid_channels, out_channels, down=down)
        self.relu = nn.ReLU(inplace=True)
        if down:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0) # size 줄이기
        else:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.changedim = nn.Sequential(conv_layer, nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def make_layer(in_channels, mid_channels, out_channels, repeats, starting = False):
    layers = []
    layers.append(Bottleneck(in_channels, mid_channels, out_channels, down=True, starting=starting))
    for _ in range(1, repeats):
        layers.append(Bottleneck(out_channels, mid_channels, out_channels, down=False))
    return nn.Sequential(*layers)


class ResNet(nn.modules):
    def __init__(self, repeats:list = [3,4,6,3], num_classes = 2):
        super(ResNet, self).__init__()
        self.num_classes = num_classes

        self.conv1 = conv_1()

        base_dim = 64
        self.conv2 = make_layer(base_dim, base_dim, base_dim*4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim*4, base_dim*2, base_dim*8, repeats[1])
        self.conv4 = make_layer(base_dim*8, base_dim*4, base_dim*16, repeats[2])
        self.conv5 = make_layer(base_dim*16, base_dim*8, base_dim*32, repeats[3])
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifer = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x
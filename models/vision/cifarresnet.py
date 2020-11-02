'''
Resnet with custom dataset and pretrained by pytorch model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time

from models.utils.inspect_model import count_parameters, plot_lr_finder, train_eval, epoch_time
from models.utils.lr_finder import LRFinder
from models.vision.visionbase import VisionBase


ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])


class Identity(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class CIFARBasicBlock(nn.Module):


    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            identity_fn = lambda x : F.pad(x[:, :, ::2, ::2],
                                           [0, 0, 0, 0, in_channels // 2, in_channels // 2])
            downsample = Identity(identity_fn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class CIFARResNet(nn.Module):

    cifar_resnet20_config = ResNetConfig(block=CIFARBasicBlock,
                                         n_blocks=[3, 3, 3],
                                         channels=[16, 32, 64])

    cifar_resnet32_config = ResNetConfig(block=CIFARBasicBlock,
                                         n_blocks=[5, 5, 5],
                                         channels=[16, 32, 64])

    cifar_resnet44_config = ResNetConfig(block=CIFARBasicBlock,
                                         n_blocks=[7, 7, 7],
                                         channels=[16, 32, 64])

    cifar_resnet56_config = ResNetConfig(block=CIFARBasicBlock,
                                         n_blocks=[9, 9, 9],
                                         channels=[16, 32, 64])

    cifar_resnet110_config = ResNetConfig(block=CIFARBasicBlock,
                                          n_blocks=[18, 18, 18],
                                          channels=[16, 32, 64])

    cifar_resnet1202_config = ResNetConfig(block=CIFARBasicBlock,
                                           n_blocks=[20, 20, 20],
                                           channels=[16, 32, 64])

    def __init__(self, config, output_dim):
        super().__init__()

        block, layers, channels = config
        self.in_channels = channels[0]

        assert len(layers) == len(channels) == 3
        assert all([i == j*2 for i, j in zip(channels[1:], channels[:-1])])

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self.get_resnet_layer(block, layers[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, layers[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, layers[2], channels[2], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):

        layers = []

        if self.in_channels != channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(channels, channels))

        self.in_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h

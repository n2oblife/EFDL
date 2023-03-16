import math
import pickle
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import numpy as np 
import torchvision.transforms as transforms
from torchvision import models
import torch 
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

## Normalization adapted for CIFAR
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,])

# ------------------------------------
# Model definition

class MBConv(nn.Module):
    def __init__(self,conv, in_channel, stride): # TODO complete the arguments
        super(FusedMBConv, self).__init__()
        inter_channels = 4*in_channel
        self.conv1 = nn.Conv2d(in_channel, inter_channels, 1)
        self.dconv = nn.Conv2d(inter_channels, inter_channels, 3, groups=inter_channels)
        self.SE =
        self.conv2 =

    def forward(self, x):
        out = self.dconv(F.relu(self.conv1(x)))
        out = self.conv2(F.relu(self.SE(out)))
        out = torch.cat([out,x],1)
        return out

class FusedMBConv(nn.Module):
    def __init__(self, in_channel): # TODO complete the arguments
        super(FusedMBConv,self).__init__()
        self.conv1 =
        self.SE =
        self.conv2 =
    
    def forward(self,x):
        out = self.SE(F.relu(self.conv1(x)))
        out = self.conv2(out)
        out = torch.cat([out,x],1)
        return out

class EfficientNet(nn.Module):

import torch
from torch import nn
from torch.functional import F
from torch.nn import Conv2d, MaxPool2d, LeakyReLU, BatchNorm2d, Linear, Flatten
from torchvision.io import read_image
from torchvision.transforms import Resize
from matplotlib import pyplot as plt


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)

        return x


class YOLO(nn.Module):
    """ 
    YOLO V1 network
    """

    def __init__(self):
        super(YOLO, self).__init__()
    

    def forward(self, x):
        
        return x
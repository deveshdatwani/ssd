from torch import nn 
from model import YOLO
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.nn import MSELoss
from utils import *
import torch


class Criterion(nn.Module):
    """
    This class inherits from the Module class and calculates loss value for a given prediction
    The forward function implments the loss function of YOLO model as described in the paper
    """

    def __init__(self, lambda_coord=5, lambda__nooj=0.5, S=7, B=2, C=20, im_width=448, im_height=448):
        self.lambda_coord = lambda_coord    
        self.lambda_noobj = lambda__nooj
        self.S = S
        self.B = B
        self.C = C
        self.im_wdith = im_width
        self.im_height = im_height
        self.mse_loss = MSELoss(reduction="sum")

    def forward(self, predictions, target):
        
        # This is not going to be easy  

        return None


if __name__ == "__main__":


    out = torch.rand((1,7,7,30))
    target = torch.rand((1, 7, 7, 30))
    criterion = Criterion()
    criterion(out, target)

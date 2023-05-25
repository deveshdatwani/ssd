from torch import nn 
from model import YOLO
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.nn import MSELoss
from utils import *
import torch


class Criterion(nn.Module):
    '''
    Implments loss function of YOLO V1 model as per the paper: www.arxiv.org/pdf/1506.02640.pdf 
    '''
    def __init__(
            self, 
            lambda_coord=5, 
            lambda__nooj=0.5, 
            S=7, B=2, C=20, 
            im_width=448, 
            im_height=448
        ):
        self.lambda_coord = lambda_coord    
        self.lambda_noobj = lambda__nooj
        self.S = S
        self.B = B
        self.C = C
        self.mse_loss = MSELoss(reduction="sum")


    def forward(self, predictions, target):
        
        return None


if __name__ == "__main__":
    out = torch.rand((1,7,7,30))
    target = torch.rand((1, 7, 7, 30))
    criterion = Criterion()
    criterion(out, target)

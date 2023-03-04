import torch 
from torch import nn 
from torch.optim import optimizer
from torchvision.models import resnet18


class SSD():
    def __init__(self, ):
        self.backboneNetwork = resnet18(weights='DEFAULT') 

    def forward(self, x):
        loc = None
        classification  = None
        
        return loc, classification

    def train(self):
        return None

    def eval(self):
        return self.backboneNetwork.eval


if __name__ == "__main__":
    model = SSD()
    print(model.eval())
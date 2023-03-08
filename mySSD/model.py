import torch 
from torch import nn 
from torch.optim import optimizer
from torchvision.models import resnet18


class SSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.backboneNetwork = resnet18(weights='DEFAULT') 

    def forward(self, x):
        backboneOutput = self.backboneNetwork(x) 
        return backboneOutput
    

if __name__ == "__main__":
    model = SSD()
    print(model.eval())
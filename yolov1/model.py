
import torch
from torch import nn
from torch.functional import F
from torch.nn import Conv2d, MaxPool2d, LeakyReLU, BatchNorm2d, Linear, Flatten
from torchvision.io import read_image
from torchvision.transforms import Resize
from matplotlib import pyplot as plt



class YOLO(nn.Module):
    """ 
    author: devesh datwani

    This class builds a pytorch model based on the first version of YOLO network
    This class will be pretty inflexible and pretty straightforward
    Thereforce, no choice of backbone, 

    """

    def __init__(self):
        super(YOLO, self).__init__()

        self.c_layer1 = Conv2d(3, 64, 7, 2, padding=3)
        self.c_layer2 = Conv2d(64, 192, 3, 1, padding='same')
        self.c_layer3 = Conv2d(192, 128, 1, 1, padding='same')
        self.c_layer4 = Conv2d(128, 256, 3, 1, padding='same')
        self.c_layer5 = Conv2d(256, 256, 1, 1, padding='same')
        self.c_layer6 = Conv2d(256, 512, 3, 1, padding='same')
        self.c_layer7 = Conv2d(512, 256, 1, 1, padding='same')
        self.c_layer8 = Conv2d(256, 512, 3, 1, padding='same')
        self.c_layer9 = Conv2d(512, 512, 1, 1, padding='same')
        self.c_layer10 = Conv2d(512, 1024, 3, 1, padding='same')
        self.c_layer11 = Conv2d(1024, 512, 1, 1, padding='same')
        self.c_layer12 = Conv2d(512, 1024, 3, 1, padding='same')               
        self.c_layer13 = Conv2d(1024, 1024, 3, 2, padding=1)
        self.c_layer14 = Conv2d(1024, 1024, 3, 1, padding='same')
        
        self.linear1 = Linear(7*7*1024, 1024)
        self.linear2 = Linear(1024, 7*7*30)

        self.mp_layer1 = MaxPool2d(2, stride=2)

        self.leaky_relu = LeakyReLU(0.1)
        
        self.flatten = Flatten(0, -1)

        # self.batchnorm = BatchNorm2d()
        

    def forward(self, x) :
        x = self.c_layer1(x)
        x = self.mp_layer1(x)
        x = self.c_layer2(x)
        x = self.mp_layer1(x)
        x = self.c_layer3(x)
        x = self.c_layer4(x)
        x = self.c_layer5(x)
        x = self.c_layer6(x)
        x = self.mp_layer1(x)
        
        # x4
        for i in range(4):
            x = self.c_layer7(x)
            x = self.c_layer8(x)

        x = self.c_layer9(x)
        x = self.c_layer10(x)
        x = self.mp_layer1(x)

        # x2
        for i in range(2):
            x = self.c_layer11(x)
            x = self.c_layer12(x)

        x = self.c_layer13(x)
        x = self.c_layer14(x)
        x = self.c_layer14(x)
        x = self.flatten(x)
        x = self.linear1(x) 
        x = self.linear2(x)
        x = x.view(30, 7, 7)

        return x


if __name__ == "__main__":
    
    def test():
        model = YOLO()
        image = read_image("/home/deveshdatwani/plane.jpg")
        resized_image = Resize((448, 448), antialias=True)(image)
        out = model(resized_image.float())
        
        return None

    test()
    

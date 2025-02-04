import torch
from torch import nn
from config import architecture_config


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

    def __init__(self, in_channels=3, architectire_config=None, **kwargs):
        super(YOLO, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.split_size = 7 
        self.num_boxes = 2 
        self.num_classes = 20
        self._darknet = self._create_conv_layers(self.architecture)
        self._fcs = self._create_fcs(self.split_size, self.num_boxes, self.num_classes)
    

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for i, x in enumerate(architecture):

            if type(x) == tuple:    
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]

            elif type(x) == str: layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            elif type(x) == list: 
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1]
        
        return nn.Sequential(*layers) 
    

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        fcs = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(1024 * S * S, 496),
                            nn.Dropout(0.0),
                            nn.LeakyReLU(0.1),
                            nn.Linear(496, S * S * (C + B * 5))
            )
        
        return fcs


    def forward(self, x):
        x = self._darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self._fcs(x)

        return x
import torch
from utils import *


if __name__ == '__main__':
    output = torch.rand((2,7,7,30))
    print(f'output {output.shape}')
    IoU(output, 9, 0)


    
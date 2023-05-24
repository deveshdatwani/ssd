import cv2
import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from matplotlib import pyplot as plt


def IoU(bbox1: torch.tensor, bbox2: torch.tensor, target: torch.tensor):
    '''
    Args: bbox1 shape: batch_size, 7, 7, 4
          bbox2 shape: batch_size, 7, 7, 4 
          target shape: batch_size, 7, 7, 4
    '''

    xybbox1 = bbox1[:,:,:,:2]
    whbbox1 = bbox1[:,:,:,2:4]

    leftbbox1 = xybbox1 - (whbbox1 // 2)
    rightbbox1 = xybbox1 + (whbbox1 // 2)

    print(leftbbox1.shape)

    xybbox2 = bbox1[:,:,:,:2]
    whbbox2 = bbox1[:,:,:,2:4]

    leftbbox2 = xybbox1 - (whbbox1 // 2)
    rightbbox2 = xybbox1 + (whbbox1 // 2)

    return None


def visualize_sample(image: torch.tensor, label: np.ndarray):
    '''
    Args: Image: np.ndarray H * W * C
    '''

    numpy_image = image.numpy().transpose(1, 2, 0).astype(np.uint8).copy()
    h, w, _ = numpy_image.shape
    
    for c, x, y, width, height in label:
        x1 = int(x*w - ((width*w) / 2)) 
        y1 = int(y*h - ((height*h) / 2)) 
        x2 = int(x*w + ((width*w) / 2)) 
        y2 = int(y*h + ((height*h) / 2)) 

        numpy_image = cv2.rectangle(numpy_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(numpy_image)
    plt.show()

    return None
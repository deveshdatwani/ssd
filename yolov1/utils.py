import os
import cv2
import torch
import numpy as np
torch.manual_seed(0)
from torch import nn
from math import sqrt
from matplotlib import pyplot as plt


def IoU(bbox1: torch.tensor, bbox2: torch.tensor, target: torch.tensor):
    xybbox1 = bbox1[:,:,:,:2]
    whbbox1 = bbox1[:,:,:,2:4]

    leftbbox1 = xybbox1 - (whbbox1 // 2)
    rightbbox1 = xybbox1 + (whbbox1 // 2)

    xybbox2 = bbox1[:,:,:,:2]
    whbbox2 = bbox1[:,:,:,2:4]

    leftbbox2 = xybbox1 - (whbbox1 // 2)
    rightbbox2 = xybbox1 + (whbbox1 // 2)

    return None


def visualize_sample(image: torch.tensor, label: np.ndarray):
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


class dataVisualizer():
    def __init__(self, cols=5, rows=2, data_directory=None, number_of_images=25) -> None:
        self._cols = cols
        self._rows = rows
        self._data_directory = data_directory
        self._figsize = (10,10)
        self.number_of_images = number_of_images


    def visualize(self) -> None:
        image_cell = int(sqrt(self.number_of_images))
        image_list = os.listdir(self._data_directory)
        
        for i in range(self.number_of_images):
            random_id = np.random.randint(0, len(image_list)-1)
            image_name = image_list[random_id]
            image = cv2.imread(os.path.join(self._data_directory, image_name))
            plt.subplot(image_cell, image_cell, i+1)
            plt.imshow(image)
        
        plt.show()
        
    
    def set_grid_size(self, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows


    def set_figsize(self, figsize: tuple) -> None:
        self._figsize = figsize


def intersection_over_union():

    return None
import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from random import randint
from utils import visualize_sample
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image


class imageDataset(Dataset):
    '''
    Author: Devesh Datwani
    Class for loading fetching images for training
    '''
    
    def __init__(self, data_directory="/home/deveshdatwani/Datasets/voc/images", annotation_file_address="/home/deveshdatwani/Datasets/voc") -> object:
        self._annotations = pd.read_csv(os.path.join(annotation_file_address, "train.csv")).to_numpy()
        self._image_dataset = data_directory
        self._label_directory = os.path.join(annotation_file_address, "labels")
        self.S = 7 # grid size
        self.B = 2 # n bounding boxes
        self.C = 20 # n classes
        self.w = 448 
        self.h = 448


    def __len__(self):
        return len(self._annotations)
    

    def __getitem__(self, idx):

        image_path = os.path.join(self._image_dataset, self._annotations[idx, 0])
        image = read_image(image_path)
        label = np.loadtxt(os.path.join(self._label_directory, self._annotations[idx, 1]))
        sample = {"image": image, "label": label}
        
        return sample
    
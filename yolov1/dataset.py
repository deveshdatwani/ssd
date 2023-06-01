import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from random import randint
from utils import visualize_sample
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image


class VOCDatase(Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None 
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.C = C
        self.S = S
        self.B = B


    def __len__(self):
        return len(self.annotations)
        

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) 
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace('\n', '').split()] 
            boxes.append([class_label, x, y, w, h])

            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = Image.open(img_path)
            boxes = torch.tensor(boxes)

            if self.transform:
                image, box = self.transform(image, boxes) 
            
            label_matrix = torch.zeros((self.S, self.S, self.C + 5))

            for box in boxes:
                class_label, x, y, w, h = box.tolist()
                class_label = int(class_label)

                i, j = int(self.S * y), int(self.S * x) 
                width_cell, height_cell = w * self.S, h * self.S

                # It is easier to train ''' OFFSET ''' than what is being implemented 

                if label_matrix[i,j,20] == 0:
                    label_matrix[i, j, 20] = 1
                    box_coordinates = torch.tensor([i, j, width_cell, height_cell]) 
                    label_matrix[i, j, 21:25] = box_coordinates
                    label_matrix[i, j, class_label] = 1
            
            return image, label_matrix

        
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import os
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import cv2


class imageDataset(Dataset):
    """
    author: devesh datwani

    Class for loading fetching images for training
    As per pytorch, __len__ and __getitem__ have been overloaded to return the length of the dataset and also index a sample
    There is an option of writing transforms for data augmentation. The transforms have to be applied to the annotations as well 
    ** In this first iteration, I do not want to introduce any transform due to limited compute, but this can be explored later ** 
    """
    
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
    
    
    def draw_bbox(self, image, label):
        _, h, w = image.shape
        numpy_image = image.numpy().transpose(1, 2, 0)
        
        for c, x, y, width, height in label:
            x1 = int(x*w - ((width*w) / 2))
            y1 = int(y*h - ((height*h) / 2))
            x2 = int(x*w + ((width*w) / 2))
            y2 = int(y*h + ((height*h) / 2))
            numpy_image = cv2.rectangle(numpy_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return numpy_image
    

    def transform(self, image):
        transformed_image = None
        
        return image


    def encode(self, labels, image):
        tensor_label = np.zeros((self.S * self.S, self.C + 4 + 1))
        numpy_image = self.draw_bbox(image, labels)
        plt.imshow(numpy_image)
        plt.show()
        
        return tensor_label


    def __getitem__(self, idx):

        image_path = os.path.join(self._image_dataset, self._annotations[idx, 0])
        image = read_image(image_path)
        label = np.loadtxt(os.path.join(self._label_directory, self._annotations[idx, 1]))
        self.encode(label, image)
        sample = {"image": image, "label": label}
        
        return sample
    


if __name__ == "__main__":
    # testing
    trainset = imageDataset()
    sample = trainset[1000]
    ix_image, ix_label = sample.values()
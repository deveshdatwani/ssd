from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import os
import numpy as np


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


    def __len__(self):
        return len(self._annotations)
    

    def transform(self, image):
        transformed_image = None
        
        return image
    

    def __getitem__(self, idx):

        image_path = os.path.join(self._image_dataset, self._annotations[idx, 0])
        image = read_image(image_path)
        label = np.loadtxt(os.path.join(self._label_directory, self._annotations[idx, 1]))
        
        return image, label
    

if __name__ == "__main__":
    trainset = imageDataset()
    ix_image, ix_label = trainset[200]
    print(ix_label)

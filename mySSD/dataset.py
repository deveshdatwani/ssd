from torch import tensor
from torch.utils.data import Dataset
import os
import json
import PIL
import numpy as np
from PIL import Image


class dataSet(Dataset):
    
    def __init__(self, root, annotationFile):
        self.rootDir = root
        with open(annotationFile) as gtFile:
            self.annotationsFile = json.load(gtFile)

    def __len__(self):
        return len(self.annotationsFile)

    def __getitem__(self, idx):
        imageName = os.path.join(self.rootDir, self.annotationsFile[idx][0])
        image = Image.open("hopper.jpg")
        image = np.asarray(image)
        


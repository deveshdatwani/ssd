from matplotlib import pyplot as plt
from math import sqrt
import os
from random import randint
import cv2


class dataVisualizer():
    """
    author: devesh datwani 
    
    This class creates a visualizer for an image dataset through matplotlib
    The constructor requires a dataset address when initializing an object
    The visualize member function should be used to visualize random images in the dataset
    To change the grid size, the member function set_grid_size should be used
    """
    
    
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
            random_id = randint(0, len(image_list)-1)
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



if __name__ == "__main__":
    dataset_address = "/home/deveshdatwani/Datasets/voc/images"
    visualizer = dataVisualizer(data_directory=dataset_address) 
    visualizer.visualize()
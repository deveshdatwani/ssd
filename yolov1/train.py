import torch
import torchvision
import argparse
from model import YOLO
from torch.optim import SGD
from utils import *



class yoloTrainer():
    """    
    author: devesh datwani

    Class for training the yolo network
    Constructor is aimed at making the trainer flexible and modular
    Batch size, learning rate, decay and much more can be changed while building the trainer    
    """


    def __init__(self, batch_size, epochs, learning_rate,
                 momentum, num_workers, weight_decay, 
                 save_folder="./weights/"):
        
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._num_workers = num_workers
        self._save_folder = save_folder
        self._weight_decay = weight_decay
        self._pre_train = True


    def loss(self):
        return None 


    def criterion(self):
        return None


    def train(self):
        model = YOLO()
        print("Looking for weights in weight folder")
        
        if not self._save_folder:
            print("No weights found, starting training")
        
        self._optimizer = SGD(model.parameters(), self._learning_rate, self._momentum, weight_decay=self._weight_decay)

        return None



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="argument parser for training yolo network")
    parser.add_argument('--batch-size', default=4, type=int, help="set batch size for mini batch gradient descent")
    parser.add_argument('--epochs', default=10, type=int, help="set number of epochs to train for")
    parser.add_argument('--learning-rate', default=0.00001, type=float, help="set learning rate of the network")
    parser.add_argument('--momentum', default=0.9, type=float, help="set momentume or update rule")
    parser.add_argument('--weight-decay', default=0.00005, type=float, help="set weight decay")
    parser.add_argument('--num-workers', default=4, type=int, help="set number of workers for parellel compute")

    args = parser.parse_args()
    trainer = yoloTrainer(args.batch_size, args.epochs, args.learning_rate,
                          args.momentum, args.num_workers, args.weight_decay)
    
    print(trainer._weight_decay)
    
    # trainer.train()
    

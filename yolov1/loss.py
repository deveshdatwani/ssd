from torch import nn 
from model import YOLO
from torchvision.io import read_image
from torchvision.transforms import Resize


class Criterion(nn.Module):
    """
    author: devesh datwani

    This class inherits from the Module class and calculates loss value for a given prediction
    The forward function implments the loss function of YOLO model as described in the paper
    """

    def __init__(self, lambda_coord=5, lambda__nooj=0.5, S=7, B=2, C=20):
        self.lambda_coord = lambda_coord    
        self.lambda_noobj = lambda__nooj
        self.S = S
        self.B = B
        self.C = C

    def __forward__(self, x_output, target):
        loss = x_output - target


        return loss


if __name__ == "__main__":

    def test():
        model = YOLO()
        image = read_image("/home/deveshdatwani/plane.jpg")
        resized_image = Resize((448, 448), antialias=True)(image)
        out = model(resized_image.float())
        criterion = Criterion()

        return None
    
    test()
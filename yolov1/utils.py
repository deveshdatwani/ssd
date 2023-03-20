from torch import nn

""" 
author: devesh datwani

Utils for clean code
"""

class IOU(nn.Module):
    """
    author: devesh datwani 

    This class implements an intersection over union calculator through inheriting a Pytorch module

    Args:
        box1_x (float): centre x coordinate of box1 
        box1_y (float): centre y coordinate of box1
        box1_h (float): height of box1 
        box1_w (float): width of box1
        
        box2_x (float): centre x coordinate of box2 
        box2_y (float): centre y coordinate of box2 
        box1_h (float): height of box2 
        box1_w (float): width of box2

    Return: 
        area of intersection
     
    """

    def __init__(self):
        super(IOU, self).__init__()
        pass


    def forward(self, box1_x, box1_y, box1_h, box1_w, box2_x, box2_y, box2_h, box2_w):
        box1_top_left_x = box1_x - box1_w // 2        
        box1_top_left_y = box1_y - box1_h // 2
        
        box1_bottom_left_x = box1_x + box1_w // 2        
        box1_bottom_left_y = box1_y + box1_h // 2

        box2_top_left_x = box2_x - box2_w // 2        
        box2_top_left_y = box2_y - box2_h // 2
        box2_bottom_left_x = box2_x + box2_w // 2        
        box2_bottom_left_y = box2_y + box2_h // 2

        top_x = max(box1_top_left_x, box2_top_left_x)
        top_y = max(box1_top_left_y, box2_top_left_y)

        bottom_x = min(box1_bottom_left_x, box2_bottom_left_x)
        bottom_y = min(box1_bottom_left_y, box2_bottom_left_y)

        intersection = (bottom_x - top_x) * (bottom_y - top_y)

        return intersection


if __name__ == "__main__":

    iou_cal = IOU()
    print(iou_cal(600, 600, 200, 200, 450, 700, 200, 200))
    
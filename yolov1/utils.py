from torch import nn

""" 
author: devesh datwani

Utils for clean code
"""

def iou(box1_x, box1_y, box1_h, box1_w, box2_x, box2_y, box2_h, box2_w):
    
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

def iouTensor(target, predictions):
    """ 
    Args: predictions: batch_size x 49 x bbox1 x bbox2
          target: batch_size x 49 x bbox 
    """
    bbox1 = predictions[:,:,1:5]
    bbox2 = predictions[:,:,5:9]
    targetbox = target[:,:,:4]

    box1topLeftX = bbox1[...,0] - bbox1[...,2] // 2 
    box1topLefty = bbox1[...,1] - bbox1[...,3] // 2

    box1bottomRightX = bbox2[...,0] + bbox1[...,2] // 2 
    box1bottomRighty = bbox2[...,1] + bbox1[...,3] // 2



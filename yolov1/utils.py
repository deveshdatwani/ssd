from torch import nn

""" 
author: devesh datwani

utils for clean code
"""

import torch
torch.manual_seed(0)

def IoU(bbox1, bbox2, target):
    """
    Args: bbox1: batch size, 7, 7, 4
          bbox2: batch size, 7, 7, 4 
    """

    xybbox1 = bbox1.reshape(-1, 49, 4)[:,:,:2]
    whbbox1 = bbox1.reshape(-1, 49, 4)[:,:,2:4]

    xybbox2 = bbox2.reshape(-1, 49, 4)[:,:,:2]
    whbbox2 = bbox2.reshape(-1, 49, 4)[:,:,2:4]

    xyleftbbox1 = xybbox1 + (whbbox1 // 2)
    xyrightbbox1 = xybbox1 - (whbbox1 // 2)

    return None


def objectExists(target):

    return torch.nonzero(target[:,:,0])


bbox1 = torch.randint(0, 2, size=(1,49,4))


see = torch.zeros(size=(1,49,4))[torch.nonzero(bbox1[:,:,0])]
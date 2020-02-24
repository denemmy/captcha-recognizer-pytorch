import numpy as np
import torch

def labels_to_imgs(img_shape, labels):
    return torch.zeros(img_shape, dtype=torch.float32)
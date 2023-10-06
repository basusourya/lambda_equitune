import torch
import numpy as np


class RandomRot90(object):
    """
    Random rotation along given axis in multiples of 90
    """
    def __init__(self, dim1=1, dim2=2):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        k = np.random.randint(0, 4)
        out = torch.rot90(sample, k=k, dims=[self.dim1, self.dim2])
        return out
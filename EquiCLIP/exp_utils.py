import os
import random
import torch
import numpy as np

from torch import cos, sin


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_transform_images(images, group_name="rot90"):
    if group_name == "":
        return torch.stack([images])
    elif group_name == "rot90":
        group_transformed_images = []
        for i in range(4):
            g_images = torch.rot90(images, k=i, dims=(-2, -1))
            group_transformed_images.append(g_images)
        group_transformed_images = torch.stack(group_transformed_images, dim=0)
        return group_transformed_images
    elif group_name == "flip":
        group_transformed_images = []
        for i in range(2):
            if i == 0:
                g_images = images
            else:
                g_images = torch.flip(images, dims=(-2, -1))
            group_transformed_images.append(g_images)
        group_transformed_images = torch.stack(group_transformed_images, dim=0)
        return group_transformed_images
    else:
        raise NotImplementedError


class RandomRot90(object):
    """
    Random rotation along given axis in multiples of 90
    """
    def __init__(self, dim1=-2, dim2=-1):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        k = np.random.randint(0, 4)
        out = torch.rot90(sample, k=k, dims=[self.dim1, self.dim2])
        return out


class RandomFlip(object):
    """
    Random rotation along given axis in multiples of 90
    """
    def __init__(self, dim1=-2, dim2=-1):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        k = np.random.randint(0, 2)
        if k == 1:
            out = torch.flip(sample, dims=(self.dim1, self.dim2))
        else:
            out = sample
        return out


random_rot90 = RandomRot90()
random_flip = RandomFlip()


def random_transformed_images(x, data_transformations=""):
    if data_transformations == "rot90":
        x = random_rot90(x)
    elif data_transformations == "flip":
        x = random_flip(x)
    else:
        x = x
    return x

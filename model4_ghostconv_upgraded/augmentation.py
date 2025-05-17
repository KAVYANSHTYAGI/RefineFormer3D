import torch
import random
import numpy as np

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            axis = torch.randint(0, 3, (1,)).item()  # 0=D, 1=H, 2=W
            image = torch.flip(image, dims=[axis + 1])  # image (C,D,H,W) --> axis+1
            label = torch.flip(label, dims=[axis])      # label (D,H,W) --> axis
        return image, label


class RandomRotation3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(1) < self.p:
            dims = sorted(random.sample([0, 1, 2], 2))  # ensure distinct axes
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=(dims[0]+1, dims[1]+1))  # +1 for image (C, D, H, W)
            label = torch.rot90(label, k, dims=(dims[0], dims[1]))      # label (D, H, W)
        return image, label


class RandomNoise3D:
    def __init__(self, p=0.3, noise_std=0.01):
        self.p = p
        self.noise_std = noise_std

    def __call__(self, image, label):
        if random.random() < self.p:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        return image, label

class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class TumorCoreContrast:
    def __init__(self, p=0.5, scale=1.5):
        """
        Increases intensity contrast in regions labeled as tumor core (label=1)
        Args:
            p (float): probability of applying the transform
            scale (float): how much to amplify contrast (e.g., 1.5 = +50%)
        """
        self.p = p
        self.scale = scale

    def __call__(self, image, label):
        if random.random() < self.p:
            # Create a mask where label == 1 (tumor core)
            mask = (label == 1).unsqueeze(0)  # shape: (1, D, H, W)
            image = image + mask * (image * (self.scale - 1.0))
        return image, label

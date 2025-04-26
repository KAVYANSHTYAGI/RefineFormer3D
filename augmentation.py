import torch
import random
import numpy as np

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            axis = random.choice([2, 3, 4])  # D, H, W axis in input
            image = torch.flip(image, dims=[axis-1])  # image: (C, D, H, W)
            label = torch.flip(label, dims=[axis-1])  # label: (D, H, W)
        return image, label

class RandomRotation3D:
    def __init__(self, p=0.5, angles=[90, 180, 270]):
        self.p = p
        self.angles = angles

    def __call__(self, image, label):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # rotate 90, 180, 270 degrees
            axis = random.choice([(2,3), (2,4), (3,4)])  # rotate in any 2D plane
            image = torch.rot90(image, k, dims=(axis[0]-1, axis[1]-1))
            label = torch.rot90(label, k, dims=(axis[0]-1, axis[1]-1))
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

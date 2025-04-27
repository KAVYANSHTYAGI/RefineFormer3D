import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import random
from augmentation import Compose3D, RandomFlip3D, RandomRotation3D, RandomNoise3D

# --- remapping for BraTS 2017 ---
def remap_labels(label):
    label = label.clone()

    new_label = torch.zeros_like(label)

    new_label[label == 1] = 1  # Tumor Core
    new_label[label == 3] = 1  # Enhancing Tumor
    new_label[label == 2] = 2  # Edema

    return new_label

class BraTSDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        root_dirs: List of paths [HGG path, LGG path] for training
                   or [Validation path] for validation
        """
        self.patient_paths = []
        for root_dir in root_dirs:
            patients = sorted(os.listdir(root_dir))
            for p in patients:
                self.patient_paths.append(os.path.join(root_dir, p))

        self.transform = transform

    def __len__(self):
        return len(self.patient_paths)

    def __getitem__(self, idx):
        patient_path = self.patient_paths[idx]
        patient_id = os.path.basename(patient_path)

        # Load all 4 modalities
        flair = nib.load(os.path.join(patient_path, patient_id + '_flair.nii.gz')).get_fdata()
        t1 = nib.load(os.path.join(patient_path, patient_id + '_t1.nii.gz')).get_fdata()
        t1ce = nib.load(os.path.join(patient_path, patient_id + '_t1ce.nii.gz')).get_fdata()
        t2 = nib.load(os.path.join(patient_path, patient_id + '_t2.nii.gz')).get_fdata()

        # Stack into 4-channel volume
        image = np.stack([flair, t1, t1ce, t2], axis=0)  # (C, H, W, D)

        # Load label
        label = nib.load(os.path.join(patient_path, patient_id + '_seg.nii.gz')).get_fdata()

        # Transpose to (C, D, H, W) and (D, H, W)
        image = np.transpose(image, (0, 3, 1, 2))  # (C, D, H, W)
        label = np.transpose(label, (2, 0, 1))     # (D, H, W)

        # Normalize images
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Remap labels
        label = remap_labels(label)

        # Apply augmentation
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

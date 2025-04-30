import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import random
from augmentation import Compose3D, RandomFlip3D, RandomRotation3D, RandomNoise3D

def center_crop_3d(volume, crop_size=(96, 96, 96)):
    D, H, W = volume.shape[-3:]
    d, h, w = crop_size
    d1, h1, w1 = (D - d) // 2, (H - h) // 2, (W - w) // 2
    return volume[..., d1:d1 + d, h1:h1 + h, w1:w1 + w]

class BraTSDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        self.patient_dirs = []
        for root_dir in root_dirs:
            self.patient_dirs += sorted([
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])

        self.transform = transform

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_path = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_path)

        try:
            flair = nib.load(os.path.join(patient_path, patient_id + '_flair.nii')).get_fdata()
            t1 = nib.load(os.path.join(patient_path, patient_id + '_t1.nii')).get_fdata()
            t1ce = nib.load(os.path.join(patient_path, patient_id + '_t1ce.nii')).get_fdata()
            t2 = nib.load(os.path.join(patient_path, patient_id + '_t2.nii')).get_fdata()
            seg = nib.load(os.path.join(patient_path, patient_id + '_seg.nii')).get_fdata()
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {patient_id} due to missing files: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Try next patient

        image = np.stack([flair, t1, t1ce, t2], axis=0)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        image = center_crop_3d(image, crop_size=(96, 96, 96))
        seg = center_crop_3d(seg, crop_size=(96, 96, 96))
        seg = seg.astype(np.int64)

        # ✅ Clip invalid class values to expected range [0, 1, 2]
        seg = np.clip(seg, 0, 2)
        assert np.max(seg) < 3, f"Still found label >= 3 after clip: {np.unique(seg)}"

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(seg, dtype=torch.long)

        if self.transform:
            image, label = self.transform(image, label)

        # ✅ Debug print (optional)
        print("✅ [DEBUG] image:", type(image), image.shape, image.dtype)
        print("✅ [DEBUG] label:", type(label), label.shape, label.dtype)
        print("✅ [DEBUG] label unique:", torch.unique(label))

        return image, label

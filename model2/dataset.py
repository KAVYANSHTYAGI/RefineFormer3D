import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import nibabel.processing
import numpy as np
from augmentation import Compose3D, RandomFlip3D, RandomRotation3D, RandomNoise3D

def load_and_resample_nii(path, voxel_size=(1.0, 1.0, 1.0)):
    nii = nib.load(path)
    resampled = nib.processing.resample_to_output(nii, voxel_sizes=voxel_size)
    return resampled.get_fdata()

def center_crop_3d(volume, crop_size=(128, 128, 128)):
    D, H, W = volume.shape[-3:]
    d, h, w = crop_size
    d1, h1, w1 = (D - d) // 2, (H - h) // 2, (W - w) // 2
    return volume[..., d1:d1 + d, h1:h1 + h, w1:w1 + w]

class BraTSDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        self.patient_dirs = []
        for root in root_dirs:
            for subdir in sorted(os.listdir(root)):
                full_path = os.path.join(root, subdir)
                if os.path.isdir(full_path):
                    files = os.listdir(full_path)
                    if any(f.endswith("_flair.nii") for f in files):
                        self.patient_dirs.append(full_path)

        print(f"✅ Total valid patient directories found: {len(self.patient_dirs)}")
        self.transform = transform

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_path = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_path)

        try:
            flair = load_and_resample_nii(os.path.join(patient_path, patient_id + '_flair.nii'))
            t1    = load_and_resample_nii(os.path.join(patient_path, patient_id + '_t1.nii'))
            t1ce  = load_and_resample_nii(os.path.join(patient_path, patient_id + '_t1ce.nii'))
            t2    = load_and_resample_nii(os.path.join(patient_path, patient_id + '_t2.nii'))
            seg   = load_and_resample_nii(os.path.join(patient_path, patient_id + '_seg.nii'))
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {patient_id} due to missing files: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Try next patient

        image = np.stack([flair, t1, t1ce, t2], axis=0)

        for c in range(image.shape[0]):
            mean = np.mean(image[c])
            std = np.std(image[c])
            image[c] = (image[c] - mean) / std if std != 0 else image[c] - mean

        image = center_crop_3d(image, crop_size=(128, 128, 128))
        seg = center_crop_3d(seg, crop_size=(128, 128, 128))

        seg = np.rint(seg).astype(np.int64)
        seg = np.clip(seg, 0, 2)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(seg, dtype=torch.long)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

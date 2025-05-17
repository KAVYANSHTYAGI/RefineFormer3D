import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from model5_upgrade import RefineFormer3D
from dataset import BraTSDataset
from optimizer import get_optimizer, get_scheduler
from augmentation import Compose3D, RandomFlip3D, RandomRotation3D, RandomNoise3D
from losses import RefineFormer3DLoss
from config import DEVICE, IN_CHANNELS, NUM_CLASSES, BASE_LR, WEIGHT_DECAY, NUM_EPOCHS

def pad_input_for_windows(x, window_size=(2, 2, 2)):
    _, _, D, H, W = x.shape
    pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
    pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
    pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

def dice_score(pred, target):
    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + 1e-5) / (union + 1e-5)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_list, wt_dice_list, tc_dice_list, et_dice_list = [], [], [], []
    total_correct, total_voxels = 0, 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            inputs = pad_input_for_windows(inputs, window_size=(2, 2, 2))
            targets = targets.to(device=device, dtype=torch.long)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs["main"], dim=1)

            total_correct += (preds == targets).sum().item()
            total_voxels += preds.numel()

            for i in range(preds.shape[0]):
                p = preds[i].cpu().numpy()
                g = targets[i].cpu().numpy()
                for c in range(NUM_CLASSES):
                    dice_list.append(dice_score(p == c, g == c))
                wt_dice_list.append(dice_score(p > 0, g > 0))
                tc_dice_list.append(dice_score((p == 1) | (p == 3), (g == 1) | (g == 3)))
                et_dice_list.append(dice_score(p == 3, g == 3))

    avg_loss = val_loss / len(dataloader.dataset)
    avg_dice = np.mean(dice_list)
    acc = total_correct / total_voxels
    print("\n📊 Evaluation Results:")
    print(f"Avg Dice (mean over all classes): {avg_dice:.4f}")
    print(f"WT Dice (Whole Tumor):           {np.mean(wt_dice_list)*100:.2f}%")
    print(f"TC Dice (Tumor Core):            {np.mean(tc_dice_list)*100:.2f}%")
    print(f"ET Dice (Enhancing Tumor):       {np.mean(et_dice_list)*100:.2f}%")
    print(f"Overall Accuracy:                {acc*100:.2f}%")
    return avg_loss, avg_dice

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        inputs = pad_input_for_windows(inputs, window_size=(2, 2, 2))
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            if torch.isnan(outputs["main"]).any():
                print(" Model output contains NaNs ")
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)

def main():
    device = DEVICE
    model = RefineFormer3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    optimizer = get_optimizer(model, base_lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer)
    criterion = RefineFormer3DLoss()
    scaler = GradScaler(device='cuda')

    train_transform = Compose3D([
        RandomFlip3D(p=0.5),
        RandomRotation3D(p=0.5),
        RandomNoise3D(p=0.3),
    ])

    train_dataset = BraTSDataset(
        root_dirs=["/mnt/m2ssd/research project/Lightweight 3D Vision Transformers for Medical Imaging/dataset/BRATS_SPLIT/train"],
        transform=train_transform,
    )
    val_dataset = BraTSDataset(
        root_dirs=["/mnt/m2ssd/research project/Lightweight 3D Vision Transformers for Medical Imaging/dataset/BRATS_SPLIT/val"],
        transform=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=14, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch [{epoch}/{NUM_EPOCHS}] ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Avg Dice: {val_dice:.4f}")
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

    torch.save(model.state_dict(), "final_model.pt")
    print(" Final model saved as 'final_model.pt'")

if __name__ == "__main__":
    main()
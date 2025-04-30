import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import RefineFormer3D
from losses import RefineFormer3DLoss
from optimizer import get_optimizer, get_scheduler
from metrics import dice_coefficient, iou_score, precision_recall_f1, hausdorff_distance
from dataset import BraTSDataset
from augmentation import Compose3D, RandomFlip3D, RandomRotation3D, RandomNoise3D
from config import DEVICE, IN_CHANNELS, NUM_CLASSES, BASE_LR, WEIGHT_DECAY, NUM_EPOCHS
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():  # Mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate_one_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0

    dice_all, iou_all, hausdorff_all = [], [], []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast():  # Optional mixed precision inference
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            preds = outputs["main"]

            dice_scores, _ = dice_coefficient(preds, targets, num_classes)
            iou_scores, _ = iou_score(preds, targets, num_classes)
            hausdorff = hausdorff_distance(preds, targets)

            dice_all.append(np.mean(dice_scores))
            iou_all.append(np.mean(iou_scores))
            hausdorff_all.append(hausdorff)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    mean_dice = np.nanmean(dice_all)
    mean_iou = np.nanmean(iou_all)
    mean_hausdorff = np.nanmean(hausdorff_all)

    return epoch_loss, mean_dice, mean_iou, mean_hausdorff

def save_checkpoint(state, save_dir, filename="best_model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))

def main():
    # ==================== CONFIG ====================
    device = torch.device(DEVICE)
    save_dir = "./checkpoints"
    best_dice = 0.0
    scaler = GradScaler()

    # ==================== MODEL ====================
    model = RefineFormer3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model = model.to(device)

    # ==================== OPTIMIZER + LOSS ====================
    optimizer = get_optimizer(model, base_lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, mode="cosine", T_max=NUM_EPOCHS)
    criterion = RefineFormer3DLoss()

    # ==================== DATASET + DATALOADER ====================
    train_transform = Compose3D([
        RandomFlip3D(p=0.5),
        RandomRotation3D(p=0.5),
        RandomNoise3D(p=0.3),
    ])

    train_dataset = BraTSDataset(
    root_dirs=[
        "/mnt/m2ssd/research project/Lightweight 3D Vision Transformers for Medical Imaging/dataset/BraTs2017/BRATS2017/Brats17TrainingData/HGG",   # 🔥 Replace this
        "/mnt/m2ssd/research project/Lightweight 3D Vision Transformers for Medical Imaging/dataset/BraTs2017/BRATS2017/Brats17TrainingData/LGG"    # 🔥 Replace this
    ],
    transform=train_transform,
    )   

    val_dataset = BraTSDataset(
    root_dirs=[
        "/mnt/m2ssd/research project/Lightweight 3D Vision Transformers for Medical Imaging/dataset/BraTs2017/BRATS2017/Brats17ValidationData"      # 🔥 Replace this
    ],
    transform=None,
    )


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # ==================== TRAINING LOOP ====================
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch [{epoch}/{NUM_EPOCHS}] ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_dice, val_iou, val_hd = validate_one_epoch(model, val_loader, criterion, device, NUM_CLASSES)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Hausdorff: {val_hd:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
            }, save_dir)
            print(f"✅ Saved Best Model (Dice: {best_dice:.4f})")

if __name__ == "__main__":
    main()

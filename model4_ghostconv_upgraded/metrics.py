import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Dice Coefficient
def dice_coefficient(preds, targets, num_classes):
    smooth = 1e-5
    preds = torch.argmax(preds, dim=1)
    dice_scores = []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = torch.sum(pred_c * target_c)
        union = torch.sum(pred_c) + torch.sum(target_c)

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    mean_dice = np.mean(dice_scores)
    return dice_scores, mean_dice

# Intersection over Union (IoU)
def iou_score(preds, targets, num_classes):
    smooth = 1e-5
    preds = torch.argmax(preds, dim=1)
    iou_scores = []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = torch.sum(pred_c * target_c)
        union = torch.sum(pred_c) + torch.sum(target_c) - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())

    mean_iou = np.mean(iou_scores)
    return iou_scores, mean_iou

# Precision, Recall, F1-Score
def precision_recall_f1(preds, targets, num_classes):
    preds = torch.argmax(preds, dim=1)
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        TP = (pred_c & target_c).sum().item()
        FP = (pred_c & (~target_c)).sum().item()
        FN = ((~pred_c) & target_c).sum().item()

        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)

    return precision_list, recall_list, f1_list, mean_precision, mean_recall, mean_f1

# Hausdorff Distance (surface distance)
def hausdorff_distance(preds, targets, spacing=(1.0, 1.0, 1.0)):
    preds = torch.argmax(preds, dim=1)
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    batch_hd = []
    skipped = 0
    skipped_ids = []

    for i in range(preds.shape[0]):
        print(f"Sample {i} pred shape: {preds[i].shape}, target shape: {targets[i].shape}")

        pred_voxels = np.argwhere(preds[i] > 0)
        target_voxels = np.argwhere(targets[i] > 0)

        if pred_voxels.shape[0] == 0 or target_voxels.shape[0] == 0:
            batch_hd.append(np.nan)
            skipped += 1
            skipped_ids.append(f"sample_{i}: empty mask")
            continue

        if pred_voxels.shape[1] != len(spacing) or target_voxels.shape[1] != len(spacing):
            msg = f"sample_{i}: shape mismatch (pred {pred_voxels.shape}, target {target_voxels.shape})"
            print(f"⚠️ Skipping {msg}")
            batch_hd.append(np.nan)
            skipped += 1
            skipped_ids.append(msg)
            continue

        pred_voxels = pred_voxels * np.array(spacing)
        target_voxels = target_voxels * np.array(spacing)

        hd_pred_to_target = directed_hausdorff(pred_voxels, target_voxels)[0]
        hd_target_to_pred = directed_hausdorff(target_voxels, pred_voxels)[0]
        hd = max(hd_pred_to_target, hd_target_to_pred)

        batch_hd.append(hd)

    if skipped > 0:
        print(f"ℹ️ Skipped {skipped} samples due to issues. Writing details to 'skipped_samples.txt'")
        with open("skipped_samples.txt", "w") as f:
            for msg in skipped_ids:
                f.write(msg + "\n")

    mean_hd = np.nanmean(batch_hd)
    return mean_hd

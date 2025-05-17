import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import RefineFormer3D
from metrics import dice_coefficient, iou_score, precision_recall_f1, hausdorff_distance
import os
from tqdm import tqdm
import numpy as np

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from {checkpoint_path}")

def inference(model, dataloader, device, num_classes, save_preds=False, save_dir="./predictions"):
    model.eval()

    dice_all, iou_all, hausdorff_all = []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Inference")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs["main"]
            preds_softmax = F.softmax(preds, dim=1)
            preds_labels = torch.argmax(preds_softmax, dim=1)

            # Metrics
            dice_scores, _ = dice_coefficient(preds, targets, num_classes)
            iou_scores, _ = iou_score(preds, targets, num_classes)
            hausdorff = hausdorff_distance(preds, targets)

            dice_all.append(np.mean(dice_scores))
            iou_all.append(np.mean(iou_scores))
            hausdorff_all.append(hausdorff)

            # Optional: save predictions
            if save_preds:
                pred_numpy = preds_labels.cpu().numpy()[0]  # single batch
                np.save(os.path.join(save_dir, f"pred_{idx}.npy"), pred_numpy)

    mean_dice = np.nanmean(dice_all)
    mean_iou = np.nanmean(iou_all)
    mean_hausdorff = np.nanmean(hausdorff_all)

    print("\n🔎 Final Evaluation Results:")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Mean IoU Score: {mean_iou:.4f}")
    print(f"Mean Hausdorff Distance: {mean_hausdorff:.4f}")

    return mean_dice, mean_iou, mean_hausdorff

def main():
    # ==================== CONFIG ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 4
    num_classes = 3
    from config import SAVE_DIR, PRED_SAVE_DIR

    checkpoint_path = os.path.join(SAVE_DIR, "best_model.pth")
    save_preds = True  # You can add SAVE_PREDS = True in config.py too if you want


    # ==================== MODEL ====================
    model = RefineFormer3D(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    load_checkpoint(model, checkpoint_path, device)

    # ==================== DATALOADER ====================
    # Replace this with your test/validation dataset
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # ==================== INFERENCE ====================
    inference(model, test_loader, device, num_classes, save_preds=save_preds, save_dir="./predictions")

if __name__ == "__main__":
    main()

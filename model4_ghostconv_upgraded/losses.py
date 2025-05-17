import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Ensure targets and logits spatial shapes match
        # targets: (B, D, H, W) or (B,1,D,H,W)
        if targets.ndim == 5:
            targets = targets.squeeze(1)
        assert targets.shape[1:] == logits.shape[2:], \
            f"Target spatial shape {targets.shape[1:]} doesn't match logits spatial shape {logits.shape[2:]}"
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice.mean()

# Tversky Loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        if targets.ndim == 5:
            targets = targets.squeeze(1)
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        TP = torch.sum(probs * targets_onehot, dims)
        FP = torch.sum(probs * (1 - targets_onehot), dims)
        FN = torch.sum((1 - probs) * targets_onehot, dims)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return torch.pow((1 - tversky), self.gamma).mean()


# Cross Entropy
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        if targets.ndim == 5:
            targets = targets.squeeze(1)
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"[CrossEntropy] Expected torch.Tensor, but got {type(targets)}")
        if targets.dtype != torch.long:
            raise TypeError(f"[CrossEntropy] Expected targets dtype torch.long, but got {targets.dtype}")
        return F.cross_entropy(logits, targets)

# Boundary Loss (Optional)
def compute_boundary_loss(preds, targets):
    if targets.ndim == 5:
        targets = targets.squeeze(1)
    preds = torch.softmax(preds, dim=1)
    grad_preds = (
        torch.abs(preds[:, :, :-1, :, :] - preds[:, :, 1:, :, :]).mean() +
        torch.abs(preds[:, :, :, :-1, :] - preds[:, :, :, 1:, :]).mean() +
        torch.abs(preds[:, :, :, :, :-1] - preds[:, :, :, :, 1:]).mean()
    )
    return grad_preds

# Full Combined Loss
class RefineFormer3DLoss(nn.Module):
    def __init__(self, dice_weight=1.0, tversky_weight=1.0, ce_weight=1.0, boundary_weight=1.0, aux_weight=0.4):
        super(RefineFormer3DLoss, self).__init__()
        self.dice = DiceLoss()
        self.tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        self.ce = SoftCrossEntropyLoss()
        self.boundary_weight = boundary_weight
        self.aux_weight = aux_weight

        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.ce_weight = ce_weight

    def forward(self, outputs, targets):
        # Allow plain tensor outputs
        if isinstance(outputs, torch.Tensor):
            outputs = {"main": outputs}

        main_output = outputs.get("main")
        
        if torch.isnan(main_output).any():
            print("❌ NaN detected in main output")

        if torch.any(torch.isinf(main_output)):
            print("❌ Inf detected in main output")



        if main_output is None:
            raise KeyError("[Loss] 'main' key not found in model outputs")

        # Squeeze channel if present
        if targets.ndim == 5:
            targets = targets.squeeze(1)

        # Basic checks
        assert isinstance(targets, torch.Tensor), f"Expected Tensor, got {type(targets)}"
        assert targets.dtype == torch.long, f"Expected long dtype, got {targets.dtype}"
        num_classes = main_output.shape[1]
        assert torch.max(targets) < num_classes, \
            f"Target value too high: got max {torch.max(targets)}, expected < {num_classes}"

        # Compute main loss
        loss_main = (
            self.dice_weight * self.dice(main_output, targets) +
            self.tversky_weight * self.tversky(main_output, targets) +
            self.ce_weight * self.ce(main_output, targets)
        )

        # Boundary loss
        loss_boundary = self.boundary_weight * compute_boundary_loss(main_output, targets)

        # Auxiliary losses
        aux_loss = 0.0
        for aux_key in ["aux2", "aux3"]:
            aux_out = outputs.get(aux_key)
            if aux_out is not None:
                # ➤ Upsample aux output to match target spatial dims
                aux_out = F.interpolate(
                    aux_out,
                    size=targets.shape[1:],          # (D, H, W)
                    mode="trilinear",
                    align_corners=False
                )
                aux_loss += self.dice(aux_out, targets) + self.ce(aux_out, targets)
        aux_loss = self.aux_weight * aux_loss


        return loss_main + aux_loss + loss_boundary

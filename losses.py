import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice.mean()

        return dice_loss

# Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        TP = torch.sum(probs * targets_onehot, dims)
        FP = torch.sum(probs * (1 - targets_onehot), dims)
        FN = torch.sum((1 - probs) * targets_onehot, dims)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1. - tversky.mean()

        return tversky_loss

# Cross Entropy
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)

# Boundary Loss (Optional, for structures like tumor borders)
def compute_boundary_loss(preds, targets):
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
        self.tversky = TverskyLoss()
        self.ce = SoftCrossEntropyLoss()
        self.boundary_weight = boundary_weight
        self.aux_weight = aux_weight

        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.ce_weight = ce_weight

    def forward(self, outputs, targets):
        """
        outputs: dict from RefineFormer3D forward pass
        targets: ground truth masks (B, D, H, W)
        """

        loss_main = self.dice_weight * self.dice(outputs["main"], targets) + \
                    self.tversky_weight * self.tversky(outputs["main"], targets) + \
                    self.ce_weight * self.ce(outputs["main"], targets)

        loss_boundary = self.boundary_weight * compute_boundary_loss(outputs["main"], targets)

        # Auxiliary Losses
        aux_loss = 0
        for aux_key in ["aux2", "aux3", "aux4"]:
            aux_loss += self.dice(outputs[aux_key], targets) + self.ce(outputs[aux_key], targets)

        aux_loss = self.aux_weight * aux_loss

        total_loss = loss_main + aux_loss + loss_boundary

        return total_loss

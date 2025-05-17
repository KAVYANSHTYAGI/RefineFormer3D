import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

def get_optimizer(model, base_lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999)):
    """
    Creates AdamW optimizer for RefineFormer3D

    Args:
        model: nn.Module, the model whose parameters we want to optimize
        base_lr: float, initial learning rate
        weight_decay: float, weight decay for AdamW
        betas: tuple, AdamW betas

    Returns:
        optimizer: torch.optim.Optimizer
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=betas
    )
    return optimizer

def get_scheduler(optimizer, mode="cosine", T_max=100, min_lr=1e-6, patience=10):
    """
    Creates learning rate scheduler for RefineFormer3D training

    Args:
        optimizer: optimizer object
        mode: str, 'cosine' for CosineAnnealingLR, 'plateau' for ReduceLROnPlateau
        T_max: int, number of epochs for cosine decay
        min_lr: float, minimum learning rate
        patience: int, patience for plateau mode

    Returns:
        scheduler: torch.optim.lr_scheduler
    """
    if mode == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_lr
        )
    elif mode == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=patience,
            min_lr=min_lr
        )
    else:
        raise ValueError("Scheduler mode must be 'cosine' or 'plateau'!")

    return scheduler

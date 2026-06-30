import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def freeze_model(model: nn.Module):
    """
    Freeze all model parameters.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """
    Unfreeze all model parameters.
    """
    for param in model.parameters():
        param.requires_grad = True


def move_to_device(
    model: nn.Module,
    device: torch.device,
):
    """
    Move model to CPU/GPU.
    """
    return model.to(device)
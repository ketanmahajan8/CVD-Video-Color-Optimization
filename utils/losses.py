# D:\CVD_GAN\utils\losses.py
from __future__ import annotations
import torch


def gan_lsgan_loss(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """
    Least Squares GAN loss (more stable than BCE for pix2pix).
    """
    targets = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return torch.mean((logits - targets) ** 2)


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    img: (B,3,H,W)
    """
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw
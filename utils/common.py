# D:\CVD_GAN\utils\common.py
from __future__ import annotations
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic can reduce speed; keep it off for GAN stability.
    torch.backends.cudnn.benchmark = True


def save_checkpoint(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu"):
    # Safe for our own checkpoints; avoids pickle executing arbitrary code in future defaults.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch versions may not support weights_only
        return torch.load(path, map_location=map_location)
# D:\CVD_GAN\utils\data.py
from __future__ import annotations
from pathlib import Path
from typing import List
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


class ImageFolder256(Dataset):
    def __init__(self, folder: str, limit: int = 0, seed: int = 123):
        self.paths = list_images(folder)
        if limit and limit > 0:
            self.paths = self.paths[:limit]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {folder}")
        random.Random(seed).shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.uint8, copy=True)  # <- FIX (writable)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return x, str(p)
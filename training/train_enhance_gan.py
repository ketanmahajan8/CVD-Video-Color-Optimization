from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
from tqdm import tqdm

from training.models import UNetGenerator, PatchDiscriminator
from utils.common import save_checkpoint

TYPE_LIST = ["protan", "deutan", "tritan"]


def make_type_map(H: int, W: int, tname: str, device: torch.device) -> torch.Tensor:
    idx = TYPE_LIST.index(tname)
    t = torch.zeros((1, 3, H, W), device=device, dtype=torch.float32)
    t[:, idx, :, :] = 1.0
    return t


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
    return t


class EnhancePairDataset(Dataset):
    """
    Provides (x, tname, y_target) where:
      x: original image tensor (3,H,W) in [0,1]
      y_target: enhanced target for chosen type (3,H,W) in [0,1]
      tname: one of ["protan","deutan","tritan"]
    """
    def __init__(self, x_dir: Path, y_root: Path, limit: int | None = None, seed: int = 123):
        self.x_dir = Path(x_dir)
        self.y_root = Path(y_root)
        self.seed = seed

        self.x_paths = sorted([p for p in self.x_dir.glob("*.jpg")] + [p for p in self.x_dir.glob("*.png")])
        if limit is not None and limit > 0:
            self.x_paths = self.x_paths[: min(limit, len(self.x_paths))]

        if len(self.x_paths) == 0:
            raise RuntimeError(f"No images in {self.x_dir}")

        # verify at least one target exists
        sample = self.x_paths[0].name
        ok = any((self.y_root / t / sample).exists() for t in TYPE_LIST)
        if not ok:
            raise RuntimeError(f"Targets not found. Expected {self.y_root}/<type>/{sample}")

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x_path = self.x_paths[idx]

        # deterministic balance: 0->protan,1->deutan,2->tritan, repeat
        tname = TYPE_LIST[idx % 3]

        y_path = self.y_root / tname / x_path.name
        if not y_path.exists():
            # fallback: random type if a file missing
            tname = random.choice(TYPE_LIST)
            y_path = self.y_root / tname / x_path.name

        x = pil_to_tensor01(Image.open(x_path))
        y = pil_to_tensor01(Image.open(y_path))

        return x, tname, y


def gan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    # LSGAN
    return 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()


def gan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    # LSGAN
    return 0.5 * ((d_fake - 1) ** 2).mean()


def main():
    print("RUNNING FILE:", __file__)
    print("DEBUG: discriminator call should be D(x_cond, y)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--x_dir", type=str, default=r"D:\CVD_GAN\dataset\original_256")
    ap.add_argument("--y_dir", type=str, default=r"D:\CVD_GAN\dataset\enh_targets_256")
    ap.add_argument("--out_dir", type=str, default=r"D:\CVD_GAN\checkpoints\enhance_gan_conditional")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=123)

    # Loss weights
    ap.add_argument("--lambda_l1", type=float, default=100.0, help="target matching weight")
    ap.add_argument("--lambda_id", type=float, default=10.0, help="keep close to original")
    ap.add_argument("--lambda_gan", type=float, default=1.0, help="GAN realism weight")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    x_dir = Path(args.x_dir)
    y_dir = Path(args.y_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = EnhancePairDataset(x_dir=x_dir, y_root=y_dir, limit=args.limit, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # Conditional: x(3) + tmap(3) = 6 channels
    G = UNetGenerator(in_channels=6, out_channels=3).to(device)

    # IMPORTANT: PatchDiscriminator.forward() concatenates [x_cond (6) | y (3)] = 9 channels.
    # So D must be created with in_channels=9.
    D = PatchDiscriminator(in_channels=9).to(device)

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    l1 = nn.L1Loss()

    best_loss = 1e9

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}", ncols=110)
        for x, tname, y_tgt in pbar:
            x = x.to(device)
            y_tgt = y_tgt.to(device)

            # build tmap batch
            B, _, H, W = x.shape
            tmap_list = []
            for tn in tname:
                tmap_list.append(make_type_map(H, W, tn, device).squeeze(0))
            tmap = torch.stack(tmap_list, dim=0)  # (B,3,H,W)

            x_cond = torch.cat([x, tmap], dim=1)  # (B,6,H,W)

            # -----------------
            # Train D
            # -----------------
            with torch.no_grad():
                y_fake = G(x_cond)

            d_real = D(x_cond, y_tgt)
            d_fake = D(x_cond, y_fake)

            lossD = gan_d_loss(d_real, d_fake)

            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()

            # -----------------
            # Train G
            # -----------------
            y_fake = G(x_cond)
            d_fake2 = D(x_cond, y_fake)

            loss_gan = gan_g_loss(d_fake2) * args.lambda_gan
            loss_l1 = l1(y_fake, y_tgt) * args.lambda_l1
            loss_id = l1(y_fake, x) * args.lambda_id

            lossG = loss_gan + loss_l1 + loss_id

            optG.zero_grad(set_to_none=True)
            lossG.backward()
            optG.step()

            pbar.set_postfix(
                D=float(lossD.item()),
                G=float(lossG.item()),
                l1=float(loss_l1.item()),
                id=float(loss_id.item()),
                gan=float(loss_gan.item()),
            )

        # Save checkpoints each epoch
        ckpt_path = out_dir / "latest.pth"
        save_checkpoint(str(ckpt_path), {"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch})

        # Choose best by generator loss estimate (last batch shown is fine for student project)
        if lossG.item() < best_loss:
            best_loss = lossG.item()
            save_checkpoint(str(out_dir / "best.pth"), {"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch})

    print("Done. Saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
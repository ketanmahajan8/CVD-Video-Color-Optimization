# D:\CVD_GAN\training\train_precomp.py
from __future__ import annotations

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from training.models import UNetGenerator, PatchDiscriminator
from simulate.cvd_simulation import simulate_cvd
from utils.data import ImageFolder256
from utils.losses import gan_lsgan_loss, total_variation_loss
from utils.common import seed_everything, save_checkpoint


# ---------------------------
# Conditioning helper
# ---------------------------
def make_type_maps(batch_size: int, h: int, w: int, t_idx: torch.Tensor) -> torch.Tensor:
    """
    Create one-hot type maps of shape (B,3,H,W).
    t_idx: (B,) with values 0,1,2 for protan,deutan,tritan
    """
    device = t_idx.device
    maps = torch.zeros((batch_size, 3, h, w), device=device, dtype=torch.float32)
    for k in range(3):
        maps[:, k, :, :] = (t_idx == k).float().view(-1, 1, 1)
    return maps


def idx_to_name(t_idx: int) -> str:
    return ["protan", "deutan", "tritan"][t_idx]


# ---------------------------
# Train
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=r"D:\CVD_GAN\dataset\original_256")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--lambda_rec", type=float, default=100.0)
    ap.add_argument("--lambda_id", type=float, default=50.0)
    ap.add_argument("--lambda_tv", type=float, default=0.05)
    ap.add_argument("--lambda_gan", type=float, default=0.1)

    ap.add_argument("--severity", type=float, default=1.0, help="CVD sim severity (0..1)")
    ap.add_argument("--val_split", type=float, default=0.1)

    ap.add_argument("--outdir", type=str, default=r"D:\CVD_GAN\checkpoints\conditional_3type")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Dataset
    ds = ImageFolder256(args.data, limit=args.limit, seed=args.seed)
    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Models
    G = UNetGenerator(in_channels=6, out_channels=3).to(device)
    D = PatchDiscriminator(in_channels=9).to(device)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    l1 = nn.L1Loss()

    best_val = float("inf")
    os.makedirs(args.outdir, exist_ok=True)

    def run_val():
        G.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x, _paths in val_loader:
                x = x.to(device, non_blocking=True)  # (B,3,H,W)
                B, _, H, W = x.shape
                t_idx = torch.randint(0, 3, (B,), device=device)
                t_maps = make_type_maps(B, H, W, t_idx)
                x_cond = torch.cat([x, t_maps], dim=1)  # (B,6,H,W)

                y = G(x_cond)  # (B,3,H,W)

                # simulate per sample by looping types (cheap at 256)
                sy_list = []
                for i in range(B):
                    tname = idx_to_name(int(t_idx[i].item()))
                    sy_list.append(simulate_cvd(y[i], tname, severity=args.severity))
                sy = torch.stack(sy_list, dim=0)

                loss_rec = l1(sy, x)
                loss_id = l1(y, x)
                loss_tv = total_variation_loss(y)
                loss = args.lambda_rec * loss_rec + args.lambda_id * loss_id + args.lambda_tv * loss_tv

                total += float(loss.item()) * B
                count += B
        G.train()
        return total / max(count, 1)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        G.train()
        D.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=110)
        for x, _paths in pbar:
            x = x.to(device, non_blocking=True)  # (B,3,H,W)
            B, _, H, W = x.shape

            # Randomly assign a deficiency type per sample
            t_idx = torch.randint(0, 3, (B,), device=device)
            t_maps = make_type_maps(B, H, W, t_idx)
            x_cond = torch.cat([x, t_maps], dim=1)  # (B,6,H,W)

            # -------------------------
            # (1) Train D
            # -------------------------
            with torch.no_grad():
                y_fake = G(x_cond)

            d_real = D(x_cond, x)        # real pair: (x_cond, x)
            d_fake = D(x_cond, y_fake)   # fake pair: (x_cond, y_fake)

            loss_D = 0.5 * (gan_lsgan_loss(d_real, True) + gan_lsgan_loss(d_fake, False))

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # -------------------------
            # (2) Train G
            # -------------------------
            y = G(x_cond)
            d_fake_for_g = D(x_cond, y)
            loss_gan = gan_lsgan_loss(d_fake_for_g, True)

            # simulate S_t(y) per sample
            sy_list = []
            for i in range(B):
                tname = idx_to_name(int(t_idx[i].item()))
                sy_list.append(simulate_cvd(y[i], tname, severity=args.severity))
            sy = torch.stack(sy_list, dim=0)

            loss_rec = l1(sy, x)
            loss_id = l1(y, x)
            loss_tv = total_variation_loss(y)

            loss_G = (
                args.lambda_rec * loss_rec
                + args.lambda_id * loss_id
                + args.lambda_tv * loss_tv
                + args.lambda_gan * loss_gan
            )

            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            pbar.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{loss_G.item():.3f}",
                "rec": f"{loss_rec.item():.4f}",
                "id": f"{loss_id.item():.4f}",
                "gan": f"{loss_gan.item():.3f}",
            })

        # Validation + checkpoint
        val_loss = run_val()
        elapsed = time.time() - start

        ckpt_latest = {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "args": vars(args),
            "val_loss": val_loss,
        }
        save_checkpoint(os.path.join(args.outdir, "latest.pth"), ckpt_latest)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(args.outdir, "best.pth"), ckpt_latest)

        print(f"\nEpoch {epoch} done | val_loss={val_loss:.4f} | best={best_val:.4f} | time={elapsed/60:.1f} min\n")

    print("Training complete.")
    print(f"Saved to: {args.outdir}")


if __name__ == "__main__":
    main()
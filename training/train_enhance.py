from __future__ import annotations

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from training.models import UNetGenerator, PatchDiscriminator
from simulate.cvd_simulation import simulate_cvd
from utils.data import ImageFolder256
from utils.common import seed_everything, save_checkpoint
from utils.losses import gan_lsgan_loss, total_variation_loss

TYPE_NAMES = ["protan", "deutan", "tritan"]


def idx_to_name(t_idx: int) -> str:
    return TYPE_NAMES[t_idx]


def make_type_maps(batch_size: int, h: int, w: int, t_idx: torch.Tensor) -> torch.Tensor:
    maps = torch.zeros((batch_size, 3, h, w), device=t_idx.device, dtype=torch.float32)
    for k in range(3):
        maps[:, k, :, :] = (t_idx == k).float().view(-1, 1, 1)
    return maps


def gradient_mag(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) in [0,1]
    returns: (B,1,H,W) gradient magnitude
    """
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)

    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return g


def simulate_batch_grad(img: torch.Tensor, t_idx: torch.Tensor, severity: float) -> torch.Tensor:
    """
    Gradient-enabled simulation for y = G(x).
    """
    B = img.size(0)
    out = []
    for i in range(B):
        tname = idx_to_name(int(t_idx[i].item()))
        out.append(simulate_cvd(img[i], tname, severity=severity))
    return torch.stack(out, dim=0)


@torch.no_grad()
def simulate_batch_nograd(img: torch.Tensor, t_idx: torch.Tensor, severity: float) -> torch.Tensor:
    """
    No-gradient simulation for x baseline (and for validation).
    """
    B = img.size(0)
    out = []
    for i in range(B):
        tname = idx_to_name(int(t_idx[i].item()))
        out.append(simulate_cvd(img[i], tname, severity=severity))
    return torch.stack(out, dim=0)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default=r"D:\CVD_GAN\dataset\original_256")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--outdir", type=str, default=r"D:\CVD_GAN\checkpoints\conditional_3type_enhance")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pth) to resume from")

    ap.add_argument("--severity", type=float, default=1.0)

    # Default weights (you can override from CLI)
    ap.add_argument("--lambda_id", type=float, default=30.0)
    ap.add_argument("--lambda_cvd", type=float, default=20.0)
    ap.add_argument("--lambda_edge", type=float, default=15.0)
    ap.add_argument("--lambda_gan", type=float, default=0.1)
    ap.add_argument("--lambda_tv", type=float, default=0.05)

    ap.add_argument("--edge_delta", type=float, default=0.010, help="Minimum desired edge gain in S-space")

    args = ap.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = ImageFolder256(args.data, limit=args.limit, seed=args.seed)
    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    G = UNetGenerator(in_channels=6, out_channels=3).to(device)
    D = PatchDiscriminator(in_channels=9).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    l1 = nn.L1Loss()

    # Resume (optional)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        if "opt_G" in ckpt:
            opt_G.load_state_dict(ckpt["opt_G"])
        if "opt_D" in ckpt:
            opt_D.load_state_dict(ckpt["opt_D"])
        print(f"Resumed from: {args.resume}")

    best_val = float("inf")
    os.makedirs(args.outdir, exist_ok=True)

    def compute_edge_loss(sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
        """
        Enforce: mean(|∇S(y)|) - mean(|∇S(x)|) >= edge_delta
        """
        gx = gradient_mag(sx).mean()
        gy = gradient_mag(sy).mean()
        gain = gy - gx
        delta = torch.tensor(args.edge_delta, device=device, dtype=gain.dtype)
        return torch.relu(delta - gain)

    def run_val():
        G.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                B, _, H, W = x.shape
                t_idx = torch.randint(0, 3, (B,), device=device)
                t_maps = make_type_maps(B, H, W, t_idx)
                x_cond = torch.cat([x, t_maps], dim=1)

                y = G(x_cond)

                sx = simulate_batch_nograd(x, t_idx, args.severity)
                sy = simulate_batch_nograd(y, t_idx, args.severity)

                loss_id = l1(y, x)
                loss_cvd = l1(sy, sx)
                loss_edge = compute_edge_loss(sx, sy)
                loss_tv = total_variation_loss(y)

                loss = (
                    args.lambda_id * loss_id
                    + args.lambda_cvd * loss_cvd
                    + args.lambda_edge * loss_edge
                    + args.lambda_tv * loss_tv
                )

                total += float(loss.item()) * B
                count += B

        G.train()
        return total / max(count, 1)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        G.train()
        D.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=110)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            B, _, H, W = x.shape

            t_idx = torch.randint(0, 3, (B,), device=device)
            t_maps = make_type_maps(B, H, W, t_idx)
            x_cond = torch.cat([x, t_maps], dim=1)

            # ---- Train D ----
            with torch.no_grad():
                y_fake = G(x_cond)

            d_real = D(x_cond, x)
            d_fake = D(x_cond, y_fake)

            loss_D = 0.5 * (gan_lsgan_loss(d_real, True) + gan_lsgan_loss(d_fake, False))

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # ---- Train G ----
            y = G(x_cond)
            d_fake_for_g = D(x_cond, y)
            loss_gan = gan_lsgan_loss(d_fake_for_g, True)

            sx = simulate_batch_nograd(x, t_idx, args.severity)
            sy = simulate_batch_grad(y, t_idx, args.severity)  # MUST be grad-enabled

            loss_id = l1(y, x)
            loss_cvd = l1(sy, sx)
            loss_edge = compute_edge_loss(sx, sy)
            loss_tv = total_variation_loss(y)

            loss_G = (
                args.lambda_id * loss_id
                + args.lambda_cvd * loss_cvd
                + args.lambda_edge * loss_edge
                + args.lambda_tv * loss_tv
                + args.lambda_gan * loss_gan
            )

            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            pbar.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{loss_G.item():.3f}",
                "id": f"{loss_id.item():.4f}",
                "cvd": f"{loss_cvd.item():.4f}",
                "edge": f"{loss_edge.item():.4f}",
            })

        val_loss = run_val()
        elapsed = time.time() - start

        ckpt_out = {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "args": vars(args),
            "val_loss": val_loss,
        }
        save_checkpoint(os.path.join(args.outdir, "latest.pth"), ckpt_out)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(args.outdir, "best.pth"), ckpt_out)

        print(f"\nEpoch {epoch} done | val_loss={val_loss:.4f} | best={best_val:.4f} | time={elapsed/60:.1f} min\n")

    print("Training complete.")
    print(f"Saved to: {args.outdir}")


if __name__ == "__main__":
    main()
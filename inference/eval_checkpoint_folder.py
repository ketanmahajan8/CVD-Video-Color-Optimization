from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from training.models import UNetGenerator
from simulate.cvd_simulation import simulate_cvd
from utils.common import load_checkpoint

TYPE_NAMES = ["protan", "deutan", "tritan"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_type_maps(B: int, H: int, W: int, tname: str, device: str) -> torch.Tensor:
    idx = TYPE_NAMES.index(tname)
    maps = torch.zeros((B, 3, H, W), device=device, dtype=torch.float32)
    maps[:, idx, :, :] = 1.0
    return maps


def to_uint8_img(t: torch.Tensor) -> np.ndarray:
    t = torch.clamp(t, 0, 1)
    return (t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def l1_u8(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def gradient_energy(img_u8: np.ndarray) -> float:
    x = img_u8.astype(np.float32) / 255.0
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    ypad = np.pad(y, ((1, 1), (1, 1)), mode="edge")
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    H, W = y.shape
    for i in range(H):
        for j in range(W):
            patch = ypad[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * kx)
            gy[i, j] = np.sum(patch * ky)

    g = np.sqrt(gx * gx + gy * gy + 1e-8)
    return float(np.mean(np.abs(g)))


def save_panel(out_path: Path, x_u8, sx_u8, y_u8, sy_u8):
    panel = np.concatenate([x_u8, sx_u8, y_u8, sy_u8], axis=1)  # [x | S(x) | y | S(y)]
    Image.fromarray(panel).save(str(out_path))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--type", choices=["all"] + TYPE_NAMES, default="all")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.ckpt, map_location=device)

    G = UNetGenerator(in_channels=6, out_channels=3).to(device)
    if isinstance(ckpt, dict) and "G" in ckpt:
        G.load_state_dict(ckpt["G"])
    else:
        G.load_state_dict(ckpt)
    G.eval()

    paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found in {in_dir}")

    types = TYPE_NAMES if args.type == "all" else [args.type]
    report_lines = []

    for tname in types:
        t_dir = out_dir / tname
        t_dir.mkdir(parents=True, exist_ok=True)

        dist_yx, dist_s, edge_gain = [], [], []

        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
            arr = np.array(img, dtype=np.uint8, copy=True)

            x = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
            H, W = x.shape[1], x.shape[2]

            sx = simulate_cvd(x, tname, severity=args.severity)

            tmap = make_type_maps(1, H, W, tname, device=device)
            x_cond = torch.cat([x.unsqueeze(0), tmap], dim=1)
            y = G(x_cond).squeeze(0)

            sy = simulate_cvd(y, tname, severity=args.severity)

            x_u8 = to_uint8_img(x)
            sx_u8 = to_uint8_img(sx)
            y_u8 = to_uint8_img(y)
            sy_u8 = to_uint8_img(sy)

            dist_yx.append(l1_u8(y_u8, x_u8))     # did it change?
            dist_s.append(l1_u8(sy_u8, sx_u8))    # change in simulated view

            e_before = gradient_energy(sx_u8)
            e_after = gradient_energy(sy_u8)
            edge_gain.append(e_after - e_before)  # contrast gain in simulated view

            save_panel(t_dir / f"panel_{i:03d}_{p.stem}.png", x_u8, sx_u8, y_u8, sy_u8)

        def avg(v): return float(np.mean(v))
        def med(v): return float(np.median(v))

        line = (
            f"{tname}: "
            f"mean|y-x|={avg(dist_yx):.4f} (med {med(dist_yx):.4f}), "
            f"mean|S(y)-S(x)|={avg(dist_s):.4f} (med {med(dist_s):.4f}), "
            f"edge_gain={avg(edge_gain):+.5f} (med {med(edge_gain):+.5f})"
        )
        
        print(line)
        report_lines.append(line)

    (out_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("Saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
# D:\CVD_GAN\simulate\cvd_simulation.py
"""
CVD simulation functions S_protan / S_deutan / S_tritan.

We implement:
- Protan + Deutan: Viénot et al. 1999 (fast single matrix on *linear RGB*)
- Tritan: Brettel et al. 1997 (needs piecewise projection; we use the
          precomputed parameters approach popularized by DaltonLens)

Why linear RGB?
Because these algorithms are defined for linear-light values; applying them
directly on gamma-corrected sRGB is wrong. (DaltonLens explicitly highlights this.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


DefType = Literal["protan", "deutan", "tritan"]


# -------------------------
# sRGB <-> linearRGB (torch)
# -------------------------
def srgb_to_linear(rgb_srgb: torch.Tensor) -> torch.Tensor:
    """
    rgb_srgb: (..., 3) in [0, 1] (sRGB gamma-encoded)
    returns:  (..., 3) linear RGB
    """
    rgb_srgb = torch.clamp(rgb_srgb, 0.0, 1.0)
    a = 0.055
    threshold = 0.04045
    low = rgb_srgb / 12.92
    high = torch.pow((rgb_srgb + a) / (1.0 + a), 2.4)
    return torch.where(rgb_srgb <= threshold, low, high)


def linear_to_srgb(rgb_linear: torch.Tensor) -> torch.Tensor:
    """
    rgb_linear: (..., 3) linear RGB (not necessarily clipped)
    returns:    (..., 3) sRGB in [0, 1]
    """
    rgb_linear = torch.clamp(rgb_linear, 0.0, 1.0)
    a = 0.055
    threshold = 0.0031308
    low = rgb_linear * 12.92
    high = (1.0 + a) * torch.pow(rgb_linear, 1.0 / 2.4) - a
    return torch.where(rgb_linear <= threshold, low, high)


# ---------------------------------------
# Viénot 1999 matrices (linearRGB domain)
# (numbers copied from DaltonLens reference
#  implementation / libDaltonLens.c)
# ---------------------------------------
_VIENOT_MATS = {
    "protan": torch.tensor(
        [
            [0.11238, 0.88762, 0.00000],
            [0.11238, 0.88762, -0.00000],
            [0.00401, -0.00401, 1.00000],
        ],
        dtype=torch.float32,
    ),
    "deutan": torch.tensor(
        [
            [0.29275, 0.70725, 0.00000],
            [0.29275, 0.70725, -0.00000],
            [-0.02234, 0.02234, 1.00000],
        ],
        dtype=torch.float32,
    ),
}


@dataclass(frozen=True)
class BrettelParams:
    rgb_cvd_from_rgb_1: torch.Tensor  # (3,3)
    rgb_cvd_from_rgb_2: torch.Tensor  # (3,3)
    sep_plane_normal_rgb: torch.Tensor  # (3,)


# -------------------------------------------------------
# Brettel 1997 precomputed params (linearRGB domain)
# (numbers copied from DaltonLens reference implementation)
# -------------------------------------------------------
_BRETTEL = {
    "tritan": BrettelParams(
        rgb_cvd_from_rgb_1=torch.tensor(
            [
                [1.01277, 0.13548, -0.14826],
                [-0.01243, 0.86812, 0.14431],
                [0.07589, 0.80500, 0.11911],
            ],
            dtype=torch.float32,
        ),
        rgb_cvd_from_rgb_2=torch.tensor(
            [
                [0.93678, 0.18979, -0.12657],
                [0.06154, 0.81526, 0.12320],
                [-0.37562, 1.12767, 0.24796],
            ],
            dtype=torch.float32,
        ),
        sep_plane_normal_rgb=torch.tensor([0.03901, -0.02788, -0.01113], dtype=torch.float32),
    )
}


def _apply_mat(rgb_linear: torch.Tensor, mat3x3: torch.Tensor) -> torch.Tensor:
    """
    rgb_linear: (..., 3)
    mat3x3: (3,3)
    returns: (..., 3)
    """
    # (...,3) @ (3,3)^T
    return rgb_linear @ mat3x3.t()


def simulate_cvd(
    img_srgb: torch.Tensor,
    deficiency: DefType,
    severity: float = 1.0,
) -> torch.Tensor:
    """
    img_srgb: (N,3,H,W) or (3,H,W) tensor in [0,1], sRGB
    returns:  same shape, simulated CVD view in sRGB [0,1]

    severity in [0,1]:
      0 -> identity (no deficiency)
      1 -> full dichromat-like simulation (strongest)
    """
    if deficiency not in ("protan", "deutan", "tritan"):
        raise ValueError(f"Unknown deficiency: {deficiency}")

    severity = float(severity)
    if not (0.0 <= severity <= 1.0):
        raise ValueError("severity must be in [0,1]")

    x = img_srgb
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (1,3,H,W)
    if x.dim() != 4 or x.size(1) != 3:
        raise ValueError("img_srgb must have shape (3,H,W) or (N,3,H,W)")

    # (N,H,W,3)
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_lin = srgb_to_linear(x_nhwc)

    if deficiency in ("protan", "deutan"):
        mat = _VIENOT_MATS[deficiency].to(device=x_lin.device, dtype=x_lin.dtype)
        y_lin = _apply_mat(x_lin, mat)

        # severity interpolation in linear space (DaltonLens does this)
        if severity < 1.0:
            y_lin = severity * y_lin + (1.0 - severity) * x_lin

    else:
        params = _BRETTEL["tritan"]
        m1 = params.rgb_cvd_from_rgb_1.to(device=x_lin.device, dtype=x_lin.dtype)
        m2 = params.rgb_cvd_from_rgb_2.to(device=x_lin.device, dtype=x_lin.dtype)
        n = params.sep_plane_normal_rgb.to(device=x_lin.device, dtype=x_lin.dtype)

        dot = x_lin[..., 0] * n[0] + x_lin[..., 1] * n[1] + x_lin[..., 2] * n[2]
        use_m1 = (dot >= 0).unsqueeze(-1)  # (...,1)

        y1 = _apply_mat(x_lin, m1)
        y2 = _apply_mat(x_lin, m2)
        y_lin = torch.where(use_m1, y1, y2)

        if severity < 1.0:
            y_lin = severity * y_lin + (1.0 - severity) * x_lin

    y_srgb = linear_to_srgb(y_lin)
    y = y_srgb.permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)

    if img_srgb.dim() == 3:
        return y.squeeze(0)
    return y


# Convenience wrappers
def S_protan(img_srgb: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    return simulate_cvd(img_srgb, "protan", severity)


def S_deutan(img_srgb: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    return simulate_cvd(img_srgb, "deutan", severity)


def S_tritan(img_srgb: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    return simulate_cvd(img_srgb, "tritan", severity)


if __name__ == "__main__":
    # Quick sanity: run on one image path and save outputs.
    # Usage:
    #   python simulate\cvd_simulation.py D:\CVD_GAN\dataset\original_256\000000000139.jpg
    import sys
    from pathlib import Path
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python simulate\\cvd_simulation.py <path_to_image>")
        raise SystemExit(1)

    in_path = Path(sys.argv[1])
    img = Image.open(in_path).convert("RGB")
    x = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy())
    x = x.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0  # (3,H,W)

    out_dir = Path("outputs") / "images" / "cvd_sim_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(t: torch.Tensor, name: str):
        t = torch.clamp(t, 0, 1)
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(out_dir / name)

    save(x, "x.png")
    save(S_protan(x), "S_protan.png")
    save(S_deutan(x), "S_deutan.png")
    save(S_tritan(x), "S_tritan.png")

    print(f"Saved to: {out_dir.resolve()}")
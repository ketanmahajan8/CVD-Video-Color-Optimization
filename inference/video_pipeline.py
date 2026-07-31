from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from training.models import UNetGenerator
from utils.common import load_checkpoint

TYPE_NAMES = ["protan", "deutan", "tritan"]


def make_type_map_1(H: int, W: int, tname: str, device: str) -> torch.Tensor:
    idx = TYPE_NAMES.index(tname)
    t = torch.zeros((1, 3, H, W), device=device, dtype=torch.float32)
    t[:, idx, :, :] = 1.0
    return t


def bgr_to_tensor01(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor01_to_bgr(img: torch.Tensor) -> np.ndarray:
    img = torch.clamp(img, 0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
    rgb = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def lab_boost_bgr(frame_bgr: np.ndarray, cvd_type: str, strength: float) -> np.ndarray:
    if strength <= 0:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if cvd_type in ("protan", "deutan"):
        a = lab[..., 1] - 128.0
        a *= (1.0 + strength)
        lab[..., 1] = np.clip(a + 128.0, 0, 255)
    else:
        b = lab[..., 2] - 128.0
        b *= (1.0 + strength)
        lab[..., 2] = np.clip(b + 128.0, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def unsharp_mask_bgr(
    img_bgr: np.ndarray,
    amount: float = 0.5,
    radius: float = 1.0,
    threshold: int = 3,
) -> np.ndarray:
    """
    Simple unsharp mask (edge sharpening).
    - amount: 0..1 (typical 0.3..0.8)
    - radius: blur sigma (typical 0.8..1.5)
    - threshold: avoid sharpening noise (0..10 typical). Higher => less sharpening on flat areas.
    """
    if amount <= 0:
        return img_bgr

    blurred = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)

    if threshold > 0:
        diff = cv2.absdiff(img_bgr, blurred)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        keep = mask > threshold
        out = img_bgr.copy()
        out[keep] = sharp[keep]
        return out

    return sharp


@torch.no_grad()
def process_video(
    in_video: str,
    out_video: str,
    ckpt_path: str,
    cvd_type: str,
    model_size: int = 256,
    alpha: float = 1.0,
    smooth: float = 0.35,
    lab_boost: float = 0.25,
    sharpen: float = 0.5,          # NEW: 0 disables; try 0.4..0.8
    sharpen_radius: float = 1.0,   # NEW: try 0.8..1.5
    sharpen_thresh: int = 3,       # NEW: try 2..6
    progress_callback=None,
):
    # ---- Validate args ----
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if not (0.0 <= smooth <= 1.0):
        raise ValueError("smooth must be in [0, 1]")
    if lab_boost < 0.0:
        raise ValueError("lab_boost must be >= 0")
    if sharpen < 0.0:
        raise ValueError("sharpen must be >= 0")
    if sharpen_radius <= 0.0:
        raise ValueError("sharpen_radius must be > 0")
    if sharpen_thresh < 0:
        raise ValueError("sharpen_thresh must be >= 0")
    if cvd_type not in TYPE_NAMES:
        raise ValueError(f"cvd_type must be one of {TYPE_NAMES}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    G = UNetGenerator(in_channels=6, out_channels=3).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    cap = cv2.VideoCapture(in_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_video}")

    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_raw if total_frames_raw > 0 else None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w0, h0))

    tmap = make_type_map_1(model_size, model_size, cvd_type, device=device)
    prev_out = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rs = cv2.resize(frame, (model_size, model_size), interpolation=cv2.INTER_AREA)
        x = bgr_to_tensor01(frame_rs, device=device)
        x_cond = torch.cat([x, tmap], dim=1)
        y = G(x_cond)

        # Blend with original (strength control)
        if alpha < 1.0:
            y = alpha * y + (1.0 - alpha) * x

        # Anti-flicker smoothing in model output space
        if smooth > 0.0:
            if prev_out is None:
                prev_out = y
            else:
                prev_out = smooth * prev_out + (1.0 - smooth) * y
            y_use = prev_out
        else:
            y_use = y

        out_small = tensor01_to_bgr(y_use)
        out_small = lab_boost_bgr(out_small, cvd_type, lab_boost)

        out_full = cv2.resize(out_small, (w0, h0), interpolation=cv2.INTER_CUBIC)

        # NEW: edge sharpening after resize/boost (helps crispness)
        if sharpen > 0:
            out_full = unsharp_mask_bgr(
                out_full,
                amount=float(sharpen),
                radius=float(sharpen_radius),
                threshold=int(sharpen_thresh),
            )

        writer.write(out_full)

        frame_idx += 1
        if progress_callback is not None:
            progress_callback(frame_idx, total_frames)

    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--in_video", type=str, required=True)
    ap.add_argument("--out_video", type=str, required=True)
    ap.add_argument("--type", type=str, choices=TYPE_NAMES, required=True)
    ap.add_argument("--model_size", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.35)
    ap.add_argument("--lab_boost", type=float, default=0.25)

    # NEW
    ap.add_argument("--sharpen", type=float, default=0.5)
    ap.add_argument("--sharpen_radius", type=float, default=1.0)
    ap.add_argument("--sharpen_thresh", type=int, default=3)

    args = ap.parse_args()

    process_video(
        in_video=args.in_video,
        out_video=args.out_video,
        ckpt_path=args.ckpt,
        cvd_type=args.type,
        model_size=args.model_size,
        alpha=args.alpha,
        smooth=args.smooth,
        lab_boost=args.lab_boost,
        sharpen=args.sharpen,
        sharpen_radius=args.sharpen_radius,
        sharpen_thresh=args.sharpen_thresh,
        progress_callback=None,
    )


if __name__ == "__main__":
    main()
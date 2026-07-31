import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from simulate.cvd_simulation import simulate_cvd

# -------- helpers --------
def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= max(total, 1):
        raise ValueError(f"frame_idx out of range. total_frames={total}, got={frame_idx}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame_bgr


def bgr_to_tensor01(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).to(device)  # (3,H,W)
    return t


def tensor01_to_u8rgb(img: torch.Tensor) -> np.ndarray:
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def save_u8rgb(path: Path, arr: np.ndarray):
    Image.fromarray(arr).save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_video", required=True, help="input/original video path")
    ap.add_argument("--enh_video", required=True, help="enhanced output video path")
    ap.add_argument("--type", required=True, choices=["protan", "deutan", "tritan"])
    ap.add_argument("--frame", type=int, default=100, help="frame index to compare (0-based)")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--outdir", default=r"D:\CVD_GAN\outputs\images\video_frame_compare")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # read same frame index from both videos
    f1 = read_frame(args.orig_video, args.frame)
    f2 = read_frame(args.enh_video, args.frame)

    # convert to tensors
    x = bgr_to_tensor01(f1, device=device)
    y = bgr_to_tensor01(f2, device=device)

    # simulate CVD view
    sx = simulate_cvd(x, args.type, severity=args.severity)
    sy = simulate_cvd(y, args.type, severity=args.severity)

    # convert to uint8 images
    x_u8 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    y_u8 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
    sx_u8 = tensor01_to_u8rgb(sx)
    sy_u8 = tensor01_to_u8rgb(sy)

    # save individual images
    save_u8rgb(outdir / "01_original_frame.png", x_u8)
    save_u8rgb(outdir / "02_enhanced_frame.png", y_u8)
    save_u8rgb(outdir / f"03_sim_{args.type}_original.png", sx_u8)
    save_u8rgb(outdir / f"04_sim_{args.type}_enhanced.png", sy_u8)

    # 2x2 grid: [x | y] on top, [S(x) | S(y)] bottom
    top = np.concatenate([x_u8, y_u8], axis=1)
    bot = np.concatenate([sx_u8, sy_u8], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    save_u8rgb(outdir / f"grid_{args.type}_frame{args.frame:06d}.png", grid)

    print("Saved to:", outdir.resolve())
    print("Grid format:")
    print("[ original | enhanced ]")
    print(f"[ S_{args.type}(original) | S_{args.type}(enhanced) ]")


if __name__ == "__main__":
    main()
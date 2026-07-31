from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2
import numpy as np


TYPE_LIST = ["protan", "deutan", "tritan"]


def lab_axis_boost_bgr(img_bgr: np.ndarray, axis: str, strength: float) -> np.ndarray:
    """
    Boost opponent color axis in LAB.
    axis='a' boosts red-green; axis='b' boosts blue-yellow.
    strength ~ 0.2..0.6 recommended (too high looks fake).
    """
    if strength <= 0:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if axis == "a":
        a = lab[..., 1] - 128.0
        a *= (1.0 + strength)
        lab[..., 1] = np.clip(a + 128.0, 0, 255)
    elif axis == "b":
        b = lab[..., 2] - 128.0
        b *= (1.0 + strength)
        lab[..., 2] = np.clip(b + 128.0, 0, 255)
    else:
        raise ValueError("axis must be 'a' or 'b'")

    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def hsv_saturation_boost_bgr(img_bgr: np.ndarray, sat_gain: float, v_gain: float = 0.0) -> np.ndarray:
    """
    Small saturation/value boost for visibility without heavy tint.
    sat_gain: e.g., 0.05..0.20
    v_gain:   e.g., 0.00..0.05
    """
    if sat_gain <= 0 and v_gain <= 0:
        return img_bgr

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    s = np.clip(s * (1.0 + sat_gain), 0, 255)
    v = np.clip(v * (1.0 + v_gain), 0, 255)
    hsv[..., 1] = s
    hsv[..., 2] = v
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance_bgr_for_type(
    img_bgr: np.ndarray,
    t: str,
    rg_strength_protan: float,
    rg_strength_deutan: float,
    by_strength_tritan: float,
    sat_gain: float,
    v_gain: float,
) -> np.ndarray:
    """
    Pseudo-ground-truth enhancement targets.
    - protan/deutan: boost LAB 'a' (red-green opponent), different strengths
    - tritan: boost LAB 'b' (blue-yellow opponent)
    """
    out = img_bgr

    if t == "protan":
        out = lab_axis_boost_bgr(out, axis="a", strength=rg_strength_protan)
        out = hsv_saturation_boost_bgr(out, sat_gain=sat_gain, v_gain=v_gain)
        return out

    if t == "deutan":
        out = lab_axis_boost_bgr(out, axis="a", strength=rg_strength_deutan)
        out = hsv_saturation_boost_bgr(out, sat_gain=sat_gain * 0.9, v_gain=v_gain)
        return out

    if t == "tritan":
        out = lab_axis_boost_bgr(out, axis="b", strength=by_strength_tritan)
        out = hsv_saturation_boost_bgr(out, sat_gain=sat_gain * 0.8, v_gain=v_gain)
        return out

    raise ValueError(f"Unknown type: {t}")


def iter_images(in_dir: Path, exts: tuple[str, ...]) -> list[Path]:
    paths = []
    for e in exts:
        paths.extend(in_dir.rglob(f"*{e}"))
        paths.extend(in_dir.rglob(f"*{e.upper()}"))
    # de-dup
    uniq = {}
    for p in paths:
        uniq[str(p).lower()] = p
    paths = list(uniq.values())
    paths.sort()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # ONLY CHANGE: default is now 15000 (was 5000)
    ap.add_argument("--limit", type=int, default=15000)

    ap.add_argument("--seed", type=int, default=123)

    # Tunables (safe defaults)
    ap.add_argument("--rg_protan", type=float, default=0.45, help="LAB a-axis boost for protan targets")
    ap.add_argument("--rg_deutan", type=float, default=0.30, help="LAB a-axis boost for deutan targets")
    ap.add_argument("--by_tritan", type=float, default=0.40, help="LAB b-axis boost for tritan targets")
    ap.add_argument("--sat_gain", type=float, default=0.10, help="HSV saturation gain")
    ap.add_argument("--v_gain", type=float, default=0.02, help="HSV value gain")
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png", help="comma-separated")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    exts = tuple([s.strip() for s in args.exts.split(",") if s.strip()])
    random.seed(args.seed)

    # Create type subfolders
    for t in TYPE_LIST:
        (out_dir / t).mkdir(parents=True, exist_ok=True)

    imgs = iter_images(in_dir, exts)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {in_dir} with exts {exts}")

    if args.limit is not None and args.limit > 0:
        imgs = imgs[: min(args.limit, len(imgs))]

    print(f"[enhance_target] Found {len(imgs)} images to process.")

    for i, p in enumerate(imgs, 1):
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"WARNING: failed to read {p}")
            continue

        # Keep same filename (only base name) for simplicity
        fname = p.name

        for t in TYPE_LIST:
            out_bgr = enhance_bgr_for_type(
                img_bgr,
                t,
                rg_strength_protan=float(args.rg_protan),
                rg_strength_deutan=float(args.rg_deutan),
                by_strength_tritan=float(args.by_tritan),
                sat_gain=float(args.sat_gain),
                v_gain=float(args.v_gain),
            )
            out_path = out_dir / t / fname
            # JPG quality
            if out_path.suffix.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(str(out_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(str(out_path), out_bgr)

        if i % 200 == 0 or i == len(imgs):
            print(f"[enhance_target] {i}/{len(imgs)} done")

    print(f"[enhance_target] Finished. Targets saved to: {out_dir.resolve()}")
    print("Folders:")
    for t in TYPE_LIST:
        print(" -", (out_dir / t).resolve())


if __name__ == "__main__":
    main()
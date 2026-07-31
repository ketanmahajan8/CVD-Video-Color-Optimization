import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def center_crop_resize(img, size=256):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.BICUBIC)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in src.rglob("*")
        if p.suffix.lower() in IMG_EXTENSIONS
    ]

    if len(image_paths) == 0:
        raise RuntimeError("No images found in source folder.")

    image_paths = image_paths[:args.limit]

    print(f"Processing {len(image_paths)} images...")

    for path in tqdm(image_paths):
        try:
            with Image.open(path) as img:
                img = center_crop_resize(img, size=256)
                save_path = dst / (path.stem + ".jpg")
                img.save(save_path, quality=95)
        except Exception as e:
            print(f"Failed on {path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
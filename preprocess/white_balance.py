"""
white_balance.py ──
  ‣ white_balance_image(img_path, out_path)
  ‣ white_balance_batch(list[Path], out_dir)
  ‣ CLI 範例：
      python preprocess/white_balance.py --img src.jpg --out dst.jpg
"""
import cv2, numpy as np, argparse
from pathlib import Path
from typing import List

def _gray_world(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")  # ← 加 .astype
    L, a, b = cv2.split(lab)
    a -= (a.mean() - 128)
    b -= (b.mean() - 128)
    lab = cv2.merge([L, a, b]).clip(0, 255).astype("uint8")       # ← clip + cast
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def white_balance_image(img_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    wb  = _gray_world(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), wb)

def white_balance_batch(paths: List[Path], out_dir: Path):
    for p in paths:
        white_balance_image(p, out_dir / p.name)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    white_balance_image(Path(args.img), Path(args.out))

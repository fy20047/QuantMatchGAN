"""Batch face alignment and optional white‑balance.

Usage:
    python preprocess/run.py \
      --style_dir data/style/picasso \
      --content_dir data/content \
      --align off \
      --wb off \
      --out_dir data/processed
"""
import argparse, shutil, os
from pathlib import Path
from align_face import align_batch
from white_balance import white_balance_batch

def process_folder(src_dir: Path, dst_dir: Path, do_align: bool, do_wb: bool,
                   method: str = "dlib", padding: int = 0):
    dst_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    if do_align:
        align_batch(imgs, dst_dir, method=method, padding=padding)
    else:  # 只複製
        for p in imgs: shutil.copy2(p, dst_dir / p.name)
    if do_wb:
        white_balance_batch(list(dst_dir.iterdir()), dst_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 新版參數
    parser.add_argument("--style_dir")
    parser.add_argument("--content_dir")
    parser.add_argument("--out_dir")
    # 舊版保留
    parser.add_argument("--img")
    parser.add_argument("--out")
    # 共用選項
    parser.add_argument("--align", choices=["on","off"], default="on")
    parser.add_argument("--wb",    choices=["on","off"], default="on")
    parser.add_argument("--method", choices=["dlib","mediapipe"], default="dlib")
    parser.add_argument("--padding", type=int, default=0)
    args = parser.parse_args()

    do_align = (args.align == "on")
    do_wb    = (args.wb    == "on")

    if args.out_dir:       # 走新版
        assert args.style_dir and args.content_dir, "缺少 --style_dir 或 --content_dir"
        out_style   = Path(args.out_dir) / "style"
        out_content = Path(args.out_dir) / "content"
        process_folder(Path(args.style_dir),   out_style,   do_align, do_wb,
                       args.method, args.padding)
        process_folder(Path(args.content_dir), out_content, do_align, do_wb,
                       args.method, args.padding)
    else:                  # 走舊版
        assert args.img and args.out, "缺少 --img 或 --out"
        process_folder(Path(args.img), Path(args.out), do_align, do_wb,
                       args.method, args.padding)

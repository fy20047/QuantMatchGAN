#!/usr/bin/env python3
# tools/make_256.py
"""
將原始風格 / 內容圖批次 resize → 256×256，存到 data/sae_train/
使用方式：
    python tools/make_256.py \
           --src data/processed/style data/processed/content \
           --dst data/sae_train
"""
import os, cv2, glob, argparse
from tqdm import tqdm

def main(args):
    os.makedirs(args.dst, exist_ok=True)
    dst_paths = []
    for src_dir in args.src:
        for p in sorted(glob.glob(os.path.join(src_dir, '*'))):
            img = cv2.imread(p)
            if img is None: continue
            img256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            name = os.path.splitext(os.path.basename(p))[0]
            outp = os.path.join(args.dst, f'{name}.png')
            cv2.imwrite(outp, img256)
            dst_paths.append(outp)
    print(f'✓ 共匯出 {len(dst_paths)} 張到 {args.dst}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='+', required=True,
                    help='來源資料夾（可多個）')
    ap.add_argument('--dst', required=True, help='輸出資料夾')
    main(ap.parse_args())

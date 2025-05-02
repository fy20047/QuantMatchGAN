#!/usr/bin/env python3
"""
給 content idx，輸出橫幅 PNG：
內容 | Top1 | Top2 | Top3   （底下顯示 SRM 分數）
"""
import cv2, numpy as np, argparse
from pathlib import Path
from srm.match import srm_query     # 使用上面新 match.py

def strip_suffix(name: str):
    """把檔名最後的 '_S' 去掉；若末兩字不是 '_S' 則原封不動"""
    return name[:-2] if name.endswith('_S') else name

CONTENT_DIR    = Path('features/srm/content_feats_v2')
STYLE_IMG_DIR  = Path('data/processed/style')
CONTENT_IMG_DIR= Path('data/sae_data')

# 與 match.py 用同一排序基準
STYLE_PNG_LIST = sorted(STYLE_IMG_DIR.glob('*.png'))
STYLE_BASE = [strip_suffix(p.stem) for p in STYLE_PNG_LIST]    # .png → stem，再砍 "_S"

def load_img(path, H=256):
    img = cv2.imread(str(path))[:,:,::-1]
    h,w,_ = img.shape
    return cv2.resize(img, (int(w*H/h), H))

def main(idx, out):
    # Top-3 風格
    matches = srm_query(idx, 3)       # [(score, base), ...]
    # 內容圖
    cont_base = sorted(CONTENT_DIR.glob('*_S.npy'))[idx].stem[:-2]
    cont_img  = load_img(CONTENT_IMG_DIR/f'{cont_base}.png')

    tiles=[cont_img]; captions=['內容']
    for score, sid in matches:
        style_img = load_img(STYLE_IMG_DIR/f'{sid}.png')
        tiles.append(style_img)
        captions.append(f'SRM={score:.2f}')

    # 拼成一行
    gap, H = 10, tiles[0].shape[0]
    widths = [im.shape[1] for im in tiles]
    canvas = np.ones((H+40, sum(widths)+gap*len(tiles), 3), dtype=np.uint8)*255
    x=0
    for i,im in enumerate(tiles):
        canvas[0:H, x:x+im.shape[1]] = im
        cv2.putText(canvas, captions[i], (x+5, H+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        x += im.shape[1]+gap

    cv2.imwrite(out, canvas[:,:,::-1])
    print('✓ saved', out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, required=True, help='content idx (0-based)')
    ap.add_argument('--out', default='srm_panel.png')
    args = ap.parse_args()
    main(args.idx, args.out)

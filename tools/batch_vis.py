#!/usr/bin/env python3
"""
逐圖 Panel（Palette + AU + H + 原圖）
batch_vis.py  ——  對 feat_root 下每張圖輸出四聯圖：
   [原圖|Palette|AU-Heatmap|EdgeEntropy bar]

python tools/batch_vis.py \
  --imgs data/processed/style \
  --feat_root features/srm/style_feats \
  --out panels_style
"""
import os, glob, cv2, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm

def plot_one(img_path, feat_root, out_dir):
    base=os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)[:,:,::-1]
    au  = np.load(f'{feat_root}/{base}_E.npy')[:18]
    c   = np.load(f'{feat_root}/{base}_C.npy').reshape(4,3)
    rgb = cv2.cvtColor(c[np.newaxis,:,:].astype(np.float32), cv2.COLOR_Lab2RGB)[0]
    h   = float(np.load(f'{feat_root}/{base}_H.npy'))

    plt.figure(figsize=(6,2))
    plt.subplot(1,4,1); plt.imshow(img); plt.axis('off')
    plt.subplot(1,4,2); plt.imshow(rgb[np.newaxis,:,:]); plt.axis('off')
    plt.subplot(1,4,3); sns.heatmap(au.reshape(1,-1), cmap='Reds',
                                    yticklabels=False, xticklabels=False,
                                    cbar=False)
    plt.subplot(1,4,4); plt.bar([0], [h]); plt.ylim(0,4); plt.xticks([])
    plt.tight_layout(); plt.savefig(f'{out_dir}/{base}_panel.png'); plt.close()

if __name__ == '__main__':
    import argparse, pathlib
    ap=argparse.ArgumentParser(); ap.add_argument('--imgs',required=True)
    ap.add_argument('--feat_root',required=True); ap.add_argument('--out',default='panels')
    args=ap.parse_args(); pathlib.Path(args.out).mkdir(exist_ok=True)
    for p in tqdm(glob.glob(os.path.join(args.imgs,'*.png'))):
        plot_one(p, args.feat_root, args.out)
    print('✓ Panels →', args.out)

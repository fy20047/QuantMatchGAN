#!/usr/bin/env python3

# python tools/vis_au_pose.py \
#   --img data/processed/style/000087.png \
#   --feat_root features/srm/style_feats


import numpy as np, matplotlib.pyplot as plt, seaborn as sns, argparse, cv2, os

def main(img, feat_root):
    base = os.path.splitext(os.path.basename(img))[0]
    au_pose = np.load(f'{feat_root}/{base}_E.npy')      # 30 維
    au = au_pose[:18]

    im = cv2.imread(img)[:,:,::-1]
    plt.figure(figsize=(6,2.5))
    plt.subplot(1,2,1); plt.imshow(im); plt.axis('off'); plt.title('Image')

    plt.subplot(1,2,2)
    sns.heatmap(au.reshape(1,-1), cmap='Reds',
                yticklabels=['AU'], xticklabels=[f'AU{i+1}' for i in range(18)],
                cbar=False)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout(); plt.savefig('au_heatmap.png', dpi=300)
    print('✓ au_heatmap.png saved')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True)
    ap.add_argument('--feat_root', required=True)
    args = ap.parse_args()
    main(args.img, args.feat_root)

#!/usr/bin/env python3
"""
掃描逐圖特徵 (<img>_(S|C|E|H).npy) → 整合成 features.csv
CSV 欄位：
    img, σ (S std), H, AU_1 … AU_18, Pose_1 … Pose_12, C_1 … C_12

python tools/build_feature_csv.py --feat_root features/srm/style_feats --out_csv features_vn.csv

"""

#!/usr/bin/env python3
import os, glob, argparse, numpy as np, pandas as pd

def main(feat_root, out_csv):
    rows = []
    for f in glob.glob(os.path.join(feat_root, '*_S.npy')):
        base = os.path.basename(f)[:-6]
        z = np.load(f); sigma = np.std(np.abs(z))
        h = float(np.load(f'{feat_root}/{base}_H.npy'))
        c = np.load(f'{feat_root}/{base}_C.npy')
        e = np.load(f'{feat_root}/{base}_E.npy')

        # --------- 依長度拆 AU / Pose ----------
        if len(e) == 30:                   # v0
            au, pose = e[:18], e[18:30]
        elif len(e) == 6:                  # v2
            au, pose = e[:4],  e[4:6]      # PCA→Pose1,2
        else:
            raise ValueError(f"E dim={len(e)}; 檢查 extract_all.py")
        # ---------------------------------------

        row = {'img': base+'.png', 'sigma': sigma, 'H': h}
        row.update({f'AU{i+1}':   au[i]   for i in range(len(au))})
        row.update({f'Pose{i+1}': pose[i] for i in range(len(pose))})
        row.update({f'C{i+1}':    c[i]    for i in range(12)})
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('✓ CSV saved →', out_csv)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat_root', required=True)
    ap.add_argument('--out_csv',   default='features.csv')
    main(**vars(ap.parse_args()))


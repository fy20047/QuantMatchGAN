#!/usr/bin/env python3
"""
build_index.py ─ 把 Picasso 風格集的 S 向量寫入 Faiss cosine 索引
‧ 支援兩種輸入：
    1. 批量檔  : <feat_root>/S.npy   (N, 2048)
    2. 逐圖檔  : <feat_root>/*_S.npy (N 個 2048 向量)

用法：
    python srm/build_index.py \
           --feat_root features/srm/style_feats_k05_per \
           --out       srm/srm_S_k05.index
"""
import os, glob, faiss, numpy as np, argparse, pathlib

ap = argparse.ArgumentParser()
ap.add_argument('--feat_root', required=True, help='S 向量目錄；可 batch 或 per-image')
ap.add_argument('--out',       required=True, help='欲輸出之 faiss 索引檔路徑')
args = ap.parse_args()

feat_root = pathlib.Path(args.feat_root)
out_path  = pathlib.Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---------- 讀取向量 ----------
batch_path = feat_root / 'S.npy'
if batch_path.is_file():                         # 情況 1：批量檔
    S = np.load(batch_path).astype('float32')
else:                                            # 情況 2：逐圖檔
    vec_paths = sorted(feat_root.glob('*_S.npy'))
    if len(vec_paths) == 0:
        raise RuntimeError(f'找不到任何 S 向量於 {feat_root}')
    S = np.stack([np.load(p).astype('float32') for p in vec_paths])

# ======= 在這裡加入兩行 mean-center =======
mu = S.mean(0, keepdims=True)   # (1,2048)
S  = S - mu                     # 所有向量扣掉平均

# ---------- 建立 Faiss cosine 索引 ----------
faiss.normalize_L2(S)
index = faiss.IndexFlatIP(S.shape[1])   # 內積 = cosine (已 L2 normalize)
index.add(S)
faiss.write_index(index, str(out_path))
print(f'✓ S-index built → {out_path} (N={len(S)})')

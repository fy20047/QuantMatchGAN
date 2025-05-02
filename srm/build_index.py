#!/usr/bin/env python3
"""將 300 張 Picasso 風格圖的 S 向量 (2048) 寫入 Faiss cosine 索引
v2 流程建議：Faiss 只建 S (2048) 索引，
其餘 C/E/H′ 用 srm_utils.srm_score() 線上加權；
這樣能減少維度、查詢更快，也便於之後微調權重。"""
import faiss, numpy as np, argparse
from pathlib import Path

STYLE_DIR = Path('features/srm/style_feats_v2')
INDEX_PATH= Path('srm/srm_S.index')
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

STYLE_NPY  = sorted(STYLE_DIR.glob('*_S.npy'))          # ≤── 唯一排序依據
STYLE_BASE = [p.with_suffix('').stem for p in STYLE_NPY]  # 只砍 .npy

vecs = [np.load(p).astype('float32') for p in STYLE_NPY]
S    = np.stack(vecs)
faiss.normalize_L2(S)

index = faiss.IndexFlatIP(S.shape[1])
index.add(S)
faiss.write_index(index, str(INDEX_PATH))
print('✓ S‑index rebuilt →', INDEX_PATH)

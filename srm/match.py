#!/usr/bin/env python3
"""線上 SRM 配對：
   ① 先用 Faiss(S) 找 TOP_T 粗匹配
   ② 再用新版 SRM αβγδ 重新評分
"""
import argparse, faiss, numpy as np
from pathlib import Path
from features.srm_utils import srm_score

def strip_suffix(name: str):
    """把檔名最後的 '_S' 去掉；若末兩字不是 '_S' 則原封不動"""
    return name[:-2] if name.endswith('_S') else name

# ── 路徑 ─────────────────────────────────────────
STYLE_DIR   = Path('features/srm/style_feats_v2')
CONTENT_DIR = Path('features/srm/content_feats_v2')
INDEX_PATH  = Path('srm/srm_S.index')
TOP_T = 10      # 先抓 10 名，再重排

# 直接列舉風格圖檔名，確保與 build_index.py 用同一種排序
STYLE_NPY  = sorted(STYLE_DIR.glob('*_S.npy'))         # Path 物件列表
STYLE_BASE = [strip_suffix(p.with_suffix('').stem) for p in STYLE_NPY]   # ← 去掉最後 2 字元 "_S"

# ── 工具函式 ─────────────────────────────────────
def load_feat(root, base):
    return {
        'S': np.load(root/f'{base}_S.npy'),
        'C': np.load(root/f'{base}_C.npy'),
        'E': np.load(root/f'{base}_E.npy'),      # 7 維
        'H': float(np.load(root/f'{base}_H.npy'))
    }

def srm_query(idx_content: int, topk: int = 3):
    # 讀取 content 特徵
    cont_files = sorted(CONTENT_DIR.glob('*_S.npy'))
    c_base = cont_files[idx_content].stem[:-2]        # 去掉 _S
    feat_c = load_feat(CONTENT_DIR, c_base)
    q = feat_c['S'].copy()
    faiss.normalize_L2(q.reshape(1,-1))

    # Faiss 預選
    index = faiss.read_index(str(INDEX_PATH))
    _, I = index.search(q.reshape(1,-1), TOP_T)

    # SRM 打分
    cand = []
    for idx in I[0]:
        s_base = STYLE_BASE[idx]                      # 與 Faiss 序號對應
        feat_s = load_feat(STYLE_DIR, s_base)
        score  = srm_score(feat_c, feat_s)
        cand.append((score, s_base))
    cand.sort(reverse=True)
    return cand[:topk]

# ── CLI ─────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, required=True, help='content idx (0-based)')
    ap.add_argument('-k','--topk', type=int, default=3)
    args = ap.parse_args()

    for score, sid in srm_query(args.idx, args.topk):
        print(f"{sid}  | SRM = {score:.4f}")

#!/usr/bin/env python3
"""
vis_srm_match.py  ——  把 content idx = --idx
與其 SRM Top-k 風格圖排成一行 4 圖  (內容 + Top-3)
"""
import argparse, faiss, numpy as np, cv2, torch
from pathlib import Path
from features.sae.model import SAE            # ★
from features.extract_all import load_img, to_tensor  # util

ap = argparse.ArgumentParser()
ap.add_argument('--idx', type=int, default=0)
ap.add_argument('--s_index', required=True)
ap.add_argument('--style_feat_root', required=True)
ap.add_argument('--sae_weight', required=True)          # ★ 新增
ap.add_argument('--ksparse', type=float, default=0.05)  # 與訓練一致
ap.add_argument('--topk', type=int, default=3)
args = ap.parse_args()

# ----------- 讀取 style S 庫 & 索引 -------------
STYLE_DIR  = Path(args.style_feat_root)
vec_paths  = sorted(STYLE_DIR.glob('*_S.npy'))
STYLE_BASE = [p.stem[:-2] for p in vec_paths]
S_mat      = np.stack([np.load(p).astype('float32') for p in vec_paths])
faiss.normalize_L2(S_mat)
index      = faiss.read_index(args.s_index)

# ----------- 建立 SAE（只做一次） ---------------
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
sae = SAE(bottleneck=2048).to(dev).eval()
sae.load_state_dict(torch.load(args.sae_weight, map_location=dev), strict=False)

def encode_s(img):
    with torch.no_grad():
        z_full = sae(to_tensor(img).to(dev))[1].squeeze().cpu().numpy()
    if args.ksparse>0:
        k = int(2048*args.ksparse)
        topk = np.argpartition(np.abs(z_full), -k)[-k:]
        mask = np.zeros_like(z_full); mask[topk] = 1
        z = z_full*mask
    else:
        z = z_full
    return z.astype('float32')

# ----------- 取內容圖 → S_vec -------------------
CONTENT_DIR = Path('data/sae_data')
content_paths = sorted(CONTENT_DIR.glob('*'))
c_img = load_img(str(content_paths[args.idx]))
c_S   = encode_s(c_img)
faiss.normalize_L2(c_S.reshape(1,-1))

# ----------- 查詢 Top-k -------------------------
D,I = index.search(c_S.reshape(1,-1), args.topk+1)
ids, sims = [], []
for sid,sim in zip(I[0], D[0]):        # 排除「跟自己 cos=1」那筆
    if sim < 0.999: ids.append(sid); sims.append(sim)
    if len(ids)==args.topk: break

# ----------- 拼圖 -------------------------------
canvas = cv2.resize(c_img,(256,256))
for rank,(sid,sim) in enumerate(zip(ids,sims),1):
    sty_img = cv2.imread(f'data/processed/style/{STYLE_BASE[sid]}.png')[:,:,::-1]
    block = cv2.putText(cv2.resize(sty_img,(256,256)).copy(),
                        f'{rank}:{sim:.2f}', (5,24),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,0),2)
    canvas = np.hstack([canvas, block])

out = Path(f'srm_vis_idx{args.idx}.png')
cv2.imwrite(str(out), canvas[:,:,::-1])
print('✓ saved →', out)
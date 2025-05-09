#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_all.py —— 離線 / 線上 特徵萃取（含 k-sparse 遮罩）
--------------------------------------------------------
輸入：資料夾 (--dir)；輸出：S / C / E / H 四通道特徵
支援：
  • 批量模式 (default)      →  S.npy / C.npy / E.npy / H.npy  + meta.json
  • 逐圖模式 (--per_image) →  <img>_S.npy, <img>_C.npy, ...
新增：
  • --ksparse <float>      → 0~1，與訓練階段同樣的 top-k 遮罩比例
"""

import os, glob, json, argparse
import numpy as np, cv2, torch
from tqdm import tqdm

# =========== import 子模組 ============
from features.sae.model    import SAE
from features.palette_emd  import palette_emd
from features.sobel_energy import sobel_energy          # H′
from features.expression   import extract_au            # AU 4 維
# Landmark PCA 模型（若缺失則退化為 0）
try:
    from features.landmark_pca import extract_pca2, pca_model
    USE_PCA = True
except Exception as e:
    print("[Warn] Landmark PCA 模型未載入，Pose 2D 以 0 代替 :", e)
    USE_PCA = False
# =====================================

# ---------- util ----------
def load_img(path):
    """以 256×256 讀取圖像（BGR→RGB）"""
    img = cv2.imread(path)[:, :, ::-1]
    return cv2.resize(img, (256, 256))

def to_tensor(img):
    """轉成 [-1,1] Tensor，並加 batch 維度"""
    t = torch.from_numpy(img.transpose(2,0,1)).float()/127.5 - 1
    return t.unsqueeze(0)            # 1×3×256×256

# ---------- main ----------
def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 載入 SAE -------------------------------------------------------
    sae = SAE(bottleneck=args.bottleneck).to(dev).eval()
    sae.load_state_dict(torch.load(args.sae_weight, map_location=dev),
                        strict=False)

    paths = sorted(glob.glob(os.path.join(args.dir, '*')))
    N     = len(paths)

    # ---- 預開批量容器（僅在批量模式）--------------------------------------
    if not args.per_image:
        S = np.zeros((N, args.bottleneck), dtype=np.float32)
        C = np.zeros((N, 12),            dtype=np.float32)
        E = np.zeros((N, 7),             dtype=np.float32)
        H = np.zeros((N, 1),             dtype=np.float32)

    os.makedirs(args.out_root, exist_ok=True)

    # ======================= 逐圖處理 =======================
    for idx, p in enumerate(tqdm(paths)):
        img  = load_img(p)
        base = os.path.splitext(os.path.basename(p))[0]

        # ----- 幾何向量 S -------------------------------------------------
        with torch.no_grad():
            z_full = sae(to_tensor(img).to(dev))[1].squeeze()   # (2048,)
            z_full = z_full.cpu().numpy()

        # -- 若指定 ksparse，套與訓練相同的 top-k 遮罩 --------------------
        if args.ksparse>0:
            k = int(args.bottleneck*args.ksparse)
            shift = np.random.randint(0, args.bottleneck)
            z_roll = np.roll(z_full, shift)
            topk = np.argpartition(z_roll, -k)[-k:]
            mask = np.zeros_like(z_roll); mask[topk]=1
            mask = np.roll(mask, -shift)
            z = z_full*mask
        else:
            z = z_full


        # ----- 色彩向量 C (Palette-EMD 12D) -----------------------------
        c = palette_emd(img)

        # ----- 表情 + 視角 E  (7D) --------------------------------------
        au4 = extract_au(img)[:4]          # AU1-4 左右對稱
        pca2 = extract_pca2(img, pca_model) if USE_PCA else np.zeros(2)
        yaw  = pca2[0]                     # yaw 保留正負號
        e = np.concatenate([au4, pca2, [yaw]])   # (7,)

        # ----- 邊緣能量 H′ (1D) -----------------------------------------
        h = np.array([sobel_energy(img)], dtype=np.float32)

        # ---------- 寫檔 ----------
        if args.per_image:
            np.save(f"{args.out_root}/{base}_S.npy", z)
            np.save(f"{args.out_root}/{base}_C.npy", c)
            np.save(f"{args.out_root}/{base}_E.npy", e)
            np.save(f"{args.out_root}/{base}_H.npy", h)
        else:
            S[idx], C[idx], E[idx], H[idx] = z, c, e, h
    # ====================== end for ========================

    # ---------- 批量模式：一次存 ----------
    if not args.per_image:
        np.save(os.path.join(args.out_root, 'S.npy'), S)
        np.save(os.path.join(args.out_root, 'C.npy'), C)
        np.save(os.path.join(args.out_root, 'E.npy'), E)
        np.save(os.path.join(args.out_root, 'H.npy'), H)
        json.dump({p: i for i, p in enumerate(paths)},
                  open(os.path.join(args.out_root, 'meta.json'), 'w'), indent=2)
        print(f"✓ batch 特徵已存到 {args.out_root}")
    else:
        print(f"✓ per-image 特徵已存到 {args.out_root}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir',        required=True,  help='輸入影像資料夾')
    ap.add_argument('--out_root',   required=True,  help='特徵輸出資料夾')
    ap.add_argument('--sae_weight', required=True,  help='SAE 權重 .pth')
    ap.add_argument('--bottleneck', type=int, default=2048)
    ap.add_argument('--per_image',  action='store_true',
                    help='逐圖輸出 <img>_(S|C|E|H).npy')
    ap.add_argument('--ksparse',    type=float, default=0.10,
                    help='與訓練相同的 k-sparse 比例 (0~1)')
    args = ap.parse_args()
    main(args)

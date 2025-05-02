#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_all.py  ——  離線 / 線上 特徵萃取
支援：
    • 批量模式 (default)  :  S.npy / C.npy / E.npy / H.npy  + meta.json
    • 逐圖模式 (--per_image):
        <img>_S.npy  (2048,)   幾何向量
        <img>_C.npy  (12,)     Palette-EMD
        <img>_E.npy  (30,)     AU 18 + pose 12
        <img>_H.npy  (1,)      EdgeEntropy
"""
import os, glob, json, argparse
import numpy as np, cv2, torch
from tqdm import tqdm

# ---------- import 子模組 ----------
from features.sae.model    import SAE
from features.palette_emd  import palette_emd
# from features.edge_entropy import edge_entropy  # H
from features.sobel_energy import sobel_energy   # H′
from features.expression   import extract_au as extract_expr
from features.landmark_pca import extract_pca2, pca_model   # 預先載入 PCA
# ---- Landmark PCA (可選) ---------------------------------------------
try:
    from features.landmark_pca import extract_pca2, pca_model
    USE_PCA = True
except Exception as e:
    print("[Warn] Landmark PCA 模型未載入，Pose 2D 以 0 代替 :", e)
    USE_PCA = False

# ---------- util ----------
def load_img(path):
    img = cv2.imread(path)[:, :, ::-1]
    return cv2.resize(img, (256, 256))

def to_tensor(img):
    t = torch.from_numpy(img.transpose(2,0,1)).float()/127.5 - 1
    return t.unsqueeze(0)            # 1,3,256,256

# ---------- main ----------
def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- SAE ------------------------------------------------------------
    sae = SAE(bottleneck=args.bottleneck).to(dev).eval()
    sae.load_state_dict(torch.load(args.sae_weight, map_location=dev),
                        strict=False)

    paths = sorted(glob.glob(os.path.join(args.dir, '*')))
    N     = len(paths)

    # 預先開批量容器（僅在 batch 模式）
    if not args.per_image:
        S = np.zeros((N, args.bottleneck), dtype=np.float32)
        C = np.zeros((N, 12), dtype=np.float32)
        E = np.zeros((N, 7), dtype=np.float32)
        H = np.zeros((N, 1), dtype=np.float32)

    os.makedirs(args.out_root, exist_ok=True)

    for idx, p in enumerate(tqdm(paths)):
        img  = load_img(p)
        base = os.path.splitext(os.path.basename(p))[0]

        with torch.no_grad():
            z = sae(to_tensor(img).to(dev))[1].squeeze().cpu().numpy()

        # ---- E 通道 6→7 維，保留 yaw 符號 ----
        au = extract_expr(img)            # AU1–4 (左右對稱) ~4
        pca2 = extract_pca2(img, pca_model) if USE_PCA else np.zeros(2)
        yaw  = pca2[0]                    # 取第一主成分近似 yaw，帶正負
        e    = np.concatenate([au[:4], pca2, [yaw]])   # 7D
        c    = palette_emd(img)
        h    = np.array([sobel_energy(img)], dtype=np.float32)

        if args.per_image:
            np.save(f"{args.out_root}/{base}_S.npy", z)
            np.save(f"{args.out_root}/{base}_C.npy", c)
            np.save(f"{args.out_root}/{base}_E.npy", e)
            np.save(f"{args.out_root}/{base}_H.npy", h)
        else:
            S[idx], C[idx], E[idx], H[idx] = z, c, e, h

    # -------- save --------
    if args.per_image:
        print(f"✓ per-image特徵已存到 {args.out_root}")
    else:
        np.save(os.path.join(args.out_root, 'S.npy'), S)
        np.save(os.path.join(args.out_root, 'C.npy'), C)
        np.save(os.path.join(args.out_root, 'E.npy'), E)
        np.save(os.path.join(args.out_root, 'H.npy'), H)
        print(f"✓ batch 特徵已存到 {args.out_root}")

    json.dump({p: i for i, p in enumerate(paths)},
              open(os.path.join(args.out_root, 'meta.json'), 'w'), indent=2)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir',        required=True, help='輸入影像資料夾')
    ap.add_argument('--out_root',   required=True, help='特徵輸出資料夾')
    ap.add_argument('--per_image',  action='store_true',
                    help='逐圖輸出 <img>_(S|C|E|H).npy')
    ap.add_argument('--sae_weight', required=True, help='SAE 權重 .pth')
    ap.add_argument('--bottleneck', type=int, default=2048)
    args = ap.parse_args()
    main(args)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1) 隨機抽 50 張 Picasso
2) FFHQ 203 張 + 抽樣 Picasso 50 張  →  對齊 / 裁切 256²
3) Picasso 灰階  →  data/sae_mix256/picasso_gray
   FFHQ 彩色      →  data/sae_mix256/nat
4) 自動呼叫 features/sae/train_sae_sparse.py 開始訓練
"""

import os, random, glob, shutil, subprocess, argparse, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

# ---------- config ----------
ROOT = "/home/SlipperY/Pica/Picasso_GAN_Project"
SRC_PICASSO = f"{ROOT}/data/processed/style"
SRC_FFHQ    = f"{ROOT}/data/sae_data"

DST_NAT     = f"{ROOT}/data/sae_mix256/nat"
DST_PICASSO = f"{ROOT}/data/sae_mix256/picasso_gray"
BATCH_SZ    = 16
EPOCHS      = 12
BOTTLENECK  = 2048
# -----------------------------

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, refine_landmarks=True)

def align_crop(img, lm, size=256):
    """三點對齊：左右眼中心、嘴中心"""
    eyeL = np.mean(lm[[33, 133]], axis=0)   # 左眼
    eyeR = np.mean(lm[[362, 263]], axis=0)  # 右眼
    mouth = lm[13]                          # 下唇
    src = np.float32([eyeL, eyeR, mouth])
    dst = np.float32([[0.3,0.4],[0.7,0.4],[0.5,0.75]])*size
    M   = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (size, size),
                          flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REFLECT)

def process_one(img_path, dst_path, to_gray=False, fallback_center=True):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    res = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ok  = res.multi_face_landmarks
    if ok:
        lm = np.array([(p.x*w, p.y*h) for p in res.multi_face_landmarks[0].landmark])
        out = align_crop(img, lm)
    elif fallback_center:
        # 失敗時直接中心裁切後縮小
        side = min(h, w)
        y0 = (h-side)//2
        x0 = (w-side)//2
        out = img[y0:y0+side, x0:x0+side]
        out = cv2.resize(out, (256,256), interpolation=cv2.INTER_AREA)
    else:
        return False
    if to_gray:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.merge([g,g,g])
    cv2.imwrite(dst_path, out)
    return True

def build_dataset():
    # 1. 隨機抽 50 Picasso
    picasso_all = sorted(glob.glob(f"{SRC_PICASSO}/*"))
    sample_50   = random.sample(picasso_all, 50)

    # 建立資料夾
    Path(DST_NAT).mkdir(parents=True, exist_ok=True)
    Path(DST_PICASSO).mkdir(parents=True, exist_ok=True)

    # 2. FFHQ 全處理
    ffhq_all = sorted(glob.glob(f"{SRC_FFHQ}/*"))
    print(f"Processing {len(ffhq_all)} FFHQ...")
    for p in tqdm(ffhq_all):
        dst = f"{DST_NAT}/{Path(p).name}"
        process_one(p, dst, to_gray=False)

    # 3. Picasso 抽樣處理 (灰階)
    print("Processing 50 Picasso (gray)...")
    for p in tqdm(sample_50):
        dst = f"{DST_PICASSO}/{Path(p).name}"
        process_one(p, dst, to_gray=True)

def train_sae():
    train_script = f"{ROOT}/features/sae/train_sae_sparse.py"
    cmd = [
        "python", train_script,
        "--natural", DST_NAT,
        "--style",   DST_PICASSO,
        "--ratio_style", "0.2",            # 203:50 ≈ 80:20
        "--bottleneck", str(BOTTLENECK),
        "--epochs",    str(EPOCHS),
        "--batch",     str(BATCH_SZ),
        "--lambda_l1", "0.03",
        "--lambda_warm","2",
        "--ksparse",   "0.03",
        "--gray_mode","all"
    ]
    print("Launching SAE training …\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='重新前處理資料 (預設跳過)')
    args = parser.parse_args()
    if args.rebuild or not (os.path.exists(DST_NAT) and os.listdir(DST_NAT)):
        build_dataset()
    train_sae()

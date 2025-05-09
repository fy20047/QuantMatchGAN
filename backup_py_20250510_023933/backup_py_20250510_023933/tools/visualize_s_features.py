# 功能：
# 讀取兩個 S 特徵資料夾（Picasso / FFHQ）
# 計算每張圖的 σ(z_S)、平均稀疏度
# 以 PCA + t-SNE 轉 2D，畫散點圖
# 畫兩組 σ̄ 直方圖（辨識形變豐富度差異）

# 執行範例
# python tools/visualize_s_features.py \
#   --dir_a features/srm/style_S \
#   --dir_b features/srm/content_S \
#   --name_a Picasso --name_b FFHQ

# 你會拿到：
# hist_tsne_sigma.png – σ 直方對比圖
# tsne_tsne_sigma.png – t-SNE 2D 散點圖

# 評估指標
# σ 直方圖：理論上 Picasso 組應該平均右移（σ 較大，形變豐富）。
# t-SNE：兩組點若明顯分開，代表 SAE 的幾何空間能區分風格；
# 若混在一起，可能要再提高 Picasso 样本比例或 Lambda_L1。

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_dir(feat_dir):
    feats = []
    for p in sorted(glob.glob(os.path.join(feat_dir, '*.npy'))):
        f = np.load(p)
        if f.shape != (2048,):        # ← 只收 2048 向量
            continue
        feats.append(f)

    if not feats:
        raise RuntimeError(f'No (2048,) .npy files found in {feat_dir}. '
                        '確定有用 --per_image 抽取，並排除 batch S.npy')
    
    return np.stack(feats)   # (N,2048)

def calc_metrics(S):
    sparsity = np.mean(np.abs(S) < 5e-3)     # 全體 0 比率
    sigma    = np.std(np.abs(S), axis=1)     # 每張 σ
    return sparsity, sigma

def visualize(S_a, S_b, name_a, name_b, out_fig='tsne_sigma.png'):
    # 1. σ 直方圖 ----------------------------------------------------------------
    _, sigma_a = calc_metrics(S_a)
    _, sigma_b = calc_metrics(S_b)
    plt.figure(figsize=(6,4))
    plt.hist(sigma_a, bins=30, alpha=.6, label=f'{name_a} σ')
    plt.hist(sigma_b, bins=30, alpha=.6, label=f'{name_b} σ')
    plt.xlabel('σ(z_S)'); plt.ylabel('#images'); plt.legend(); plt.tight_layout()
    plt.savefig('hist_'+out_fig, dpi=300)

    # 2. t-SNE -------------------------------------------------------------------
    feats = np.concatenate([S_a, S_b], 0)
    label = np.array([0]*len(S_a) + [1]*len(S_b))
    pca = PCA(n_components=50).fit_transform(feats)
    ts  = TSNE(n_components=2, perplexity=30, init='pca',
               learning_rate='auto').fit_transform(pca)
    plt.figure(figsize=(5,5))
    plt.scatter(ts[label==0,0], ts[label==0,1], s=8, alpha=.7, label=name_a)
    plt.scatter(ts[label==1,0], ts[label==1,1], s=8, alpha=.7, label=name_b)
    plt.axis('off'); plt.legend(); plt.tight_layout()
    plt.savefig('tsne_'+out_fig, dpi=300)
    print('✓ Figures saved → hist_'+out_fig+' , tsne_'+out_fig)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir_a', required=True, help='Picasso S 目錄')
    ap.add_argument('--dir_b', required=True, help='FFHQ  S 目錄')
    ap.add_argument('--name_a', default='Picasso')
    ap.add_argument('--name_b', default='FFHQ')
    args = ap.parse_args()

    S_a = load_dir(args.dir_a)
    S_b = load_dir(args.dir_b)

    spa_a, _ = calc_metrics(S_a)
    spa_b, _ = calc_metrics(S_b)
    print(f'{args.name_a} sparsity={spa_a:.3f} | {args.name_b} sparsity={spa_b:.3f}')

    visualize(S_a, S_b, args.name_a, args.name_b)

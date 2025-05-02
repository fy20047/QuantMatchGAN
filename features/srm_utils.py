"""
新版 SRM 計分工具  (α,β,γ,δ) = (0.50, 0.30, 0.15, 0.05)
E 向量長度 = 6   (AU1–4 + PCA1,PCA2)
H' 以差值 → 相似度   sim_H = 1 - |Hi-Hj| / Hmax
"""
import numpy as np
from numpy.linalg import norm

# ---- 超參數 -----------------------------------------------------------
ALPHA  = 0.50      # S (geometry)
BETA   = 0.30      # C (palette-EMD)
GAMMA  = 0.15      # E (AU+Pose PCA)
DELTA  = 0.05      # H' (Sobel energy)
H_MAX  = 600.0     # 依資料集最大值設定，可動態計算

# ---- 基本相似度 -------------------------------------------------------
def cos_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

def h_sim(h_i, h_j, h_max=H_MAX):
    return 1.0 - abs(h_i - h_j) / h_max

# ---- 主函式 ----------------------------------------------------------
def srm_score(feat_i, feat_j):
    """
    feat dict 需包含:
        S : (2048,) np.ndarray
        C : (12,)
        E : (6,)
        H : float  (H′)
    """
    s_sim = cos_sim(feat_i['S'], feat_j['S'])
    c_sim = cos_sim(feat_i['C'], feat_j['C'])
    e_sim = cos_sim(feat_i['E'], feat_j['E'])
    h_sim_ = h_sim(feat_i['H'], feat_j['H'])

    return (ALPHA * s_sim +
            BETA  * c_sim +
            GAMMA * e_sim +
            DELTA * h_sim_)

import cv2, numpy as np
from scipy.spatial.distance import jensenshannon

# ---------- 1. 125 維 Lab 直方圖 ----------
def lab_hist(img: np.ndarray, bins: int = 5) -> np.ndarray:
    """
    將輸入 BGR 圖像轉 Lab，並在 (L,a,b) 各自等距分成 5 段，
    輸出 shape=(125,) 的機率直方圖 (總和 = 1)。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2],
                        None,
                        [bins, bins, bins],
                        [0, 100,  -128, 127,  -128, 127])  # L, a, b 範圍
    hist = hist.flatten().astype(np.float32)
    return hist / (hist.sum() + 1e-8)

# ---------- 2. Jensen-Shannon Divergence ----------
def js_div(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    JS Divergence (對稱，已開根號)；回傳值區間 [0, ln(2)]。
    這裡直接平方，把距離化成「類似 MSE」效果，後續做 z-score。
    """
    return float(jensenshannon(h1, h2) ** 2)

# 方便外部直接 import palette_emd 同名函式
palette_emd = None   # 先預留, 由 palette_emd.py 實際提供
from features.palette_emd import palette_emd   # 實際函式

import cv2, numpy as np
from pyemd import emd

# ---------- 1. 取 K 個主色 ----------
def _get_palette(img: np.ndarray, k: int = 4) -> np.ndarray:
    """BGR → K×3 Lab 主色, 依 hue 角度排序"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, _, centers = cv2.kmeans(lab, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    a, b = centers[:, 1], centers[:, 2]
    hue = np.arctan2(b, a)
    order = np.argsort(hue)
    return centers[order]                              # (k,3) Lab

# ---------- 2. Earth-Mover 距離 ----------
def _emd_palette(p1: np.ndarray, p2: np.ndarray) -> float:
    k = len(p1)
    w = np.full(k, 1.0 / k, dtype=np.float64)
    dist = np.linalg.norm(p1[:, None] - p2[None, :], axis=-1).astype(np.float64)
    return float(emd(w, w, dist))

# ---------- 3. 相似度 (距離 → RBF) ----------
def palette_similarity(p1: np.ndarray, p2: np.ndarray, sigma: float = 25.0) -> float:
    d = _emd_palette(p1, p2)
    return float(np.exp(- (d ** 2) / (2 * sigma ** 2)))

# ---------- 4. 12-維向量（給 extract_all 用） ----------
def palette_emd(img: np.ndarray, k: int = 4) -> np.ndarray:
    """
    將圖像轉成 12-D 向量 [L1,a1,b1, …, L4,a4,b4]，供 SRM 做 cosine。
    """
    pal = _get_palette(img, k)             # (k,3)
    return pal.astype(np.float32).reshape(-1)   # 12, dtype float32

## ✦ 2. 專案實作雛型

### 2-1  新增 `features/sobel_energy.py`

import cv2, numpy as np

def sobel_energy(img_rgb):
    """回傳 Sobel x+y 梯度 RMS 作線條能量."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # 自適應 Canny 閾值：取中位數 ±20 %
    v = np.median(gray); lower = int(max(0, 0.8*v)); upper = int(min(255, 1.2*v))
    edges = cv2.Canny(gray, lower, upper)
    # Sobel
    sx = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(edges, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.sqrt(sx**2 + sy**2).mean()          # RMS
    return float(energy)
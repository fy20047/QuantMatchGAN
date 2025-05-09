import cv2, numpy as np

def edge_entropy(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 轉灰階  
    edges = cv2.Canny(gray, 50, 150)                   # 邊緣偵測  
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)   # 極座標 Hough  
    theta = [l[0][1] for l in lines] if lines is not None else []
    if not theta:                     # 若偵測不到線，H = 0
        return 0.0
    hist, _ = np.histogram(theta, bins=18, range=(0, np.pi))
    p = hist / np.sum(hist)           # 方向機率分布  
    return float(-(p * np.log(p + 1e-8)).sum())  # Shannon entropy

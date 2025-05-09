"""
align_face.py ──
  ‣ align_image(img_path, out_path, method='dlib'|'mediapipe', padding=0.1)
  ‣ align_batch(list[Path], out_dir, method, padding)
  ‣ CLI 範例：
      python preprocess/align_face.py --img src.jpg --out dst.jpg --method mediapipe
"""

import cv2, dlib, mediapipe as mp, argparse
import numpy as np
from pathlib import Path
from typing import List

## ────────────────── helper ──────────────────
_DLIB_DET  = dlib.get_frontal_face_detector()
_DLIB_PRED = dlib.shape_predictor("preprocess/weights/shape_predictor_68_face_landmarks.dat")
_MP_FACE   = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def _detect_landmarks(img, method):
    if method == "dlib":
        face = _DLIB_DET(img, 1)[0]
        lm   = _DLIB_PRED(img, face)
        return np.array([(lm.part(i).x, lm.part(i).y) for i in range(68)])
    else:
        res = _MP_FACE.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise ValueError("No face detected by mediapipe")
        pts = res.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        return np.array([(int(p.x*w), int(p.y*h)) for p in pts])

def align_image(img_path: Path, out_path: Path,
                method: str = "dlib", padding: float = 0.1,
                out_size: int = 1024):
    img = cv2.imread(str(img_path))
    pts = _detect_landmarks(img, method)

    # 三基準點：雙眼中心 + 嘴中心
    if method == "dlib":
        eye_l, eye_r = pts[36:42].mean(0), pts[42:48].mean(0)
        mouth        = pts[48:60].mean(0)
    else:
        eye_l, eye_r, mouth = pts[33], pts[263], pts[13]

    src = np.float32([eye_l, eye_r, mouth])
    ref = np.float32([
        [0.3*out_size, 0.35*out_size],
        [0.7*out_size, 0.35*out_size],
        [0.5*out_size, 0.65*out_size]
    ])
    M   = cv2.getAffineTransform(src, ref)
    aligned = cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_CUBIC)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), aligned)

def align_batch(paths: List[Path], out_dir: Path,
                method: str = "dlib", padding: float = 0.1):
    for p in paths:
        dst = out_dir / p.name
        try:
            align_image(p, dst, method, padding)
        except Exception as e:
            print(f"[!] {p.name} skipped: {e}")

## ────────────────── CLI ──────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--method", choices=["dlib","mediapipe"], default="dlib")
    ap.add_argument("--padding", type=float, default=0.1)
    args = ap.parse_args()
    align_image(Path(args.img), Path(args.out), args.method, args.padding)

#!/usr/bin/env python3
"""
批量產生 SRM 配對橫幅：
Content | Style1 | Style2 | Style3
底下顯示：TOTAL=0.82 (S=0.91 C=0.78 E=0.60 H=0.55)

使用方式（範例，預設 v2）：
python batch_vis_srm.py -k 3 \
    --s_index faiss/style_v3.index \
    --style_feat_root features/style/v3
"""
import cv2, json, argparse
import numpy as np
from pathlib import Path

# --- 導入 srm.match（改用模組匯入，才能動態覆寫變數） ---
import srm.match as sm
from srm.match import srm_query          # 只需要函式本身即可

from features.srm_utils import (
    ALPHA, BETA, GAMMA, DELTA,
    cos_sim, h_sim
)

# ----- 預設資料夾（可被 CLI 參數覆寫） -----
CONTENT_DIR    = sm.CONTENT_DIR               # 固定
STYLE_DIR      = sm.STYLE_DIR                 # 會視 --style_feat_root 覆寫
OUT_DIR        = Path('panel_out')
OUT_DIR.mkdir(exist_ok=True)

CONTENT_IMG_D  = Path('data/sae_data')
STYLE_IMG_D    = Path('data/processed/style')  # 僅用於顯示原圖

def load_feat(root: Path, base: str):
    """讀取單張（content / style）所有特徵"""
    return {
        'S': np.load(root / f'{base}_S.npy').astype('float32'),
        'C': np.load(root / f'{base}_C.npy').astype('float32'),
        'E': np.load(root / f'{base}_E.npy').astype('float32'),
        'H': float(np.load(root / f'{base}_H.npy'))
    }

def srm_components(fC, fS):
    s_sim = cos_sim(fC['S'], fS['S'])
    c_sim = cos_sim(fC['C'], fS['C'])
    e_sim = cos_sim(fC['E'], fS['E'])
    h_sim_ = h_sim(fC['H'], fS['H'])
    total  = (ALPHA * s_sim +
              BETA  * c_sim +
              GAMMA * e_sim +
              DELTA * h_sim_)
    return total, s_sim, c_sim, e_sim, h_sim_

def load_img(path: Path, H: int = 256):
    img = cv2.imread(str(path))[:, :, ::-1]  # BGR→RGB
    h, w, _ = img.shape
    return cv2.resize(img, (int(w * H / h), H))

# ----- 主流程 -----
def run(topk=3):
    cont_files = sorted(CONTENT_DIR.glob('*_S.npy'))

    for idx, f in enumerate(cont_files):
        c_base = f.stem[:-2]
        feat_c = load_feat(CONTENT_DIR, c_base)

        # 取得 SRM 相似度前 k 名
        matches = srm_query(idx, topk)        # [(score, style_base), ...]
        panels  = [load_img(CONTENT_IMG_D / f'{c_base}.png')]
        captions= ['Content']

        # 依序加入 k 張風格圖
        for score, s_base in matches:
            feat_s = load_feat(STYLE_DIR, s_base)
            total, s_sim, c_sim, e_sim, h_sim_ = srm_components(feat_c, feat_s)

            txt = (f"TOTAL={total:.2f}\n"
                   f"S={s_sim:.2f} C={c_sim:.2f}\n"
                   f"E={e_sim:.2f} H={h_sim_:.2f}")
            img = load_img(STYLE_IMG_D / f'{s_base}.png')
            panels.append(img)
            captions.append(txt)

        # 拼圖
        gap, H = 10, panels[0].shape[0]
        widths = [im.shape[1] for im in panels]
        canvas = np.ones((H + 60, sum(widths) + gap * len(panels), 3),
                         dtype=np.uint8) * 255
        x = 0
        for im, cap in zip(panels, captions):
            canvas[0:H, x:x + im.shape[1]] = im
            for ln, txt in enumerate(cap.split('\n')):
                cv2.putText(canvas, txt,
                            (x + 3, H + 22 + 18 * ln),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 1, cv2.LINE_AA)
            x += im.shape[1] + gap

        out_path = OUT_DIR / f'{idx:04d}_{c_base}.png'
        cv2.imwrite(str(out_path), canvas[:, :, ::-1])  # RGB→BGR
        print('✓', out_path)

# ----- CLI -----
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', '--topk', type=int, default=3,
                    help='每張 Content 顯示的 Style 數量')
    ap.add_argument('--s_index', type=str, default=None,
                    help='Faiss index 檔路徑（.index），若留空則用 srm.match 預設值')
    ap.add_argument('--style_feat_root', type=str, default=None,
                    help='Style 特徵 .npy 檔所在資料夾')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # --- 動態覆寫路徑 ---
    if args.style_feat_root:
        STYLE_DIR = Path(args.style_feat_root)
        sm.STYLE_DIR = STYLE_DIR                # 同步給 srm.match
        print(f'[INFO] 使用 Style 特徵資料夾：{STYLE_DIR}')

    if args.s_index:
        sm.S_INDEX = args.s_index               # srm.match 中需有 S_INDEX 變數
        print(f'[INFO] 使用 Faiss index：{sm.S_INDEX}')

    run(topk=args.topk)

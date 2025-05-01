"""線上輸入內容圖，計算 SRM，回傳 Top‑k 風格圖 ID 與權重。"""
import argparse, faiss, numpy as np, json
from pathlib import Path

INDEX_PATH = Path('srm') / 'srm.index'
FEAT_DIR   = Path('data/features')
STYLE_META = Path('data/style/style_meta.json')  # 存風格圖檔名順序

ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2


def load_vec(file):
    return np.load(FEAT_DIR / file).astype('float32')

def srm_query(idx_content: int, topk: int = 3):
    """idx_content 為內容圖在 features 中的序號"""
    # 讀取向量
    S_c = load_vec('S_content.npy')[idx_content]
    C_c = load_vec('C_content.npy')[idx_content]
    E_c = load_vec('E_content.npy')[idx_content]
    vec_c = np.concatenate([S_c, C_c, E_c]).reshape(1, -1)
    faiss.normalize_L2(vec_c)

    # 載入索引並查詢
    index = faiss.read_index(str(INDEX_PATH))
    D, I = index.search(vec_c, topk)  # 內積越大越相似

    # 讀取 style 檔名
    meta = json.load(open(STYLE_META))
    hits = []
    for score, idx in zip(D[0], I[0]):
        hits.append({'style_id': int(idx), 'filename': meta[idx], 'srm': float(score)})
    return hits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, required=True, help='index of content vector')
    parser.add_argument('-k', '--topk', type=int, default=3)
    args = parser.parse_args()

    results = srm_query(args.idx, args.topk)
    for h in results:
        print(h)
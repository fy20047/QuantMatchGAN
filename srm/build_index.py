"""離線將 300 張 Picasso 風格圖的 (S, C, E) 向量寫入 Faiss 索引。"""
import argparse, faiss, numpy as np, pathlib
from pathlib import Path

DATA_DIR = Path('data/features')
INDEX_PATH = Path('srm') / 'srm.index'

def load_vectors(name: str):
    return np.load(DATA_DIR / f'{name}.npy').astype('float32')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=str(INDEX_PATH))
    args = parser.parse_args()

    S = load_vectors('S')
    C = load_vectors('C')
    E = load_vectors('E')
    vecs = np.concatenate([S, C, E], axis=1)      # [N, 2048+125+30]

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)                # 內積 = cosine (已 L2 norm)
    index.add(vecs)
    faiss.write_index(index, args.index)
    print(f'Index saved to {args.index}')
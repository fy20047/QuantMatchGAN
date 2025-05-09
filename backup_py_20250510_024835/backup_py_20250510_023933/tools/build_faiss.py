#!/usr/bin/env python3
"""
1) 讀 style_feats_v2 逐圖 S 向量 → 建立 Faiss cosine 索引
2) 示範查詢：輸入 content S 向量，回傳 top-k 風格圖檔名＋ SRM 分數
"""
import faiss, numpy as np, glob, os, argparse, json
from features.srm_utils import srm_score

def load_feats(root):
    feats={}
    for f in glob.glob(f"{root}/*_S.npy"):
        base=os.path.basename(f)[:-6]
        feats[base]={
            'S': np.load(f).astype('float32'),
            'C': np.load(f"{root}/{base}_C.npy").astype('float32'),
            'E': np.load(f"{root}/{base}_E.npy").astype('float32'),
            'H': float(np.load(f"{root}/{base}_H.npy"))
        }
    return feats

def build_faiss(vecs):
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)           # inner product = cosine 因先行 L2 norm
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index

def main(style_root, content_root, topk, out_json):
    style_feats = load_feats(style_root)
    st_keys = list(style_feats.keys())
    style_matrix = np.stack([style_feats[k]['S'] for k in st_keys]).astype('float32')
    faiss.normalize_L2(style_matrix)
    index = build_faiss(style_matrix)

    # 查詢每張 content，找 top-k，並用 SRM 重新排序
    content_feats = load_feats(content_root)
    result={}
    for base, feat in content_feats.items():
        q = feat['S'].astype('float32').copy()
        faiss.normalize_L2(q.reshape(1,-1))
        D,I = index.search(q.reshape(1,-1), topk*2)   # 先抓多一點
        cand=[st_keys[idx] for idx in I[0]]
        scored=[ (srm_score(feat, style_feats[c]), c) for c in cand ]
        scored.sort(reverse=True)                     # 依 SRM 由大→小
        result[base] = scored[:topk]                  # [(score, style_id), ...]

    json.dump(result, open(out_json,'w'), indent=2)
    print("✓ saved matching list →", out_json)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--style_root',  default='features/srm/style_feats_v2')
    ap.add_argument('--content_root',default='features/srm/content_feats_v2')
    ap.add_argument('-k','--topk', type=int, default=3)
    ap.add_argument('--out_json', default='srm_matches_v2.json')
    main(**vars(ap.parse_args()))

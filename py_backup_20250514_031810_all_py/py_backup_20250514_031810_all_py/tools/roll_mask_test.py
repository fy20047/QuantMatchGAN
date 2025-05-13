import glob, numpy as np, json, sys, os, faiss
root = 'features/srm/style_feats_k20_roll'
vecs = []
act_cnt = np.zeros(2048, int)

for p in glob.glob(f'{root}/*_S.npy'):
    z = np.load(p); vecs.append(z)
    act_cnt += (z != 0)

vecs = np.stack(vecs).astype('float32')
print('→  有被啟用的維度數：', (act_cnt>0).sum())
print('→  單維最高啟用次數：', act_cnt.max())

faiss.normalize_L2(vecs)
index = faiss.IndexFlatIP(2048); index.add(vecs)
D,_ = index.search(vecs, 2)       # 最近鄰 (自己除外)
print('→  平均最近鄰 cos ：', float(D[:,1].mean()))

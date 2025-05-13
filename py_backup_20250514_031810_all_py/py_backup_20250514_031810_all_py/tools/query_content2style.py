# #!/usr/bin/env python3
# import cv2, faiss, argparse, glob, numpy as np, pandas as pd
# from pathlib import Path
# from features.srm_utils import ALPHA,BETA,GAMMA,DELTA,cos_sim,h_sim

# # ---------- util ----------
# def load_vec(root: Path, base: str, key: str):
#     # 讀取單通道特徵；檔名格式 <base>_(S|C|E|H).npy
#     return np.load(root/f'{base}_{key}.npy').astype('float32')

# def build_index(vec_dir: Path):
#     # ① 擷取 style 資料夾所有 S 向量 → ② L2 normalize → ③ 建 Faiss 內積索
#     vecs, names = [], []
#     for p in vec_dir.glob('*_S.npy'):
#         base = p.stem[:-2]     # 去掉 '_S
#         vecs.append(np.load(p))
#         names.append(base)
#     vecs = np.vstack(vecs).astype('float32')   # (N, 2048)
#     faiss.normalize_L2(vecs)
#     idx = faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs)
#     return idx, names, vecs.mean(0)  # vecs.mean(0)為全庫平均向量，用來 mean-center

# def find_img(base: str, folder: Path):
#     for ext in ('.png','.jpg','.jpeg','.PNG','.JPG','.JPEG'):
#         p = folder / f"{base}{ext}"
#         if p.exists(): return cv2.imread(str(p))
#     return None

# def pad_to(v, n):                      # 對齊維度不足時補 0
#     return np.pad(v, (0, n-len(v)), 'constant') if len(v)<n else v

# def query_one(c_base,c_root,idx,names,mu,k=5):
#     # 給定一張 content → 回傳前 k 名 style (IP, base)
#     v = load_vec(c_root,c_base,'S') - mu
#     v = v/np.linalg.norm(v)
#     D,I = idx.search(v[None],k)  # (1,k) scores & idx
#     return list(zip(D[0], [names[i] for i in I[0]]))

# def srm_scores(c_root,s_root,c_base,s_base):
#     # 計算四通道相似度並加權
#     fc={k:load_vec(c_root,c_base,k) for k in ['S','C','E','H']}
#     fs={k:load_vec(s_root,s_base,k) for k in ['S','C','E','H']}
#     s = cos_sim(fc['S'],fs['S'])
#     c = cos_sim(fc['C'],fs['C'])
#     dim=max(len(fc['E']),len(fs['E']))
#     e = cos_sim(pad_to(fc['E'],dim), pad_to(fs['E'],dim))
#     h = h_sim(fc['H'],fs['H'])
#     total = ALPHA*s + BETA*c + GAMMA*e + DELTA*h
#     return total,s,c,e,h

# def save_grid(c_base,sty_list,c_dir,s_dir,out_dir):
#     # 把 1 張 content + k 張 style 拼成水平圖；若 style 圖缺失發 WARN
#     out_dir.mkdir(parents=True,exist_ok=True)
#     img_c=find_img(c_base,c_dir)
#     if img_c is None:
#         print(f"[WARN] content img missing: {c_base}"); return
#     panel=[cv2.resize(img_c[:,:,::-1],(256,256))]
#     for s in sty_list:
#         img_s=find_img(s,s_dir)
#         if img_s is None:
#             print(f"[WARN] style img missing: {s}"); return
#         panel.append(cv2.resize(img_s[:,:,::-1],(256,256)))
#     cv2.imwrite(str(out_dir/f'{c_base}.png'),cv2.hconcat(panel)[:,:,::-1])

# # ---------- main ----------
# def main(a):
#     c_root,s_root=Path(a.content_feat),Path(a.style_feat)
#     idx,names,mu=build_index(s_root)
#     rows=[]
#     for p in c_root.glob('*_S.npy'):
#         c_base=p.stem[:-2]
#         v=load_vec(c_root,c_base,'S')-mu; v/=np.linalg.norm(v)
#         D,I=idx.search(v[None],a.k); top=list(zip(D[0],[names[i] for i in I[0]]))
#         if a.vis_dir: save_grid(c_base,[n for _,n in top],Path(a.content_img),Path(a.style_img),Path(a.vis_dir))
#         for ip,s_base in top:
#             fc={k:load_vec(c_root,c_base,k) for k in ['S','C','E','H']}
#             fs={k:load_vec(s_root,s_base,k) for k in ['S','C','E','H']}
#             total,s,c,e,h=srm_scores(fc,fs)
#             rows.append(dict(content=c_base,style=s_base,IP=float(ip),
#                              total=float(total),S=s,C=c,E=e,H=h))
#     pd.DataFrame(rows).to_csv(a.out_csv,index=False)
#     print("✓ saved", a.out_csv, "rows=", len(rows))

# if __name__=="__main__":
#     ap=argparse.ArgumentParser()
#     ap.add_argument('--content_feat',required=True)
#     ap.add_argument('--style_feat',required=True)
#     ap.add_argument('--content_img',default='data/sae_data')
#     ap.add_argument('--style_img',default='data/style/picasso')
#     ap.add_argument('-k',type=int,default=5)
#     ap.add_argument('--out_csv',required=True)
#     ap.add_argument('--vis_dir',default=None)
#     a=ap.parse_args(); main(a)
#!/usr/bin/env python3
import cv2, faiss, argparse, glob, numpy as np, pandas as pd
from pathlib import Path
from features.srm_utils import ALPHA,BETA,GAMMA,DELTA,cos_sim,h_sim

def load_vec(root:Path, base:str, key:str):
    return np.load(root/f'{base}_{key}.npy').astype('float32')

def build_index(vec_dir:Path):
    vecs,names=[],[]
    for p in vec_dir.glob('*_S.npy'):
        names.append(p.stem[:-2]); vecs.append(np.load(p))
    vecs=np.vstack(vecs).astype('float32')
    faiss.normalize_L2(vecs)
    idx=faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs)
    return idx,names,vecs.mean(0)

def find_img(base:str, folder:Path):
    for ext in('.png','.jpg','.jpeg','.PNG','.JPG','.JPEG'):
        p=folder/f'{base}{ext}';  # <img_name>.ext
        if p.exists(): return cv2.imread(str(p))
    return None

def pad(v,n): return np.pad(v,(0,n-len(v)),'constant') if len(v)<n else v

def srm_score(fc,fs):
    s=cos_sim(fc['S'],fs['S']); c=cos_sim(fc['C'],fs['C'])
    d=max(len(fc['E']),len(fs['E']))
    e=cos_sim(pad(fc['E'],d), pad(fs['E'],d))
    h=h_sim(fc['H'],fs['H'])
    return ALPHA*s+BETA*c+GAMMA*e+DELTA*h, s,c,e,h

def save_grid(c_base, styles, c_dir, s_dir, out):
    img_c=find_img(c_base,c_dir)
    if img_c is None: print("[MISS] content", c_base); return
    panel=[cv2.resize(img_c[:,:,::-1],(256,256))]
    for s in styles:
        img_s=find_img(s,s_dir)
        if img_s is None: print("[MISS] style", s); return
        panel.append(cv2.resize(img_s[:,:,::-1],(256,256)))
    out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/f'{c_base}.png'),cv2.hconcat(panel)[:,:,::-1])

def main(a):
    c_root,s_root=Path(a.content_feat),Path(a.style_feat)
    idx,names,mu=build_index(s_root)
    rows=[]
    for p in c_root.glob('*_S.npy'):
        c_base=p.stem[:-2]
        v=load_vec(c_root,c_base,'S')-mu; v/=np.linalg.norm(v)
        D,I=idx.search(v[None],a.k); top=list(zip(D[0],[names[i] for i in I[0]]))
        if a.vis_dir: save_grid(c_base,[n for _,n in top],Path(a.content_img),Path(a.style_img),Path(a.vis_dir))
        for ip,s_base in top:
            fc={k:load_vec(c_root,c_base,k) for k in ['S','C','E','H']}
            fs={k:load_vec(s_root,s_base,k) for k in ['S','C','E','H']}
            total,s,c,e,h=srm_score(fc,fs)
            rows.append(dict(content=c_base,style=s_base,IP=float(ip),
                             total=float(total),S=s,C=c,E=e,H=h))
    pd.DataFrame(rows).to_csv(a.out_csv,index=False)
    print("✓ saved", a.out_csv, "rows=", len(rows))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--content_feat',required=True)
    ap.add_argument('--style_feat',required=True)
    ap.add_argument('--content_img',default='data/sae_data')
    ap.add_argument('--style_img',default='data/style/picasso')
    ap.add_argument('-k',type=int,default=5)
    ap.add_argument('--out_csv',required=True)
    ap.add_argument('--vis_dir',default=None)
    a=ap.parse_args(); main(a)

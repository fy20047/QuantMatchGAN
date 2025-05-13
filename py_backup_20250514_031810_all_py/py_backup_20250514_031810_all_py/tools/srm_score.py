#!/usr/bin/env python3
import argparse, json, numpy as np, csv, os, pathlib
from features.srm_utils import ALPHA,BETA,GAMMA,DELTA,cos_sim,h_sim

def load_vec(root, base, key):
    return np.load(f"{root}/{base}_{key}.npy").astype('float32')

ap=argparse.ArgumentParser()
ap.add_argument('--pair_json', required=True)
ap.add_argument('--style_root', required=True)
ap.add_argument('--content_root', required=True)
ap.add_argument('--alpha',type=float,default=ALPHA)
ap.add_argument('--beta', type=float,default=BETA)
ap.add_argument('--gamma',type=float,default=GAMMA)
ap.add_argument('--delta',type=float,default=DELTA)
ap.add_argument('--out_csv', required=True)
args=ap.parse_args()

pairs=json.load(open(args.pair_json))
rows=[]
for s_path, c_list in pairs.items():
    s_base=pathlib.Path(s_path).stem.replace('.png','').replace('.jpg','')
    fS={k:load_vec(args.style_root,s_base,k) for k in ['S','C','E','H']}
    for c_item in c_list: 
        # 支援三種情況： 
        # 1. "cont00042.png" 
        # 2. ["cont00042.png", 0.83] 
        # 3. [0.83, 42]      ← 遇到這種就 skip 
        if isinstance(c_item, str): 
            c_path = c_item 
        elif isinstance(c_item, list) and isinstance(c_item[0], str): 
            c_path = c_item[0] 
        else: 
            continue          # 純 float/ int → 無法解析，跳過 
        c_base = pathlib.Path(c_path).stem.split('.')[0]
        fC={k:load_vec(args.content_root,c_base,k) for k in ['S','C','E','H']}
        S=cos_sim(fC['S'],fS['S']); C=cos_sim(fC['C'],fS['C'])
        E=cos_sim(fC['E'],fS['E']); H=h_sim(fC['H'],fS['H'])
        total=args.alpha*S+args.beta*C+args.gamma*E+args.delta*H
        rows.append([s_base,c_base,total,S,C,E,H])
with open(args.out_csv,'w',newline='') as f:
    csv.writer(f).writerows([['style','content','total','S','C','E','H']]+rows)
print("✓ saved", args.out_csv)

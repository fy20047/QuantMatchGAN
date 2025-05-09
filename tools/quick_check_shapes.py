# quick_check_shapes.py
import os, glob, numpy as np, sys
root = sys.argv[1]
cnt = {}
for p in glob.glob(os.path.join(root,'*.npy')):
    shape = np.load(p).shape
    cnt.setdefault(shape, []).append(os.path.basename(p))
for s, files in cnt.items():
    print(f"{s}: {len(files)} files")
    if s != (2048,):
        print("  →", files[:5], "...")

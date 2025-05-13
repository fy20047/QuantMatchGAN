#!/usr/bin/env python3
"""
Sparse Auto -  Encoder (SAE) training script – *latest version*
-----------------------------------------------------------
• 支援兩種資料介面：
    1. 舊版 → --data <folder>    (單一路徑，全部圖片混合)
    2. 新版 → --natural / --style (分開路徑，可用 --ratio_style 控制比例)
• 特色：
    – k -  sparse 遮罩（--ksparse）
    – L1 粗線 warm -  up（--lambda_warm）
    – 全灰階 / 僅 style 灰階 / 不灰階（--gray_mode all|style|none）
"""

import os, glob, random, argparse, cv2, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 確保可以 import features.sae.model
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from features.sae.model import SAE

# —————————— Dataset ——————————
class MixedFolder256(Dataset):
    def __init__(self, nat_root, sty_root=None, ratio_style=0.3, gray_mode="all"):
        self.nat = sorted(glob.glob(os.path.join(nat_root, '*')))
        self.sty = sorted(glob.glob(os.path.join(sty_root, '*'))) if sty_root else []
        if len(self.nat)==0 and len(self.sty)==0:
            raise ValueError("natural / style 資料夾皆為空，無法訓練 SAE")
        self.r    = ratio_style
        self.gray_mode = gray_mode

        self.aug_affine = transforms.RandomAffine(
            degrees=25, translate=(.02,.02), scale=(.8,1.2), shear=15, fill=0)
        self.aug_persp  = transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
        self.hflip      = transforms.RandomHorizontalFlip(p=.5)

    def __len__(self):
        return max(len(self.nat), len(self.sty)) if self.sty else len(self.nat)

    def _read(self, path, is_style=False):
        img = cv2.imread(path)[:,:,::-1]
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        # 圖像增強
        img = self.hflip(torch.from_numpy(img.copy())).cpu().numpy()
        img = self.aug_affine(torch.from_numpy(img.copy())).numpy()
        img = self.aug_persp(torch.from_numpy(img.copy())).numpy()

        # 灰階處理
        if self.gray_mode == "all" or (self.gray_mode=="style" and is_style):
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([g,g,g], -1)
        img = torch.from_numpy(img.transpose(2,0,1)).float()/127.5 - 1
        return img

    def __getitem__(self, idx):
        use_style = self.sty and random.random() < self.r
        if use_style:
            path = random.choice(self.sty)
            return self._read(path, is_style=True)
        else:
            path = self.nat[idx % len(self.nat)]
            return self._read(path, is_style=False)

class Folder256(Dataset):
    def __init__(self, root, gray_mode="none"):
        self.paths = sorted(glob.glob(os.path.join(root, '*')))
        self.hflip = transforms.RandomHorizontalFlip(p=.5)
        self.gray_mode = gray_mode
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])[:,:,::-1]
        img = cv2.resize(img,(256,256), interpolation=cv2.INTER_AREA)
        img = self.hflip(torch.from_numpy(img.copy())).numpy()
        if self.gray_mode == "all":
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([g,g,g], -1)
        img = torch.from_numpy(img.transpose(2,0,1)).float()/127.5 - 1
        return img

# ——————— argparse ———————
ap = argparse.ArgumentParser()
ap.add_argument('--natural')
ap.add_argument('--style')
ap.add_argument('--ratio_style', type=float, default=0.3)
ap.add_argument('--gray_mode', choices=['all','style','none'], default='style')
ap.add_argument('--data')

ap.add_argument('--bottleneck', type=int, default=2048)
ap.add_argument('--epochs',     type=int, default=10)
ap.add_argument('--batch',      type=int, default=16)
ap.add_argument('--lr',         type=float, default=3e-5)
ap.add_argument('--lambda_warm',type=int, default=2)

ap.add_argument('--ksparse',    type=float, default=0.20)   # ↑ 0.10 → 0.20
ap.add_argument('--lambda_l1',  type=float, default=0.05)   # ↑ 0.03 → 0.05
ap.add_argument('--lambda_dc',  type=float, default=0.10)   # ← 新增 decorrelation 權重

ap.add_argument('--save',       default='features/sae/sae_sparse2048.pth')
args = ap.parse_args()

# ————— dataset —————
if args.data:
    dataset = Folder256(args.data, gray_mode=args.gray_mode)
else:
    if not args.natural:
        ap.error('--natural 不能為空')
    dataset = MixedFolder256(args.natural, args.style, args.ratio_style,
                             gray_mode=args.gray_mode)

dloader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                     num_workers=4, drop_last=True, pin_memory=True)

# ————— model —————
model = SAE(bottleneck=args.bottleneck).cuda()
opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

# 設定控制參數
k_percent = args.ksparse if args.ksparse>0 else None
warm_epochs = max(1, args.lambda_warm)

# 訓練
for epoch in range(1, args.epochs+1):
    pbar = tqdm(dloader, desc=f"[E{epoch}/{args.epochs}]")
    agg = {'mse':0,'l1':0,'dc':0,'sp':0,'sigma':0,'true_sp':0,'nz':0}
    for imgs in pbar:
        imgs = imgs.cuda()
        imgs64 = F.interpolate(imgs, 64, mode='bilinear', align_corners=False)

        # Encoder 生成系統向量 z
        z = model.encode(imgs)

        # k-sparse masking: 保留 top-k 經常活躍維度
        if k_percent:
            k = int(z.shape[1]*k_percent)
            # --- 隨機循環位移，讓每 batch 挑到不同維度 -------------
            # 每張圖各自隨機 shift，避免 batch 同步 
            shifts = torch.randint(0, z.size(1), (z.size(0),1), 
                                   device=z.device)              # B×1 
            idx = (torch.arange(z.size(1), device=z.device)[None] + shifts) % z.size(1) 
            z_roll = torch.gather(z, 1, idx)                     # 已 roll 
 
            topk  = z_roll.topk(k, dim=1).indices 
            mask  = torch.zeros_like(z_roll).scatter_(1, topk, 1.) 
            z_sparse = z_roll * mask                             # ★ 再也不 roll-back
        else:
            z_sparse = z

        # Decoder 重建圖像
        recon = model.decode(z_sparse)
        loss_mse = F.mse_loss(recon, imgs64)
        loss_l1  = z_sparse.abs().mean()
        lam = args.lambda_l1 * min(1.0, epoch / warm_epochs)
        # ---------- decorrelation loss ----------
        z_norm  = F.normalize(z_sparse, dim=1)
        corr    = torch.matmul(z_norm, z_norm.T)
        loss_dc = torch.triu(corr, diagonal=1).pow(2).mean()
        loss = loss_mse + lam * loss_l1 + args.lambda_dc * loss_dc

        opt.zero_grad(); loss.backward(); opt.step()

        # 精密統計
        sigma = torch.std(z_sparse, dim=1).mean().item()
        nonzero = (z_sparse.abs() > 1e-4).float().sum(dim=1)
        true_sp = (nonzero / z.shape[1]).mean().item()

        agg['mse']+=loss_mse.item()
        agg['l1'] +=loss_l1.item()
        # agg['sp'] +=sparsity
        agg['dc'] += loss_dc.item()
        agg['sigma']+=sigma
        agg['true_sp']+=true_sp
        agg['nz']+=nonzero.mean().item()

    n=len(dloader)
    print(f"L={agg['mse']/n:.4f}  k={args.ksparse:.2f}  true_sp={agg['true_sp']/n:.3f}  "
          f"nz={agg['nz']/n:.1f}  σ̄={agg['sigma']/n:.3f}  dc={agg['dc']/n:.4f}")

# 儲存模型
save_dir = os.path.dirname(args.save) or '.'
os.makedirs(save_dir, exist_ok=True)          # ★ 用 save_dir
torch.save(model.state_dict(), args.save)
print(f"✓ SAE saved → {args.save}")

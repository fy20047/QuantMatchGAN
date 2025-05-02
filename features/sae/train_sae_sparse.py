#!/usr/bin/env python3
"""
Sparse Auto‑Encoder (SAE) training script – *latest version*
-----------------------------------------------------------
• 支援兩種資料介面：
    1. 舊版 → --data <folder>    （單一路徑，全部圖片混合）
    2. 新版 → --natural / --style （分開路徑，可用 --ratio_style 控制比例）
• 特色：
    – k‑sparse 遮罩（--ksparse）
    – L1 稀疏 warm‑up （--lambda_warm）
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

# ───────────────────────── Dataset ──────────────────────────
class MixedFolder256(Dataset):
    def __init__(self, nat_root, sty_root=None, ratio_style=0.3, gray_mode="all"):
        """nat_root: 自然臉資料夾 (必填)
           sty_root: 形變臉資料夾 (可 None)  
           ratio_style: 每張取 style 的機率 (0~1)  
           gray_mode  : all / style / none  
        """
        self.nat = sorted(glob.glob(os.path.join(nat_root, '*')))
        self.sty = sorted(glob.glob(os.path.join(sty_root, '*'))) if sty_root else []
        if len(self.nat)==0 and len(self.sty)==0:
            raise ValueError("natural / style 資料夾皆為空，無法訓練 SAE")
        self.r    = ratio_style
        self.gray_mode = gray_mode

        self.aug_affine = transforms.RandomAffine(
            degrees=25, translate=(.02,.02), scale=(.8,1.2), shear=15, fill=0)
        self.aug_persp  = transforms.RandomPerspective(distortion_scale=0.4, p=0.5)
        self.hflip      = transforms.RandomHorizontalFlip(p=.5)

    def __len__(self):
        return max(len(self.nat), len(self.sty)) if self.sty else len(self.nat)

    def _read(self, path, is_style=False):
        img = cv2.imread(path)[:,:,::-1]   # BGR → RGB
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        # Augment
        img = self.hflip(torch.from_numpy(img.copy())).numpy()
        img = self.aug_affine(torch.from_numpy(img.copy())).numpy()
        img = self.aug_persp(torch.from_numpy(img.copy())).numpy()

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
    """舊版：單一路徑、不分自然/風格"""
    def __init__(self, root):
        self.paths = sorted(glob.glob(os.path.join(root, '*')))
        self.hflip = transforms.RandomHorizontalFlip(p=.5)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])[:,:,::-1]
        img = cv2.resize(img,(256,256), interpolation=cv2.INTER_AREA)
        img = self.hflip(torch.from_numpy(img.copy())).numpy()
        img = torch.from_numpy(img.transpose(2,0,1)).float()/127.5 - 1
        return img

# ───────────────────────── argparse ─────────────────────────
ap = argparse.ArgumentParser()
# 新版
ap.add_argument('--natural',      help='自然臉資料夾')
ap.add_argument('--style',        help='形變臉資料夾 (Picasso)')
ap.add_argument('--ratio_style', type=float, default=0.3,
                help='每 batch 使用 style 圖的機率')
ap.add_argument('--gray_mode',    choices=['all','style','none'], default='all')
# 舊版 (單一路徑)
ap.add_argument('--data', help='[Deprecated] 若仍想用單一路徑，填這個')

# 訓練超參
ap.add_argument('--bottleneck', type=int, default=2048)
ap.add_argument('--epochs',     type=int, default=10)
ap.add_argument('--batch',      type=int, default=16)
ap.add_argument('--lr',         type=float, default=3e-5)
ap.add_argument('--lambda_l1',  type=float, default=0.03)
ap.add_argument('--lambda_warm',type=int,   default=2,
                help='前幾 epoch 線性遞增 λ')
ap.add_argument('--ksparse',    type=float, default=0.03,
                help='保留活躍百分比 (0~1)，<=0 停用 k‑sparse')
ap.add_argument('--save',       default='features/sae/sae_sparse2048.pth')
args = ap.parse_args()

# ───────────────────────── dataset loader ───────────────────
if args.data:   # 舊版介面
    dataset = Folder256(args.data)
else:
    if not args.natural:
        ap.error('--natural 不能為空')
    dataset = MixedFolder256(args.natural, args.style, args.ratio_style,
                             gray_mode=args.gray_mode)

dloader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                     num_workers=4, drop_last=True, pin_memory=True)

# ───────────────────────── model / optim ───────────────────
model = SAE(bottleneck=args.bottleneck).cuda()
# ── monkey-patch encode / decode（若 class 裡本來就沒有） ──
if not hasattr(model, "encode"):
    def _encode(self, x):
        _, z = self.forward(x)     # self.forward 回傳 recon, z
        return z
    model.encode = _encode.__get__(model, SAE)   # 綁定到 instance

if not hasattr(model, "decode"):
    # 大多數實作都有 self.decoder，若叫 self.dec 請改這行
    def _decode(self, z):
        return self.decoder(z)
    model.decode = _decode.__get__(model, SAE)
# ────────────────────────────────────────────────────────────
opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

# ───────────────────────── training loop ───────────────────
k_percent = args.ksparse if args.ksparse>0 else None
warm_epochs = max(1, args.lambda_warm)

for epoch in range(1, args.epochs+1):
    pbar = tqdm(dloader, desc=f"[E{epoch}/{args.epochs}]")
    agg = {'mse':0,'l1':0,'sp':0,'sigma':0}
    for imgs in pbar:
        imgs = imgs.cuda()
        imgs64 = F.interpolate(imgs, 64, mode='bilinear', align_corners=False)

        # -- Encoder 取 z ---------------------------------------------------
        z = model.encode(imgs)                  # ← 只要一次

        # -- k-sparse -------------------------------------------------------
        if k_percent:
            k = int(z.shape[1] * k_percent)
            topk = torch.topk(z.abs(), k, dim=1).indices
            mask = torch.zeros_like(z).scatter_(1, topk, 1.)
            z_sparse = z * mask
            sparsity = 1 - k_percent
        else:
            z_sparse = z
            sparsity = (z.abs() < 5e-3).float().mean().item()

        # -- Decoder 用 z_sparse -------------------------------------------
        recon = model.decode(z_sparse)          # **只傳遮罩後的向量**
        loss_mse = F.mse_loss(recon, imgs64)
        loss_l1  = z_sparse.abs().mean()
        lam = args.lambda_l1 * min(1.0, epoch / warm_epochs)
        loss = loss_mse + lam * loss_l1

        opt.zero_grad(); loss.backward(); opt.step()

        sigma = torch.std(z_sparse.abs(), dim=1).mean().item()
        agg['mse']+=loss_mse.item(); agg['l1']+=loss_l1.item()
        agg['sp'] +=sparsity;        agg['sigma']+=sigma

    n=len(dloader)
    print(f"L={agg['mse']/n:.4f}  sparsity={agg['sp']/n:.3f}  σ̄={agg['sigma']/n:.3f}")

# ───────────────────────── save ─────────────────────────────
os.makedirs(os.path.dirname(args.save), exist_ok=True)
torch.save(model.state_dict(), args.save)
print(f"✓ SAE saved → {args.save}")


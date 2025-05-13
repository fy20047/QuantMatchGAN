# """Train Sparse Auto‑Encoder and save state_dict as .pth"""
# import argparse, pathlib, torch, torchvision.transforms as T
# from torch.utils.data import DataLoader
# from features.sae.model import SAE
# from PIL import Image
# from pathlib import Path
# import torch.nn.functional as F

# class ImgFolder(torch.utils.data.Dataset):
#     def __init__(self, root):
#         self.paths = [p for p in Path(root).iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'}]
#         self.tf = T.Compose([
#             T.Resize(256),
#             T.CenterCrop(256),
#             T.ToTensor(),
#         ])
#     def __len__(self):
#         return len(self.paths)
#     def __getitem__(self, idx):
#         img = Image.open(self.paths[idx]).convert('RGB')
#         return self.tf(img)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--img_dir', required=True)
#     ap.add_argument('--epochs', type=int, default=200)
#     ap.add_argument('--save_path', default='features/sae/sae.pth')
#     ap.add_argument('--lr', type=float, default=1e-4)
#     ap.add_argument('--log_csv', default=None, help='Path to save training loss as CSV')

#     args = ap.parse_args()

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"[INFO] Training on device: {device}")
#     net = SAE().to(device)
#     opt = torch.optim.Adam(net.parameters(), lr=args.lr)
#     mse = torch.nn.MSELoss()
#     loader = DataLoader(ImgFolder(args.img_dir), batch_size=8, shuffle=True)

#     for epoch in range(args.epochs):
#         for x in loader:
#             print(f"[DEBUG] x type: {type(x)}") 
#             x = x.to(device)
#             recon, z = net(x)
#             if recon.shape[-1] != x.shape[-1]:         # 尺寸不符時，將 x down‑sample
#                 x_ = F.interpolate(x, size=recon.shape[-2:],
#                                 mode='bilinear', align_corners=False)
#             else:
#                 x_ = x
#             loss = mse(recon, x_) + 1e-4 * z.abs().mean()
#             opt.zero_grad(); loss.backward(); opt.step()
#         if (epoch+1)%50==0:
#             print(f'Epoch {epoch+1}/{args.epochs}  L={loss.item():.4f}')

#     Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save(net.state_dict(), args.save_path)
#     print('[Done] SAE saved ->', args.save_path)
"""Train Sparse Auto‑Encoder and save state_dict as .pth"""
import argparse, csv, os, torch, torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

from features.sae.model import SAE

# ─────────────────── Dataset ───────────────────
class ImgFolder(torch.utils.data.Dataset):
    def __init__(self, root):
        self.paths = [p for p in Path(root).iterdir()
                      if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
        self.tf = T.Compose([
            T.Resize(256), T.CenterCrop(256), T.ToTensor()
        ])
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.tf(img)

# ──────────────────── Train ─────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--save_path', default='features/sae/sae.pth')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bottleneck', type=int, default=2048)
    ap.add_argument('--λ_sparse', type=float, default=1e-4)
    ap.add_argument('--perceptual', action='store_true')
    ap.add_argument('--λ_percep', type=float, default=0.1)
    ap.add_argument('--log_csv', default=None,
                    help='Path to loss.csv (epoch, avg_loss)')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] device = {dev}')
    net = SAE(bottleneck=args.bottleneck).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    loader = DataLoader(ImgFolder(args.img_dir), batch_size=8, shuffle=True)

    # optional perceptual loss
    if args.perceptual:
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(dev)
        for p in vgg.parameters(): p.requires_grad = False

    # CSV header
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        if not Path(args.log_csv).exists():
            with open(args.log_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'avg_loss'])

    for ep in range(1, args.epochs + 1):
        ep_loss, n = 0.0, 0
        for x in loader:
            x = x.to(dev)
            recon, z = net(x)
            if recon.shape[-1] != x.shape[-1]:
                x_ = F.interpolate(x, size=recon.shape[-2:],
                                   mode='bilinear', align_corners=False)
            else:
                x_ = x
            loss = mse(recon, x_) + args.λ_sparse * z.abs().mean()

            if args.perceptual:
                with torch.no_grad(): fx = vgg(x_)
                frec = vgg(recon)
                loss += args.λ_percep * F.l1_loss(frec, fx)

            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * x.size(0); n += x.size(0)

        avg = ep_loss / n
        if ep % 10 == 0:
            print(f'Epoch {ep:3d}/{args.epochs}  loss={avg:.4f}')

        # append to CSV
        if args.log_csv:
            with open(args.log_csv, 'a', newline='') as f:
                csv.writer(f).writerow([ep, avg])

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.save_path)
    print('[Done] SAE weight saved ->', args.save_path)

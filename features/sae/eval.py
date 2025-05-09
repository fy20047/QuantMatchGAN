"""
SAE 評估工具 (DEBUG 版)
產出：
  loss_curve.png   recon_grid.png
  sigma_hist.png   tsne.png (可選)
  info.json        (avg_loss, avg_sparsity, avg_sigma)
"""
import os, json, argparse, glob, numpy as np
import torch, torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
from features.sae.model import SAE

THRESH = 5e-3

# ---------- util ----------
class ImgFolder(torch.utils.data.Dataset):
    def __init__(self, root):
        self.paths = sorted(glob.glob(os.path.join(root, '*')))
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = plt.imread(self.paths[idx])[:, :, :3]          # H,W,3 0-1
        img = torch.from_numpy(img.transpose(2,0,1)).float() # 3,H,W
        img = img * 2 - 1                                    # 0-1 → -1~1
        img = F.interpolate(img.unsqueeze(0), 256,
                            mode='bilinear', align_corners=False)
        return img.squeeze()                                 # 3,256,256

def l2n(x): return x / (x.norm(dim=1, keepdim=True) + 1e-8)
def denorm(t): return (t + 1) * 0.5                         # -1~1→0~1

# ---------- main ----------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SAE(bottleneck=int(args.bottleneck)).to(dev)
    net.load_state_dict(torch.load(args.sae_weight, map_location=dev),
                        strict=False)
    net.eval()

    ds = ImgFolder(args.img_dir)
    dl = torch.utils.data.DataLoader(ds, batch_size=64,
                                     shuffle=False, num_workers=4)

    # -------- encode loop --------
    zs_list, sig_list = [], []
    for step, x in enumerate(tqdm(dl, desc='encode')):
        x = x.to(dev)
        with torch.no_grad():
            recon, z = net(x)                 # (recon , z)

        if step == 0:                         # DEBUG 一次
            print("DEBUG recon :", recon.shape)   # (B,3,64,64)
            print("DEBUG z     :", z.shape)       # (B, bottleneck)
        zs_list.append(z.cpu())
        sig_list.append(torch.std(z, dim=1).cpu())  # (B,)
    zs     = torch.cat(zs_list)               # (N, bottleneck)
    sigmas = torch.cat(sig_list)              # (N,)
    print("DEBUG zs size :", zs.shape)        # e.g. (300,2048)
    print("DEBUG σ size  :", sigmas.shape)    # (300,)

    # -------- sparsity / loss --------
    sparsity = (zs.abs() < THRESH).float().mean().item()

    n_samp = min(128, len(ds))
    idxs   = torch.randint(0, len(ds), (n_samp,))
    x_samp = torch.stack([ds[i] for i in idxs]).to(dev)
    with torch.no_grad():
        recon_s, _ = net(x_samp)
    x_cmp = F.interpolate(x_samp, recon_s.shape[-1],
                          mode='bilinear', align_corners=False)
    loss = F.mse_loss(recon_s, x_cmp).item()

    json.dump(dict(avg_loss=loss,
                   avg_sparsity=sparsity,
                   avg_sigma=float(sigmas.mean())),
              open(os.path.join(args.out_dir,'info.json'),'w'), indent=2)

    # -------- sigma hist --------
    plt.figure()
    sns.histplot(sigmas.numpy(), kde=args.kde, bins=40)
    plt.title('sigma(z_S)')
    plt.savefig(f'{args.out_dir}/sigma_hist.png'); plt.close()

    # -------- recon grid --------
    # 取最後一個 batch 的 recon 做展示
    recon_v = F.interpolate(recon, x.shape[-1],
                            mode='bilinear', align_corners=False)
    grid = vutils.make_grid(torch.cat([denorm(x).cpu(),
                                       denorm(recon_v).cpu()], 0),
                            nrow=8)
    vutils.save_image(grid, f'{args.out_dir}/recon_grid.png',
                      normalize=False)

    # -------- t-SNE --------
    if args.tsne:
        z_emb = TSNE(n_components=2, init='pca',
                     perplexity=30).fit_transform(l2n(zs).numpy())
        plt.figure(figsize=(5,5))
        sc=plt.scatter(z_emb[:,0], z_emb[:,1],
                       c=sigmas.numpy(), cmap='viridis', s=4)
        plt.colorbar(sc,label='σ'); plt.axis('off')
        plt.title('t-SNE of z_S')
        plt.savefig(f'{args.out_dir}/tsne.png'); plt.close()

    print('✓ report saved to', args.out_dir)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir',   required=True, help='風格資料夾')
    ap.add_argument('--sae_weight',required=True, help='SAE 權重 .pth')
    ap.add_argument('--bottleneck', default=2048, help='SAE 瓶頸維度 512 / 2048')
    ap.add_argument('--out_dir',   required=True, help='報表輸出資料夾')
    ap.add_argument('--kde',  action='store_true', help='sigma_hist 加 KDE')
    ap.add_argument('--tsne', action='store_true', help='輸出 t-SNE')
    args = ap.parse_args()
    main(args)

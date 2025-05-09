# """
# Evaluate SAE training quality:
# 1) loss curve 2) sparsity ratio 3) σ(zS) 分布
# 4) recon vs input grid 5) t-SNE of zS
# """
# import torch, torchvision.utils as vutils, matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from features.sae.model import SAE
# from features.sae.train import ImgFolder           # 共用資料集類別
# import seaborn as sns, numpy as np, os, json
# import torch.nn.functional as F

# def load_sae(weight, device='cpu'):
#     net = SAE().to(device)
#     net.load_state_dict(torch.load(weight, map_location=device))
#     net.eval()
#     return net

# def recon_grid(net, loader, device, save):
#     x = next(iter(loader)).to(device)[:8]  # 取 8 張
#     with torch.no_grad():
#         recon, _ = net(x)
#         if recon.shape[-1] != x.shape[-1]:  # 若解析度不一致
#             recon = F.interpolate(recon, size=x.shape[-2:], mode='bilinear', align_corners=False)

#     grid = vutils.make_grid(
#         torch.cat([x, recon], 0), nrow=8, normalize=True, scale_each=True)
#     vutils.save_image(grid, save)

# def sparsity_stats(net, loader, device):
#     zs, sparsity = [], []
#     with torch.no_grad():
#         for x in loader:
#             z = net.encode(x.to(device))
#             zs.append(z.cpu())
#             sparsity.append((z.abs() < 1e-3).float().mean().item())
#     z_all = torch.cat(zs, 0)
#     sigma = z_all.abs().std(dim=1)       # σ(zS)
#     return np.mean(sparsity), sigma.numpy()

# if __name__ == '__main__':
#     import argparse, pathlib
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--img_dir', required=True)
#     ap.add_argument('--sae_weight', required=True)
#     ap.add_argument('--out_dir', default='features/sae/report')
#     args = ap.parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     net = load_sae(args.sae_weight, device)
#     loader = DataLoader(ImgFolder(args.img_dir), batch_size=16)

#     # 1) Reconstruct grid
#     recon_grid(net, loader, device, f'{args.out_dir}/recon_grid.png')

#     # 2) Sparsity & σ
#     sparsity, sigma = sparsity_stats(net, loader, device)
#     np.save(f'{args.out_dir}/sigma.npy', sigma)
#     with open(f'{args.out_dir}/stats.json', 'w') as f:
#         json.dump({'sparsity': sparsity,
#                    'sigma_mean': float(np.mean(sigma)),
#                    'sigma_std': float(np.std(sigma))}, f)

#     # 3) σ 分布直方圖
#     plt.figure(); sns.histplot(sigma, kde=False, bins=30)
#     plt.xlabel(r'$\sigma(z_S)$'); plt.savefig(f'{args.out_dir}/sigma_hist.png')
#     print('[Done] report saved to', args.out_dir)

"""
Evaluate SAE training quality
Outputs:
  1) loss_curve.png
  2) recon_grid.png
  3) sigma_hist.png (+kde)
  4) tsne.png       (optional)
  5) stats.json     {sparsity, σ_mean, σ_std, q1, q3}
"""
import argparse, os, json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import torch, torch.nn.functional as F, torchvision.utils as vutils
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import seaborn as sns
from features.sae.model import SAE
from features.sae.train import ImgFolder   # 共用資料集類別

# ---------- helper ----------
def load_model(weight, device):
    net = SAE().to(device)
    net.load_state_dict(torch.load(weight, map_location=device))
    net.eval(); return net

def plot_loss(csv, out):
    if not csv or not Path(csv).exists(): return
    epochs, loss = np.loadtxt(csv, delimiter=',', unpack=True)
    plt.figure(); plt.plot(epochs, loss); plt.xlabel('epoch'); plt.ylabel('loss')
    plt.savefig(out); plt.close()

def recon_grid(net, loader, device, out):
    x = next(iter(loader)).to(device)[:8]
    with torch.no_grad():
        rec, _ = net(x)
        if rec.shape[-1] != x.shape[-1]:
            rec = F.interpolate(rec, size=x.shape[-2:], mode='bilinear', align_corners=False)
    grid = vutils.make_grid(torch.cat([x, rec], 0),
                            nrow=8, normalize=True, scale_each=True)
    vutils.save_image(grid, out)

def sparsity_sigma(net, loader, device):
    zs, spars = [], []
    with torch.no_grad():
        for x in loader:
            z = net.encode(x.to(device))
            zs.append(z.cpu())
            spars.append((z.abs() < 1e-3).float().mean().item())
    z_all = torch.cat(zs, 0)
    sigma = z_all.abs().std(dim=1)
    return np.mean(spars), sigma.numpy(), z_all.numpy()

# ---------- main ----------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--sae_weight', required=True)
    ap.add_argument('--out_dir', default='features/sae/report/loss2048/')
    ap.add_argument('--log_csv', default=None, help='loss.csv path (epoch,loss)')
    ap.add_argument('--kde', action='store_true', help='add KDE curve')
    ap.add_argument('--tsne', action='store_true', help='plot t-SNE')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(args.sae_weight, dev)
    loader = DataLoader(ImgFolder(args.img_dir), batch_size=16)

    # 1) Loss curve
    plot_loss(args.log_csv, f'{args.out_dir}/loss_curve.png')

    # 2) Recon grid
    recon_grid(net, loader, dev, f'{args.out_dir}/recon_grid.png')

    # 3) Sparsity & σ
    sparsity, sigma, z_all = sparsity_sigma(net, loader, dev)
    stats = {
        'sparsity': float(sparsity),
        'sigma_mean': float(np.mean(sigma)),
        'sigma_std': float(np.std(sigma)),
        'q1': float(np.percentile(sigma,25)),
        'q3': float(np.percentile(sigma,75))
    }
    with open(f'{args.out_dir}/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    np.save(f'{args.out_dir}/sigma.npy', sigma)

    plt.figure()
    sns.histplot(sigma, kde=args.kde, bins=30); plt.xlabel(r'$\sigma(z_S)$')
    for t in [0.08,0.20,0.25]:
        plt.axvline(t, ls='--')
    plt.savefig(f'{args.out_dir}/sigma_hist.png'); plt.close()

    # 4) t‑SNE (optional)
    if args.tsne:
        tsne = TSNE(n_components=2, perplexity=30, metric='cosine', init='pca', random_state=0)
        emb = tsne.fit_transform(z_all)
        plt.figure(figsize=(5,5))
        sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=sigma, palette='viridis', s=12, legend=False)
        plt.title('t‑SNE of zS (color = σ)')
        plt.savefig(f'{args.out_dir}/tsne.png'); plt.close()

    print('[Done] report saved to', args.out_dir)

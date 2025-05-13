# 功能：
# 從指定資料夾隨機挑 N 張圖
# 用訓好的 SAE Encoder+Decoder 還原 256² → 64² 影像
# 拼接 GT vs Recon，存 PNG，目視檢查「是否只還原幾何，忽略色彩」

# 評估：
# 幾何輪廓應大致對齊，顏色/紋理被模糊，代表 SAE 專注結構。
# 若重建看起來完全模糊不成形 → 可能 λ_L1 太大或 bottleneck 太小。
# 若顏色被完美復原 → SAE 仍在記顏色特徵，需提高灰階比例或 λ。

#!/usr/bin/env python3
import os, glob, random, argparse, cv2, torch
import numpy as np
from torchvision.utils import make_grid, save_image
from features.sae.model import SAE

def preprocess(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img,(256,256))
    t   = torch.from_numpy(img.transpose(2,0,1)).float()/127.5 - 1
    return t

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', required=True)
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--bottleneck', type=int, default=2048)
    ap.add_argument('--n', type=int, default=4)
    args = ap.parse_args()

    paths = random.sample(glob.glob(os.path.join(args.img_dir,'*')), args.n)
    imgs  = torch.stack([preprocess(p) for p in paths]).cuda()

    model = SAE(bottleneck=args.bottleneck).cuda().eval()
    model.load_state_dict(torch.load(args.sae, map_location='cuda'))
    with torch.no_grad():
        recon, _ = model(imgs)
    # 升回 256×256 方便對照
    recon_up = torch.nn.functional.interpolate(recon, size=256, mode='bilinear', align_corners=False)
    grid = make_grid(torch.cat([imgs, recon_up], 0), nrow=args.n, normalize=True, value_range=(-1,1))
    save_image(grid, 'recon_cmp.png')
    print('✓ recon_cmp.png saved')

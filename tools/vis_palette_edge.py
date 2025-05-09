import numpy as np, cv2, os, argparse, matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def show_palette(c):
    # c.shape = (12,)  → 四主色 * Lab
    colors = c.reshape(4,3)
    rgb = cv2.cvtColor(colors[np.newaxis,:,:], cv2.COLOR_Lab2RGB)[0]
    plt.figure(figsize=(2,1))
    plt.imshow(rgb[np.newaxis,:,:]); plt.axis('off')

def main(img_path, feat_root):
    base = os.path.splitext(os.path.basename(img_path))[0]
    c = np.load(f'{feat_root}/{base}_C.npy')
    h = float(np.load(f'{feat_root}/{base}_H.npy'))
    img = cv2.imread(img_path)[:,:,::-1]

    plt.subplot(1,3,1); plt.imshow(img); plt.title('Image'); plt.axis('off')
    plt.subplot(1,3,2); show_palette(c); plt.title('Palette')
    plt.subplot(1,3,3); plt.bar([0], [h]); plt.ylim(0,4); plt.title('EdgeEntropy')
    plt.savefig('demo_vis.png', dpi=300); print('→ demo_vis.png saved')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True)
    ap.add_argument('--feat_root', required=True)
    main(**vars(ap.parse_args()))

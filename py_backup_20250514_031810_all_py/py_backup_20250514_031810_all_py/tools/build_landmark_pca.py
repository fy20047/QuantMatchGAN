# python tools/build_landmark_pca.py \
#   --dir data/sae_data        # 或把 style/ 與 content/都丟進同一資料夾

# tools/build_landmark_pca.py
import glob, cv2, mediapipe as mp, numpy as np, joblib, argparse, pathlib

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def lm468(img):
    res = mp_face.process(img)
    if not res.multi_face_landmarks:
        return None
    pts = [(pt.x, pt.y, pt.z) for pt in res.multi_face_landmarks[0].landmark]
    return np.array(pts).flatten()          # 468×3 → 1404

def main(dir_in, out_pkl):
    mats = []
    for p in glob.glob(f"{dir_in}/*.png") + glob.glob(f"{dir_in}/*.jpg"):
        img = cv2.imread(p)[:,:,::-1]
        v = lm468(img)
        if v is not None:
            mats.append(v)
    X = np.stack(mats)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=16).fit(X)
    pathlib.Path(out_pkl).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(pca, out_pkl)
    print("✓ PCA saved →", out_pkl)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='輸入圖資料夾(風格+內容混合)')
    ap.add_argument('--out', default='features/pca/landmark16.pkl')
    args = ap.parse_args(); main(args.dir, args.out)

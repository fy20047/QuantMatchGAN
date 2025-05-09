# features/landmark_pca.py
import mediapipe as mp, numpy as np, joblib, os
import joblib, os

PCA_PATH = os.path.join(os.path.dirname(__file__), 'pca/landmark16.pkl')
pca_model = joblib.load(PCA_PATH)

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def extract_pca2(img_rgb, pca_model):
    res = mp_face.process(img_rgb)
    if not res.multi_face_landmarks: return np.zeros(2)   # fallback
    lm = np.array([(pt.x, pt.y, pt.z) for pt in res.multi_face_landmarks[0].landmark]).flatten()
    return pca_model.transform(lm[None,:])[0,:2]          # 2 維

# ☑ 訓練 PCA：對 300 張 style + 200 張 content 抽 Landmark → sklearn.PCA(n_components=16)
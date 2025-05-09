"""
features/expression.py
────────────────────────────────────────────────────────
extract_au(img) → np.ndarray shape=(30,)
‖ 0 : eye-open   (0~1)
‖ 1 : mouth-open (0~1)
‖ 2 : yaw   (-1~1 → 映射 0~1)
‖ 3 : pitch (-1~1 → 映射 0~1)
‖ 4–29 皆為 0（保留佔位）
"""
import cv2, numpy as np, mediapipe as mp

# ── Mediapipe FaceMesh ───────────────────────────────
_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, refine_landmarks=False)

# landmark index（FaceMesh 468 點編號）
IDX = {
    'eye_top_L': 159, 'eye_bot_L': 145,
    'eye_top_R': 386, 'eye_bot_R': 374,
    'mouth_top':  13, 'mouth_bot':  14,
    'mouth_mid':  17,              # ← 新增：下唇中點
    'eye_outer_L': 33, 'eye_outer_R': 263,
    'nose_tip': 1, 'chin': 199,
    
}

# SolvePnP 6 個點（新版 OpenCV 最低要求）
PNP_ID  = [IDX['nose_tip'], IDX['chin'],
           IDX['eye_outer_L'], IDX['eye_outer_R'],
           IDX['mouth_top'], IDX['mouth_mid']]  

MODEL_3D = np.array([
    [ 0.0,   0.0,  0.0],   # nose tip
    [ 0.0, -63.6, -12.5],  # chin
    [-43.3, 32.7, -26],    # left eye
    [ 43.3, 32.7, -26],    # right eye
    [ 0.0, -28.9, -24.1],  # upper lip
    [ 0.0, -39.5, -24.1],   # lower lip (下唇，向下平移 10.6 mm)
], dtype=np.float32)

# ── 主函式 ────────────────────────────────────────────
def extract_au(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    res = _face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return np.zeros(30, dtype=np.float32)

    lm = res.multi_face_landmarks[0].landmark
    def pt(i):                       # landmark → (x,y) 像素座標
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

    # 1️⃣ 眼睛張開度（左右平均 / 眼寬正規化）
    eye_h = (np.linalg.norm(pt(IDX['eye_top_L']) - pt(IDX['eye_bot_L'])) +
             np.linalg.norm(pt(IDX['eye_top_R']) - pt(IDX['eye_bot_R']))) / 2
    eye_w = np.linalg.norm(pt(IDX['eye_outer_L']) - pt(IDX['eye_outer_R']))
    eye_open = np.clip(eye_h / (eye_w + 1e-6), 0, 1)

    # 2️⃣ 嘴巴張開度（上下唇距 / 唇寬）
    mouth_h = np.linalg.norm(pt(IDX['mouth_top']) - pt(IDX['mouth_bot']))
    lip_w   = eye_w * 0.9  # 眼寬近似唇寬，避免額外點
    mouth_open = np.clip(mouth_h / (lip_w + 1e-6), 0, 1)

    # 3️⃣-4️⃣ Yaw / Pitch（SolvePnP，失敗則 0.5）
    img_pts = np.array([pt(i) for i in PNP_ID])
    cam_mtx = np.array([[w, 0, w/2],
                        [0, w, h/2],
                        [0, 0,   1]], dtype=np.float32)
    try:
        ok, rvec, _ = cv2.solvePnP(MODEL_3D, img_pts, cam_mtx, None,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False
        
    if ok:
        rot, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot[0, 0]**2 + rot[1, 0]**2)
        yaw   = np.arctan2(rot[2, 1], rot[2, 2]) / np.pi   # -1~1
        pitch = np.arctan2(-rot[2, 0], sy) / np.pi         # -1~1
        yaw_n   = (yaw   + 1) / 2                          # 映射 0~1
        pitch_n = (pitch + 1) / 2
    else:
        yaw_n = pitch_n = 0.5

    # 組合向量
    E = np.zeros(30, dtype=np.float32)
    E[0], E[1], E[2], E[3] = eye_open, mouth_open, yaw_n, pitch_n
    return E

# ========= 別名，供 extract_all.py 使用 =========
extract_expr = extract_au

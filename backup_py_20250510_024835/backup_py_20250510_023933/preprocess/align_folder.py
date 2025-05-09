import os
import PIL
from tqdm import tqdm
import dlib
import numpy as np
import scipy
import scipy.ndimage
from PIL import Image

# 引入 FFHQ 對齊方法
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    raise FileNotFoundError(f"找不到 {SHAPE_PREDICTOR_PATH}，請先下載！")

def get_landmark(filepath, predictor):
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
    t = list(shape.parts())
    lm = np.array([[pt.x, pt.y] for pt in t])
    return lm

def align_face(filepath, predictor, output_size=256):
    lm = get_landmark(filepath, predictor)
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = Image.open(filepath)
    transform_size = output_size
    enable_padding = True

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(img.size[0] / shrink)), int(np.rint(img.size[1] / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img

if __name__ == "__main__":
    input_dir = '/home/SlipperY/Pica/Picasso_GAN_Project/data/content'
    output_dir = '/home/SlipperY/Pica/Picasso_GAN_Project/data/processed/content'

    os.makedirs(output_dir, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png'}
    filenames = [f for f in os.listdir(input_dir) if os.path.splitext(f)[-1].lower() in exts]

    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    print(f"共找到 {len(filenames)} 張圖片，開始處理...")

    for fname in tqdm(filenames):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            aligned_img = align_face(input_path, predictor)
            aligned_img.save(output_path)
        except Exception as e:
            print(f"❌ 錯誤處理 {input_path}: {e}")

    print(f"✅ 全部處理完畢！輸出到：{output_dir}")

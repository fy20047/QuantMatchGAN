Picasso_GAN_Project/
├── env/                  # conda & python 版本管理
│   └── environment.yml
├── data/
│   ├── content/          # 使用者輸入臉部圖像 (1024x1024)
│   └── style/            # 300 張 Picasso 風格圖 (1024x1024)
├── preprocess/
│   ├── align_face.py     # 臉部對齊 (dlib/mediapipe)
│   ├── white_balance.py  # 可選白平衡 (--wb flag)
│   └── run.py            # 一鍵執行前處理
├── features/
│   ├── sae/
│   │   ├── model.py      # SAE Encoder/Decoder 架構
│   │   └── train.py      # 訓練 SAE
│   ├── edge_entropy.py   # 計算 H
│   ├── color_hist.py     # Lab 直方圖 & JS
│   ├── palette_emd.py    # Palette‑EMD (可選)
│   ├── expression.py     # AU + landmark 萃取
│   └── extract_all.py    # 產出 (S,C,E,H,JS,σ)
├── srm/
│   ├── build_index.py    # 建立 Faiss 索引
│   ├── match.py          # SRM 計算 & Top‑k 檢索
│   └── params.yaml       # αβγ、k、閾值設定
├── dualstylegan/
│   ├── configs/          # 官方訓練 yaml (已改路徑)
│   ├── inference.py      # 加 w 權重、intrinsic/extrinsic z+
│   └── weights/          # fine‑tuned checkpoint
├── eval/
│   ├── pi.py             # PI 計算 (H,JS,σ)
│   ├── fid.py            # FID_Picasso 包裝器
│   ├── survey.xlsx       # 問卷模板
│   └── run_eval.py       # 統計檢定 & p 值
└── README.md             # 專案說明

執行流程

# 0. 啟動 conda
conda activate picasso_gan

# 1. 前處理：對齊 + (可選) 白平衡
python preprocess/run.py \
       --style_dir data/style/picasso \
       --content_dir data/content \
       --out_dir data/processed \
       --align on \
       --wb on

# 2. (首次) 訓練 SAE
python features/sae/train.py \
       --img_dir data/processed/style \
       --epochs 200 \
       --save_path features/sae/sae.pth

# 3. 批次特徵萃取
python features/extract_all.py \
       --dir data/processed/style \
       --out features/style_feats.npy
python features/extract_all.py \
       --dir data/processed/content \
       --out features/content_feats.npy

# 4. 自動配對 + 生成
python srm/build_index.py --params srm/params.yaml       # 建索引 (一次)
python srm/match.py --idx 0 -k 3 > latents/ex_z.json      # 為第 0 張內容圖配對
python gan/generate.py --latent_json latents/ex_z.json \
       --out_dir output/gen

# 5. 評估
python eval/pi_gain.py --gen output/gen/pi.npy --content features/pi_content.npy
python eval/fid_pi.py  --gen_dir output/gen --pi_file output/gen/pi_stats.json


<!-- 1. 資料前處理 (可關閉白平衡)
python preprocess/run.py --input_dir data --wb off

2. 訓練 SAE（一次性）
python features/sae/train.py --img_dir data/style

3. 特徵萃取
python features/extract_all.py --content data/content --style data/style

4. 建立索引
python srm/build_index.py --style_feats features/style.npy

5. SRM 配對 & 生成
python srm/match.py --content_feats features/content.npy --k 3

6. 評估
python eval/run_eval.py --gen_dir results --content_dir data/content --style_dir data/style -->


研究亮點

SRM 自動配對：結構/色彩/表情三向量最近鄰，αβγ 可於 srm/params.yaml 調整。

PI 指數：eval/pi.py 內定 0.4/0.35/0.25 權重，可用 hyperopt 重新尋優。

可重現：所有隨機種子、路徑、超參數皆在 YAML 檔集中管理。

📄 詳細方法學、指標公式與實驗結果，請參閱 README.md 末的「論文整合說明」章節。
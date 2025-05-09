# 不用 ydata-profiling 的「全體可視化」版本
# 以下腳本一次輸出三張 PNG，檔案小、適合論文：

#!/usr/bin/env python3
# tools/quick_overview.py
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np, argparse

ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); args=ap.parse_args()
df=pd.read_csv(args.csv)

# 1. σ vs H scatter
plt.figure(figsize=(12,3))
plt.subplot(1,3,1)
sns.scatterplot(data=df, x='sigma', y='H'); plt.title('σ vs H')

# 2. AU heatmap (均值)
au_cols=[c for c in df.columns if c.startswith('AU')]
plt.subplot(1,3,2)
sns.barplot(x=au_cols, y=df[au_cols].mean()); plt.xticks(rotation=45,ha='right')
plt.title('平均 AU 強度')

# 3. Correlation heatmap (精選欄)
pick=['sigma','H']+au_cols[:4]+['C1','C2','C3']
corr=df[pick].corr()
plt.subplot(1,3,3)
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Pearson r (精選)')

plt.tight_layout(); plt.savefig('feature_overview.png',dpi=200)
print('✓ feature_overview.png 產生完成')

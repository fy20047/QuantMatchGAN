#!/usr/bin/env python3

# python tools/gen_feature_report.py --csv features.csv
# xdg-open feature_dashboard.html   # Linux，macOS 用 open，Windows 直接雙擊


"""
讀 features.csv → ydata-profiling HTML 報告
每列顯示統計、相關性矩陣、可篩選資料表。

python tools/gen_feature_report.py --csv features.csv
xdg-open feature_dashboard.html

"""
#!/usr/bin/env python3
"""
讀 features.csv → ydata-profiling HTML 報告
每列顯示統計、相關性矩陣、可篩選資料表。
"""
import argparse, pandas as pd
from ydata_profiling import ProfileReport

ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True)
args = ap.parse_args()

df = pd.read_csv(args.csv)
profile = ProfileReport(df,
                        title="PicassoGAN Feature Overview",
                        explorative=True,
                        correlations={"pearson": {"calculate": True}})
profile.to_file("feature_dashboard.html")
print("✓ feature_dashboard.html 已生成（用瀏覽器打開）")




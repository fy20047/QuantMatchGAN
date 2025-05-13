#!/bin/bash

# 獲取時間戳
timestamp=$(date +"%Y%m%d_%H%M%S")
new_dir="backup_py_$timestamp"

# 創建備份目錄
mkdir "$new_dir"

# 複製所有 .py 文件，不包含之前的備份資料夾
find . -type f -name "*.py" -not -path "./backup_py_*/*" -exec cp --parents {} "$new_dir" \;

echo "所有 .py 文件已複製到 $new_dir"

# 切換到專案目錄
cd ~/Pica/Picasso_GAN_Project

# 添加並推送到 GitHub
git add "$new_dir"
git commit -m "Added .py files backup with timestamp $timestamp"
git push origin main

echo "備份完成：$new_dir"

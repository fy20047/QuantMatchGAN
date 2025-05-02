#!/bin/bash
cd "$(dirname "$0")"

timestamp=$(date '+%m%d')

# 加入所有變動
git add .

# 提交 commit，加上時間戳
git commit -m "📦 Backup on $timestamp"

# 推送到 GitHub
git push origin main

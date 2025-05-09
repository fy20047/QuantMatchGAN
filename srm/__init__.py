# srm/__init__.py  ——  確保整個專案可被任何子模組 import
import sys, pathlib

# 專案根路徑 = 目前檔案的父層再上一層
_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))   # ★ 關鍵：把根目錄塞進 sys.path

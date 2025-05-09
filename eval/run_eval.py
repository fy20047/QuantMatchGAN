"""統計檢定：比較兩組指標，輸出 t 值與 p 值"""
import scipy.stats as st, argparse, numpy as np

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_a', required=True)
    ap.add_argument('--file_b', required=True)
    args = ap.parse_args()
    a = np.loadtxt(args.file_a)
    b = np.loadtxt(args.file_b)
    t, p = st.ttest_ind(a, b, equal_var=False)
    print(f"t = {t:.3f},  p = {p:.5f}")
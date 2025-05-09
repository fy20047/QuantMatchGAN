"""整合計算 FID-Picasso 以及 PI 統計摘要"""
import argparse, subprocess, pathlib, json
from pathlib import Path

def run_fid(real_dir, gen_dir):
    cmd = [
        'torch-fid', str(real_dir), str(gen_dir)
    ]
    return float(subprocess.check_output(cmd))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real_dir', default='data/style')
    ap.add_argument('--gen_dir', required=True)
    ap.add_argument('--pi_file', required=True)
    args = ap.parse_args()
    fid = run_fid(args.real_dir, args.gen_dir)
    pi = json.loads(Path(args.pi_file).read_text())
    print(json.dumps({'FID_Picasso': fid, **pi}, indent=2))
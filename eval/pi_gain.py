"""計算 PI-Gain (PI_gen - PI_content)"""
import numpy as np, argparse, pathlib
from pathlib import Path

def load_pi(path):
    return np.load(path)

def compute_gain(gen_pi, content_pi):
    return (gen_pi - content_pi).mean()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', required=True)
    ap.add_argument('--content', required=True)
    args = ap.parse_args()
    gain = compute_gain(load_pi(args.gen), load_pi(args.content))
    print(f'PI-Gain: {gain:.3f}')
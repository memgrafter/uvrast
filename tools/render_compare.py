"""Compare two render folders view by view (PSNR, identical pixels) and two score.json files.

  python tools/render_compare.py DIR_A DIR_B [--score A/score.json B/score.json]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--score", nargs=2)
    x = ap.parse_args()
    names = sorted(n for n in os.listdir(x.a) if n.endswith(".png"))
    rows = []
    for n in names:
        a = np.asarray(Image.open(os.path.join(x.a, n)).convert("RGBA"))
        b = np.asarray(Image.open(os.path.join(x.b, n)).convert("RGBA"))
        rows.append(dict(view=n, psnr=psnr(a, b), identical=bool(np.array_equal(a, b)),
                         pixels_differ=int((a != b).any(-1).sum()), max_diff=int(np.abs(a.astype(int) - b.astype(int)).max())))
    out = dict(views=len(rows), identical=sum(r["identical"] for r in rows), min_psnr=min(r["psnr"] for r in rows),
               max_pixels_differ=max(r["pixels_differ"] for r in rows), rows=rows)
    if x.score:
        sa, sb = (json.load(open(p)) for p in x.score)
        out["score_a"], out["score_b"] = sa, sb
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, default=str))
    return out


if __name__ == "__main__":
    main()

"""Compare uvrast with recorded black-box probe outputs (from probe_nvdiffrast.py). Runs anywhere (CPU is fine).

  python tools/check_probe.py PROBE_DIR [case ...]

Per case: coverage and triangle-id mismatches (and how many lie exactly on an edge), barycentric and interpolation
bit equality, and the first mismatches with their geometry, to find the rule that differs.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uvrast import raster, rules  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def check_case(path, show=5):
    d = np.load(path)
    T = int(d["T"]); pos = torch.from_numpy(d["pos"]).to(DEV); tri = torch.from_numpy(d["faces"]).to(DEV)
    want = np.zeros((T * T, 4), np.float32); want[d["cov_idx"]] = d["rast_cov"]
    # the probe used to_glb's chunk loop: later chunks overwrite earlier ones
    got = torch.zeros(T * T, 4, device=DEV)
    for i in range(0, tri.shape[0], 100000):
        r = raster.rasterize(pos, tri[i:i + 100000], T, T).view(-1, 4)
        m = r[:, 3] > 0
        r[:, 3] += i
        got = torch.where(m[:, None], r, got)
    got = got.cpu().numpy()
    wi, gi = want[:, 3].astype(np.int64), got[:, 3].astype(np.int64)
    bad = np.nonzero(wi != gi)[0]
    same = (wi == gi) & (wi > 0)
    out = dict(case=os.path.basename(path), T=T, faces=len(tri), covered=int((wi > 0).sum()), id_or_cover_mismatch=len(bad),
               bary_bit_equal=f"{int((want[same, :2] == got[same, :2]).all(1).sum())}/{int(same.sum())}",
               bary_max_abs=float(np.abs(want[same, :2] - got[same, :2]).max()) if same.any() else 0.0,
               deterministic=bool(d["deterministic"]))
    X = rules.to_fixed(pos[:, 0], T).cpu(); Y = rules.to_fixed(pos[:, 1], T).cpu()
    f = tri.long().cpu()
    for k in bad[:show]:
        r, c = divmod(int(k), T)
        lines = [f"  texel r{r} c{c}: nvdiffrast id {wi[k] - 1}, uvrast id {gi[k] - 1}"]
        for fid in {wi[k] - 1, gi[k] - 1} - {-1}:
            t = f[fid]
            xs, ys = X[t] / rules.SUB - c - 0.5, Y[t] / rules.SUB - r - 0.5
            lines.append(f"    tri {fid}: verts rel. to centre (px) {[(round(float(x), 5), round(float(y), 5)) for x, y in zip(xs, ys)]}"
                         f" raw x {pos[t, 0].tolist()} y {pos[t, 1].tolist()}")
        out.setdefault("examples", []).append("\n".join(lines))
    if "interp_cov" in d:
        attr = torch.from_numpy(d["attr"]).to(DEV)
        ip = raster.interpolate(attr, torch.from_numpy(want.reshape(T, T, 4)).to(DEV), tri).view(-1, attr.shape[1]).cpu().numpy()
        wantip = d["interp_cov"]
        out["interp_bit_equal"] = f"{int((ip[d['cov_idx']] == wantip).all(1).sum())}/{len(wantip)}"
    return out


def check_interp(path):
    d = np.load(path)
    rast = torch.from_numpy(d["rast"][0]).to(DEV)
    got = raster.interpolate(torch.from_numpy(d["attr"][0]).to(DEV), rast, torch.from_numpy(d["tri"]).to(DEV)).cpu().numpy()
    want = d["out"][0]
    eq = (got == want).all(-1)
    return dict(case="interp", bit_equal=f"{int(eq.sum())}/{eq.size}", max_abs=float(np.abs(got - want).max()))


def main():
    pdir = sys.argv[1]
    names = sys.argv[2:] or sorted(n[:-4] for n in os.listdir(pdir) if n.endswith(".npz"))
    for n in names:
        p = os.path.join(pdir, n + ".npz")
        if n == "interp":
            print(check_interp(p))
        elif n in ("centres", "depth", "big-sizes"):
            continue
        else:
            r = check_case(p)
            ex = r.pop("examples", [])
            print(r)
            for e in ex:
                print(e)


if __name__ == "__main__":
    main()

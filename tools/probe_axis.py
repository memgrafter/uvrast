"""Black-box probe (EVALUATION venv only; public API only): move test triangles across the grid centre in whole-pixel
steps and record where nvdiffrast's coverage differs from uvrast's exact test.

  python -I tools/probe_axis.py BIG_SIZES.npz OUT.npz
"""
import os, sys
import numpy as np
import torch
import nvdiffrast.torch as dr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uvrast import raster  # noqa: E402

T = 4096


def main():
    src = np.load(sys.argv[1])
    ctx = dr.RasterizeCudaContext()
    out = []
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    for i in (0, 9, 25, 28, 49):
        base = src["tris"][i].astype(np.float64) * T          # pixels
        for axis in (0, 1):
            for step in range(-160, 161, 4):
                v = base.copy(); v[:, axis] += step
                uv = (v / T).astype(np.float32)
                pos = torch.cat([torch.from_numpy(uv).cuda() * 2 - 1, torch.zeros(3, 1, device="cuda"), torch.ones(3, 1, device="cuda")], 1)
                nv = dr.rasterize(ctx, pos[None].contiguous(), tri, resolution=[T, T])[0][0, ..., 3] > 0
                ours = raster.rasterize_ids(pos, tri.long(), T, T).view(T, T) >= 0
                bad = torch.nonzero(nv != ours).cpu().numpy()
                for r, c in bad:
                    out.append((i, axis, step, r, c, int(nv[r, c])))
                print(i, axis, step, len(bad), flush=True)
    np.save(sys.argv[2], np.array(out, dtype=np.int64).reshape(-1, 6))


if __name__ == "__main__":
    main()

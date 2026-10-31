"""Black-box probe of nvdiffrast's coverage for large triangles (EVALUATION venv only; calls the public API only).
For each test triangle, records the texels where nvdiffrast's coverage differs from uvrast's exact fixed-point test.

  python -I tools/probe_big.py OUT.npz
"""
import os, sys
import numpy as np
import torch
import nvdiffrast.torch as dr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uvrast import raster  # noqa: E402

T = 4096


def main():
    ctx = dr.RasterizeCudaContext()
    rng = np.random.default_rng(11)
    tris, recs = [], []
    for size in (64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4000):
        for k in range(24):
            c = rng.uniform(size / 2 + 1, T - size / 2 - 1, 2)
            v = c + rng.uniform(-size / 2, size / 2, (3, 2))
            if k % 4 == 3:                      # same shape moved: translation dependence
                v = v - v.min(0) + rng.uniform(0, 8, 2)
            uv = (v / T).astype(np.float32)
            pos = torch.cat([torch.from_numpy(uv).cuda() * 2 - 1, torch.zeros(3, 1, device="cuda"), torch.ones(3, 1, device="cuda")], 1)
            tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
            nv = dr.rasterize(ctx, pos[None].contiguous(), tri, resolution=[T, T])[0][0, ..., 3] > 0
            ours = raster.rasterize_ids(pos, tri.long(), T, T).view(T, T) >= 0
            bad = torch.nonzero(nv != ours)
            tris.append(uv); recs.append(np.concatenate([bad.cpu().numpy(), nv[bad[:, 0], bad[:, 1]].cpu().numpy()[:, None].astype(np.int64)], 1))
            print(size, k, "mismatch", bad.shape[0], "covered", int(nv.sum()), flush=True)
    np.savez_compressed(sys.argv[1], tris=np.stack(tris), counts=np.array([len(r) for r in recs]),
                        bad=np.concatenate(recs) if recs else np.zeros((0, 3)))


if __name__ == "__main__":
    main()

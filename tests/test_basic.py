"""Self-contained tests (CPU or GPU): no nvdiffrast. Expected values come from the documented output layout and from
an independent brute-force reference (exact rational arithmetic on the same fixed-point positions)."""
import os
import sys
from fractions import Fraction

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import cases  # noqa: E402
import uvrast  # noqa: E402
from uvrast import raster, rules  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def clip(uv):
    uvs = torch.from_numpy(uv).to(DEV)
    return torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], -1).unsqueeze(0)


def brute_ids(pos, tri, H, W):
    """pixel -> covering triangles, by a direct loop over every pixel and triangle (small grids only)"""
    X = rules.to_fixed(pos[:, 0].cpu(), W).tolist(); Y = rules.to_fixed(pos[:, 1].cpu(), H).tolist()
    ids = np.full(H * W, -1)
    for f, (a, b, c) in enumerate(tri.tolist()):
        xs, ys = torch.tensor([[X[a], X[b], X[c]]]), torch.tensor([[Y[a], Y[b], Y[c]]])
        ar = rules.area2(xs, ys)
        if int(ar) == 0:
            continue
        for r in range(H):
            for cidx in range(W):
                if bool(rules.inside(xs, ys, ar, torch.tensor([cidx]), torch.tensor([r]))[0]):
                    p = r * W + cidx
                    ids[p] = f if ids[p] < 0 else (max(ids[p], f) if rules.TIE == "amax" else min(ids[p], f))
    return ids


def test_layout_and_empty():
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], np.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEV)
    rast, db = uvrast.rasterize(uvrast.RasterizeCudaContext(), clip(uv), tri, resolution=[8, 8])
    assert rast.shape == (1, 8, 8, 4) and db.shape == (1, 8, 8, 0)
    ids = rast[0, ..., 3]
    assert ids[0, 0] == 0 and ids[1, 1] == 1 and ids[7, 7] == 0     # row 0 is the bottom (v near 0)
    assert ids[5, 1] == 1 and ids[1, 5] == 1 and ids[6, 6] == 0   # (6, 1) is exactly on the long edge
    empty = rast[0][ids == 0]
    assert bool((empty == 0).all())


def test_barycentrics_reproduce_vertices():
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], np.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEV)
    pos = clip(uv)
    rast, _ = uvrast.rasterize(None, pos, tri, resolution=[64, 64])
    xy, _ = uvrast.interpolate(pos[..., :2].contiguous(), rast, tri)
    m = rast[0, ..., 3] > 0
    r, c = torch.nonzero(m, as_tuple=True)
    want = torch.stack([(2 * c + 1).float() / 64 - 1, (2 * r + 1).float() / 64 - 1], 1)
    assert torch.allclose(xy[0][m], want, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_brute_force(seed):
    uv, f = cases.cells_case(64, seed=seed)
    pos = clip(uv)[0]
    tri = torch.from_numpy(f).to(DEV).long()
    got = raster.rasterize_ids(pos, tri, 64, 64).cpu().numpy()
    assert np.array_equal(got, brute_ids(pos, tri, 64, 64))


def test_shared_edges_cover_once_and_no_gaps():
    """a square split into a fan of triangles through texel centres: every texel inside is covered exactly once"""
    n = 32
    g = np.linspace(4.5, 27.5, 6)
    xx, yy = np.meshgrid(g, g)
    v = np.stack([xx.ravel(), yy.ravel()], 1) / n
    q = [(i * 6 + j, i * 6 + j + 1, (i + 1) * 6 + j, (i + 1) * 6 + j + 1) for i in range(5) for j in range(5)]
    f = np.array([t for a, b, c, d in q for t in ((a, b, c), (c, b, d))], np.int32)
    pos = clip(v.astype(np.float32))[0]
    tri = torch.from_numpy(f).to(DEV).long()
    X, Y = rules.to_fixed(pos[:, 0], n), rules.to_fixed(pos[:, 1], n)
    hits = torch.zeros(n * n, dtype=torch.int64)
    for k in range(len(f)):
        xs, ys = X[tri[k]][None].cpu(), Y[tri[k]][None].cpu()
        r, c = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        ins = rules.inside(xs.expand(n * n, 3), ys.expand(n * n, 3), rules.area2(xs, ys).expand(n * n), c.ravel(), r.ravel())
        hits += ins.long()
    r, c = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    strict = (c > 4) & (c < 27) & (r > 4) & (r < 27)
    assert bool((hits.view(n, n)[strict] == 1).all())
    assert int(hits.max()) == 1


def test_band_splitting_and_passes(monkeypatch):
    """small budgets force row bands and many passes; the result must not change"""
    uv, f = cases.big_case(256)
    pos = clip(uv)[0]; tri = torch.from_numpy(f).to(DEV).long()
    ref = raster.rasterize(pos, tri, 256, 256)
    monkeypatch.setattr(raster, "MAX_ITEM", 300)
    monkeypatch.setattr(raster, "CANDIDATE_BUDGET", 1000)
    assert torch.equal(raster.rasterize(pos, tri, 256, 256), ref)


def test_rejects_unsupported():
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEV)
    pos = clip(np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], np.float32))
    bad = pos.clone(); bad[..., 3] = 2
    with pytest.raises(NotImplementedError):
        uvrast.rasterize(None, bad, tri, [8, 8])
    with pytest.raises(NotImplementedError):
        uvrast.rasterize(None, pos, tri, [8, 8], ranges=torch.tensor([[0, 1]]))


def test_install_shim():
    uvrast.install()
    import nvdiffrast.torch as dr
    assert dr.rasterize is uvrast.rasterize and dr.interpolate is uvrast.interpolate


def test_fma32_is_correctly_rounded():
    """fma32 against exact rational arithmetic, on random values and on constructed float32 rounding ties"""
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(2000, generator=g) * 10 ** torch.randint(-3, 4, (2000,), generator=g)).float()
    b = (torch.randn(2000, generator=g) * 10 ** torch.randint(-3, 4, (2000,), generator=g)).float()
    c = (torch.randn(2000, generator=g) * 10 ** torch.randint(-3, 4, (2000,), generator=g)).float()
    # ties: a*b exactly half an ulp of c away from c, plus a tiny third term that decides the rounding
    one = torch.tensor([1.0, 1.0, 3.0, 5.0])
    eps = torch.tensor([2.0 ** -24, -(2.0 ** -24), 2.0 ** -23, -(2.0 ** -22)])
    a = torch.cat([a, (1 + eps) * 1.0, torch.tensor([1.0 + 2 ** -23])])
    b = torch.cat([b, one * (1 + 2 ** -23), torch.tensor([1.0 + 2 ** -23])])
    c = torch.cat([c, -one, torch.tensor([-1.0])])
    got = rules.fma32(a, b, c)
    for x, y, z, r in zip(a.tolist(), b.tolist(), c.tolist(), got.tolist()):
        exact = Fraction(x) * Fraction(y) + Fraction(z)
        cand = np.float32(float(exact))
        lo, hi = np.nextafter(cand, np.float32(-np.inf)), np.nextafter(cand, np.float32(np.inf))
        best = min((cand, lo, hi), key=lambda q: (abs(Fraction(float(q)) - exact), int(np.float32(q).view(np.int32)) & 1))
        assert r == float(best), (x, y, z, r, float(best))

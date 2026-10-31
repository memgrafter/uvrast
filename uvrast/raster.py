"""Point-sampled triangle rasterizer for UV layouts, in PyTorch (any device).

Output layout (public nvdiffrast API docs): per pixel (u, v, z/w, triangle_id + 1), all zero where no triangle covers
the pixel centre; the first vertex of a triangle has (u, v) = (1, 0), the second (0, 1), the third (0, 0). Row 0 is the
bottom row (y = -1).

Method: every triangle's pixel-centre bounding box is expanded to (triangle, pixel) candidates, each candidate gets an
exact integer inside test on fixed-point vertex positions, and the winner per pixel is chosen with one scatter.
Barycentrics are then computed for the winners only.
"""
import numpy as np
import torch

from . import rules

CANDIDATE_BUDGET = 1 << 24      # (triangle, pixel) candidates per pass; bounds the temporary memory
MAX_ITEM = 1 << 20              # a triangle with a larger bounding box is split into row bands


def _bbox(X, Y, W, H):
    """pixel index ranges [c0, c1] x [r0, r1] whose centres lie in the fixed-point bounding box, clamped"""
    S, half = rules.SUB, rules.SUB // 2
    fl = lambda a: torch.div(a, S, rounding_mode="floor")
    c0 = -fl(-(X.min(1).values - half)); c1 = fl(X.max(1).values - half)
    r0 = -fl(-(Y.min(1).values - half)); r1 = fl(Y.max(1).values - half)
    return c0.clamp(min=0), c1.clamp(max=W - 1), r0.clamp(min=0), r1.clamp(max=H - 1)


def _work_items(c0, c1, r0, r1):
    """(face, c0, ncols, r0, nrows) items; faces with large boxes are split into row bands of at most MAX_ITEM pixels"""
    face = torch.arange(c0.numel(), device=c0.device)
    nc, nr = (c1 - c0 + 1).clamp(min=0), (r1 - r0 + 1).clamp(min=0)
    keep = (nc > 0) & (nr > 0)
    face, c0, nc, r0, nr = face[keep], c0[keep], nc[keep], r0[keep], nr[keep]
    band = torch.clamp(MAX_ITEM // nc, min=1)
    nb = torch.div(nr + band - 1, band, rounding_mode="floor")
    if bool((nb > 1).any()):
        rep = torch.repeat_interleave(torch.arange(face.numel(), device=face.device), nb)
        k = torch.arange(rep.numel(), device=face.device) - torch.repeat_interleave(torch.cumsum(nb, 0) - nb, nb)
        start = r0[rep] + k * band[rep]
        nr = torch.minimum(band[rep], r0[rep] + nr[rep] - start)
        face, c0, nc, r0 = face[rep], c0[rep], nc[rep], start
    return face, c0, nc, r0, nr


def _passes(count):
    """split items into consecutive groups of at most CANDIDATE_BUDGET candidates (an item larger than the budget
    gets a group of its own)"""
    cum = torch.cumsum(count, 0)
    bounds, start, total = [], 0, int(cum[-1]) if cum.numel() else 0
    base = 0
    while start < count.numel():
        end = int(torch.searchsorted(cum, base + CANDIDATE_BUDGET, right=True))
        end = max(end, start + 1)
        bounds.append((start, end))
        base = int(cum[end - 1]); start = end
    return bounds, total


def _clip(poly, axis, bound, keep_less):
    """Sutherland-Hodgman against one axis-aligned line; poly: list of (point float64[2], fixed int pair or None)"""
    out = []
    for k in range(len(poly)):
        (p, pf), (q, qf) = poly[k], poly[(k + 1) % len(poly)]
        pin = p[axis] <= bound if keep_less else p[axis] >= bound
        qin = q[axis] <= bound if keep_less else q[axis] >= bound
        if pin:
            out.append((p, pf))
        if pin != qin:
            x = p + (bound - p[axis]) / (q[axis] - p[axis]) * (q - p)
            x[axis] = bound
            out.append((x, None))
    return out


def _tile_split(pos, tri, X, Y, H, W):
    """Triangles that cross a sub-grid border (every rules.TILE pixels) or the grid border are clipped per sub-grid,
    the new vertices snapped, and each clipped polygon fan-triangulated (probe: nvdiffrast does this above 2048 px).
    Returns X, Y [F', 3] int64 and owner [F'] (the original triangle of each row)."""
    S, ts = rules.SUB, rules.TILE * rules.SUB
    U = ((pos[:, 0].double() + 1) * (0.5 * W * S))[tri]
    V = ((pos[:, 1].double() + 1) * (0.5 * H * S))[tri]
    def crosses(P, n):
        lo, hi = P.min(1).values, P.max(1).values
        out = (lo < 0) | (hi > n * S)
        for b in range(ts, n * S, ts):
            out |= (lo < b) & (hi > b)
        return out
    split = crosses(U, W) | crosses(V, H)
    owner = torch.arange(tri.shape[0], device=tri.device)
    if not bool(split.any()):
        return X, Y, owner
    ids = torch.nonzero(split).squeeze(1).cpu().numpy()
    u, v = U[split].cpu().numpy(), V[split].cpu().numpy()
    xf, yf = X[split].cpu().numpy(), Y[split].cpu().numpy()
    fx, fy, fo = [], [], []
    for k, f in enumerate(ids):
        poly0 = [(np.array([u[k, j], v[k, j]]), (int(xf[k, j]), int(yf[k, j]))) for j in range(3)]
        for tx in range(max(0, int(u[k].min() // ts)), min((W * S - 1) // ts, int(u[k].max() // ts)) + 1):
            for ty in range(max(0, int(v[k].min() // ts)), min((H * S - 1) // ts, int(v[k].max() // ts)) + 1):
                poly = poly0
                for axis, b, less in ((0, tx * ts, False), (0, min((tx + 1) * ts, W * S), True),
                                      (1, ty * ts, False), (1, min((ty + 1) * ts, H * S), True)):
                    poly = _clip(poly, axis, b, less) if len(poly) >= 3 else poly
                if len(poly) < 3:
                    continue
                q = [pf if pf is not None else (int(np.rint(p[0])), int(np.rint(p[1]))) for p, pf in poly]
                for j in range(1, len(q) - 1):
                    fx.append((q[0][0], q[j][0], q[j + 1][0])); fy.append((q[0][1], q[j][1], q[j + 1][1])); fo.append(f)
    keep = ~split
    dev = X.device
    if not fo:
        return X[keep], Y[keep], owner[keep]
    return (torch.cat([X[keep], torch.tensor(fx, dtype=torch.int64, device=dev)]),
            torch.cat([Y[keep], torch.tensor(fy, dtype=torch.int64, device=dev)]),
            torch.cat([owner[keep], torch.tensor(fo, dtype=torch.int64, device=dev)]))


def rasterize_ids(pos, tri, H, W):
    """triangle index (0-based) per pixel, -1 where empty; pos [V, 4] float32 clip space with z = 0, w = 1 (so every
    triangle has the same depth and the tie rule alone picks the winner), tri [F, 3] int64"""
    X, Y = rules.to_fixed(pos[:, 0], W), rules.to_fixed(pos[:, 1], H)
    X, Y, owner = _tile_split(pos, tri, X[tri], Y[tri], H, W)   # [F', 3]; owner: original triangle per row
    area = rules.area2(X, Y)                           # twice the signed area, int64
    c0, c1, r0, r1 = _bbox(X, Y, W, H)
    drawn = area != 0
    c1 = torch.where(drawn, c1, c0 - 1)
    face, c0, nc, r0, nr = _work_items(c0, c1, r0, r1)
    empty = (1 << 62) if rules.TIE == "amin" else -1
    best = torch.full((H * W,), empty, dtype=torch.int64, device=pos.device)
    count = nc * nr
    bounds, _ = _passes(count)
    for s, e in bounds:
        f, cc0, ncc, rr0, cnt = face[s:e], c0[s:e], nc[s:e], r0[s:e], count[s:e]
        item = torch.repeat_interleave(torch.arange(f.numel(), device=f.device), cnt)
        local = torch.arange(item.numel(), device=f.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        col = cc0[item] + local % ncc[item]
        row = rr0[item] + torch.div(local, ncc[item], rounding_mode="floor")
        ff = f[item]
        inside = rules.inside(X[ff], Y[ff], area[ff], col, row)
        ff, pix = ff[inside], (row * W + col)[inside]
        best.scatter_reduce_(0, pix, owner[ff], reduce=rules.TIE)
    return torch.where(best == empty, -1, best)


def rasterize(pos, tri, H, W):
    """[H, W, 4] float32: (u, v, z/w, id + 1); pos [V, 4] float32 with w = 1, tri [F, 3]"""
    tri = tri.long()
    ids = rasterize_ids(pos, tri, H, W)
    out = torch.zeros(H * W, 4, dtype=torch.float32, device=pos.device)
    pix = torch.nonzero(ids >= 0).squeeze(1)
    f = ids[pix]
    row, col = torch.div(pix, W, rounding_mode="floor"), pix % W
    u, v = rules.barycentrics(pos[:, 0][tri[f]], pos[:, 1][tri[f]], col, row, W, H)
    out[pix, 0], out[pix, 1] = u, v
    # out[:, 2] (z/w) stays 0: the public wrapper only accepts z = 0, w = 1 (a UV layout)
    out[pix, 3] = (f + 1).float()
    return out.view(H, W, 4)


def interpolate(attr, rast, tri):
    """[H, W, A]: attributes at each pixel from (u, v, id); zero where id = 0. attr [V, A], rast [H, W, 4]"""
    H, W = rast.shape[:2]
    r = rast.reshape(-1, 4)
    out = torch.zeros(H * W, attr.shape[1], dtype=attr.dtype, device=attr.device)
    pix = torch.nonzero(r[:, 3] > 0).squeeze(1)
    f = r[pix, 3].long() - 1
    t = tri.long()[f]
    out[pix] = rules.blend(r[pix, 0:1], r[pix, 1:2], attr[t[:, 0]], attr[t[:, 1]], attr[t[:, 2]])
    return out.view(H, W, -1)

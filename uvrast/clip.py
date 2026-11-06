"""General clip-space rasterization (perspective or orthographic views), added for MV-Adapter's texturing
(branch texturing-port, csc-1k57). The UV-layout path in raster.py is unchanged.

Written from the public API documentation and the callers' code only (no nvdiffrast source):
- the view frustum is x/w, y/w, z/w all in [-1, 1]; each pixel shows the closest surface along the ray through the
  pixel centre; a larger z/w is further away; pixels with no triangle get zeros in every channel;
- output per pixel (u, v, z/w, triangle_id + 1); vertex 0 has (u, v) = (1, 0), vertex 1 (0, 1), vertex 2 (0, 0);
  row 0 is the bottom row (y/w = -1).

Rules chosen here (the documentation does not state them):
- coverage: the same fixed-point (1/16 px) positions and exact integer edge test with a top-left fill rule as the
  UV-layout path (rules.inside), applied to x/w and y/w; so shared edges have no cracks and no double cover;
- depth: z/w interpolated linearly in screen space (it is affine there) from the unsnapped x/w, y/w; a fragment with
  z/w outside [-1, 1] is not drawn (for vertices with w > 0 this is the near/far plane clip); the smallest z/w wins,
  equal z/w: the higher triangle index wins (as the UV-layout tie rule);
- (u, v): perspective-correct: screen-space barycentrics (rules.barycentrics) divided by each vertex's w and
  normalised; when the triangle's three w are equal (orthographic) the screen-space values are returned as they are;
- triangles entirely outside one frustum plane are dropped; a kept triangle with a vertex at w <= 0 (it would need
  clipping against w = 0) raises NotImplementedError, so a caller never gets a silently wrong image.
"""
import torch

from . import raster, rules

MAX_NDC = 1 << 12   # a kept triangle with |x/w| or |y/w| above this raises (guard-band clipping is not implemented)
ID_BITS = 24


def _sortable(z):
    """float32 -> int64 with the same order (for one integer scatter-min of (depth, triangle))"""
    z = z + 0.0                                    # -0.0 -> +0.0
    b = z.view(torch.int32).to(torch.int64)
    return torch.where(b >= 0, b + (1 << 31), (1 << 31) - 1 - (b & 0x7FFFFFFF))


def rasterize(pos, tri, H, W):
    """[H, W, 4] float32 (u, v, z/w, id + 1); pos [V, 4] float32 clip space, tri [F, 3] int"""
    tri = tri.long()
    F = tri.shape[0]
    out = torch.zeros(H * W, 4, dtype=torch.float32, device=pos.device)
    if F == 0:
        return out.view(H, W, 4)
    P = pos[tri]                                   # [F, 3, 4]
    x, y, z, w = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    outside = ((x < -w).all(1) | (x > w).all(1) | (y < -w).all(1) | (y > w).all(1) | (z < -w).all(1) | (z > w).all(1))
    keep = torch.nonzero(~outside).squeeze(1)
    if keep.numel() == 0:
        return out.view(H, W, 4)
    if bool((w[keep] <= 0).any()):
        raise NotImplementedError("a triangle in the view has a vertex with w <= 0 (clipping against w = 0 is not supported)")
    xd, yd, zd, wd = (a[keep].double() for a in (x, y, z, w))
    xn, yn, zn = xd / wd, yd / wd, zd / wd         # float64 NDC
    if bool((xn.abs() > MAX_NDC).any() | (yn.abs() > MAX_NDC).any()):
        raise NotImplementedError(f"a triangle in the view has |x/w| or |y/w| > {MAX_NDC} (guard-band clipping is not supported)")
    X, Y = rules.to_fixed(xn, W), rules.to_fixed(yn, H)
    area = rules.area2(X, Y)
    c0, c1, r0, r1 = raster._bbox(X, Y, W, H)
    c1 = torch.where(area != 0, c1, c0 - 1)
    item_face, c0, nc, r0, nr = raster._work_items(c0, c1, r0, r1)
    empty = 1 << 62
    best = torch.full((H * W,), empty, dtype=torch.int64, device=pos.device)
    count = nc * nr
    bounds, _ = raster._passes(count)
    for s, e in bounds:
        f, cc0, ncc, rr0, cnt = item_face[s:e], c0[s:e], nc[s:e], r0[s:e], count[s:e]
        item = torch.repeat_interleave(torch.arange(f.numel(), device=f.device), cnt)
        local = torch.arange(item.numel(), device=f.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        col = cc0[item] + local % ncc[item]
        row = rr0[item] + torch.div(local, ncc[item], rounding_mode="floor")
        ff = f[item]
        ins = rules.inside(X[ff], Y[ff], area[ff], col, row)
        ff, col, row = ff[ins], col[ins], row[ins]
        # screen-space barycentrics (float64, unsnapped NDC) -> z/w at the pixel centre
        px = (2 * col + 1).double() / W - 1
        py = (2 * row + 1).double() / H - 1
        ax, ay = xn[ff] - px[:, None], yn[ff] - py[:, None]
        e0 = ax[:, 1] * ay[:, 2] - ay[:, 1] * ax[:, 2]
        e1 = ax[:, 2] * ay[:, 0] - ay[:, 2] * ax[:, 0]
        e2 = ax[:, 0] * ay[:, 1] - ay[:, 0] * ax[:, 1]
        zz = ((e0 * zn[ff, 0] + e1 * zn[ff, 1] + e2 * zn[ff, 2]) / (e0 + e1 + e2)).float()
        ok = (zz >= -1) & (zz <= 1)
        fid = keep[ff[ok]]
        key = (_sortable(zz[ok]) << ID_BITS) | ((1 << ID_BITS) - 1 - fid)
        best.scatter_reduce_(0, (row * W + col)[ok], key, reduce="amin")
    pix = torch.nonzero(best != empty).squeeze(1)
    f = (1 << ID_BITS) - 1 - (best[pix] & ((1 << ID_BITS) - 1))
    row, col = torch.div(pix, W, rounding_mode="floor"), pix % W
    t = tri[f]
    wv = pos[:, 3][t]                               # [N, 3] float32
    xs, ys, zs = pos[:, 0][t] / wv, pos[:, 1][t] / wv, pos[:, 2][t] / wv
    u, v = rules.barycentrics(xs, ys, col, row, W, H)          # screen space, float32
    wt = (1 - u) - v
    zw = zs[:, 2] + u * (zs[:, 0] - zs[:, 2]) + v * (zs[:, 1] - zs[:, 2])   # exact when the three z/w are equal
    a0, a1, a2 = u / wv[:, 0], v / wv[:, 1], wt / wv[:, 2]
    ssum = a0 + a1 + a2
    flat = (wv[:, 0] == wv[:, 1]) & (wv[:, 1] == wv[:, 2])
    out[pix, 0] = torch.where(flat, u, a0 / ssum)
    out[pix, 1] = torch.where(flat, v, a1 / ssum)
    out[pix, 2] = zw
    out[pix, 3] = (f + 1).float()
    return out.view(H, W, 4)


def texture(tex, uv, filter_mode, boundary_mode):
    """tex [Bt, Ht, Wt, C] (Bt = 1 or the minibatch), uv [B, H, W, 2] -> [B, H, W, C].
    Documented conventions: (s, t) = (0, 0) is the bottom-left corner, row 0 of the tensor is the bottom row;
    'wrap' takes the fractional part, 'clamp' clamps to the centres of the boundary texels, 'zero' extends the texture
    with zeros. Texel (i, j) has its centre at ((j + 0.5) / Wt, (i + 0.5) / Ht) (the usual convention; assumed)."""
    B, H, W, _ = uv.shape
    Bt, Ht, Wt, C = tex.shape
    if Bt not in (1, B):
        raise ValueError("tex minibatch must be 1 or the uv minibatch")
    flat = tex.reshape(Bt, Ht * Wt, C)
    if Bt == 1 and B > 1:
        flat = flat.expand(B, -1, -1)
    s = uv[..., 0].reshape(B, -1).float()
    t = uv[..., 1].reshape(B, -1).float()

    def fetch(i, j):
        if boundary_mode == "wrap":
            i, j, valid = torch.remainder(i, Ht), torch.remainder(j, Wt), None
        elif boundary_mode == "clamp":
            i, j, valid = i.clamp(0, Ht - 1), j.clamp(0, Wt - 1), None
        else:   # zero
            valid = (i >= 0) & (i < Ht) & (j >= 0) & (j < Wt)
            i, j = i.clamp(0, Ht - 1), j.clamp(0, Wt - 1)
        val = torch.gather(flat, 1, (i * Wt + j).unsqueeze(-1).expand(-1, -1, C))
        return val if valid is None else val * valid.unsqueeze(-1).to(val.dtype)

    if boundary_mode == "wrap":
        s, t = s - torch.floor(s), t - torch.floor(t)
    if filter_mode == "nearest":
        j = torch.floor(s * Wt).long()
        i = torch.floor(t * Ht).long()
        if boundary_mode == "wrap":
            j, i = j.clamp(max=Wt - 1), i.clamp(max=Ht - 1)   # s = 1 - eps rounding up to Wt
        out = fetch(i, j)
    else:
        fs, ft = s * Wt - 0.5, t * Ht - 0.5
        if boundary_mode == "clamp":
            fs, ft = fs.clamp(0, Wt - 1), ft.clamp(0, Ht - 1)
        j0, i0 = torch.floor(fs), torch.floor(ft)
        bs, bt = (fs - j0).unsqueeze(-1), (ft - i0).unsqueeze(-1)
        j0, i0 = j0.long(), i0.long()
        out = ((fetch(i0, j0) * (1 - bs) + fetch(i0, j0 + 1) * bs) * (1 - bt)
               + (fetch(i0 + 1, j0) * (1 - bs) + fetch(i0 + 1, j0 + 1) * bs) * bt)
    return out.reshape(B, H, W, C)

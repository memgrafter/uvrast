"""Caller-shaped tests (texturing-port): the way MV-Adapter renders position/normal maps (6 views in one minibatch,
attributes broadcast from [1, V, C]) and the way TRELLIS.2's texturing bakes (UV layout -> surface positions).
Expected values are analytic: a sphere's ray hit point, its silhouette disc, and a plane's surface."""
import math

import numpy as np
import torch

import uvrast

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sphere(r=0.4, n=5):
    """octahedron subdivided n times, projected onto the sphere"""
    V = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(n):
        cache, F2 = {}, []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                V.append((V[a] + V[b]) / 2); cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        F = F2
    V = np.array(V); V = V / np.linalg.norm(V, axis=1, keepdims=True) * r
    return torch.tensor(V, dtype=torch.float32, device=DEV), torch.tensor(F, dtype=torch.int32, device=DEV)


def look_at(eye):
    """camera-to-world as MV-Adapter builds it (right, up, -lookat; world up +z)"""
    eye = np.asarray(eye, float); f = -eye / np.linalg.norm(eye)
    right = np.cross(f, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -f, eye
    return c2w


def ortho(s, n=0.1, f=100.0):
    P = np.eye(4); P[0, 0] = P[1, 1] = 1 / s; P[2, 2] = -2 / (f - n); P[2, 3] = -(f + n) / (f - n)
    return P


def persp(fovy, n=0.1, f=100.0):
    t = 1 / math.tan(math.radians(fovy) / 2)
    return np.array([[t, 0, 0, 0], [0, t, 0, 0], [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, -1, 0]])


def render_positions(V, F, mvps, H):
    pos = torch.cat([V, torch.ones_like(V[:, :1])], 1)
    clip = torch.stack([pos @ torch.tensor(m.T, dtype=torch.float32, device=DEV) for m in mvps]).contiguous()
    rast, _ = uvrast.rasterize(uvrast.RasterizeGLContext(device=DEV), clip, F, (H, H))
    xyz, _ = uvrast.interpolate(V[None], rast, F)        # [1, V, 3] broadcast over the views (MV-Adapter's call)
    return rast, xyz


def test_six_ortho_views_hit_the_front_of_a_sphere():
    r, H, s = 0.4, 96, 0.55
    V, F = sphere(r)
    eyes = [(0, -1.8, 0), (1.8, 0, 0), (0, 1.8, 0), (-1.8, 0, 0), (0, 0.0001, 1.8), (0, 0.0001, -1.8)]
    c2ws = [look_at(e) for e in eyes]
    rast, xyz = render_positions(V, F, [ortho(s) @ np.linalg.inv(c) for c in c2ws], H)
    sag = r * (1 - math.cos(math.pi / 2 / 2 ** 5 * 1.5))      # bound on facet depth error (planar triangles)
    for k, c in enumerate(c2ws):
        m = rast[k, ..., 3] > 0
        p = xyz[k][m].double().cpu().numpy()
        # each hit is on the sphere (up to the facet error) and on the camera side
        rad = np.linalg.norm(p, axis=1)
        assert rad.max() <= r + 1e-5 and rad.min() >= r - sag
        assert (p @ c[:3, 2] > -1e-6).all()                    # c2w z axis points back to the camera
        # and on the ray through the pixel centre: its in-plane coordinates are the pixel centre's
        row, col = torch.nonzero(m, as_tuple=True)
        xc = ((2 * col + 1).double() / H - 1).cpu().numpy() * s
        yc = ((2 * row + 1).double() / H - 1).cpu().numpy() * s
        # interior pixels: exact up to float32; pixels whose barycentrics were clamped (centre inside the 1/16 px
        # snapped triangle but outside the unsnapped one, the UV path's rule) sit on the edge: within 0.25 px
        bc = rast[k][m][:, :2].double().cpu().numpy(); edge = (bc.min(1) == 0) | (bc.sum(1) >= 1 - 1e-6)
        e = np.maximum(np.abs(p @ c[:3, 0] - xc), np.abs(p @ c[:3, 1] - yc))
        assert e[~edge].max() < 1e-4 and e[edge].max() < 0.25 * 2 * s / H
        # silhouette = the disc of radius r (polygon inside the circle; edge pixels may differ)
        dist = np.hypot(xc, yc)
        assert dist.max() <= r + s / H
        rr, cc = np.meshgrid((2 * np.arange(H) + 1) / H - 1, (2 * np.arange(H) + 1) / H - 1, indexing="ij")
        inside = np.hypot(cc * s, rr * s) < r - sag - 2 * s / H
        assert m.cpu().numpy()[inside].all()


def test_perspective_view_hits_the_sphere():
    r, H = 0.4, 96
    V, F = sphere(r)
    c = look_at((0.3, -2.0, 0.5))
    rast, xyz = render_positions(V, F, [persp(40) @ np.linalg.inv(c)], H)
    m = rast[0, ..., 3] > 0
    p = xyz[0][m].double().cpu().numpy()
    sag = r * (1 - math.cos(math.pi / 2 / 2 ** 5 * 1.5))
    rad = np.linalg.norm(p, axis=1)
    assert rad.max() <= r + 1e-5 and rad.min() >= r - sag
    # visible side only: the point faces the camera
    eye = c[:3, 3]
    assert (np.einsum("ij,ij->i", p, eye - p) > -1e-4).all()
    # the hit is on the ray through its pixel centre (perspective-correct interpolation)
    w2c = np.linalg.inv(c); q = p @ w2c[:3, :3].T + w2c[:3, 3]
    t = 1 / math.tan(math.radians(20))
    row, col = torch.nonzero(m, as_tuple=True)
    xc = ((2 * col + 1).double() / H - 1).cpu().numpy(); yc = ((2 * row + 1).double() / H - 1).cpu().numpy()
    bc = rast[0][m][:, :2].double().cpu().numpy(); edge = (bc.min(1) == 0) | (bc.sum(1) >= 1 - 1e-6)
    e = np.maximum(np.abs(t * q[:, 0] / -q[:, 2] - xc), np.abs(t * q[:, 1] / -q[:, 2] - yc))
    assert e[~edge].max() < 2e-4 and e[edge].max() < 0.25 * 2 / H     # clamped edge pixels: within 0.25 px


def test_trellis_texturing_bake_shape():
    """TRELLIS.2's postprocess_mesh: uvs * 2 - 1 with z = 0, w = 1, [1, V, 4]; interpolate vertices [1, V, 3] -> the
    texel's surface point. A plane with a planar map: the point must equal the texel centre mapped back."""
    n = 17
    g = np.linspace(0, 1, n)
    uu, vv = np.meshgrid(g, g, indexing="xy")
    uv = np.stack([uu.ravel(), vv.ravel()], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j; F += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    verts = np.stack([uv[:, 0] * 2.0 - 0.3, uv[:, 1] * 0.5 + 0.1, 0.25 * np.ones(len(uv))], 1)
    uvs = torch.tensor(uv, dtype=torch.float32, device=DEV)
    faces = torch.tensor(F, dtype=torch.int32, device=DEV)
    pos = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], -1).unsqueeze(0)
    rast, _ = uvrast.rasterize(uvrast.RasterizeCudaContext(), pos, faces, resolution=[64, 64])
    out = uvrast.interpolate(torch.tensor(verts, dtype=torch.float32, device=DEV).unsqueeze(0), rast, faces)[0][0]
    m = rast[0, ..., 3] > 0
    assert bool(m.all())                                  # the planar map covers the whole square
    row, col = torch.nonzero(m, as_tuple=True)
    u = (col.double() + 0.5) / 64; v = (row.double() + 0.5) / 64
    p = out[m].double()
    assert float((p[:, 0] - (u * 2 - 0.3)).abs().max()) < 1e-6
    assert float((p[:, 1] - (v * 0.5 + 0.1)).abs().max()) < 1e-6
    assert float((p[:, 2] - 0.25).abs().max()) < 1e-6

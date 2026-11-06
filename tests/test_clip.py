"""Tests of the clip-space path (clip.py), texture() and the API additions of branch texturing-port.
Expected values are analytic (no nvdiffrast): known depths, pixel-centre reprojection of perspective-correct
attributes, exact coverage identities, hand-computed texture samples."""
import math

import pytest
import torch

import uvrast
from uvrast import clip, raster

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEV)


def centres(H, W):
    r, c = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    return (2 * c + 1).float() / W - 1, (2 * r + 1).float() / H - 1   # NDC x, y of each pixel centre; row 0 = bottom


def test_uv_layout_still_uses_the_original_path():
    pos = T([[[-0.8, -0.8, 0, 1], [0.8, -0.7, 0, 1], [-0.6, 0.9, 0, 1]]])
    tri = T([[0, 1, 2]], torch.int32)
    rast, db = uvrast.rasterize(uvrast.RasterizeCudaContext(), pos, tri, (32, 32))
    assert torch.equal(rast[0], raster.rasterize(pos[0], tri, 32, 32)) and db.shape == (1, 32, 32, 0)


def test_depth_test_and_constant_depth():
    # far square at z/w = 0.5 (two triangles), near triangle at z/w = -0.25 in front of part of it (orthographic, w = 1)
    pos = T([[[-0.9, -0.9, 0.5, 1], [0.9, -0.9, 0.5, 1], [0.9, 0.9, 0.5, 1], [-0.9, 0.9, 0.5, 1],
              [-0.5, -0.5, -0.25, 1], [0.5, -0.5, -0.25, 1], [0.0, 0.6, -0.25, 1]]])
    tri = T([[0, 1, 2], [0, 2, 3], [4, 5, 6]], torch.int32)
    rast, _ = uvrast.rasterize(uvrast.RasterizeGLContext(), pos, tri, (40, 40))
    ids, z = rast[0, ..., 3], rast[0, ..., 2]
    assert set(ids.unique().tolist()) == {0.0, 1.0, 2.0, 3.0}
    assert bool((z[ids == 3] == -0.25).all()) and bool((z[(ids == 1) | (ids == 2)] == 0.5).all())
    assert bool((rast[0][ids == 0] == 0).all())
    # the same with the triangle order reversed: the nearer triangle still wins (depth, not order)
    rast2, _ = uvrast.rasterize(None, pos, T([[4, 5, 6], [0, 2, 3], [0, 1, 2]], torch.int32), (40, 40))
    assert torch.equal(rast2[0, ..., 3] == 1, ids == 3)


def test_far_plane_discard():
    # a triangle tilted in depth from z/w = 0 to z/w = 2: only its part with z/w <= 1 is drawn
    pos = T([[[-1, -1, 0, 1], [1, -1, 2, 1], [-1, 1, 0, 1]]])
    rast, _ = uvrast.rasterize(None, pos, T([[0, 1, 2]], torch.int32), (16, 16))
    x, y = centres(16, 16)
    drawn = rast[0, ..., 3] > 0
    assert bool((x[drawn] <= 0).all())                # z/w = x + 1 here
    assert bool(((rast[0, ..., 2] - (x + 1)).abs()[drawn] < 1e-6).all())


def test_perspective_correct_attributes_reproject_to_pixel_centres():
    # a triangle in camera space, projected with a perspective matrix; interpolating the CLIP position as an attribute
    # must give a point whose x/w, y/w is the pixel centre (true only with perspective-correct barycentrics)
    n, f, s = 0.1, 10.0, 1 / math.tan(math.radians(25))
    proj = T([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, -1, 0]])
    cam = T([[-1.0, -0.8, -1.5, 1], [1.2, -0.5, -4.0, 1], [-0.2, 1.0, -2.5, 1]])
    clip_pos = (cam @ proj.T)[None].contiguous()
    tri = T([[0, 1, 2]], torch.int32)
    rast, _ = uvrast.rasterize(None, clip_pos, tri, (64, 64))
    out, _ = uvrast.interpolate(clip_pos, rast, tri)
    m = rast[0, ..., 3] > 0
    assert int(m.sum()) > 300
    x, y = centres(64, 64)
    px, py = out[0, ..., 0] / out[0, ..., 3], out[0, ..., 1] / out[0, ..., 3]
    assert float((px - x)[m].abs().max()) < 2e-5 and float((py - y)[m].abs().max()) < 2e-5
    # and z/w in rast is the interpolated point's z/w
    assert float((rast[0, ..., 2] - out[0, ..., 2] / out[0, ..., 3])[m].abs().max()) < 2e-5
    # camera-space attribute: the interpolated point lies on the triangle's plane
    pc, _ = uvrast.interpolate(cam[None, :, :3].contiguous(), rast, tri)
    nrm = torch.linalg.cross(cam[1, :3] - cam[0, :3], cam[2, :3] - cam[0, :3])
    assert float(((pc[0][m] - cam[0, :3]) @ nrm).abs().max()) < 1e-4


def test_shared_edges_cover_each_pixel_once():
    # a convex quad split along either diagonal covers the same pixels, and no pixel twice
    q = T([[-0.83, -0.71, 0.1, 1], [0.77, -0.9, 0.1, 1], [0.91, 0.63, 0.1, 1], [-0.6, 0.88, 0.1, 1]])
    H = W = 48
    a, _ = uvrast.rasterize(None, q[None], T([[0, 1, 2], [0, 2, 3]], torch.int32), (H, W))
    b, _ = uvrast.rasterize(None, q[None], T([[0, 1, 3], [1, 2, 3]], torch.int32), (H, W))
    assert torch.equal(a[0, ..., 3] > 0, b[0, ..., 3] > 0)
    # each triangle alone: the two masks are disjoint and their union is the quad
    t0, _ = uvrast.rasterize(None, q[None], T([[0, 1, 2]], torch.int32), (H, W))
    t1, _ = uvrast.rasterize(None, q[None], T([[0, 2, 3]], torch.int32), (H, W))
    m0, m1 = t0[0, ..., 3] > 0, t1[0, ..., 3] > 0
    assert not bool((m0 & m1).any()) and torch.equal(m0 | m1, a[0, ..., 3] > 0)


def test_culling_and_w_le_zero():
    tri = T([[0, 1, 2]], torch.int32)
    behind = T([[[-0.5, -0.5, 0, -1], [0.5, -0.5, 0, -1], [0, 0.5, 0, -1]]])   # all w < 0: x > w for all -> dropped
    r, _ = uvrast.rasterize(None, behind, tri, (8, 8))
    assert bool((r == 0).all())
    crossing = T([[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, -0.5]]])
    with pytest.raises(NotImplementedError):
        uvrast.rasterize(None, crossing, tri, (8, 8))


def test_minibatch_and_attribute_broadcast():
    pos = T([[[-0.9, -0.9, 0.2, 1], [0.9, -0.9, 0.2, 1], [0, 0.9, 0.2, 1]],
             [[-0.9, 0.9, 0.3, 1], [0.9, 0.9, 0.3, 1], [0, -0.9, 0.3, 1]]])
    tri = T([[0, 1, 2]], torch.int32)
    rast, _ = uvrast.rasterize(None, pos, tri, (16, 16))
    attr = T([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]])
    out, _ = uvrast.interpolate(attr, rast, tri)
    assert out.shape == (2, 16, 16, 3)
    for i in range(2):
        m = rast[i, ..., 3] > 0
        assert bool(((out[i].sum(-1) - 1).abs()[m] < 1e-6).all()) and bool((out[i][~m] == 0).all())
    assert not torch.equal(rast[0, ..., 3], rast[1, ..., 3])


def test_texture_conventions():
    # 2 x 2 texture; row 0 is the bottom row: tex[0, 0] at (s, t) near (0, 0)
    tex = T([[[[1.0], [2.0]], [[3.0], [4.0]]]])     # [1, 2, 2, 1]
    uv = T([[[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]]])   # texel centres
    for fm in ("nearest", "linear", "auto"):
        assert uvrast.texture(tex, uv, filter_mode=fm)[0, 0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    mid = T([[[[0.5, 0.5], [0.5, 0.25], [0.375, 0.25]]]])
    assert uvrast.texture(tex, mid)[0, 0, :, 0].tolist() == [2.5, 1.5, 1.25]
    # wrap: fractional part; s = 0 sits between the last and the first texel column
    w = uvrast.texture(tex, T([[[[1.25, 0.25], [0.0, 0.25], [-0.75, 1.75]]]]), boundary_mode="wrap")
    assert w[0, 0, :, 0].tolist() == [1.0, 1.5, 3.0]
    c = uvrast.texture(tex, T([[[[-3.0, 0.25], [0.0, 0.25], [5.0, 5.0]]]]), boundary_mode="clamp")
    assert c[0, 0, :, 0].tolist() == [1.0, 1.0, 4.0]
    z = uvrast.texture(tex, T([[[[-3.0, 0.25], [0.0, 0.25]]]]), boundary_mode="zero")
    assert z[0, 0, :, 0].tolist() == [0.0, 0.5]
    with pytest.raises(NotImplementedError):
        uvrast.texture(tex, uv, filter_mode="linear-mipmap-linear")
    with pytest.raises(NotImplementedError):
        uvrast.antialias(None, None, None, None)


def test_gl_context_signature():
    ctx = uvrast.RasterizeGLContext(output_db=False, mode="automatic", device=DEV)
    assert ctx.device.type == torch.device(DEV).type

"""The conventions that decide which pixels a triangle covers and the exact float32 values written.
HYPOTHESES until the black-box probes (tools/probe_nvdiffrast.py, tools/fit_rules.py) settle them; see README."""
import torch

SUB_BITS = 4                  # fixed-point vertex precision: 2^SUB_BITS steps per pixel (probe: 1/16 px, ties to even)
SUB = 1 << SUB_BITS
TILE = 2048                  # sub-grid size in pixels: triangles crossing a sub-grid border are clipped there (probe)
TIE = "amax"                 # equal depth: the higher triangle index wins ("amax") or the lower ("amin")


def to_fixed(x, n):
    """clip coordinate (float32, -1..1 across n pixels) -> int64 position in 1/SUB pixel units"""
    return torch.round((x + 1) * (0.5 * n) * SUB).to(torch.int64)


def area2(X, Y):
    return (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])


def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _owns_zero(ax, ay, bx, by, sign):
    """fill rule for a centre exactly on edge a->b of a counter-clockwise (sign-corrected) triangle"""
    dx, dy = (bx - ax) * sign, (by - ay) * sign
    return (dy < 0) | ((dy == 0) & (dx > 0))


def inside(X, Y, area, col, row):
    px, py = col * SUB + SUB // 2, row * SUB + SUB // 2
    sign = torch.sign(area)
    ok = torch.ones_like(px, dtype=torch.bool)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = _edge(X[:, a], Y[:, a], X[:, b], Y[:, b], px, py) * sign
        ok &= (e > 0) | ((e == 0) & _owns_zero(X[:, a], Y[:, a], X[:, b], Y[:, b], sign))
    return ok


def fma32(a, b, c):
    """float32 a * b + c with ONE rounding (a fused multiply-add), exact on every device that has float64: the product
    is exact in float64; TwoSum gives the float64 sum's error, which decides the rare float32 rounding ties."""
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bv = s - p
    err = (p - (s - bv)) + (cd - bv)
    r = s.float()
    rd = r.double()
    toward = torch.where(s > rd, torch.full_like(r, float("inf")), torch.full_like(r, float("-inf")))
    other = torch.nextafter(r, toward)
    tie = (s != rd) & (s == (rd + other.double()) * 0.5) & (err != 0)
    wrong = tie & ((err > 0) == (other.double() > rd))
    return torch.where(wrong, other, r)


def _cross(ax, ay, bx, by):
    return fma32(ax, by, -(ay * bx))


def barycentrics(x, y, col, row, W, H):
    """(u, v) float32 at pixel centres from the UNSNAPPED clip positions (probe: bit-exact on every covered texel).
    x, y [N, 3]: clip x, y of the winning triangle's vertices. Pixel centre in clip units (exact for power-of-two
    sizes). Each vertex relative to the centre; e_i = cross product of the other two (one fused multiply-add);
    u = e0 / (e0 + e1 + e2) as e0 * (1 / sum); then u, v are clamped to [0, 1] and scaled by 1 / (u + v) when
    u + v > 1."""
    px = (2 * col + 1).float() / W - 1
    py = (2 * row + 1).float() / H - 1
    rx, ry = x - px[:, None], y - py[:, None]
    e0 = _cross(rx[:, 1], ry[:, 1], rx[:, 2], ry[:, 2])
    e1 = _cross(rx[:, 2], ry[:, 2], rx[:, 0], ry[:, 0])
    e2 = _cross(rx[:, 0], ry[:, 0], rx[:, 1], ry[:, 1])
    inv = 1 / ((e0 + e1) + e2)
    u = (e0 * inv).clamp(0, 1)
    v = (e1 * inv).clamp(0, 1)
    s = u + v
    r = 1 / s
    over = s > 1
    return torch.where(over, u * r, u), torch.where(over, v * r, v)


def blend(u, v, a0, a1, a2):
    """attribute at (u, v) (probe: bit-exact): w = (1 - u) - v; fma(w, a2, fma(u, a0, v * a1))"""
    w = (1 - u) - v
    return fma32(w.expand_as(a2), a2, fma32(u.expand_as(a0), a0, v * a1))

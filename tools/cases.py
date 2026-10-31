"""Test inputs for the black-box probes and the synthetic stress tests. No rasterizer is imported here.

Every case is (uv, faces, T): uv float32 [V, 2] in [0, 1] like xatlas output, faces int32 [F, 3]. Callers turn uv into
clip positions the same way TRELLIS.2's to_glb does: [u*2-1, v*2-1, 0, 1] in float32 on the device.
"""
import numpy as np

CELL = 16  # pixels per cell side; one test shape per cell, so shapes in different cells never overlap


def _cells(T):
    n = T // CELL
    cy, cx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return cx.ravel(), cy.ravel()


def _to_uv(px, T):
    """pixel units (0 = left/bottom border of the grid, T = right/top border) -> uv in float32, as float64 / T"""
    return (np.asarray(px, np.float64) / T).astype(np.float32)


def _shape(kind, rng):
    """One shape in a 16 px cell, in pixel units relative to the cell's corner: (vertices [k, 2], faces [m, 3])."""
    tri = np.array([[0, 1, 2]])
    quad = np.array([[0, 1, 2], [2, 1, 3]])
    if kind == "float":         # arbitrary positions
        return rng.uniform(1, 15, (3, 2)), tri
    if kind == "centre":        # vertices on texel centres
        return rng.integers(1, 15, (3, 2)) + 0.5, tri
    if kind == "corner":        # vertices on texel corners
        return rng.integers(1, 16, (3, 2)).astype(np.float64), tri
    if kind == "q16":           # 1/16 px grid
        return rng.integers(16, 240, (3, 2)) / 16.0, tri
    if kind == "q256":          # 1/256 px grid
        return rng.integers(256, 3840, (3, 2)) / 256.0, tri
    if kind == "q4096":         # 1/4096 px grid
        return rng.integers(4096, 61440, (3, 2)) / 4096.0, tri
    if kind == "degenerate":    # zero area: collinear or repeated vertex
        a, b = rng.uniform(1, 15, 2), rng.uniform(1, 15, 2)
        third = a if rng.random() < 0.5 else a + (b - a) * rng.choice([0.25, 0.5])
        return np.stack([a, b, third]), tri
    if kind == "sliver":        # long and very thin
        a = rng.uniform(1, 4, 2); b = rng.uniform(12, 15, 2); d = b - a
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * rng.choice([1e-4, 1e-3, 1e-2, 0.1])
        return np.stack([a, b, (a + b) / 2 + n]), tri
    if kind == "row_edge":      # an edge along a row of texel centres, shared by two triangles
        y = rng.integers(3, 13) + 0.5; x0, x1 = rng.integers(1, 5), rng.integers(11, 15)
        return np.array([[x0, y - rng.uniform(1, 3)], [x0, y], [x1, y], [x1, y + rng.uniform(1, 3)]]), np.array([[0, 2, 1], [1, 2, 3]])
    if kind == "col_edge":      # an edge along a column of texel centres, shared by two triangles
        x = rng.integers(3, 13) + 0.5; y0, y1 = rng.integers(1, 5), rng.integers(11, 15)
        return np.array([[x - rng.uniform(1, 3), y0], [x, y0], [x, y1], [x + rng.uniform(1, 3), y1]]), np.array([[0, 1, 2], [1, 3, 2]])
    if kind == "diag_edge":     # a quad split on a diagonal through texel centres (|slope| 1 or 2)
        s = rng.choice([1, 2, -1, -2]); x0 = 4.5; y0 = 7.5 - 2 * s
        p0 = np.array([x0, y0]); p1 = p0 + [4, 4 * s] if abs(s) == 1 else p0 + [3, 3 * s]
        side = np.array([-(p1 - p0)[1], (p1 - p0)[0]]); side = side / np.abs(side).max() * 3
        return np.stack([p0, p1, p0 + side, p1 - side]), np.array([[0, 1, 2], [1, 0, 3]])
    if kind == "quad_float":    # a quad with a shared diagonal at arbitrary positions
        return rng.uniform(1, 15, (4, 2)), quad
    if kind == "fan_centre":    # 6 triangles around a vertex on a texel centre
        c = rng.integers(5, 11, 2) + 0.5
        ang = np.sort(rng.uniform(0, 2 * np.pi, 6)); ring = c + 4 * np.stack([np.cos(ang), np.sin(ang)], 1)
        v = np.concatenate([[c], ring]); f = np.array([[0, 1 + i, 1 + (i + 1) % 6] for i in range(6)])
        return v, f
    raise ValueError(kind)


KINDS = ["float", "centre", "corner", "q16", "q256", "q4096", "degenerate", "sliver", "row_edge", "col_edge",
         "diag_edge", "quad_float", "fan_centre"]


def cells_case(T, seed=0, kinds=KINDS, flip_some=True):
    """One shape per 16 px cell over the whole T x T grid; kinds cycle over the cells; half the triangles flipped (CW)."""
    rng = np.random.default_rng(seed)
    cx, cy = _cells(T)
    verts, faces = [], []
    nv = 0
    for i, (x, y) in enumerate(zip(cx, cy)):
        v, f = _shape(kinds[i % len(kinds)], rng)
        if flip_some and rng.random() < 0.5:
            f = f[:, [0, 2, 1]]
        verts.append(v + [x * CELL, y * CELL]); faces.append(f + nv); nv += len(v)
    px = np.concatenate(verts)
    faces = np.concatenate(faces).astype(np.int32)
    perm = rng.permutation(len(faces))  # face order unrelated to position, as in a real mesh
    return _to_uv(px, T), faces[perm]


def border_case(T, seed=1):
    """Triangles with uv exactly 0 or 1 (the grid border) and triangles that extend past it."""
    rng = np.random.default_rng(seed)
    pts, faces = [], []
    for k in range(64):
        e = rng.integers(0, 4); t = rng.uniform(0.05, 0.95)
        a = np.array([[t, 0.0], [t + 0.02, 0.0], [t + 0.01, 0.03]])
        if k % 4 == 1:
            a[:, 1] = 1 - a[:, 1]
        if k % 4 == 2:
            a = a[:, ::-1]
        if k % 4 == 3:
            a = 1 - a[:, ::-1]
        if k % 8 == 7:
            a = a * 1.05 - 0.025
        pts.append(a); faces.append(np.array([[0, 1, 2]]) + 3 * k)
    pts.append(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float64)); faces.append(np.array([[0, 1, 2]]) + 3 * 64)
    return np.concatenate(pts).astype(np.float32), np.concatenate(faces).astype(np.int32)


def big_case(T, seed=2):
    """A few large triangles (hundreds to thousands of texels per side): barycentric precision far from the vertices."""
    rng = np.random.default_rng(seed)
    uv = rng.uniform(0, 1, (12, 2)).astype(np.float32)
    return uv, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], np.int32)


def overlap_case():
    """Identical and overlapping triangles at the same depth (z = 0), for the depth tie-break."""
    base = np.array([[0.1, 0.1], [0.6, 0.15], [0.3, 0.7]])
    uv = np.concatenate([base, base, base + 0.05, base[::-1]]).astype(np.float32)
    return uv, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], np.int32)


def sweep_case(T, steps=np.arange(-512, 513), unit=1 / 4096):
    """Edges moved across a texel centre in small steps (unit px): vertical, horizontal and a slope-1/3 edge, one per
    cell. Shows the position precision (vertex snapping) and the rule for a centre exactly on an edge."""
    cx, cy = _cells(T)
    verts, faces = [], []
    i = 0
    for kind in range(3):
        for s in steps:
            d = s * unit
            if kind == 0:   # vertical left edge at x = 8.5 + d
                v = np.array([[8.5 + d, 2.0], [14.0, 8.0], [8.5 + d, 14.0]])
            elif kind == 1:  # horizontal bottom edge at y = 8.5 + d
                v = np.array([[2.0, 8.5 + d], [14.0, 8.5 + d], [8.0, 14.0]])
            else:            # slope 1/3 edge through (8.5, 8.5 + d)
                v = np.array([[2.5, 6.5 + d], [14.5, 10.5 + d], [2.5, 14.0]])
            verts.append(v + [cx[i] * CELL, cy[i] * CELL]); faces.append(np.array([[0, 1, 2]]) + 3 * i); i += 1
    return _to_uv(np.concatenate(verts), T), np.concatenate(faces).astype(np.int32)

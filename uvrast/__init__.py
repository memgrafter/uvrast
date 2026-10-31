"""uvrast: a clean-room stand-in for the two nvdiffrast calls in TRELLIS.2's GLB texture bake (o_voxel to_glb).

Same names as nvdiffrast.torch: RasterizeCudaContext, rasterize, interpolate. Only what to_glb uses is supported:
a UV layout (z = 0, w = 1), no antialiasing, no texture sampling, no gradients, no image-space derivatives.
Anything else raises, so a caller can never get a silently different result.

  import uvrast; uvrast.install()   # before o_voxel is imported: sys.modules["nvdiffrast.torch"] = uvrast
"""
import sys
import types

import torch

from . import raster


class RasterizeCudaContext:
    """No state is needed; kept for the API. device: as in nvdiffrast (None = the current CUDA device)."""
    def __init__(self, device=None):
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None and torch.cuda.is_available() \
            else torch.device(device if device is not None else "cpu") if not isinstance(device, int) else torch.device("cuda", device)


def _check(pos, tri):
    if pos.dtype != torch.float32 or tri.dtype != torch.int32:
        raise TypeError("pos must be float32 and tri int32")
    if pos.dim() != 3 or pos.shape[-1] != 4 or tri.dim() != 2 or tri.shape[1] != 3:
        raise ValueError("only instanced mode is supported: pos [minibatch, V, 4], tri [F, 3]")
    if not (bool((pos[..., 2] == 0).all()) and bool((pos[..., 3] == 1).all())):
        raise NotImplementedError("only a UV layout is supported: z = 0 and w = 1 for every vertex")
    if tri.shape[0] >= 1 << 24:
        raise ValueError("at most 2^24 - 1 triangles per call (ids are stored in float32)")


def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    """(rast [minibatch, H, W, 4] = (u, v, z/w, triangle_id + 1), db): db has 0 channels (derivatives not computed)"""
    if ranges is not None:
        raise NotImplementedError("range mode is not supported")
    _check(pos, tri)
    H, W = int(resolution[0]), int(resolution[1])
    out = torch.stack([raster.rasterize(p, tri, H, W) for p in pos])
    return out, out.new_zeros(out.shape[:3] + (0,))


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """(out [minibatch, H, W, A], empty derivatives); zero where no triangle"""
    if rast_db is not None or diff_attrs is not None:
        raise NotImplementedError("image-space derivatives are not supported")
    if attr.dim() == 2:
        raise NotImplementedError("range mode is not supported")
    a = attr.expand(rast.shape[0], -1, -1) if attr.shape[0] == 1 else attr
    out = torch.stack([raster.interpolate(a[i], rast[i], tri) for i in range(rast.shape[0])])
    return out, out.new_zeros(out.shape[:3] + (0,))


def install():
    """Make `import nvdiffrast.torch as dr` load uvrast. Call before o_voxel is imported."""
    pkg = types.ModuleType("nvdiffrast")
    pkg.torch = sys.modules[__name__]
    sys.modules["nvdiffrast"] = pkg
    sys.modules["nvdiffrast.torch"] = sys.modules[__name__]

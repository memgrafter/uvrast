"""uvrast: a clean-room stand-in for the nvdiffrast.torch calls of TRELLIS.2's GLB texture bake (o_voxel to_glb),
TRELLIS.2's texturing pipeline and MV-Adapter's texturing.

Same names as nvdiffrast.torch: RasterizeCudaContext, RasterizeGLContext, rasterize, interpolate, texture.
- rasterize: a UV layout (z = 0, w = 1 for every vertex) goes through the original path (raster.py, unchanged);
  any other clip-space input goes through clip.py (depth test, perspective-correct barycentrics).
- texture: 2D, 'nearest' or 'linear' ('auto' = 'linear'), boundary 'wrap', 'clamp' or 'zero'; no mipmaps.
- Not supported (raise): range mode, antialias, gradients, image-space derivatives, mipmaps, cube maps, triangles
  in view with a vertex at w <= 0. A caller never gets a silently different result.

  import uvrast; uvrast.install()   # before o_voxel is imported: sys.modules["nvdiffrast.torch"] = uvrast
"""
import sys
import types

import torch

from . import clip, raster


class RasterizeCudaContext:
    """No state is needed; kept for the API. device: as in nvdiffrast (None = the current CUDA device)."""
    def __init__(self, device=None):
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None and torch.cuda.is_available() \
            else torch.device(device if device is not None else "cpu") if not isinstance(device, int) else torch.device("cuda", device)


class RasterizeGLContext(RasterizeCudaContext):
    """Same as RasterizeCudaContext (no OpenGL is used); output_db and mode are accepted for the API only."""
    def __init__(self, output_db=True, mode="automatic", device=None):
        super().__init__(device)


def _check(pos, tri):
    """True for a UV layout (z = 0 and w = 1 for every vertex), False for general clip-space input"""
    if pos.dtype != torch.float32 or tri.dtype != torch.int32:
        raise TypeError("pos must be float32 and tri int32")
    if pos.dim() != 3 or pos.shape[-1] != 4 or tri.dim() != 2 or tri.shape[1] != 3:
        raise ValueError("only instanced mode is supported: pos [minibatch, V, 4], tri [F, 3]")
    if tri.shape[0] >= 1 << 24:
        raise ValueError("at most 2^24 - 1 triangles per call (ids are stored in float32)")
    return bool((pos[..., 2] == 0).all()) and bool((pos[..., 3] == 1).all())


def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    """(rast [minibatch, H, W, 4] = (u, v, z/w, triangle_id + 1), db): db has 0 channels (derivatives not computed)"""
    if ranges is not None:
        raise NotImplementedError("range mode is not supported")
    uv_layout = _check(pos, tri)
    H, W = int(resolution[0]), int(resolution[1])
    fn = raster.rasterize if uv_layout else clip.rasterize
    out = torch.stack([fn(p, tri, H, W) for p in pos])
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


def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode="auto", boundary_mode="wrap", max_mip_level=None):
    """2D texture sampling without mipmaps: [minibatch, H, W, C]"""
    if uv_da is not None or mip_level_bias is not None or mip is not None:
        raise NotImplementedError("mipmaps (uv_da, mip_level_bias, mip) are not supported")
    if filter_mode == "auto":
        filter_mode = "linear"
    if filter_mode not in ("linear", "nearest"):
        raise NotImplementedError(f"filter_mode {filter_mode!r} is not supported (only 'linear', 'nearest', 'auto')")
    if boundary_mode not in ("wrap", "clamp", "zero"):
        raise NotImplementedError(f"boundary_mode {boundary_mode!r} is not supported (cube maps are not)")
    if tex.dim() != 4 or uv.dim() != 4 or uv.shape[-1] != 2:
        raise ValueError("only 2D textures: tex [minibatch, H, W, C], uv [minibatch, H, W, 2]")
    return clip.texture(tex.float(), uv, filter_mode, boundary_mode)


def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    raise NotImplementedError("antialias is not supported by uvrast")


def install():
    """Make `import nvdiffrast.torch as dr` load uvrast. Call before o_voxel is imported."""
    pkg = types.ModuleType("nvdiffrast")
    pkg.torch = sys.modules[__name__]
    sys.modules["nvdiffrast"] = pkg
    sys.modules["nvdiffrast.torch"] = sys.modules[__name__]

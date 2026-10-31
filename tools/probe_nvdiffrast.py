"""Black-box probes of nvdiffrast through its public Python API (EVALUATION venv only; NVIDIA's licence allows
"research or evaluation purposes"). This file never reads nvdiffrast's source; it only calls
nvdiffrast.torch.RasterizeCudaContext, rasterize and interpolate, and saves inputs and outputs for analysis.

  python -I tools/probe_nvdiffrast.py OUT_DIR

Writes OUT_DIR/<case>.npz with the inputs (uv, faces, T) and nvdiffrast's outputs. The outputs stay outside this
repository: they are evaluation data, used only to find and check the behaviour.
"""
import os
import sys

import numpy as np
import torch
import nvdiffrast.torch as dr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases  # noqa: E402

CHUNK = 100000  # the chunk size TRELLIS.2's to_glb uses


def clip_pos(uv):
    """to_glb's own expression: [u*2-1, v*2-1, 0, 1], float32 on the GPU"""
    uvs = torch.from_numpy(uv).cuda()
    return torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=-1).unsqueeze(0)


def bake_rast(ctx, uv, faces, T):
    """to_glb's chunk loop, as written in TRELLIS.2 (MIT)"""
    pos = clip_pos(uv)
    f = torch.from_numpy(faces).cuda()
    rast = torch.zeros((1, T, T, 4), device="cuda", dtype=torch.float32)
    for i in range(0, f.shape[0], CHUNK):
        rc, _ = dr.rasterize(ctx, pos, f[i:i + CHUNK], resolution=[T, T])
        m = rc[..., 3:4] > 0
        rc[..., 3:4] += i
        rast = torch.where(m, rc, rast)
    return pos, f, rast


def run_case(ctx, out, name, uv, faces, T, attr=None):
    pos, f, rast = bake_rast(ctx, uv, faces, T)
    _, _, rast2 = bake_rast(ctx, uv, faces, T)
    same = bool(torch.equal(rast, rast2))
    if attr is None:
        attr = torch.cat([pos[0, :, :2], torch.from_numpy(np.random.default_rng(5).normal(0, 0.3, (len(uv), 1)).astype(np.float32)).cuda()], 1)
    a = attr.unsqueeze(0).contiguous()
    ip = dr.interpolate(a, rast, f)[0]
    single = None
    if len(faces) <= CHUNK:
        single = dr.rasterize(ctx, pos, f, resolution=[T, T])[0]
    keep = rast[0].reshape(-1, 4)
    cov = torch.nonzero(keep[:, 3] > 0).squeeze(1)
    np.savez_compressed(os.path.join(out, f"{name}.npz"), uv=uv, faces=faces, T=T, pos=pos[0].cpu().numpy(),
                        attr=attr.cpu().numpy(), cov_idx=cov.cpu().numpy(), rast_cov=keep[cov].cpu().numpy(),
                        interp_cov=ip[0].reshape(-1, ip.shape[-1])[cov].cpu().numpy(), deterministic=same,
                        single_equal=(None if single is None else bool(torch.equal(single, rast))))
    print(f"{name}: T={T} faces={len(faces)} covered={cov.numel()} deterministic={same}", flush=True)


def interp_probe(ctx, out):
    """interpolate() on synthetic rast values: finds its arithmetic (order of operations) bit for bit"""
    g = torch.Generator(device="cpu").manual_seed(7)
    H, W, V, Fn = 256, 256, 3000, 1000
    attr = (torch.randn(1, V, 3, generator=g) * torch.tensor([0.5, 3.0, 100.0])).cuda()
    tri = torch.randint(0, V, (Fn, 3), generator=g, dtype=torch.int32).cuda()
    u = torch.rand(1, H, W, generator=g); v = torch.rand(1, H, W, generator=g) * (1 - u)
    ids = torch.randint(0, Fn + 1, (1, H, W), generator=g).float()
    rast = torch.stack([u, v, torch.zeros_like(u), ids], -1).cuda().contiguous()
    ip = dr.interpolate(attr, rast, tri)[0]
    np.savez_compressed(os.path.join(out, "interp.npz"), attr=attr.cpu().numpy(), tri=tri.cpu().numpy(),
                        rast=rast.cpu().numpy(), out=ip.cpu().numpy())
    print("interp: done", flush=True)


def centres_probe(ctx, out):
    """A non-square grid covered by two big triangles: interpolating x and y gives each pixel's sample point"""
    pos = torch.tensor([[[-3, -3, 0, 1], [3, -3, 0, 1], [-3, 3, 0, 1], [3, 3, 0, 1]]], dtype=torch.float32).cuda()
    tri = torch.tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.int32).cuda()
    rast = dr.rasterize(ctx, pos, tri, resolution=[6, 10])[0]
    ip = dr.interpolate(pos[..., :2].contiguous(), rast, tri)[0]
    np.savez_compressed(os.path.join(out, "centres.npz"), rast=rast.cpu().numpy(), xy=ip.cpu().numpy())
    print("centres:", ip[0, :, 0, 1].cpu().numpy(), ip[0, 0, :, 0].cpu().numpy(), flush=True)


def depth_probe(ctx, out):
    """Overlapping triangles with equal and different z, in one call"""
    uv, faces = cases.overlap_case()
    res = {}
    for zs in ([0, 0, 0, 0], [0.5, 0.2, 0.0, 0.2], [0.2, 0.5, 0.5, 0.0]):
        pos = clip_pos(uv)
        pos[0, :, 2] = torch.tensor(np.repeat(zs, 3), dtype=torch.float32).cuda()
        for order in ("fwd", "rev"):
            f = torch.from_numpy(faces if order == "fwd" else faces[::-1].copy()).cuda()
            r = [dr.rasterize(ctx, pos, f, resolution=[64, 64])[0] for _ in range(3)]
            res[f"z{zs}_{order}"] = r[0][0, ..., 3].cpu().numpy()
            res[f"z{zs}_{order}_det"] = all(torch.equal(r[0], x) for x in r[1:])
            res[f"z{zs}_{order}_z"] = r[0][0, ..., 2].cpu().numpy()
    np.savez_compressed(os.path.join(out, "depth.npz"), **res)
    print("depth: done", flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ctx = dr.RasterizeCudaContext()
    centres_probe(ctx, out)
    interp_probe(ctx, out)
    depth_probe(ctx, out)
    for T in (1024, 2048, 4096):
        uv, f = cases.cells_case(T, seed=T)
        run_case(ctx, out, f"cells{T}", uv, f, T)
        uv, f = cases.border_case(T)
        run_case(ctx, out, f"border{T}", uv, f, T)
        uv, f = cases.big_case(T)
        run_case(ctx, out, f"big{T}", uv, f, T)
    uv, f = cases.sweep_case(4096)
    run_case(ctx, out, "sweep4096", uv, f, 4096)
    uv, f = cases.sweep_case(1024, unit=1 / 1024)
    run_case(ctx, out, "sweep1024", uv, f, 1024)


if __name__ == "__main__":
    main()

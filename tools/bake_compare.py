# prepare(), bake() and glb_mesh() below are copied from TRELLIS.2 (o-voxel/o_voxel/postprocess.py), under this licence:
#
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

"""Equality proof: bake the same UV mesh with nvdiffrast and with uvrast, and compare every intermediate and output.
EVALUATION venv only (nvdiffrast: NVIDIA, "research or evaluation purposes only"); outputs stay outside this repo.

  python -I tools/bake_compare.py --mesh DECODED.pt --out DIR [--sizes 1024,2048,4096] [--decimation 500000] [--glb 4096]

Protocol: the decoded TRELLIS.2 mesh is cleaned, remeshed, simplified and UV-unwrapped ONCE
(CuMesh), then baked twice at each texture size, once per rasterizer, in the same process with the same BVH, so only
the rasterizer differs. The two halves of o_voxel.postprocess.to_glb are copied below from TRELLIS.2 @ 75fbf018
(o-voxel/o_voxel/postprocess.py, MIT License, Copyright (c) Microsoft Corporation), unchanged except that the
rasterizer module is a parameter and the intermediate results are returned.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import cv2
import trimesh
import trimesh.visual
from PIL import Image
import cumesh
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as nvdr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import uvrast  # noqa: E402
from uvrast import rules  # noqa: E402


def prepare(vertices, faces, aabb, voxel_size, decimation_target, remesh=True, remesh_band=1, remesh_project=0):
    """to_glb up to and including the UV unwrap (TRELLIS.2, MIT)"""
    aabb = torch.tensor(np.array(aabb), dtype=torch.float32, device="cuda")
    voxel_size = torch.tensor(np.array([voxel_size] * 3), dtype=torch.float32, device="cuda")
    grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    vertices = vertices.cuda(); faces = faces.cuda()
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()
    bvh = cumesh.cuBVH(vertices, faces)
    assert remesh
    center = aabb.mean(dim=0)
    scale = (aabb[1] - aabb[0]).max().item()
    resolution = grid_size.max().item()
    mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(vertices, faces, center=center,
                                                       scale=(resolution + 3 * remesh_band) / resolution * scale,
                                                       resolution=resolution, band=remesh_band, project_back=remesh_project,
                                                       verbose=False, bvh=bvh))
    mesh.simplify(decimation_target, verbose=False)
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={"threshold_cone_half_angle_rad": np.radians(90.0), "refine_iterations": 0,
                               "global_iterations": 1, "smooth_strength": 1},
        return_vmaps=True, verbose=False)
    out_vertices, out_faces, out_uvs, out_vmaps = out_vertices.cuda(), out_faces.cuda(), out_uvs.cuda(), out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    return dict(vertices=vertices, faces=faces, bvh=bvh, aabb=aabb, voxel_size=voxel_size, grid_size=grid_size,
                out_vertices=out_vertices, out_faces=out_faces, out_uvs=out_uvs, out_normals=out_normals)


def bake(dr, p, attr_volume, coords, attr_layout, texture_size):
    """to_glb's texture bake (TRELLIS.2, MIT), with timings of the two rasterizer calls"""
    out_vertices, out_faces, out_uvs = p["out_vertices"], p["out_faces"], p["out_uvs"]
    torch.cuda.synchronize(); t0 = time.time()
    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device="cuda", dtype=torch.float32)
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(ctx, uvs_rast, out_faces[i:i + 100000], resolution=[texture_size, texture_size])
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)
    torch.cuda.synchronize(); t1 = time.time()
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    torch.cuda.synchronize(); t2 = time.time()
    valid_pos = pos[mask]
    _, face_id, uvw = p["bvh"].unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = p["vertices"][p["faces"][face_id.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device="cuda")
    attrs[mask] = grid_sample_3d(attr_volume, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                                 shape=torch.Size([1, attr_volume.shape[1], *p["grid_size"].tolist()]),
                                 grid=((valid_pos - p["aabb"][0]) / p["voxel_size"]).reshape(1, -1, 3), mode="trilinear")
    torch.cuda.synchronize(); t3 = time.time()
    mask_np = mask.cpu().numpy()
    raw = {}
    for k in ("base_color", "metallic", "roughness", "alpha"):
        raw[k] = np.clip(attrs[..., attr_layout[k]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    mask_inv = (~mask_np).astype(np.uint8)
    final = dict(base_color=cv2.inpaint(raw["base_color"], mask_inv, 3, cv2.INPAINT_TELEA),
                 metallic=cv2.inpaint(raw["metallic"], mask_inv, 1, cv2.INPAINT_TELEA)[..., None],
                 roughness=cv2.inpaint(raw["roughness"], mask_inv, 1, cv2.INPAINT_TELEA)[..., None],
                 alpha=cv2.inpaint(raw["alpha"], mask_inv, 1, cv2.INPAINT_TELEA)[..., None])
    t4 = time.time()
    times = dict(rasterize_s=t1 - t0, interpolate_s=t2 - t1, bvh_and_sample_s=t3 - t2, maps_and_inpaint_s=t4 - t3)
    return dict(rast=rast[0], pos=pos, attrs=attrs, raw=raw, final=final, times=times)


def glb_mesh(p, final, remesh=True):
    """to_glb's material and mesh (TRELLIS.2, MIT)"""
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([final["base_color"], final["alpha"]], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(final["metallic"]), final["roughness"], final["metallic"]], axis=-1)),
        metallicFactor=1.0, roughnessFactor=1.0, alphaMode="OPAQUE", doubleSided=True if not remesh else False)
    v = p["out_vertices"].cpu().numpy().copy(); n = p["out_normals"].cpu().numpy().copy(); uv = p["out_uvs"].cpu().numpy().copy()
    v[:, 1], v[:, 2] = v[:, 2], -v[:, 1]
    n[:, 1], n[:, 2] = n[:, 2], -n[:, 1]
    uv[:, 1] = 1 - uv[:, 1]
    return trimesh.Trimesh(vertices=v, faces=p["out_faces"].cpu().numpy(), vertex_normals=n, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))


def edge_zero(X, Y, col, row):
    px, py = col * rules.SUB + rules.SUB // 2, row * rules.SUB + rules.SUB // 2
    z = torch.zeros_like(px, dtype=torch.bool)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        z |= ((X[:, b] - X[:, a]) * (py - Y[:, a]) - (Y[:, b] - Y[:, a]) * (px - X[:, a])) == 0
    return z


def compare(p, A, B, T, voxel):
    ra, rb = A["rast"], B["rast"]
    ia, ib = ra[..., 3].long(), rb[..., 3].long()
    ca, cb = ia > 0, ib > 0
    both = ca & cb
    res = dict(T=T, covered_nv=int(ca.sum()), covered_uv=int(cb.sum()), cover_only_nv=int((ca & ~cb).sum()),
               cover_only_uv=int((cb & ~ca).sum()), id_mismatch=int((both & (ia != ib)).sum()))
    # texels that differ: is the centre exactly on an edge of either triangle (the only place a tie rule matters)?
    diff = (ca != cb) | (both & (ia != ib))
    r, c = torch.nonzero(diff, as_tuple=True)
    fx = rules.to_fixed(p["out_uvs"][:, 0] * 2 - 1, T); fy = rules.to_fixed(p["out_uvs"][:, 1] * 2 - 1, T)
    tri = p["out_faces"].long()
    tie = torch.zeros(r.numel(), dtype=torch.bool, device=r.device)
    for ids in (ia[r, c], ib[r, c]):
        k = ids > 0
        t = tri[(ids[k] - 1)]
        tie[k] |= edge_zero(fx[t], fy[t], c[k], r[k])
    res["mismatch_on_edge"] = int(tie.sum()); res["mismatch_not_on_edge"] = int((~tie).sum())
    same = both & (ia == ib)
    for ch, name in ((0, "u"), (1, "v")):
        d = (ra[..., ch] - rb[..., ch]).abs()[same]
        res[f"bary_{name}_max_abs"] = float(d.max()) if d.numel() else 0.0
        res[f"bary_{name}_bit_equal"] = int((ra[..., ch][same] == rb[..., ch][same]).sum())
    dp = (A["pos"] - B["pos"]).abs()[same]
    res["matching_texels"] = int(same.sum())
    res["pos_max_abs"] = float(dp.max()) if dp.numel() else 0.0
    res["pos_max_voxels"] = res["pos_max_abs"] / voxel
    res["pos_bit_equal_texels"] = int((A["pos"][same] == B["pos"][same]).all(1).sum())
    for stage in ("raw", "final"):
        for k in ("base_color", "metallic", "roughness", "alpha"):
            a = A[stage][k].astype(np.int16); b = B[stage][k].astype(np.int16)
            d = np.abs(a - b).max(axis=-1) if a.ndim == 3 else np.abs(a - b)
            d = d.reshape(T, T)
            mse = float(((a - b).astype(np.float64) ** 2).mean())
            res[f"{stage}.{k}"] = dict(over_1=int((d > 1).sum()), differ=int((d > 0).sum()), max=int(d.max()),
                                       psnr=(float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="1024,2048,4096")
    ap.add_argument("--decimation", type=int, default=500000)
    ap.add_argument("--glb", type=int, default=4096, help="texture size of the two GLBs written (0 = none)")
    ap.add_argument("--repeat", type=int, default=3, help="timed bakes per rasterizer at the largest size")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = torch.load(a.mesh, weights_only=False)
    t = time.time()
    p = prepare(m["vertices"], m["faces"], [[-0.5] * 3, [0.5] * 3], m["voxel_size"], a.decimation)
    prep_s = time.time() - t
    np.savez(os.path.join(a.out, "unwrap.npz"), vertices=p["out_vertices"].cpu().numpy(), faces=p["out_faces"].cpu().numpy(),
             uvs=p["out_uvs"].cpu().numpy())
    attrs, coords = m["attrs"].cuda(), m["coords"].cuda()
    report = dict(mesh=a.mesh, prop=m.get("prop"), decimation=a.decimation, faces=int(p["out_faces"].shape[0]),
                  vertices=int(p["out_vertices"].shape[0]), prepare_s=prep_s, voxel_size=float(m["voxel_size"]), sizes={})
    for T in [int(s) for s in a.sizes.split(",")]:
        A = bake(nvdr, p, attrs, coords, m["layout"], T)
        B = bake(uvrast, p, attrs, coords, m["layout"], T)
        r = compare(p, A, B, T, float(m["voxel_size"]))
        r["times_nvdiffrast"], r["times_uvrast"] = A["times"], B["times"]
        if T == max(int(s) for s in a.sizes.split(",")):
            for name, dr in (("nvdiffrast", nvdr), ("uvrast", uvrast)):
                ts = [bake(dr, p, attrs, coords, m["layout"], T)["times"] for _ in range(a.repeat)]
                r[f"repeat_{name}"] = ts
        if T == a.glb:
            glb_mesh(p, A["final"]).export(os.path.join(a.out, f"nvdiffrast-{T}.glb"))
            glb_mesh(p, B["final"]).export(os.path.join(a.out, f"uvrast-{T}.glb"))
            np.save(os.path.join(a.out, f"diffmask-{T}.npy"), ((A["rast"][..., 3] != B["rast"][..., 3])).cpu().numpy())
        report["sizes"][str(T)] = r
        print(json.dumps({k: v for k, v in r.items() if not k.startswith("repeat")}), flush=True)
        del A, B
        torch.cuda.empty_cache()
    json.dump(report, open(os.path.join(a.out, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

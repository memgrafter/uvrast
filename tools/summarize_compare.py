"""One table from many bake_compare.py results.

  python tools/summarize_compare.py CMP_DIR
"""
import glob
import json
import os
import sys

MAPS = ("base_color", "metallic", "roughness", "alpha")


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "compare.json"))):
        r = json.load(open(f))
        for T, s in sorted(r["sizes"].items(), key=lambda kv: int(kv[0])):
            mism = s["cover_only_nv"] + s["cover_only_uv"] + s["id_mismatch"]
            raw = sum(s[f"raw.{k}"]["differ"] for k in MAPS)
            over = sum(s[f"final.{k}"]["over_1"] for k in MAPS)
            differ = sum(s[f"final.{k}"]["differ"] for k in MAPS)
            psnr = min(s[f"final.{k}"]["psnr"] for k in MAPS)
            rn = [x["rasterize_s"] + x["interpolate_s"] for x in s.get("repeat_nvdiffrast", [])]
            ru = [x["rasterize_s"] + x["interpolate_s"] for x in s.get("repeat_uvrast", [])]
            rows.append(dict(mesh=os.path.basename(os.path.dirname(f)), faces=r["faces"], T=int(T), covered=s["covered_nv"],
                             mismatch=mism, on_edge=s["mismatch_on_edge"], pos_equal=s["pos_bit_equal_texels"] == s["matching_texels"],
                             pos_max_voxels=s["pos_max_voxels"], raw_texels_differ=raw, final_texels_differ=differ,
                             final_over_1=over, final_min_psnr=psnr,
                             nv_ms=min(rn) * 1000 if rn else None, uv_ms=min(ru) * 1000 if ru else None))
    hdr = f"{'mesh':26s} {'faces':>7s} {'T':>5s} {'covered':>9s} {'id/cov diff':>11s} {'pos exact':>9s} {'raw diff':>8s} {'final diff':>10s} {'>1/255':>6s} {'min PSNR':>8s} {'nv ms':>6s} {'uv ms':>6s}"
    print(hdr)
    for x in rows:
        nv = f"{x['nv_ms']:6.1f}" if x["nv_ms"] is not None else "     -"
        uv = f"{x['uv_ms']:6.1f}" if x["uv_ms"] is not None else "     -"
        print(f"{x['mesh']:26s} {x['faces']:7d} {x['T']:5d} {x['covered']:9d} {x['mismatch']:11d} {str(x['pos_equal']):>9s} "
              f"{x['raw_texels_differ']:8d} {x['final_texels_differ']:10d} {x['final_over_1']:6d} {x['final_min_psnr']:8.1f} {nv} {uv}")
    tot = sum(x["covered"] for x in rows)
    print(f"total covered texels {tot}, id/coverage mismatches {sum(x['mismatch'] for x in rows)}, "
          f"bakes with any final-map texel over 1/255: {sum(x['final_over_1'] > 0 for x in rows)} of {len(rows)}")


if __name__ == "__main__":
    main()

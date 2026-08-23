#!/usr/bin/env python3
"""VGGT usability gate for the v2 unified-lift design.

The premise under test: a nadir satellite crop, fed to VGGT as one more posed
view alongside the street views, yields depth whose structure reflects real
building heights (nadir depth ~= camera height minus structure height), and
VGGT's estimated camera geometry is consistent with our calibrated world.

Reads one cached tile (street views + satellite + dense LiDAR points), runs a
single VGGT forward over all views, and reports:
  1. Pearson correlation of the satellite-view depth map (resampled to the BEV
     grid) vs the dense-LiDAR h_mean map, on covered cells.  |r| >= ~0.3 with
     the expected sign is the go signal for v2.
  2. Camera-center RMSE after best similarity alignment of VGGT extrinsics to
     GT world poses (gauge sanity for per-tile metric anchoring).
  3. Wall time and GPU memory for a full-tile forward.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/shizhm/Lenovo/vggt")

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from world3d.unified_bev.data import load_cached_unified_bev
from world3d.unified_bev.geometry import height_statistics

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    denom = float(a.norm() * b.norm())
    return float((a * b).sum() / denom) if denom > 1e-8 else 0.0


def umeyama_rigid(src: torch.Tensor, dst: torch.Tensor) -> float:
    """RMSE of dst after best similarity fit of src->dst (both (N,3))."""
    src_c, dst_c = src - src.mean(0), dst - dst.mean(0)
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, s, vh = torch.linalg.svd(cov)
    d = torch.sign(torch.det(u @ vh))
    dvec = torch.cat([torch.ones_like(s[:2]), d[None]])
    r = u @ torch.diag(dvec) @ vh
    scale = float((s * dvec).sum() / (src_c ** 2).sum())
    aligned = (scale * (src_c @ r.T)) + dst.mean(0)
    return float(((aligned - dst) ** 2).sum(dim=1).sqrt().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/shizhm/Downloads/vggt.pt")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max_views", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--no_sat", action="store_true", help="street views only (isolate satellite-view poisoning)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    model = VGGT().to(device)
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.eval()

    ds = load_cached_unified_bev(args.cache)
    s = ds[args.index]
    views = s["source_rgb"][: args.max_views]          # (V,3,96,160) in [0,1]
    sat = s["satellite"]                                # (3,512,512) in [0,1]
    n_street = views.shape[0]

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    parts = [F.interpolate(views, size=(args.resolution, args.resolution), mode="bilinear")]
    if not args.no_sat:
        parts.append(F.interpolate(sat[None], size=(args.resolution, args.resolution), mode="bilinear"))
    batch = torch.cat(parts, dim=0)
    batch = ((batch - mean) / std).to(device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        tokens, ps_idx = model.aggregator(batch[None])
        depth_map, depth_conf = model.depth_head(tokens, batch[None], ps_idx)
        pose_enc = model.camera_head(tokens)[-1]
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    extri, _ = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
    extri = extri[0].float().cpu()
    depth_map = depth_map[0].float().cpu()

    # --- Gate 1: nadir-view depth vs dense-LiDAR height map ---
    bev = 128
    nadir = F.interpolate(depth_map[-1, ..., 0][None, None], size=(bev, bev), mode="bilinear")[0, 0]
    pts = s["source_points_world"][: args.max_views]
    valid = s["source_points_valid"][: args.max_views]
    flat_p = pts[valid]
    h_mean, _ = height_statistics(
        flat_p[None, None], torch.ones(1, flat_p.shape[0], dtype=torch.bool),
        s["origin_xy"][None], 0.5, bev, bev,
    )
    covered = (h_mean[0, 0] != 0)
    r = pearson(nadir[covered], h_mean[0, 0][covered])
    r_all = pearson(nadir, h_mean[0, 0])

    # --- Gate 2: camera geometry vs GT world poses ---
    centers_vggt = torch.stack([
        -extri[i, :3, :3].T @ extri[i, :3, 3] for i in range(n_street)
    ])
    gt = s["source_T_world_cam"][:n_street, :3, 3]
    rmse = umeyama_rigid(centers_vggt, gt)

    print(f"tile_index={args.index} street_views={n_street}+sat resolution={args.resolution}")
    print(f"forward: {elapsed:.1f}s, peak GPU {peak_gb:.1f} GB")
    print(f"GATE1 nadir-depth vs LiDAR h_mean: pearson_r={r:+.3f} (covered cells "
          f"{int(covered.sum())}/{bev*bev}; r_all={r_all:+.3f})  [expect negative: depth~height-H]")
    print(f"GATE2 camera centers after sim-align: RMSE={rmse:.2f} m over {n_street} views")
    conf_street = depth_conf[0, :n_street, ..., 0] if depth_conf.dim() == 4 else depth_conf[0, :n_street]
    print(f"depth_conf: street mean={float(conf_street.mean()):.1f}"
          + (f", sat mean={float(depth_conf[0,-1,...,0].mean() if depth_conf.dim()==4 else depth_conf[0,-1].mean()):.1f}" if not args.no_sat else " (no_sat)"))


if __name__ == "__main__":
    main()

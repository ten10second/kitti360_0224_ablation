#!/usr/bin/env python3
"""Experiment 2: Vertical Complexity Stratification.

Does the satellite contribution grow with scene vertical complexity?  Theory:
the cross-view gap is elevation-induced, so Gain_sat = M(sat) - M(xy) should
slope upward from flat regions (roads, sigma(h) ~ 0) to high-vertical regions
(building edges, tall sigma(h)) -- if instead the gain is flat across strata,
the satellite is supplying generic cues, not elevation information.

Strata are defined per BEV cell by local height spread: sigma over the 3x3
neighborhood of the dense-LiDAR relative-height DEM (the observable the
recovery is judged against).  Metrics per stratum: probe Pearson gain and
latent-L1 gain (both sat minus xy) plus the probe error reduction vs gnd.
Pure analysis over the saved probe model + frozen checkpoints on unseen 0003.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import UnifiedBEVDataset, load_cached_unified_bev
from world3d.unified_bev.geometry import height_statistics
from world3d.unified_bev.models import (
    GroundBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
)
from scripts.probe_unified_bev_height import HeightProbe, dem_from_sample, spearman


def local_sigma(h: torch.Tensor) -> torch.Tensor:
    x = h[None, None]
    pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    sq = torch.zeros(1, 9, *h.shape)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sq[0, k] = pad[0, 0, 1 + dy:1 + dy + h.shape[0], 1 + dx:1 + dx + h.shape[1]]
            k += 1
    return sq[0].std(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b_sat", required=True)
    ap.add_argument("--stage_b_xy", required=True)
    ap.add_argument("--probe_ckpt", required=True, help="saved probe weights from Exp1")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--n_sparse", type=int, default=1)
    ap.add_argument("--eval_manifest", default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl")
    ap.add_argument("--eval_drive", default="2013_05_28_drive_0003_sync")
    ap.add_argument("--eval_samples", type=int, default=32)
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--strata", default="0.15,0.45", help="sigma(h) thresholds flat/med/high")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    th = [float(x) for x in args.strata.split(",")]

    ds = load_cached_unified_bev(args.cache)
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    ground.load_state_dict(a["ground"]); ground.eval()
    bs = torch.load(args.stage_b_sat, map_location=device, weights_only=False)
    bx = torch.load(args.stage_b_xy, map_location=device, weights_only=False)
    sat = HeightMapSatellitePrior(bev_height=ds.bev_size, bev_width=ds.bev_size,
                                  **bs.get("config", {}).get("sat_encoder_kwargs", {})).to(device)
    sat.load_state_dict(bs["satellite_encoder"]); sat.eval()
    comp_sat = LatentCompletion(mode="residual", bev_height=ds.bev_size,
                                bev_width=ds.bev_size, tile_size_m=ds.tile_size_m).to(device)
    comp_sat.load_state_dict(bs["completion"]); comp_sat.eval()
    comp_xy = LatentCompletion(mode="coordinate_only", bev_height=ds.bev_size,
                               bev_width=ds.bev_size, tile_size_m=ds.tile_size_m).to(device)
    comp_xy.load_state_dict(bx["completion"]); comp_xy.eval()
    probe = HeightProbe().to(device)
    probe.load_state_dict(torch.load(args.probe_ckpt, map_location=device, weights_only=False))
    probe.eval()

    ev = UnifiedBEVDataset(
        args.eval_manifest, lidar_root=args.lidar_root, drive=args.eval_drive,
        min_target_spacing_m=5.0, max_samples=args.eval_samples,
        dense_source_count=8, sparse_source_count=args.n_sparse,
        image_size=ds.image_size, max_points_per_view=4096,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ev, batch_size=1, shuffle=False, num_workers=0)

    names = ("flat", "medium", "high")
    acc = {n: {"r_sat": [], "r_xy": [], "r_gnd": [], "lat_sat": [], "lat_xy": [],
               "mae_sat": [], "mae_xy": [], "cells": 0} for n in names}
    with torch.no_grad():
        for batch in loader:
            s = {k: (v[0] if torch.is_tensor(v) else v) for k, v in batch.items()}
            dem, covered = dem_from_sample(s, ds.bev_resolution_m, ds.bev_size)
            if not covered.any():
                continue
            views = s["source_rgb"][None].to(device)
            pts = s["source_points_world"][None].to(device)
            uv = s["source_points_uv"][None].to(device)
            val = s["source_points_valid"][None].to(device)
            org = s["origin_xy"][None].to(device)
            z_star, _ = ground(views, pts, uv, val, org, ds.bev_resolution_m)
            sp = slice(0, args.n_sparse * ds.views_per_frame)
            z_gnd, cov = ground(views[:, sp], pts[:, sp], uv[:, sp], val[:, sp],
                                org, ds.bev_resolution_m)
            prior, _, _ = sat(s["satellite"][None].to(device), z_gnd, ds.tile_size_m, 0.196)
            z_sat = comp_sat(prior, z_gnd, cov, args.n_sparse, 8)
            z_xy = comp_xy(torch.zeros_like(prior), z_gnd, cov, args.n_sparse, 8)

            sigma = local_sigma(dem)
            stratum = torch.zeros_like(sigma, dtype=torch.long)
            stratum[sigma > th[0]] = 1
            stratum[sigma > th[1]] = 2
            dem_d = dem.to(device)
            p_sat = probe(z_sat)[0, 0]
            p_xy = probe(z_xy)[0, 0]
            p_gnd = probe(z_gnd)[0, 0]
            lat_sat = (z_sat[0] - z_star[0]).abs().mean(dim=0)
            lat_xy = (z_xy[0] - z_star[0]).abs().mean(dim=0)
            m = covered.to(device)

            def pearson_at(p, t, mask):
                a = p[mask] - p[mask].mean()
                b = t[mask] - t[mask].mean()
                d = float(a.norm() * b.norm())
                return float((a * b).sum() / d) if d > 1e-8 else 0.0

            for si, name in enumerate(names):
                sel = m & (stratum.to(device) == si)
                if int(sel.sum()) < 16:
                    continue
                acc[name]["r_sat"].append(pearson_at(p_sat, dem_d, sel))
                acc[name]["r_xy"].append(pearson_at(p_xy, dem_d, sel))
                acc[name]["r_gnd"].append(pearson_at(p_gnd, dem_d, sel))
                acc[name]["mae_sat"].append(float((p_sat[sel] - dem_d[sel]).abs().mean()))
                acc[name]["mae_xy"].append(float((p_xy[sel] - dem_d[sel]).abs().mean()))
                acc[name]["lat_sat"].append(float(lat_sat[sel].mean()))
                acc[name]["lat_xy"].append(float(lat_xy[sel].mean()))
                acc[name]["cells"] += int(sel.sum())

    m = lambda xs: sum(xs) / max(1, len(xs))
    print(f"=== Vertical stratification on {args.eval_drive} (Ns={args.n_sparse}, "
          f"sigma thresholds {th}) ===")
    print(f"{'stratum':>7s} {'cells':>7s} {'r_sat':>7s} {'r_xy':>7s} {'dR':>7s} "
          f"{'mae_sat':>8s} {'mae_xy':>8s} {'dMAE':>7s} {'lat_sat':>8s} {'lat_xy':>8s}")
    for name in names:
        r = acc[name]
        if not r["r_sat"]:
            print(f"{name:>7s}  (too few cells)")
            continue
        print(f"{name:>7s} {r['cells']:>7d} {m(r['r_sat']):>+7.3f} {m(r['r_xy']):>+7.3f} "
              f"{m(r['r_sat'])-m(r['r_xy']):>+7.3f} {m(r['mae_sat']):>8.3f} {m(r['mae_xy']):>8.3f} "
              f"{m(r['mae_xy'])-m(r['mae_sat']):>+7.3f} {m(r['lat_sat']):>8.3f} {m(r['lat_xy']):>8.3f}")


if __name__ == "__main__":
    main()

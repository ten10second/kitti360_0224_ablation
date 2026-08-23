#!/usr/bin/env python3
"""Diagnostic: decoder readout gain along the satellite correction direction.

For each tile the completion writes a correction c = z_hat - z_gnd.  We
compare the frozen decoder's response to three latent perturbations of equal
L1 norm added to z_gnd:

  c_sat    : the correction the trained completion actually wrote
  c_ideal  : the direction z* - z_gnd (what full recovery would add)
  c_rand   : a random direction (the decoder's average sensitivity)

gain = ||render(z_gnd + c) - render(z_gnd)||_1 / ||c||_1

If gain(c_sat) << gain(c_ideal) as Ns grows, the satellite correction is
written into decoder-insensitive directions -- a direct measurement of the
latent-vs-render dialect mismatch behind the poor exchange rate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    LatentCompletion,
    SatelliteBEVEncoder,
)


def l1(t: torch.Tensor) -> float:
    return float(t.abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--max_points", type=int, default=4096)
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    ap.add_argument("--sparse_list", default="1,2,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive,
        min_target_spacing_m=args.min_target_spacing_m, max_samples=args.max_samples,
        dense_source_count=args.dense_sources, sparse_source_count=1,
        image_size=(160, 96), max_points_per_view=args.max_points,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder = ColumnFieldDecoder(
        hidden=a.get("config", {}).get("hidden", 128),
        samples=a.get("config", {}).get("ray_samples", 24),
    ).to(device)
    ground.load_state_dict(a["ground"])
    decoder.load_state_dict(a["decoder"])
    ground.eval(); decoder.eval()
    b = torch.load(args.stage_b, map_location=device, weights_only=False)
    completion = LatentCompletion(
        mode=b.get("config", {}).get("fusion", "residual"),
        bev_height=ds.bev_size, bev_width=ds.bev_size, tile_size_m=ds.tile_size_m,
    ).to(device)
    completion.load_state_dict(b["completion"])
    completion.eval()
    sat = SatelliteBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    sat.load_state_dict(b["satellite_encoder"])
    sat.eval()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    batches = []
    with torch.no_grad():
        for batch in loader:
            batches.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            if len(batches) >= args.max_samples:
                break

    print(f"tiles={len(batches)}  gains per unit latent L1 (target-view RGB L1):")
    for n_sparse in [int(x) for x in args.sparse_list.split(",")]:
        g = {"sat": [], "ideal": [], "rand": [], "ratio": []}
        with torch.no_grad():
            for batch in batches:
                z_star, _ = ground(
                    batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
                    batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
                )
                sp = slice(0, n_sparse * ds.views_per_frame)
                z_sparse, sparse_mask = ground(
                    batch["source_rgb"][:, sp], batch["source_points_world"][:, sp],
                    batch["source_points_uv"][:, sp], batch["source_points_valid"][:, sp],
                    batch["origin_xy"], ds.bev_resolution_m,
                )
                z_sat = sat(batch["satellite"], ds.tile_size_m, 0.196)
                z_hat = completion(z_sat, z_sparse, sparse_mask, n_sparse, args.dense_sources)

                def render(z):
                    return decoder.render(
                        z, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
                        tile_size_m=ds.tile_size_m, image_size=ds.image_size,
                    )[0]

                base = render(z_sparse)
                c_sat = z_hat - z_sparse
                norm = c_sat.abs().mean()
                c_ideal = (z_star - z_sparse)
                c_ideal = c_ideal * (norm / c_ideal.abs().mean().clamp_min(1e-8))
                noise = torch.randn(c_sat.shape, generator=gen).to(device)
                c_rand = noise * (norm / noise.abs().mean())
                out = {}
                for tag, c in (("sat", c_sat), ("ideal", c_ideal), ("rand", c_rand)):
                    out[tag] = (render(z_sparse + c) - base).abs().mean() / norm.clamp_min(1e-8)
                g["sat"].append(float(out["sat"]))
                g["ideal"].append(float(out["ideal"]))
                g["rand"].append(float(out["rand"]))
                g["ratio"].append(float(out["sat"] / out["ideal"].clamp_min(1e-8)))
        n = len(g["sat"])
        m = lambda k: sum(g[k]) / n
        print(f"Ns={n_sparse}: gain_sat={m('sat'):.4f}  gain_ideal={m('ideal'):.4f}  "
              f"gain_rand={m('rand'):.4f}  ratio_sat/ideal={m('ratio'):.3f}  "
              f"(ratio<1 on {sum(r < 1 for r in g['ratio'])}/{n})")


if __name__ == "__main__":
    main()

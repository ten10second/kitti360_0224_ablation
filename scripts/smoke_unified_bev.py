#!/usr/bin/env python3
"""Real-data tensor smoke test for the unified BEV pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    LatentCompletion,
    SatelliteBEVEncoder,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--dense_sources", type=int, default=4)
    ap.add_argument("--max_points", type=int, default=512)
    args = ap.parse_args()
    ds = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive, max_samples=1,
        dense_source_count=args.dense_sources, sparse_source_count=2,
        image_size=(160, 96), max_points_per_view=args.max_points,
    )
    sample = ds[0]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}
    ground = GroundBEVEncoder(latent_channels=16, bev_height=ds.bev_size, bev_width=ds.bev_size)
    z, mask = ground(
        batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
        batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
    )
    decoder = ColumnFieldDecoder(latent_channels=16, hidden=32, samples=4)
    rgb, depth, opacity = decoder.render(
        z, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
        tile_size_m=ds.tile_size_m, image_size=ds.image_size, far_m=30.0,
    )
    sat = SatelliteBEVEncoder(latent_channels=16, bev_height=ds.bev_size, bev_width=ds.bev_size)
    z_sat = sat(batch["satellite"], ds.tile_size_m, 0.196)
    z_hat = LatentCompletion(channels=16)(z_sat, z, mask, n_sparse=2, dense_sources=args.dense_sources)
    assert rgb.shape == batch["target_rgb"].shape
    assert depth.shape == batch["target_depth"].shape
    assert z.shape == z_sat.shape == z_hat.shape
    assert torch.isfinite(rgb).all() and torch.isfinite(depth).all() and torch.isfinite(z_hat).all()
    print({
        "drive": args.drive,
        "target_fid": sample["meta"]["target_fid"],
        "source_fids": sample["meta"]["source_fids"],
        "sat_m_per_px": 0.196,
        "coverage": float(mask.mean()),
        "rgb": tuple(rgb.shape),
        "depth": tuple(depth.shape),
        "latent": tuple(z.shape),
        "status": "PASS",
    })


if __name__ == "__main__":
    main()

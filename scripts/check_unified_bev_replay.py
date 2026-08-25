#!/usr/bin/env python3
"""Verify cached Stage-A latent replay with the frozen renderer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.checkpoints import (
    validate_stage_a_checkpoint,
    validate_stage_a_dataset,
)
from world3d.unified_bev.data import UnifiedBEVDataset, attach_dense_geometry
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    GroundDenseBEVEncoder,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="runs/unified_bev_replay/z_star.pt")
    ap.add_argument("--geometry_cache", default=None)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--max_points", type=int, default=4096)
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.stage_a, map_location=device, weights_only=False)
    validate_stage_a_checkpoint(ckpt)
    ds = UnifiedBEVDataset(args.manifest, lidar_root=args.lidar_root, drive=args.drive,
                           max_samples=1, dense_source_count=args.dense_sources,
                           sparse_source_count=2, image_size=(160, 96),
                           max_points_per_view=args.max_points,
                           min_target_spacing_m=args.min_target_spacing_m)
    if args.geometry_cache:
        ds = attach_dense_geometry(ds, args.geometry_cache)
    validate_stage_a_dataset(
        ckpt, ds, dense_geometry_attached=bool(args.geometry_cache),
    )
    sample = ds[0]
    b = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    ground_config = dict(ckpt["ground_config"])
    family = ground_config.pop("family")
    ground_cls = GroundDenseBEVEncoder if family == "dense" else GroundBEVEncoder
    ground = ground_cls(**ground_config).to(device)
    decoder = ColumnFieldDecoder(**ckpt["renderer_config"]).to(device)
    ground.load_state_dict(ckpt["ground"]); decoder.load_state_dict(ckpt["decoder"])
    ground.eval(); decoder.eval()
    with torch.no_grad():
        if family == "dense":
            z_online, mask_online = ground(
                b["source_rgb"], b["source_K"], b["dense_depth"], b["dense_conf"],
                b["source_T_world_cam"], b["origin_xy"], ds.bev_resolution_m,
            )
        else:
            z_online, mask_online = ground(
                b["source_rgb"], b["source_points_world"], b["source_points_uv"],
                b["source_points_valid"], b["origin_xy"], ds.bev_resolution_m,
            )
        rgb_online, depth_online, _ = decoder.render(z_online, b["target_K"], b["target_T_world_cam"],
                                                     b["origin_xy"], tile_size_m=ds.tile_size_m,
                                                     image_size=ds.image_size)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z_star": z_online.cpu(), "coverage": mask_online.cpu()}, cache_path)
    cached = torch.load(cache_path, map_location=device, weights_only=False)
    with torch.no_grad():
        rgb_cached, depth_cached, _ = decoder.render(cached["z_star"].to(device), b["target_K"],
                                                     b["target_T_world_cam"], b["origin_xy"],
                                                     tile_size_m=ds.tile_size_m, image_size=ds.image_size)
    z_err = float((z_online - cached["z_star"].to(device)).abs().max())
    rgb_err = float((rgb_online - rgb_cached).abs().max())
    depth_err = float((depth_online - depth_cached).abs().max())
    assert z_err < 1e-7 and rgb_err < 1e-6 and depth_err < 1e-6
    print({"cache": str(cache_path), "z_max_abs": z_err, "rgb_max_abs": rgb_err,
           "depth_max_abs": depth_err, "status": "PASS"})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M0 QA: overlay LiDAR BEV splat and view positions on the satellite BEV.

The overlay is the gate for the coordinate fix: LiDAR structure (road
corridor, building fronts) must land on the matching satellite content.  A
north/south mirror error would show as LiDAR structures on the wrong side of
the road or off the road corridor entirely.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.geometry import bilinear_splat, height_statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="runs/unified_bev_qa/qa_alignment.png")
    args = ap.parse_args()

    ds = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive, max_samples=args.sample + 1,
        dense_source_count=8, sparse_source_count=2, image_size=(160, 96), max_points_per_view=8192,
    )
    s = ds[args.sample]
    origin = s["origin_xy"]
    n = s["source_points_world"].shape[0]
    pts = s["source_points_world"].unsqueeze(0)
    valid = s["source_points_valid"].unsqueeze(0)
    ones = torch.ones(1, n, pts.shape[2], 1)
    count_bev, _ = bilinear_splat(
        ones, pts[..., :2], valid, origin_xy=origin.unsqueeze(0),
        resolution_m=ds.bev_resolution_m, height=ds.bev_size, width=ds.bev_size,
    )
    h_mean_map, _ = height_statistics(
        pts, valid, origin.unsqueeze(0), ds.bev_resolution_m, ds.bev_size, ds.bev_size,
    )
    height_mean = h_mean_map.squeeze(0).squeeze(0).numpy()
    count = count_bev.squeeze(0).squeeze(0).numpy()

    sat = s["satellite"].permute(1, 2, 0).numpy()
    tile_px = int(round(ds.tile_size_m / 0.196))
    y0 = x0 = (sat.shape[0] - tile_px) // 2
    crop = torch.from_numpy(sat[y0 : y0 + tile_px, x0 : x0 + tile_px].copy()).permute(2, 0, 1).unsqueeze(0)
    crop = torch.flip(crop, dims=[-2])
    crop = torch.nn.functional.interpolate(
        crop, size=(ds.bev_size, ds.bev_size), mode="bilinear", align_corners=False)
    sat_bev = crop.squeeze(0).permute(1, 2, 0).numpy()

    def world_to_px(xy):
        c = (xy - origin.numpy()) / ds.tile_size_m * ds.bev_size
        return float(c[0]), float(ds.bev_size - c[1])  # image row 0 = north for display

    overlay = Image.fromarray((sat_bev * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for r in range(ds.bev_size):
        for c in range(ds.bev_size):
            if count[r, c] > 0.5:
                x, y = c, r
                draw.point((x, y), fill=(255, 64, 64))
    T = s["target_T_world_cam"].numpy()
    tx, ty = world_to_px([T[0, 3], T[1, 3]])
    fwd = T[:3, 0]  # camera x-axis in world (forward)
    fx, fy = world_to_px([T[0, 3] + fwd[0] * 8, T[1, 3] + fwd[1] * 8])
    draw.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], outline=(0, 255, 255), width=3)
    draw.line([tx, ty, fx, fy], fill=(0, 255, 255), width=3)
    for i in range(n):
        Sv = s["source_T_world_cam"][i].numpy()
        px, py = world_to_px([Sv[0, 3], Sv[1, 3]])
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], outline=(255, 255, 0), width=2)

    hm = np.clip((height_mean + 4.0) / 16.0, 0, 1)
    hm_rgb = np.stack([hm, hm * 0.6, 1.0 - hm], axis=-1)
    hm_rgb[count <= 0.5] = sat_bev[count <= 0.5]

    panel = Image.new("RGB", (ds.bev_size * 3 + 20, ds.bev_size + 20), (20, 20, 20))
    panel.paste(Image.fromarray((sat_bev * 255).astype(np.uint8)), (0, 0))
    panel.paste(overlay, (ds.bev_size + 10, 0))
    panel.paste(Image.fromarray((hm_rgb * 255).astype(np.uint8)), (ds.bev_size * 2 + 20, 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out)
    print({"out": str(out), "drive": args.drive, "target_fid": s["meta"]["target_fid"],
           "coverage": float((count > 0.5).mean()), "panel": "sat | sat+lidar+views | height"})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a diagnostic panel: GT target vs dense/sparse renders + baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.models import ColumnFieldDecoder, GroundBEVEncoder


def psnr(a, b):
    return float(10.0 * torch.log10(1.0 / F.mse_loss(a, b).clamp_min(1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--sparse_sources", type=int, default=2)
    ap.add_argument("--max_points", type=int, default=2048)
    ap.add_argument("--out", default="runs/unified_bev_qa/render_panel.png")
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = UnifiedBEVDataset(args.manifest, lidar_root=args.lidar_root, drive=args.drive, min_target_spacing_m=args.min_target_spacing_m,
                           max_samples=args.samples, dense_source_count=args.dense_sources,
                           sparse_source_count=args.sparse_sources, image_size=(160, 96),
                           max_points_per_view=args.max_points)
    ckpt = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder = ColumnFieldDecoder(hidden=ckpt.get("config", {}).get("hidden", 128),
                                   samples=ckpt.get("config", {}).get("ray_samples", 24)).to(device)
    ground.load_state_dict(ckpt["ground"]); decoder.load_state_dict(ckpt["decoder"])
    ground.eval(); decoder.eval()

    rows = []
    stats = {"dense": [], "sparse": [], "mean": [], "nearest": []}
    mean_img = torch.stack([ds[i]["target_rgb"] for i in range(len(ds))]).mean(0)
    with torch.no_grad():
        for i in range(len(ds)):
            s = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
            z_star, _ = ground(s["source_rgb"], s["source_points_world"], s["source_points_uv"],
                               s["source_points_valid"], s["origin_xy"], ds.bev_resolution_m)
            sp = slice(0, args.sparse_sources)
            z_sp, _ = ground(s["source_rgb"][:, sp], s["source_points_world"][:, sp],
                             s["source_points_uv"][:, sp], s["source_points_valid"][:, sp],
                             s["origin_xy"], ds.bev_resolution_m)
            r_dense, _, _ = decoder.render(z_star, s["target_K"], s["target_T_world_cam"], s["origin_xy"],
                                           tile_size_m=ds.tile_size_m, image_size=ds.image_size)
            r_sparse, _, _ = decoder.render(z_sp, s["target_K"], s["target_T_world_cam"], s["origin_xy"],
                                            tile_size_m=ds.tile_size_m, image_size=ds.image_size)
            near = s["source_rgb"][:, int(s["source_points_valid"].sum(dim=(1, 2)).argmax())]
            gt = s["target_rgb"]
            stats["dense"].append(psnr(r_dense, gt)); stats["sparse"].append(psnr(r_sparse, gt))
            stats["mean"].append(psnr(mean_img.to(device), gt)); stats["nearest"].append(psnr(near, gt))
            rows.append(torch.cat([gt, r_dense, r_sparse, near], dim=-1).clamp(0, 1))
    panel = (torch.cat(rows, dim=-2).squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(out)
    print({k: round(sum(v) / len(v), 3) for k, v in stats.items()},
          {"panel": str(out), "layout": "GT | dense | sparse | nearest_source"})


if __name__ == "__main__":
    main()

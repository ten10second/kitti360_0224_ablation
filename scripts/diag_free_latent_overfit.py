#!/usr/bin/env python3
"""Isolate decoder capacity: jointly fit a free per-tile latent + column
decoder to a single target view.  If this cannot overfit, the queryable
decoder itself (or its optimization) is the bottleneck; if it can, the
ground-encoder pathway (feature splat) is the problem."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.models import ColumnFieldDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ray_samples", type=int, default=48)
    ap.add_argument("--z_lr", type=float, default=1e-2)
    ap.add_argument("--mlp_lr", type=float, default=2e-4)
    ap.add_argument("--latent_channels", type=int, default=64)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = UnifiedBEVDataset(args.manifest, lidar_root=args.lidar_root, drive=args.drive,
                           max_samples=1, dense_source_count=8, sparse_source_count=2,
                           image_size=(160, 96), max_points_per_view=1024,
                           min_target_spacing_m=0.0)
    s = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in ds[0].items()}

    torch.manual_seed(0)
    z_free = torch.zeros(1, args.latent_channels, ds.bev_size, ds.bev_size, device=device)
    torch.nn.init.normal_(z_free, std=0.1)
    z_free.requires_grad_(True)
    dec = ColumnFieldDecoder(latent_channels=args.latent_channels, hidden=args.hidden,
                             samples=args.ray_samples).to(device)
    opt = torch.optim.AdamW([
        {"params": [z_free], "lr": args.z_lr},
        {"params": dec.parameters(), "lr": args.mlp_lr},
    ], weight_decay=0.0)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        rgb, depth, _ = dec.render(z_free, s["target_K"], s["target_T_world_cam"], s["origin_xy"],
                                   tile_size_m=ds.tile_size_m, image_size=ds.image_size)
        rgb_loss = F.smooth_l1_loss(rgb, s["target_rgb"])
        m = s["target_depth_mask"]
        depth_loss = F.smooth_l1_loss(depth[m], s["target_depth"][m]) if m.any() else depth.mean() * 0
        loss = rgb_loss + 0.1 * depth_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 250 == 0:
            mse = float(F.mse_loss(rgb.detach(), s["target_rgb"]))
            psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-10)))
            print(f"step={step}/{args.steps} rgb={float(rgb_loss):.5f} depth={float(depth_loss):.4f} "
                  f"psnr={float(psnr):.2f} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train Stage A: dense-ground reference BEV latent + queryable renderer.

This is deliberately a small single-GPU probe.  It is the gate before Stage B
and before any full-data or multi-GPU run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.data import UnifiedBEVDataset
from world3d.unified_bev.data import load_cached_unified_bev
from world3d.unified_bev.models import ColumnFieldDecoder, GroundBEVEncoder


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--out", default="runs/unified_bev_stage_a_probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--image_width", type=int, default=160)
    ap.add_argument("--image_height", type=int, default=96)
    ap.add_argument("--max_points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ray_samples", type=int, default=24)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cache", default=None, help="prebuilt sample cache; serving from RAM, workers forced to 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    ds = (load_cached_unified_bev(args.cache) if args.cache else UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, dense_source_count=args.dense_sources,
        sparse_source_count=2, image_size=(args.image_width, args.image_height),
        max_points_per_view=args.max_points, max_samples=args.max_samples, drive=args.drive,
        min_target_spacing_m=args.min_target_spacing_m,
    ))
    if args.cache:
        args.num_workers = 0  # RAM serving; forked workers would copy the cache
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=True, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder = ColumnFieldDecoder(hidden=args.hidden, samples=args.ray_samples).to(device)
    opt = torch.optim.AdamW(list(ground.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[stage-a] device={device} samples={len(ds)} bev={ds.bev_size}x{ds.bev_size} mpp=0.196")

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        ground.load_state_dict(ck["ground"]); decoder.load_state_dict(ck["decoder"])
        step = int(ck.get("step", 0))
        print(f"[stage-a] resumed from {args.resume} at step {step}")
    running = 0.0
    iterator = iter(loader)
    t0 = time.time()
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        z_star, coverage = ground(
            batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
            batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
        )
        pred_rgb, pred_depth, _ = decoder.render(
            z_star, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
            tile_size_m=ds.tile_size_m, image_size=ds.image_size,
        )
        rgb_loss = F.smooth_l1_loss(pred_rgb, batch["target_rgb"])
        depth_mask = batch["target_depth_mask"]
        depth_loss = F.smooth_l1_loss(pred_depth[depth_mask], batch["target_depth"][depth_mask]) if depth_mask.any() else pred_depth.mean() * 0
        coverage_loss = (1.0 - coverage).mean() * 0.001
        loss = rgb_loss + 0.1 * depth_loss + coverage_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ground.parameters()) + list(decoder.parameters()), 1.0)
        opt.step()
        step += 1
        running += float(loss.detach())
        if step == 1 or step % 50 == 0:
            print(f"step={step}/{args.steps} loss={running / (50 if step >= 50 else step):.5f} "
                  f"rgb={float(rgb_loss):.5f} depth={float(depth_loss):.5f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
            running = 0.0
        if step % 500 == 0 or step == args.steps:
            torch.save({
                "ground": ground.state_dict(), "decoder": decoder.state_dict(),
                "config": vars(args), "tile_size_m": ds.tile_size_m,
                "bev_resolution_m": ds.bev_resolution_m, "bev_size": ds.bev_size,
                "image_size": ds.image_size, "step": step,
            }, out / "stage_a.pt")
    print(f"[stage-a] checkpoint={out / 'stage_a.pt'}")


if __name__ == "__main__":
    main()

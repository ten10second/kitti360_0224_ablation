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

from world3d.unified_bev.data import (
    UnifiedBEVDataset,
    attach_dense_geometry,
    load_cached_unified_bev,
    load_dense_cached_unified_bev,
)
from world3d.unified_bev.checkpoints import (
    GEOMETRY_TARGET_VERSION,
    STAGE_A_SCHEMA_VERSION,
    compute_stage_a_fingerprint,
    validate_stage_a_checkpoint,
    validate_stage_a_dataset,
)
from world3d.unified_bev.geometry import relative_height_map
from world3d.unified_bev.losses import masked_smooth_l1
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    GroundDenseBEVEncoder,
    render_multi_view,
)
from world3d.unified_bev.readouts import BEVHeightDecoder


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
    ap.add_argument("--geometry_width", type=int, default=64)
    ap.add_argument("--geometry_weight", type=float, default=0.1,
                    help="weight of the shared relative-height readout trained from dense-ground Z*")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cache", default=None, help="prebuilt sample cache; serving from RAM, workers forced to 0")
    ap.add_argument("--geometry_cache", "--m3d_cache", dest="geometry_cache", default=None,
                    help="dense geometry cache (Metric3D or joint-view VGGT)")
    ap.add_argument("--chunked", action="store_true",
                    help="route-chunk windows (cache v7); the dense lift is the union of "
                         "all chunks' lift frames and queries are the per-chunk core frames")
    ap.add_argument("--chunks_per_window", type=int, default=4)
    ap.add_argument("--chunk_arc_m", type=float, default=12.0)
    ap.add_argument("--guard_m", type=float, default=4.0)
    ap.add_argument("--frames_per_chunk", type=int, default=2)
    ap.add_argument("--max_geometry_frames", type=int, default=8)
    ap.add_argument("--window_stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.chunked and not args.geometry_cache:
        raise ValueError("--chunked requires --geometry_cache (chunk cache v7)")
    if args.cache and args.geometry_cache:
        ds = load_dense_cached_unified_bev(args.cache, args.geometry_cache)
    elif args.cache:
        ds = load_cached_unified_bev(args.cache)
    elif args.chunked:
        from world3d.unified_bev.data import ChunkedUnifiedBEVDataset, attach_chunk_geometry
        ds = attach_chunk_geometry(ChunkedUnifiedBEVDataset(
            args.manifest, lidar_root=args.lidar_root, drive=args.drive,
            chunks_per_window=args.chunks_per_window, chunk_arc_m=args.chunk_arc_m,
            guard_m=args.guard_m, frames_per_chunk=args.frames_per_chunk,
            max_geometry_frames=args.max_geometry_frames,
            window_stride=args.window_stride, max_samples=args.max_samples,
            image_size=(args.image_width, args.image_height),
            max_points_per_view=args.max_points,
        ), args.geometry_cache)
    else:
        ds = UnifiedBEVDataset(
            args.manifest, lidar_root=args.lidar_root, dense_source_count=args.dense_sources,
            sparse_source_count=2, image_size=(args.image_width, args.image_height),
            max_points_per_view=args.max_points, max_samples=args.max_samples, drive=args.drive,
            min_target_spacing_m=args.min_target_spacing_m,
        )
        if args.geometry_cache:
            ds = attach_dense_geometry(ds, args.geometry_cache)
    if args.cache:
        args.num_workers = 0  # RAM serving; forked workers would copy the cache
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=True, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    ground_config = {
        "family": "dense" if args.geometry_cache else "sparse",
        "latent_channels": 64,
        "bev_height": int(ds.bev_size),
        "bev_width": int(ds.bev_size),
        "context_blocks": 4,
    }
    if args.geometry_cache:
        ground_config.update({
            "conf_threshold": 0.3, "min_depth_m": 0.5, "max_depth_m": 60.0,
        })
        ground = GroundDenseBEVEncoder(**{
            key: ground_config[key]
            for key in (
                "latent_channels", "bev_height", "bev_width", "context_blocks",
                "conf_threshold", "min_depth_m", "max_depth_m",
            )
        }).to(device)
    else:
        ground = GroundBEVEncoder(**{
            key: ground_config[key]
            for key in ("latent_channels", "bev_height", "bev_width", "context_blocks")
        }).to(device)
    renderer_config = {
        "latent_channels": 64, "hidden": int(args.hidden), "samples": int(args.ray_samples),
    }
    geometry_decoder_config = {
        "latent_channels": 64, "width": int(args.geometry_width),
    }
    grid_config = {
        "bev_size": int(ds.bev_size),
        "bev_resolution_m": float(ds.bev_resolution_m),
        "tile_size_m": float(ds.tile_size_m),
        "views_per_frame": int(ds.views_per_frame),
        "target_views": int(ds.target_views),
        "target_view_layout_version": str(ds.target_view_layout_version),
    }
    decoder = ColumnFieldDecoder(**renderer_config).to(device)
    geometry_decoder = BEVHeightDecoder(**geometry_decoder_config).to(device)
    trainable = (
        list(ground.parameters())
        + list(decoder.parameters())
        + list(geometry_decoder.parameters())
    )
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[stage-a] device={device} samples={len(ds)} bev={ds.bev_size}x{ds.bev_size} mpp=0.196")

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        validate_stage_a_checkpoint(ck)
        validate_stage_a_dataset(
            ck, ds, dense_geometry_attached=bool(args.geometry_cache),
        )
        for key, expected in (
            ("ground_config", ground_config),
            ("renderer_config", renderer_config),
            ("geometry_decoder_config", geometry_decoder_config),
        ):
            if ck[key] != expected:
                raise RuntimeError(f"resume {key} differs from the requested architecture")
        ground.load_state_dict(ck["ground"]); decoder.load_state_dict(ck["decoder"])
        geometry_decoder.load_state_dict(ck["geometry_decoder"])
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
        if args.geometry_cache:
            z_star, coverage = ground(
                batch["source_rgb"], batch["source_K"], batch["dense_depth"],
                batch["dense_conf"], batch["source_T_world_cam"],
                batch["origin_xy"], ds.bev_resolution_m,
            )
        else:
            z_star, coverage = ground(
                batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
                batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
            )
        pred_rgb, pred_depth, _ = render_multi_view(
            decoder, z_star, batch["target_K"], batch["target_T_world_cam"],
            batch["origin_xy"],
            tile_size_m=ds.tile_size_m, image_size=ds.image_size,
        )
        rgb_loss = F.smooth_l1_loss(pred_rgb, batch["target_rgb"])
        depth_mask = batch["target_depth_mask"]
        depth_loss = F.smooth_l1_loss(pred_depth[depth_mask], batch["target_depth"][depth_mask]) if depth_mask.any() else pred_depth.mean() * 0
        h_ref, h_valid, _ = relative_height_map(
            batch["source_points_world"], batch["source_points_valid"],
            batch["origin_xy"], ds.bev_resolution_m, ds.bev_size, ds.bev_size,
        )
        h_pred = geometry_decoder(z_star)
        geometry_loss = masked_smooth_l1(h_pred, h_ref, h_valid)
        loss = rgb_loss + 0.1 * depth_loss + args.geometry_weight * geometry_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        running += float(loss.detach())
        if step == 1 or step % 50 == 0:
            print(f"step={step}/{args.steps} loss={running / (50 if step >= 50 else step):.5f} "
                  f"rgb={float(rgb_loss):.5f} depth={float(depth_loss):.5f} "
                  f"geometry={float(geometry_loss):.5f} coverage={float(coverage.mean()):.3f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
            running = 0.0
        if step % 500 == 0 or step == args.steps:
            checkpoint = {
                "schema_version": STAGE_A_SCHEMA_VERSION,
                "ground": ground.state_dict(), "decoder": decoder.state_dict(),
                "geometry_decoder": geometry_decoder.state_dict(),
                "ground_config": ground_config,
                "renderer_config": renderer_config,
                "geometry_decoder_config": geometry_decoder_config,
                "geometry_target_version": GEOMETRY_TARGET_VERSION,
                "grid_config": grid_config,
                "config": vars(args), "tile_size_m": ds.tile_size_m,
                "bev_resolution_m": ds.bev_resolution_m, "bev_size": ds.bev_size,
                "image_size": ds.image_size, "step": step,
            }
            if args.chunked:
                checkpoint["chunk_config"] = {
                    "chunks_per_window": int(ds.chunks_per_window),
                    "chunk_arc_m": float(ds.chunk_arc_m),
                    "guard_m": float(ds.guard_m),
                    "frames_per_chunk": int(ds.frames_per_chunk),
                    "chunking_version": str(ds.chunking_version),
                }
            checkpoint["fingerprint"] = compute_stage_a_fingerprint(checkpoint)
            torch.save(checkpoint, out / "stage_a.pt")
    print(f"[stage-a] checkpoint={out / 'stage_a.pt'}")


if __name__ == "__main__":
    main()

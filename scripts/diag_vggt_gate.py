#!/usr/bin/env python3
"""One-tile gate for joint-view, motion-metric VGGT geometry.

This gate deliberately excludes satellite imagery: VGGT establishes the
ground-view geometry used by Stage A, while satellite geometry enters through
the separately trained Stage-B prior.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/shizhm/Lenovo/vggt")

from vggt.models.vggt import VGGT
from world3d.unified_bev.data import load_cached_unified_bev
from scripts.build_vggt_street_cache import run_joint_subset


def lidar_depth_metrics(entry: dict, sample: dict, view_slice: slice) -> dict[str, float | int]:
    predicted = entry["depth"].float()
    errors, relatives, ratios = [], [], []
    valid_count = 0
    for local_view, source_view in enumerate(range(view_slice.start, view_slice.stop)):
        valid = sample["source_points_valid"][source_view]
        uv = sample["source_points_uv"][source_view][valid]
        points = sample["source_points_world"][source_view][valid]
        T = sample["source_T_world_cam"][source_view]
        R, t = T[:3, :3], T[:3, 3]
        target_z = ((points - t) @ R)[:, 2]
        x = uv[:, 0].round().long().clamp(0, predicted.shape[-1] - 1)
        y = uv[:, 1].round().long().clamp(0, predicted.shape[-2] - 1)
        depth = predicted[local_view, y, x]
        keep = (target_z > 0.5) & (depth > 0.5)
        if keep.any():
            target_z, depth = target_z[keep], depth[keep]
            diff = (depth - target_z).abs()
            errors.append(diff.square())
            relatives.append(diff / target_z)
            ratios.append(torch.maximum(depth / target_z, target_z / depth))
            valid_count += int(keep.sum())
    if not errors:
        return {"points": 0, "absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    return {
        "points": valid_count,
        "absrel": float(torch.cat(relatives).mean()),
        "rmse": float(torch.cat(errors).mean().sqrt()),
        "delta1": float((torch.cat(ratios) < 1.25).float().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/shizhm/Downloads/vggt.pt")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--frame_start", type=int, default=0)
    ap.add_argument("--frame_count", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--min_baseline_m", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    ds = load_cached_unified_bev(args.cache)
    sample = ds[args.index]
    start = args.frame_start * ds.views_per_frame
    stop = (args.frame_start + args.frame_count) * ds.views_per_frame
    if stop > sample["source_rgb"].shape[0]:
        raise ValueError("requested source subset exceeds the cached source pool")

    model = VGGT(enable_point=False, enable_track=False).to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    del state
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    entry = run_joint_subset(
        model,
        sample["source_rgb"][start:stop],
        sample["source_T_world_cam"][start:stop],
        args.resolution,
        args.min_baseline_m,
        ds.views_per_frame,
        device,
        gt_world_vehicle=sample["source_T_world_imu"][
            args.frame_start:args.frame_start + args.frame_count
        ],
        view_camera_ids=torch.tensor(ds.view_camera_ids),
    )
    elapsed = time.time() - t0
    metrics = lidar_depth_metrics(entry, sample, slice(start, stop))
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0)
    print(f"tile={args.index} subset={args.frame_start}:{args.frame_count} views={stop-start}")
    print(f"forward={elapsed:.2f}s peak_gpu={peak_gb:.2f}GB input_hw={entry['vggt_input_hw'].tolist()}")
    print(f"metric_scale={float(entry['metric_scale']):.6f} source={entry['scale_source']} "
          f"pairs={entry['scale_pair_count']} "
          f"relative_mad={float(entry['scale_relative_mad']):.4f} "
          f"pose_rmse={float(entry['pose_alignment_rmse_m']):.3f}m")
    print(f"LiDAR points={metrics['points']} AbsRel={metrics['absrel']:.4f} "
          f"RMSE={metrics['rmse']:.3f}m delta1={metrics['delta1']:.4f}")


if __name__ == "__main__":
    main()

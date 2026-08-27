#!/usr/bin/env python3
"""Vehicle-side global-height correction probe (follow-up to E0).

Two questions, one scene (held-out 0003):

Part 1 - residual shape diagnosis (uses ground truth; diagnostic only):
  fit a constant, then a tilted plane, to (satellite_height - lidar_height).
  constant-captured improvement  -> the map is just uniformly offset
  plane-needed improvement       -> the downhill slope is also wrong
  This decides: one scalar correction vs a slowly-varying terrain slope.

Part 2 - correction conditions evaluated on the AHEAD region (cells the
first chunk has not seen):
  oracle : offset from real LiDAR (upper bound; uses the answer)
  vggt   : offset = median(satellite - VGGT-chunk-1 measurement height)
           over the first chunk's support -- deployable inference: satellite
           gives a coarse map, the first ground packet calibrates it globally,
           evaluation happens where the car has not been yet.

If the VGGT-condition MAE approaches the centered/oracle MAE, a global
height correction belongs in the state machine as a first-class operation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.geometry import bilinear_splat  # noqa: E402
from world3d.unified_bev.losses import masked_smooth_l1  # noqa: E402
from world3d.unified_bev.models import fixed_relative_xy_encoding, unproject_dense  # noqa: E402
from world3d.unified_bev.state_models import FrozenDINOv2  # noqa: E402
from world3d.unified_bev.world_vggt import load_world_vggt_cache  # noqa: E402
from scripts.probe_world_satellite_prior import ProbeHead, load_scenes  # noqa: E402

XY_CHANNELS = 16
CONF_GATE = 0.3
MIN_DEPTH, MAX_DEPTH = 0.5, 60.0


def train_dino_head(train, ev, device, steps, seed):
    """Deterministic replica of the E0 dino arm."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dino = FrozenDINOv2().to(device).eval()
    feats = {}
    for sc in train + ev:
        with torch.no_grad():
            feats[sc["scene_id"]] = dino(sc["sat"].to(device)).cpu()
    del dino
    if device.type == "cuda":
        torch.cuda.empty_cache()
    xy = fixed_relative_xy_encoding(XY_CHANNELS, 200, 200, tile_size_m=100.0).to(device)
    torch.manual_seed(seed)
    head = ProbeHead(768 + XY_CHANNELS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-4)
    order = np.arange(len(train))
    for step in range(1, steps + 1):
        if step % len(train) == 1:
            np.random.shuffle(order)
        sc = train[order[(step - 1) % len(train)]]
        feat = F.interpolate(feats[sc["scene_id"]].to(device), size=(200, 200),
                             mode="bilinear", align_corners=False)
        pred = head(torch.cat([feat, xy], dim=1))
        m = sc["valid"].to(device)
        loss = masked_smooth_l1(pred[:, :1], sc["height"].to(device), m) + \
            masked_smooth_l1(pred[:, 1:2], sc["density"].to(device), m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head, feats, xy


def predict_satellite_height(head, feats, sc, xy, device):
    with torch.no_grad():
        feat = F.interpolate(feats[sc["scene_id"]].to(device), size=(200, 200),
                             mode="bilinear", align_corners=False)
        return head(torch.cat([feat, xy], dim=1))[0, 0].cpu()


def vggt_measurement_height(entry, origin, datum, device):
    """BEV height field of one cached VGGT chunk (same math as
    GroundMeasurementEncoder: gated unproject + z splat - datum)."""
    depth = entry["depth"].float().unsqueeze(0).to(device)
    conf = entry["conf"].float().unsqueeze(0).to(device)
    K = entry["K"].float().unsqueeze(0).to(device)
    T = entry["T_world_cam"].float().unsqueeze(0).to(device)
    gate = (conf > CONF_GATE) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    B, N, H, W = depth.shape
    pts = unproject_dense(depth, K, T).view(B, N, H * W, 3)
    gate = gate.view(B, N, H * W)
    z_abs, count = bilinear_splat(
        pts[..., 2:3], pts[..., :2], gate,
        origin_xy=origin.view(1, 2).to(device), resolution_m=0.5,
        height=200, width=200,
    )
    support = (count > 0)[0, 0].cpu()
    h_meas = ((z_abs - float(datum)) * (count > 0).float())[0, 0].cpu()
    return h_meas, support


def fit_plane(err, mask, origin_xy, resolution=0.5, size=200):
    """Least-squares tilted plane a + b*x + c*y on masked residual cells."""
    idx = torch.nonzero(mask, as_tuple=False).numpy()
    r = err.numpy()[mask.numpy()]
    xs = origin_xy[0].item() + (idx[:, 1] + 0.5) * resolution
    ys = origin_xy[1].item() + (idx[:, 0] + 0.5) * resolution
    A = np.stack([np.ones_like(xs), xs, ys], axis=1)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    fit = A @ coef
    return coef, r - fit


def mae(x):
    return float(np.abs(x).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_scenes", default="runs/world_state_e0/targets_train")
    ap.add_argument("--eval_scenes", default="runs/world_state_targets_smoke")
    ap.add_argument("--vggt_cache", default="runs/world_state_vggt_smoke")
    ap.add_argument("--out", default="runs/world_state_e0/height_correction.json")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    train = load_scenes(args.train_scenes)
    ev = load_scenes(args.eval_scenes)
    sc = ev[0]
    import warnings
    warnings.filterwarnings("ignore")
    blob = None
    from world3d.unified_bev.world_data import WorldStateSceneDataset
    ds = WorldStateSceneDataset(args.eval_scenes)
    _, _, blob = ds[0]
    cache = load_world_vggt_cache(
        args.vggt_cache, blob["scene_id"],
        str(blob["world_target_version"]), str(blob["world_target_hash"]),
    )
    print(f"[corr] scene={blob['scene_id']} device={device}")

    head, feats, xy = train_dino_head(train, ev, device, args.steps, args.seed)
    h_sat = predict_satellite_height(head, feats, sc, xy, device)
    h_meas, support1 = vggt_measurement_height(
        cache["chunks"]["1"], blob["origin_xy"], blob["z_datum_m"], device,
    )

    gt = sc["height"][0, 0]
    valid = sc["valid"][0, 0]
    err = (h_sat - gt)
    ahead = valid & ~support1
    overlap1 = support1 & valid
    print(f"[corr] cells: valid={int(valid.sum())} chunk1_vggt={int(support1.sum())} "
          f"overlap={int(overlap1.sum())} ahead={int(ahead.sum())}")

    # ---- Part 1: residual shape (diagnostic, uses gt) ----
    part1 = {}
    for region_name, region in (("valid", valid), ("ahead", ahead)):
        e = err[region].numpy()
        const_med = e - np.median(e)
        coef, plane_res = fit_plane(err, region, blob["origin_xy"].numpy())
        part1[region_name] = {
            "raw_mae": mae(e),
            "mean_bias": float(e.mean()),
            "median_bias": float(np.median(e)),
            "after_constant_median_mae": mae(const_med),
            "after_plane_mae": mae(plane_res),
            "plane_coef_abc": [float(c) for c in coef],
            "plane_slope_m_per_km": float(np.hypot(coef[1], coef[2]) * 1000),
        }
        print(f"[corr:p1 {region_name:5s}] raw={mae(e):.3f} -> const(med)={mae(const_med):.3f} "
              f"-> plane={mae(plane_res):.3f}  bias={e.mean():+.3f} "
              f"slope={np.hypot(coef[1], coef[2]) * 1000:.1f} m/km")

    # ---- Part 2: correction conditions on the AHEAD region ----
    c_oracle = float(np.median((h_sat - gt)[overlap1].numpy()))
    c_vggt = float(np.median((h_sat - h_meas)[support1].numpy()))
    part2 = {
        "ahead_cells": int(ahead.sum()),
        "correction_oracle_m": c_oracle,
        "correction_vggt_m": c_vggt,
        "raw_ahead_mae": mae(err[ahead].numpy()),
        "oracle_corrected_ahead_mae": mae((err[ahead] - c_oracle).numpy()),
        "vggt_corrected_ahead_mae": mae((err[ahead] - c_vggt).numpy()),
        "centered_ahead_mae": mae((err[ahead] - err[ahead].mean()).numpy()),
    }
    print(f"[corr:p2] offset oracle={c_oracle:+.3f} m  vggt-chunk1={c_vggt:+.3f} m  "
          f"(difference {abs(c_oracle - c_vggt):.3f} m)")
    print(f"[corr:p2] AHEAD mae: raw={part2['raw_ahead_mae']:.3f} -> "
          f"oracle={part2['oracle_corrected_ahead_mae']:.3f} -> "
          f"vggt={part2['vggt_corrected_ahead_mae']:.3f} "
          f"(perfect-centering={part2['centered_ahead_mae']:.3f})")

    results = {"scene": blob["scene_id"], "part1_residual_shape": part1,
               "part2_correction": part2, "config": vars(args)}
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[corr] wrote {args.out}")


if __name__ == "__main__":
    main()

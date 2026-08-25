#!/usr/bin/env python3
"""Build the Metric3D dense-depth cache for the unified-BEV v2 lift.

Per cached tile (64 views), for every source view:
  - the first two views of each frame are centered windows from the same
    image_00 camera; M3D runs on both original-resolution 560x376 crops and
    each prediction is resized to its matching 96x160 source view;
  - fisheye virtual crops: run M3D on the cached 96x160 view directly
    (5:3 native aspect, 96% canvas fill);
  - per view: ONE LiDAR scale anchor (median of z_lidar/d_m3d over valid
    LiDAR pixels) fixes the per-view metric drift (measured factors
    1.06-2.35x; cross-view disagreement 71% -> 11% after anchoring);
  - confidence (M3D's second output) is stored alongside.

Depth values are z-depth along the optical axis; resampling between pixel
grids does not change values, and the dataset's anisotropic K is exactly
consistent with the axis-resized images, so unprojection from the cached
96x160 grid stays geometrically exact.

Storage per tile: depth (64,96,160) fp16, conf (64,96,160) fp16,
scale (64,) fp32.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch.hub as hub
from world3d.unified_bev.data import (
    FRONT_CROP_OVERLAP,
    FRONT_CROP_WIDTH,
    UnifiedBEVDataset,
    VIEW_LAYOUT_VERSION,
    centered_two_crop_starts,
    geometry_sample_identity,
    load_cached_unified_bev,
    validate_geometry_blob_identity,
)

MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
IN_H, IN_W = 616, 1064


def m3d_run(m3d, img01: torch.Tensor, fx, max_batch: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """M3D on a [0,1] (B,3,h,w) GPU batch of same-size images -> (z-depth, conf) (B,h,w).

    Batched inference: ViT-S activations scale linearly with batch (no
    cross-image attention), so batching same-size views multiplies GPU
    utilization ~10x at negligible memory cost on a 24GB card."""
    B, _, h, w = img01.shape
    scale = min(IN_H / h, IN_W / w)
    nh, nw = int(h * scale), int(w * scale)
    ds, cs = [], []
    for b0 in range(0, B, max_batch):
        chunk = img01[b0:b0 + max_batch]
        n = chunk.shape[0]
        x = F.interpolate(chunk, size=(nh, nw), mode="bilinear", align_corners=False) * 255.0
        ph0, pw0 = (IN_H - nh) // 2, (IN_W - nw) // 2
        xp = torch.zeros(n, 3, IN_H, IN_W, device=chunk.device)
        xp[:, :, ph0:ph0 + nh, pw0:pw0 + nw] = x
        with torch.no_grad():
            d, c, _ = m3d.inference({"input": (xp - MEAN.to(chunk.device)) / STD.to(chunk.device)})
        ds.append(d[:, 0, ph0:ph0 + nh, pw0:pw0 + nw])
        cs.append(c[:, 0, ph0:ph0 + nh, pw0:pw0 + nw])
    d = torch.cat(ds)
    c = torch.cat(cs)
    d = F.interpolate(d.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
    c = F.interpolate(c.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
    if not torch.is_tensor(fx):
        fx = torch.full((B,), float(fx), device=d.device)
    return d * (fx.view(-1, 1, 1) * scale / 1000.0), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cache_m3d_front2_centered")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crop_w", type=int, default=560)
    ap.add_argument("--crop_overlap", type=int, default=72)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--eval_split", action="store_true", help="build from eval manifest/drive instead of the sample cache")
    ap.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    args = ap.parse_args()
    if args.crop_w != FRONT_CROP_WIDTH or args.crop_overlap != FRONT_CROP_OVERLAP:
        raise ValueError(
            "Metric3D crop arguments must match the dataset front2 view layout: "
            f"width={FRONT_CROP_WIDTH}, overlap={FRONT_CROP_OVERLAP}"
        )
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(hub.get_dir()) / "yvanyin_metric3d_main"
    m3d = hub.load(str(repo), "metric3d_vit_small", pretrain=True, source="local").to(device).eval()

    if args.eval_split:
        # Build M3D depth for an EVAL split directly from the dataset (these
        # tiles are not in the train sample cache); indices follow dataset order.
        ds = UnifiedBEVDataset(
            args.manifest, lidar_root=args.lidar_root, drive=args.drive,
            min_target_spacing_m=5.0, max_samples=args.max_samples,
            dense_source_count=8, sparse_source_count=8,
            image_size=(160, 96), max_points_per_view=4096,
        )
    else:
        ds = load_cached_unified_bev(args.cache)
    vpf = ds.views_per_frame
    n = len(ds)

    # manifest lookup for original perspective images
    paths = {}
    with open(args.manifest) as fh:
        for line in fh:
            if line.strip():
                it = json.loads(line)
                paths[(it["drive"], int(it["frame_index"]))] = it["image_00_path"]

    from world3d.unified_bev.data import _read_p_rect_00
    k0_cache: dict[str, torch.Tensor] = {}

    t0 = time.time()
    done = 0
    for i in range(args.start, n):
        dst = out_dir / f"{i:06d}.pt"
        s = ds[i]
        if dst.exists():
            existing = torch.load(dst, map_location="cpu", weights_only=False)
            validate_geometry_blob_identity(existing, s["meta"], context=str(dst))
            continue
        n_views = s["source_rgb"].shape[0]
        depth = torch.zeros(n_views, 96, 160)
        conf = torch.zeros(n_views, 96, 160)
        scale = torch.ones(n_views)
        perspective_crop_starts = None

        # ---- pass 1: all fisheye virtual views in ONE batched call ----
        fish_idx = [v for v in range(n_views) if v % vpf >= 2]
        if fish_idx:
            fish_imgs = torch.stack([s["source_rgb"][v] for v in fish_idx]).to(device)
            fish_fx = torch.tensor([float(s["source_K"][v][0, 0]) for v in fish_idx], device=device)
            d_all, c_all = m3d_run(m3d, fish_imgs, fish_fx)
            for k, v in enumerate(fish_idx):
                valid = s["source_points_valid"][v]
                uv = s["source_points_uv"][v][valid]
                pw = s["source_points_world"][v][valid]
                T = s["source_T_world_cam"][v]
                R, t = T[:3, :3], T[:3, 3]
                z_cam = ((pw - t) @ R)[:, 2]
                px = uv[:, 0].round().long().clamp(0, 159)
                py = uv[:, 1].round().long().clamp(0, 95)
                d_v = d_all[k].cpu(); c_v = c_all[k].cpu()
                dv_pts = d_v[py, px]
                msk = (z_cam > 0.5) & (dv_pts > 0.05) & (c_v[py, px] > 0.05)
                if int(msk.sum()) > 50:
                    sc = float((z_cam[msk] / dv_pts[msk]).median())
                    if 0.3 < sc < 4.0:
                        d_v = d_v * sc
                        scale[v] = sc
                depth[v] = d_v.clamp(0.0, 60.0).to(torch.float16).float()
                conf[v] = c_v.clamp(min=0.0).to(torch.float16).float()

        # ---- pass 2: the two calibrated image_00 crops per frame ----
        for f in range(n_views // vpf):
            drive = s["meta"]["drive"]
            drive = drive[0] if isinstance(drive, list) else drive
            fids = s["meta"]["source_fids"]
            fid = fids[f]
            fid = fid[0] if isinstance(fid, list) else fid
            src_path = paths[(drive, int(fid))]
            ddir = Path(src_path).parents[2]
            if drive not in k0_cache:
                K0 = _read_p_rect_00(ddir / "calibration" / "perspective.txt")
                k0_cache.clear()
                k0_cache[drive] = torch.from_numpy(K0.astype(np.float32))
            K0 = k0_cache[drive]
            fx0 = float(K0[0, 0])
            img = torch.from_numpy(
                np.asarray(Image.open(src_path).convert("RGB"), dtype=np.float32) / 255.0
            ).permute(2, 0, 1).to(device)
            _, W0 = img.shape[-2:]
            W = args.crop_w
            starts = centered_two_crop_starts(
                W0, W, args.crop_overlap, float(K0[0, 2]),
            )
            perspective_crop_starts = starts
            crops = torch.stack([img[..., x0:x0 + W] for x0 in starts])
            d_all, c_all = m3d_run(m3d, crops, fx0)
            for k, _ in enumerate(starts):
                v = f * vpf + k
                d_v = F.interpolate(
                    d_all[k:k + 1, None], size=(96, 160), mode="bilinear", align_corners=False,
                )[0, 0].cpu()
                c_v = F.interpolate(
                    c_all[k:k + 1, None], size=(96, 160), mode="bilinear", align_corners=False,
                )[0, 0].cpu()
                valid = s["source_points_valid"][v]
                uv = s["source_points_uv"][v][valid]
                pw = s["source_points_world"][v][valid]
                T = s["source_T_world_cam"][v]
                R, t = T[:3, :3], T[:3, 3]
                z_cam = ((pw - t) @ R)[:, 2]
                px = uv[:, 0].round().long().clamp(0, 159)
                py = uv[:, 1].round().long().clamp(0, 95)
                dv_pts = d_v[py, px]
                msk = (z_cam > 0.5) & (dv_pts > 0.05) & (c_v[py, px] > 0.05)
                if int(msk.sum()) > 50:
                    sc = float((z_cam[msk] / dv_pts[msk]).median())
                    if 0.3 < sc < 4.0:
                        d_v = d_v * sc
                        scale[v] = sc
                depth[v] = d_v.clamp(0.0, 60.0).to(torch.float16).float()
                conf[v] = c_v.clamp(min=0.0).to(torch.float16).float()

        torch.save({"depth": depth.to(torch.float16),
                    "conf": conf.to(torch.float16),
                    "scale": scale,
                    "sample_identity": geometry_sample_identity(s["meta"]),
                    "crop_policy": "front2_centered_v1",
                    "view_layout_version": VIEW_LAYOUT_VERSION,
                    "perspective_crop_starts": perspective_crop_starts}, dst)
        done += 1
        if done % 64 == 0:
            rate = done / (time.time() - t0)
            print(f"[m3d-cache] {i + 1}/{n} rate={rate * 3600:.0f}/h eta={(n - i - 1) / rate / 60:.0f}min", flush=True)
    print(f"[m3d-cache] DONE from index {args.start} in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

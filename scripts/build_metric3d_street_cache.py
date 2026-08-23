#!/usr/bin/env python3
"""Build the Metric3D dense-depth cache for the unified-BEV v2 lift.

Per cached tile (56 views), for every source view:
  - perspective views (v % views_per_frame == 0): read the ORIGINAL
    1408x376 image + original K, run M3D on three 560-wide standard-aspect
    crops (overlap 72 px, principal point shifted per crop), reassemble the
    strip, resize to 96x160;
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

Storage per tile: depth (56,96,160) fp16, conf (56,96,160) fp16,
scale (56,) fp32  (~3.6 MB; ~7 GB total).
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
from world3d.unified_bev.data import UnifiedBEVDataset, load_cached_unified_bev

MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
IN_H, IN_W = 616, 1064


def m3d_run(m3d, img01: torch.Tensor, fx: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One M3D call on a [0,1] (1,3,h,w) GPU tensor; returns (z-depth, conf) at (h,w)."""
    _, _, h, w = img01.shape
    scale = min(IN_H / h, IN_W / w)
    nh, nw = int(h * scale), int(w * scale)
    x = F.interpolate(img01, size=(nh, nw), mode="bilinear", align_corners=False) * 255.0
    ph0, pw0 = (IN_H - nh) // 2, (IN_W - nw) // 2
    xp = torch.zeros(1, 3, IN_H, IN_W, device=img01.device)
    xp[:, :, ph0:ph0 + nh, pw0:pw0 + nw] = x
    with torch.no_grad():
        d, c, _ = m3d.inference({"input": (xp - MEAN.to(img01.device)) / STD.to(img01.device)})
    d = d[0, 0, ph0:ph0 + nh, pw0:pw0 + nw]
    c = c[0, 0, ph0:ph0 + nh, pw0:pw0 + nw]
    d = F.interpolate(d[None, None], size=(h, w), mode="bilinear")[0, 0]
    c = F.interpolate(c[None, None], size=(h, w), mode="bilinear")[0, 0]
    return d * (fx * scale / 1000.0), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cache_m3d_street")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crop_w", type=int, default=560)
    ap.add_argument("--crop_overlap", type=int, default=72)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--eval_split", action="store_true", help="build from eval manifest/drive instead of the sample cache")
    ap.add_argument("--drive", default="2013_05_28_drive_0003_sync")
    ap.add_argument("--max_samples", type=int, default=32)
    args = ap.parse_args()
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
        if dst.exists():
            continue
        s = ds[i]
        n_views = s["source_rgb"].shape[0]
        depth = torch.zeros(n_views, 96, 160)
        conf = torch.zeros(n_views, 96, 160)
        scale = torch.ones(n_views)

        for v in range(n_views):
            valid = s["source_points_valid"][v]
            uv = s["source_points_uv"][v][valid]
            pw = s["source_points_world"][v][valid]
            T = s["source_T_world_cam"][v]
            R, t = T[:3, :3], T[:3, 3]
            z_cam = ((pw - t) @ R)[:, 2]
            px = uv[:, 0].round().long().clamp(0, 159)
            py = uv[:, 1].round().long().clamp(0, 95)

            if v % vpf == 0:
                # perspective: original resolution + standard-aspect crops
                drive = s["meta"]["drive"][0] if isinstance(s["meta"]["drive"], list) else s["meta"]["drive"]
                fids = s["meta"]["source_fids"]
                fid = int(fids[v // vpf][0] if isinstance(fids[v // vpf], list) else fids[v // vpf])
                src = paths[(drive, fid)]
                ddir = Path(src).parents[2]
                if drive not in k0_cache:
                    K0 = _read_p_rect_00(ddir / "calibration" / "perspective.txt")
                    k0_cache.clear()  # one drive at a time keeps this tiny
                    k0_cache[drive] = torch.from_numpy(K0.astype(np.float32))
                K0 = k0_cache[drive]
                fx0, cx0 = float(K0[0, 0]), float(K0[0, 2])
                img = torch.from_numpy(
                    np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
                ).permute(2, 0, 1).to(device)
                H0, W0 = img.shape[-2:]
                W = args.crop_w
                step = W - args.crop_overlap
                strip_d = torch.zeros(H0, W0)
                strip_c = torch.zeros(H0, W0)
                strip_w = torch.zeros(H0, W0)
                for x0 in range(0, max(W0 - W, 0) + 1, step):
                    crop = img[..., x0:x0 + W][None]
                    d_c, c_c = m3d_run(m3d, crop, fx0)
                    strip_d[:, x0:x0 + W] += d_c.cpu() * c_c.cpu()
                    strip_c[:, x0:x0 + W] += c_c.cpu()
                    strip_w[:, x0:x0 + W] += 1.0
                m = strip_w > 0
                strip_d[m] = strip_d[m] / strip_c[m]  # conf-weighted crop merge
                strip_c[m] = strip_c[m] / strip_w[m]
                # conf-weighted mean of (d*c) then /c gives mean d weighted by c.
                d_v = F.interpolate(strip_d[None, None], size=(96, 160), mode="bilinear")[0, 0]
                c_v = F.interpolate(strip_c[None, None], size=(96, 160), mode="bilinear")[0, 0]
            else:
                img = s["source_rgb"][v][None].to(device)
                fx = float(s["source_K"][v][0, 0])
                d_v, c_v = m3d_run(m3d, img, fx)
                d_v, c_v = d_v.cpu(), c_v.cpu()

            # per-view LiDAR scale anchor (one scalar)
            dv_pts = d_v[py, px]
            msk = (z_cam > 0.5) & (dv_pts > 0.05) & (c_v[py, px] > 0.05)
            if int(msk.sum()) > 50:
                sc = float((z_cam[msk] / dv_pts[msk]).median())
                if 0.3 < sc < 4.0:  # reject pathological anchors
                    d_v = d_v * sc
                    scale[v] = sc
            depth[v] = d_v.clamp(0.0, 60.0).to(torch.float16).float()
            conf[v] = c_v.clamp(min=0.0).to(torch.float16).float()

        torch.save({"depth": depth.to(torch.float16),
                    "conf": conf.to(torch.float16),
                    "scale": scale}, dst)
        done += 1
        if done % 64 == 0:
            rate = done / (time.time() - t0)
            print(f"[m3d-cache] {i + 1}/{n} rate={rate * 3600:.0f}/h eta={(n - i - 1) / rate / 60:.0f}min", flush=True)
    print(f"[m3d-cache] DONE from index {args.start} in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-tile validation: does aspect-correct preprocessing fix VGGT depth?

The sample cache stores perspective views already distorted (1408x376 ->
160x96).  This test feeds VGGT the official recipe instead: read the ORIGINAL
perspective PNG, pad to square (no aspect distortion), resize to 518; virtual
fisheye views are padded 160x96 -> square (aspect preserved).  Depth is mapped
back to the 160x96 grid and correlated against LiDAR camera-z at valid pixels.
Success: mean per-view Pearson r clearly above the ~0.30 of the distorted
pipeline (target > 0.6 to justify rebuilding the cache with this recipe).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/shizhm/Lenovo/vggt")

from vggt.models.vggt import VGGT
from world3d.unified_bev.data import load_cached_unified_bev

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / math.sqrt(float((a ** 2).sum() * (b ** 2).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/shizhm/Downloads/vggt.pt")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--views", choices=["all", "persp"], default="all")
    args = ap.parse_args()
    device = "cuda"

    model = VGGT().to(device)
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.eval()

    paths = {}
    with open(args.manifest) as fh:
        for line in fh:
            if line.strip():
                it = json.loads(line)
                paths[(it["drive"], int(it["frame_index"]))] = it["image_00_path"]

    ds = load_cached_unified_bev(args.cache)
    s = ds[args.index]
    vpf = ds.views_per_frame
    all_views = s["source_rgb"].shape[0]
    view_idx = list(range(0, all_views, vpf)) if args.views == "persp" else list(range(all_views))
    n_views = len(view_idx)
    R = args.resolution

    sq = torch.zeros(n_views, 3, R, R)
    content_box = []  # per view: (top, left, h, w) of content in the square
    for vi, v in enumerate(view_idx):
        if v % vpf == 0:  # perspective: read ORIGINAL image, official pad
            drive = s["meta"]["drive"]
            fid = int(s["meta"]["source_fids"][v // vpf])
            img = Image.open(paths[(drive, fid)]).convert("RGB")
            img = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        else:             # virtual fisheye: cached 160x96, aspect already native
            img = s["source_rgb"][v].clone()
        _, h, w = img.shape
        top, left = (R - round(h * R / max(h, w))) // 2, (R - round(w * R / max(h, w))) // 2
        ch, cw = round(h * R / max(h, w)), round(w * R / max(h, w))
        sq[vi, :, top:top + ch, left:left + cw] = F.interpolate(
            img[None], size=(ch, cw), mode="bilinear", align_corners=False)[0]
        content_box.append((top, left, ch, cw))

    batch = ((sq - MEAN) / STD)[None].to(device)
    with torch.no_grad():
        tokens, ps_idx = model.aggregator(batch)
        depth_map, _ = model.depth_head(tokens, batch, ps_idx)
    depth = depth_map[0, ..., 0].cpu()  # (V, R, R)

    rs_persp, rs_virtual = [], []
    for vi, v in enumerate(view_idx):
        top, left, ch, cw = content_box[vi]
        d_content = depth[vi, top:top + ch, left:left + cw]
        d_grid = F.interpolate(d_content[None, None], size=(96, 160), mode="bilinear")[0, 0]
        valid = s["source_points_valid"][v]
        uv = s["source_points_uv"][v][valid]
        pw = s["source_points_world"][v][valid]
        T = s["source_T_world_cam"][v]
        Rt, t = T[:3, :3], T[:3, 3]
        z_cam = ((pw - t) @ Rt)[:, 2]
        px = uv[:, 0].round().long().clamp(0, 159)
        py = uv[:, 1].round().long().clamp(0, 95)
        dv = d_grid[py, px]
        m = (z_cam > 0.5) & (dv > 0.05) & (torch.isfinite(dv))
        if int(m.sum()) > 100:
            r = pearson(z_cam[m], dv[m])
            (rs_persp if v % vpf == 0 else rs_virtual).append(r)

    def stat(name, xs):
        if xs:
            print(f"{name}: n={len(xs)} mean r={sum(xs)/len(xs):+.3f} min={min(xs):+.3f} max={max(xs):+.3f}")
        else:
            print(f"{name}: none")

    print(f"=== aspect-correct VGGT depth vs LiDAR (tile {args.index}) ===")
    stat("perspective", rs_persp)
    stat("virtual fisheye", rs_virtual)
    allr = rs_persp + rs_virtual
    if allr:
        print(f"overall: mean r={sum(allr)/len(allr):+.3f} over {len(allr)} views "
              f"(distorted-pipeline baseline ~ +0.30)")


if __name__ == "__main__":
    main()

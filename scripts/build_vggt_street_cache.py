#!/usr/bin/env python3
"""Precompute VGGT depth+confidence for all street views of the cached tiles.

Tomorrow's street-arm lift upgrade consumes per-view dense metric geometry
instead of LiDAR-only point features.  Preprocessing matches VGGT's official
loader (center pad-to-square on black, resize, ImageNet norm); the predicted
depth is mapped back to each view's original 96x160 grid for direct use with
the existing uv/splat pipeline.  Depth stays in VGGT's per-sequence gauge;
metric anchoring happens at integration time via the known GT poses + LiDAR.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/shizhm/Lenovo/vggt")

from vggt.models.vggt import VGGT
from world3d.unified_bev.data import load_cached_unified_bev

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/shizhm/Downloads/vggt.pt")
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--out", default="runs/cache_vggt_street")
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VGGT().to(device)
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.eval()

    ds = load_cached_unified_bev(args.cache)
    n = len(ds)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    t0 = time.time()
    for i in range(args.start, n):
        dst = out_dir / f"{i:06d}.pt"
        if dst.exists():
            continue
        views = ds[i]["source_rgb"]  # (V,3,96,160) in [0,1]
        v, _, h, w = views.shape
        side = max(h, w)
        pad_t, pad_l = (side - h) // 2, (side - w) // 2
        padded = F.pad(views, (pad_l, side - w - pad_l, pad_t, side - h - pad_t))
        square = F.interpolate(padded, size=(args.resolution, args.resolution), mode="bilinear")
        batch = ((square - mean) / std).to(device)
        with torch.no_grad():
            tokens, ps_idx = model.aggregator(batch[None])
            depth_map, depth_conf = model.depth_head(tokens, batch[None], ps_idx)
        depth = depth_map[0, ..., 0].cpu()            # (V, R, R)
        conf = depth_conf[0].cpu()
        if conf.dim() == 3:
            conf = conf[..., 0] if depth_map.dim() == 4 else conf
        # Back to the padded square at original scale, then crop the content box.
        back = F.interpolate(depth[None], size=(side, side), mode="bilinear")[0]
        conf_back = F.interpolate(conf[None], size=(side, side), mode="bilinear")[0]
        depth_view = back[:, pad_t:pad_t + h, pad_l:pad_l + w]
        conf_view = conf_back[:, pad_t:pad_t + h, pad_l:pad_l + w]
        torch.save({"depth": depth_view.to(torch.float16),
                    "conf": conf_view.to(torch.float16)}, dst)
        if (i + 1) % 64 == 0:
            rate = (i + 1 - args.start) / (time.time() - t0)
            print(f"[vggt-cache] {i + 1}/{n} rate={rate*3600:.0f}/h eta={(n - i - 1) / rate / 60:.0f}min", flush=True)
    print(f"[vggt-cache] DONE {n} samples in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

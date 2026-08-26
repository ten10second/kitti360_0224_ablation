#!/usr/bin/env python3
"""E0: frozen-DINOv2 satellite-feature probe for the world-state prior.

Measures the information ceiling of the satellite pathway BEFORE any state
machinery: regress the LiDAR height/density targets directly from frozen
DINOv2 features of the same 200x200 south-up satellite raster that
``SatelliteInitializer`` consumes, and compare against placebo arms.

Arms (identical readout head and training loop):
  dino    frozen DINOv2 tokens (16x16 -> 200x200) + fixed XY encoding
  xy      zero features (same shape) + fixed XY encoding      [placebo]
  shuffle frozen DINOv2 tokens, per-scene spatial permutation [layout binding]
  scratch small trainable CNN on the raw satellite raster     [encoder regime]

Baselines: per-scene mean (constant prediction of the train-set mean).

Verdict frame (pre-registered):
  dino >> xy on the held-out scene  -> satellite information exists; E1 has
                                       something to work with
  dino ~= xy                        -> DINO domain gap; swap the frozen
                                       backbone (design unchanged)
  dino fits train but fails held-out-> scene starvation; scale scenes
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.losses import masked_smooth_l1  # noqa: E402
from world3d.unified_bev.models import fixed_relative_xy_encoding  # noqa: E402
from world3d.unified_bev.state_models import FrozenDINOv2  # noqa: E402
from world3d.unified_bev.world_data import WorldStateSceneDataset  # noqa: E402

XY_CHANNELS = 16


def load_scenes(root: str):
    ds = WorldStateSceneDataset(root)
    scenes = []
    for i in range(len(ds)):
        inputs, sup, blob = ds[i]
        scenes.append({
            "scene_id": blob["scene_id"],
            "sat": inputs.satellite_bev.float(),           # (1,3,200,200)
            "height": sup.height.float(),                  # (1,1,200,200)
            "density": sup.density.float(),
            "valid": sup.world_valid.bool(),
            "route": sup.future_route_support.bool(),
        })
    return scenes


class ProbeHead(nn.Module):
    """Same shape as the SatelliteInitializer write path plus a 2-channel
    geometric readout; trainable in every arm."""

    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class ScratchEncoder(nn.Module):
    """Small trainable CNN over the raw satellite raster (reference regime)."""

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 + XY_CHANNELS, out_ch, 5, stride=2, padding=2),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch), nn.GELU(),
        )

    def forward(self, x):
        return F.interpolate(self.net(x), size=(200, 200), mode="bilinear", align_corners=False)


def evaluate(head, inputs, targets, masks, scratch_encoder=None):
    with torch.no_grad():
        pred = head(scratch_encoder(inputs) if scratch_encoder is not None else inputs)
    out = {}
    for region, mask in masks.items():
        m = mask.to(pred.device)
        if m.sum() < 256:
            out[region] = None
            continue
        h_pred, d_pred = pred[:, :1], pred[:, 1:2]
        h_gt, d_gt = targets["height"].to(pred.device), targets["density"].to(pred.device)
        h_err = (h_pred - h_gt)[m]
        d_err = (d_pred - d_gt)[m]
        vx = h_pred[m].flatten().cpu().numpy()
        vy = h_gt[m].flatten().cpu().numpy()
        if np.std(vx) < 1e-6 or np.std(vy) < 1e-6:
            pearson = None
        else:
            pearson = float(np.corrcoef(vx, vy)[0, 1])
        out[region] = {
            "height_mae": float(h_err.abs().mean()),
            "height_bias": float(h_err.mean()),
            "height_pearson": pearson,
            "density_mae": float(d_err.abs().mean()),
            "cells": int(m.sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_scenes", required=True)
    ap.add_argument("--eval_scenes", required=True)
    ap.add_argument("--out", default="runs/world_state_e0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = load_scenes(args.train_scenes)
    ev = load_scenes(args.eval_scenes)
    print(f"[e0] train scenes={len(train)} eval scenes={len(ev)} device={device}")
    if len(train) < 4:
        raise SystemExit("need at least 4 training scenes")

    dino = FrozenDINOv2().to(device).eval()
    feat_dim = dino.embed_dim
    xy_enc = fixed_relative_xy_encoding(XY_CHANNELS, 200, 200, tile_size_m=100.0).to(device)

    # precompute per-scene DINO features once (frozen)
    print("[e0] extracting frozen DINOv2 features...")
    for sc in train + ev:
        with torch.no_grad():
            sc["dino_feat"] = dino(sc["sat"].to(device)).cpu()
    del dino
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # constant baseline: predict the train-set mean everywhere
    mean_h = torch.cat([s["height"][s["valid"]] for s in train]).mean().item()
    mean_d = torch.cat([s["density"][s["valid"]] for s in train]).mean().item()

    results = {"arms": {}, "baseline_mean": {"height": mean_h, "density": mean_d}, "config": vars(args)}
    for arm in ("dino", "xy", "shuffle", "scratch"):
        torch.manual_seed(args.seed)
        rng = torch.Generator(device=device).manual_seed(args.seed)
        if arm == "scratch":
            encoder = ScratchEncoder().to(device)
            head = ProbeHead(64).to(device)
            params = list(encoder.parameters()) + list(head.parameters())
        else:
            encoder = None
            head = ProbeHead(feat_dim + XY_CHANNELS).to(device)
            params = list(head.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr)

        def make_x(sc, split):
            if arm == "scratch":
                sat = sc["sat"].to(device)
                return torch.cat([sat, xy_enc.expand(sat.shape[0], -1, -1, -1)], dim=1)
            if arm == "xy":
                feat = torch.zeros(1, feat_dim, 16, 16, device=device)
            else:
                feat = sc["dino_feat"].to(device)
                if arm == "shuffle":
                    # per-scene permutation: no stable layout->position binding
                    perm = torch.randperm(256, generator=rng, device=device)
                    feat = feat.reshape(1, feat_dim, 256)[:, :, perm].reshape(1, feat_dim, 16, 16)
            feat = F.interpolate(feat, size=(200, 200), mode="bilinear", align_corners=False)
            return torch.cat([feat, xy_enc], dim=1)

        t0 = time.time()
        order = np.arange(len(train))
        running = 0.0
        for step in range(1, args.steps + 1):
            if step % len(train) == 1:
                np.random.shuffle(order)
            sc = train[order[(step - 1) % len(train)]]
            x = make_x(sc, "train")
            pred = head(encoder(x) if encoder is not None else x)
            mask = sc["valid"].to(device)
            loss = masked_smooth_l1(pred[:, :1], sc["height"].to(device), mask) + \
                masked_smooth_l1(pred[:, 1:2], sc["density"].to(device), mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            if step % 200 == 0:
                print(f"[e0:{arm}] step={step}/{args.steps} loss={running / 200:.4f} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
                running = 0.0

        # train-fit check + held-out evaluation
        eval_rows = []
        for split, scenes in (("train", train), ("heldout", ev)):
            for sc in scenes:
                if arm == "scratch":
                    x = make_x(sc, split)
                    x = encoder(x)
                else:
                    x = make_x(sc, split)
                row = evaluate(head, x, sc, {"valid": sc["valid"], "route": sc["route"]})
                row.update({"split": split, "scene_id": sc["scene_id"]})
                eval_rows.append(row)
        results["arms"][arm] = eval_rows
        tr = [r for r in eval_rows if r["split"] == "train"]
        ho = [r for r in eval_rows if r["split"] == "heldout"]
        def _m(rows, key, region):
            vals = [r[region][key] for r in rows if r[region]]
            return float(np.mean(vals)) if vals else None
        print(f"[e0:{arm}] TRAIN fit   valid: hMAE={_m(tr, 'height_mae', 'valid'):.3f} "
              f"r={_m(tr, 'height_pearson', 'valid'):.3f}")
        print(f"[e0:{arm}] HELD-OUT    valid: hMAE={_m(ho, 'height_mae', 'valid'):.3f} "
              f"r={_m(ho, 'height_pearson', 'valid'):.3f} | "
              f"route: hMAE={_m(ho, 'height_mae', 'route'):.3f} "
              f"r={_m(ho, 'height_pearson', 'route'):.3f}", flush=True)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"[e0] wrote {out / 'summary.json'}")

    # constant-baseline reference numbers on the held-out scenes
    for sc in ev:
        v = sc["valid"]
        h_mae = float((sc["height"] - mean_h).abs()[v].mean())
        print(f"[e0:mean-baseline] {sc['scene_id'][:40]} valid hMAE={h_mae:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment 3: Decoder-only Adaptation.

Isolate whether the satellite-recovered latent contains information that a
frozen decoder cannot read.  Everything is frozen except the decoder, which
is fine-tuned on the train split while reading either satellite-recovered
latents (mode=sat) or XY-prior latents (mode=xy).  If the representation
transfer is real, AdaptedDecoder(z_sat) must beat AdaptedDecoder(z_xy) even
though the frozen decoder shows no such advantage.

Saves a Stage-A-style checkpoint (ground encoder untouched, decoder adapted)
so the standard paired evaluator consumes it unchanged.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import UnifiedBEVDataset, load_cached_unified_bev
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b", required=True)
    ap.add_argument("--mode", choices=["sat", "xy"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--sparse_source_choices", default="1,2,4,8")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ray_samples", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--render_weight", type=float, default=1.0)
    ap.add_argument("--depth_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_cached_unified_bev(args.cache)
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder = ColumnFieldDecoder(hidden=args.hidden, samples=args.ray_samples).to(device)
    ground.load_state_dict(a["ground"])
    decoder.load_state_dict(a["decoder"])
    b = torch.load(args.stage_b, map_location=device, weights_only=False)
    bcfg = b.get("config", {})
    family = bcfg.get("sat_encoder", "cnn")
    if family == "heightmap":
        sat = HeightMapSatellitePrior(bev_height=ds.bev_size, bev_width=ds.bev_size,
                                      **bcfg.get("sat_encoder_kwargs", {})).to(device)
    elif family == "vit":
        sat = SatelliteViTEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size,
                                  **bcfg.get("sat_encoder_kwargs", {})).to(device)
    else:
        sat = SatelliteBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    sat.load_state_dict(b["satellite_encoder"])
    mode = "residual" if args.mode == "sat" else "coordinate_only"
    completion = LatentCompletion(mode=mode, bev_height=ds.bev_size,
                                  bev_width=ds.bev_size, tile_size_m=ds.tile_size_m).to(device)
    # In xy mode the caller passes the coordinate_only Stage-B checkpoint, so
    # b['completion'] is already the right control; no second load needed.

    for m in (ground, sat, completion):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    iterator = iter(loader)
    choices = tuple(int(x) for x in args.sparse_source_choices.split(","))
    running, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        n_sparse = rng.choice(choices)
        with torch.no_grad():
            sp = slice(0, n_sparse * ds.views_per_frame)
            z_gnd, cov = ground(
                batch["source_rgb"][:, sp], batch["source_points_world"][:, sp],
                batch["source_points_uv"][:, sp], batch["source_points_valid"][:, sp],
                batch["origin_xy"], ds.bev_resolution_m,
            )
            if args.mode == "sat":
                if family == "heightmap":
                    prior, _, _ = sat(batch["satellite"], z_gnd, ds.tile_size_m, 0.196)
                else:
                    prior = sat(batch["satellite"], ds.tile_size_m, 0.196)
            else:
                prior = torch.zeros_like(z_gnd)
            z_hat = completion(prior, z_gnd, cov, n_sparse, args.dense_sources)
        pred_rgb, pred_depth, _ = decoder.render(
            z_hat, batch["target_K"], batch["target_T_world_cam"], batch["origin_xy"],
            tile_size_m=ds.tile_size_m, image_size=ds.image_size,
        )
        render_loss = F.smooth_l1_loss(pred_rgb, batch["target_rgb"])
        dm = batch["target_depth_mask"]
        depth_loss = F.smooth_l1_loss(pred_depth[dm], batch["target_depth"][dm]) if dm.any() else pred_depth.mean() * 0
        loss = args.render_weight * render_loss + args.depth_weight * depth_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        running += float(loss.detach())
        if step == 1 or step % 500 == 0:
            print(f"[adapt-{args.mode}] step={step}/{args.steps} loss={running/500:.5f} "
                  f"render={float(render_loss):.5f} Ns={n_sparse} elapsed={time.time()-t0:.0f}s", flush=True)
            running = 0.0
    torch.save({
        "ground": ground.state_dict(), "decoder": decoder.state_dict(),
        "config": dict(a.get("config", {})), "adapted_from": args.stage_a,
        "adapt_mode": args.mode, "steps": args.steps,
    }, out / "stage_a.pt")
    print(f"[adapt-{args.mode}] saved {out / 'stage_a.pt'}")


if __name__ == "__main__":
    main()

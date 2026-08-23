#!/usr/bin/env python3
"""DPT readout experiment: is satellite information IN the latent?

Exp1's light conv probe favored XY's regular Fourier structure, showing that
"information presence" depends on the readout family.  This experiment closes
the question with a strong dense-prediction readout: for each frozen latent
(star / sat / xy / gnd) an independent DPT-style head is trained FROM SCRATCH
on the train split and evaluated on unseen drive 0003.

If DPT(sat) > DPT(xy) -- especially at sparse Ns -- the satellite information
is present in the latent and the original ground-trained readout is simply
too weak to extract it.  If the gap persists even under a strong readout, the
information itself is absent (domain failure of the satellite pathway).

The head is a conv DPT: a 4-stage encoder pyramid over the BEV latent with
progressive upsampling fusion to a 128x128 height map (parameters ~2M, a
~20x stronger readout family than the Exp1 probe).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import UnifiedBEVDataset, load_cached_unified_bev
from world3d.unified_bev.models import (
    GroundBEVEncoder,
    HeightMapSatellitePrior,
    LatentCompletion,
)
from scripts.probe_unified_bev_height import dem_from_sample, spearman


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DPThead(nn.Module):
    """Conv DPT: encoder pyramid + progressive upsampling fusion."""

    def __init__(self, ch: int = 64, width: int = 128):
        super().__init__()
        self.stem = ConvBlock(ch, width)               # 256 @ 128
        self.stage2 = ConvBlock(width, width * 2)      # 512 @ 64
        self.stage3 = ConvBlock(width * 2, width * 4)  # 1024 @ 32
        self.stage4 = ConvBlock(width * 4, width * 4)  # 1024 @ 16
        self.proj4 = nn.Conv2d(width * 4, width, 1)
        self.proj3 = nn.Conv2d(width * 4, width, 1)
        self.proj2 = nn.Conv2d(width * 2, width, 1)
        self.fuse2 = ConvBlock(width * 2, width)
        self.fuse1 = ConvBlock(width * 2, width)
        self.out = nn.Sequential(
            nn.Conv2d(width, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, z):
        s1 = self.stem(z)
        s2 = self.stage2(F.avg_pool2d(s1, 2))
        s3 = self.stage3(F.avg_pool2d(s2, 2))
        s4 = self.stage4(F.avg_pool2d(s3, 2))
        up = lambda x, s: F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
        f4 = self.proj4(s4)                 # 256 @ 16
        f3 = self.proj3(s3) + up(f4, s3)    # 256 @ 32
        f2 = self.fuse2(torch.cat([self.proj2(s2), up(f3, s2)], dim=1))   # 256 @ 64
        f1 = self.fuse1(torch.cat([s1, up(f2, s1)], dim=1))               # 256 @ 128
        return F.softplus(self.out(f1)).clamp(max=60.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b_sat", required=True)
    ap.add_argument("--stage_b_xy", required=True)
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--n_sparse", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_manifest", default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl")
    ap.add_argument("--eval_drive", default="2013_05_28_drive_0003_sync")
    ap.add_argument("--eval_samples", type=int, default=32)
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--out_dir", default="runs/dpt_readout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_cached_unified_bev(args.cache)
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    ground.load_state_dict(a["ground"]); ground.eval()
    bs = torch.load(args.stage_b_sat, map_location=device, weights_only=False)
    bx = torch.load(args.stage_b_xy, map_location=device, weights_only=False)
    sat = HeightMapSatellitePrior(bev_height=ds.bev_size, bev_width=ds.bev_size,
                                  **bs.get("config", {}).get("sat_encoder_kwargs", {})).to(device)
    sat.load_state_dict(bs["satellite_encoder"]); sat.eval()
    comp_sat = LatentCompletion(mode="residual", bev_height=ds.bev_size,
                                bev_width=ds.bev_size, tile_size_m=ds.tile_size_m).to(device)
    comp_sat.load_state_dict(bs["completion"]); comp_sat.eval()
    comp_xy = LatentCompletion(mode="coordinate_only", bev_height=ds.bev_size,
                               bev_width=ds.bev_size, tile_size_m=ds.tile_size_m).to(device)
    comp_xy.load_state_dict(bx["completion"]); comp_xy.eval()
    for m in (ground, sat, comp_sat, comp_xy):
        for p in m.parameters():
            p.requires_grad_(False)

    def latents_of(s):
        with torch.no_grad():
            views = s["source_rgb"][None].to(device)
            pts = s["source_points_world"][None].to(device)
            uv = s["source_points_uv"][None].to(device)
            val = s["source_points_valid"][None].to(device)
            org = s["origin_xy"][None].to(device)
            z_star, _ = ground(views, pts, uv, val, org, ds.bev_resolution_m)
            sp = slice(0, args.n_sparse * ds.views_per_frame)
            z_gnd, cov = ground(views[:, sp], pts[:, sp], uv[:, sp], val[:, sp],
                                org, ds.bev_resolution_m)
            prior, _, _ = sat(s["satellite"][None].to(device), z_gnd, ds.tile_size_m, 0.196)
            z_sat_hat = comp_sat(prior, z_gnd, cov, args.n_sparse, 8)
            z_xy_hat = comp_xy(torch.zeros_like(prior), z_gnd, cov, args.n_sparse, 8)
        return {"star": z_star[0], "gnd": z_gnd[0], "sat": z_sat_hat[0], "xy": z_xy_hat[0]}

    heads = {k: DPThead().to(device) for k in ("star", "gnd", "xy", "sat")}
    opts = {k: torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=1e-4)
            for k, h in heads.items()}
    print(f"DPT params/head: {sum(p.numel() for p in heads['sat'].parameters())/1e6:.2f}M", flush=True)

    order = list(range(len(ds)))
    import random
    rng = random.Random(args.seed)
    t0 = time.time()
    step = 0
    while step < args.steps:
        rng.shuffle(order)
        for i in order:
            if step >= args.steps:
                break
            s = ds[i]
            dem, covered = dem_from_sample(s, ds.bev_resolution_m, ds.bev_size)
            if not covered.any():
                continue
            dem_d = dem.to(device); cov_d = covered.to(device)
            latents = latents_of(s)
            total = 0.0
            for k, h in heads.items():
                pred = h(latents[k][None])[0, 0]
                loss = F.smooth_l1_loss(pred[cov_d], dem_d[cov_d])
                opts[k].zero_grad(set_to_none=True); loss.backward(); opts[k].step()
                total += float(loss)
            step += 1
            if step % 500 == 0:
                print(f"dpt step={step}/{args.steps} mean_loss={total/4:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    for k, h in heads.items():
        torch.save(h.state_dict(), out_dir / f"dpt_{k}_ns{args.n_sparse}.pt")

    ev = UnifiedBEVDataset(
        args.eval_manifest, lidar_root=args.lidar_root, drive=args.eval_drive,
        min_target_spacing_m=5.0, max_samples=args.eval_samples,
        dense_source_count=8, sparse_source_count=args.n_sparse,
        image_size=ds.image_size, max_points_per_view=4096,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ev, batch_size=1, shuffle=False, num_workers=0)
    stats = {k: {"rmse": [], "mae": [], "r": [], "rho": []} for k in heads}
    with torch.no_grad():
        for batch in loader:
            s = {k: (v[0] if torch.is_tensor(v) else v) for k, v in batch.items()}
            dem, covered = dem_from_sample(s, ds.bev_resolution_m, ds.bev_size)
            if not covered.any():
                continue
            latents = latents_of(s)
            dem_d = dem.to(device); cov_d = covered.to(device)
            for k, h in heads.items():
                h.eval()
                pred = h(latents[k][None])[0, 0]
                p, t = pred[cov_d], dem_d[cov_d]
                stats[k]["rmse"].append(float(((p - t) ** 2).mean().sqrt()))
                stats[k]["mae"].append(float((p - t).abs().mean()))
                pp, tt = p - p.mean(), t - t.mean()
                dn = float(pp.norm() * tt.norm())
                stats[k]["r"].append(float((pp * tt).sum() / dn) if dn > 1e-8 else 0.0)
                stats[k]["rho"].append(spearman(p.cpu(), t.cpu()))
    print(f"\n=== DPT readout on unseen {args.eval_drive} (Ns={args.n_sparse}) ===")
    print(f"{'latent':>6s} {'RMSE':>7s} {'MAE':>7s} {'Pearson':>8s} {'Spearman':>9s}")
    for k in ("star", "gnd", "xy", "sat"):
        row = stats[k]
        m = lambda key: sum(row[key]) / len(row[key])
        print(f"{k:>6s} {m('rmse'):>7.3f} {m('mae'):>7.3f} {m('r'):>+8.3f} {m('rho'):>+9.3f}")


if __name__ == "__main__":
    main()

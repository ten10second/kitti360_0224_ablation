#!/usr/bin/env python3
"""Experiment 1: Height / Geometry Probe.

Freeze every latent (Z* dense, z_hat satellite, z_hat XY, z_gnd sparse) and
train one lightweight convolutional probe f_h(Z(x,y)) -> h(x,y) to predict the
dense-LiDAR height map.  If satellite-recovered latents carry vertical
structure beyond positional priors, the probe must read heights more
accurately from z_hat_sat than from z_hat_XY, especially at sparse Ns.

Train/eval protocol: probe trains on the geographic train split (cache) and
evaluates on unseen drive 0003 -- the probe itself is also isolation-tested.
Reports RMSE / MAE / Pearson / Spearman per cell over covered BEV cells.
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
from world3d.unified_bev.data import load_cached_unified_bev
from world3d.unified_bev.geometry import bilinear_splat, height_statistics
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    LatentCompletion,
    HeightMapSatellitePrior,
    SatelliteBEVEncoder,
    SatelliteViTEncoder,
)


def build_sat_encoder(cfg, bev, tile_m):
    family = cfg.get("sat_encoder", "cnn")
    if family == "heightmap":
        return HeightMapSatellitePrior(bev_height=bev, bev_width=bev, **cfg.get("sat_encoder_kwargs", {}))
    if family == "vit":
        return SatelliteViTEncoder(bev_height=bev, bev_width=bev, **cfg.get("sat_encoder_kwargs", {}))
    return SatelliteBEVEncoder(bev_height=bev, bev_width=bev)


def dem_from_sample(s, res_m, bev):
    """Relative-height DEM and covered-cell mask for one sample.

    KITTI-360 world-Z is absolute altitude (~115 m).  The probe target is
    height above the local ground: anchor each tile at the 10th percentile of
    covered cells (road level), then clamp noise tails.  The coverage mask is
    the splat count map, not h != 0.
    """
    pts = s["source_points_world"][None]        # (1, N, P, 3)
    valid = s["source_points_valid"][None]      # (1, N, P)
    ones = valid.float().unsqueeze(-1)          # (1, N, P, 1)
    _, cnt = bilinear_splat(
        ones, pts[..., :2], valid,
        origin_xy=s["origin_xy"][None],
        resolution_m=res_m, height=bev, width=bev,
    )
    covered = cnt[0, 0] > 0
    h_raw, _ = height_statistics(pts, valid, s["origin_xy"][None], res_m, bev, bev)
    h_raw = h_raw[0, 0]
    ground = torch.quantile(h_raw[covered].float(), 0.1)
    h_ref = (h_raw - ground).clamp(min=-2.0, max=30.0)
    return h_ref, covered


class HeightProbe(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, z):
        return F.softplus(self.net(z)).clamp(max=60.0)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = float(ra.norm() * rb.norm())
    return float((ra * rb).sum() / d) if d > 1e-8 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b_sat", required=True)
    ap.add_argument("--stage_b_xy", required=True)
    ap.add_argument("--cache", default="runs/cache_grad_2048_dir")
    ap.add_argument("--n_sparse", type=int, default=2)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_manifest", default="dataset_splits/kitti360_geofence_buffer30/test_manifest.jsonl")
    ap.add_argument("--eval_drive", default="2013_05_28_drive_0003_sync")
    ap.add_argument("--eval_samples", type=int, default=32)
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ds = load_cached_unified_bev(args.cache)
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder_cfg = a.get("config", {})
    ground.load_state_dict(a["ground"]); ground.eval()
    bs = torch.load(args.stage_b_sat, map_location=device, weights_only=False)
    bx = torch.load(args.stage_b_xy, map_location=device, weights_only=False)
    sat_cfg = bs.get("config", {})
    sat = build_sat_encoder(sat_cfg, ds.bev_size, ds.tile_size_m).to(device)
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

    probe = HeightProbe().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

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
            if sat_cfg.get("sat_encoder") == "heightmap":
                prior, _, _ = sat(s["satellite"][None].to(device), z_gnd, ds.tile_size_m, 0.196)
            else:
                prior = sat(s["satellite"][None].to(device), ds.tile_size_m, 0.196)
            z_sat_hat = comp_sat(prior, z_gnd, cov, args.n_sparse, args.dense_sources)
            z_xy_hat = comp_xy(torch.zeros_like(prior), z_gnd, cov, args.n_sparse, args.dense_sources)
        return {"star": z_star[0], "gnd": z_gnd[0], "sat": z_sat_hat[0], "xy": z_xy_hat[0]}

    t0 = time.time()
    step = 0
    while step < args.steps:
        for i in range(len(ds)):
            if step >= args.steps:
                break
            s = ds[i]
            dem, covered = dem_from_sample(s, ds.bev_resolution_m, ds.bev_size)
            if not covered.any():
                continue
            latents = latents_of(s)
            dem_d = dem.to(device); cov_d = covered.to(device)
            losses = []
            for z in latents.values():
                pred = probe(z[None])[0, 0]
                losses.append(F.smooth_l1_loss(pred[cov_d], dem_d[cov_d]))
            loss = torch.stack(losses).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            if step % 500 == 0:
                print(f"probe step={step}/{args.steps} loss={float(loss):.4f} elapsed={time.time()-t0:.0f}s", flush=True)

    torch.save(probe.state_dict(), "runs/probe_height_ns%d.pt" % args.n_sparse)
    print("probe saved: runs/probe_height_ns%d.pt" % args.n_sparse, flush=True)

    # ---- evaluate on the unseen drive ----
    from world3d.unified_bev.data import UnifiedBEVDataset
    ev = UnifiedBEVDataset(
        args.eval_manifest, lidar_root=args.lidar_root, drive=args.eval_drive,
        min_target_spacing_m=5.0, max_samples=args.eval_samples,
        dense_source_count=args.dense_sources, sparse_source_count=args.n_sparse,
        image_size=ds.image_size, max_points_per_view=4096,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ev, batch_size=1, shuffle=False, num_workers=0)
    stats = {k: {"rmse": [], "mae": [], "r": [], "rho": []} for k in ("star", "gnd", "sat", "xy")}
    with torch.no_grad():
        for batch in loader:
            s = {k: (v[0] if torch.is_tensor(v) else v) for k, v in batch.items()}
            dem, covered = dem_from_sample(s, ds.bev_resolution_m, ds.bev_size)
            if not covered.any():
                continue
            latents = latents_of(s)
            dem_d = dem.to(device); cov_d = covered.to(device)
            for name, z in latents.items():
                pred = probe(z[None])[0, 0]
                p, t = pred[cov_d], dem_d[cov_d]
                stats[name]["rmse"].append(float(((p - t) ** 2).mean().sqrt()))
                stats[name]["mae"].append(float((p - t).abs().mean()))
                pp, tt = p - p.mean(), t - t.mean()
                dn = float(pp.norm() * tt.norm())
                stats[name]["r"].append(float((pp * tt).sum() / dn) if dn > 1e-8 else 0.0)
                stats[name]["rho"].append(spearman(p.cpu(), t.cpu()))
    print(f"\n=== Height probe on unseen {args.eval_drive} (Ns={args.n_sparse}) ===")
    print(f"{'latent':>6s} {'RMSE':>7s} {'MAE':>7s} {'Pearson':>8s} {'Spearman':>9s}")
    for name in ("star", "gnd", "xy", "sat"):
        row = stats[name]
        m = lambda k: sum(row[k]) / len(row[k])
        print(f"{name:>6s} {m('rmse'):>7.3f} {m('mae'):>7.3f} {m('r'):>+8.3f} {m('rho'):>+9.3f}")


if __name__ == "__main__":
    main()

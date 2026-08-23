#!/usr/bin/env python3
"""Diagnostic: is RGB nadir round-trip supervision viable with the frozen decoder?

For each eval tile, renders each candidate latent straight down through the
frozen decoder and compares against the aligned satellite crop (grayscale):
masked L1 (same form as the training loss) and per-tile Pearson correlation
(the viability signal).  If even the dense reference latent Z* cannot be
rendered top-down with any correlation to the satellite layout, an RGB nadir
loss would be noise and the supervision design must change.

Saves a PNG panel (reference vs nadir renders of Z*/z_gnd/z_hat) for the
first two tiles.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world3d.unified_bev.data import SAT_M_PER_PX, UnifiedBEVDataset
from world3d.unified_bev.models import (
    ColumnFieldDecoder,
    GroundBEVEncoder,
    LatentCompletion,
    SatelliteBEVEncoder,
    nadir_distance,
    satellite_bev_crop,
)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    return float((a * b).sum() / denom) if denom > 1e-8 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", required=True)
    ap.add_argument("--stage_b", action="append", default=[],
                    help="repeatable: name=path/to/stage_b.pt")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--lidar_root", default="/media/shizhm/sda2/KITTI360_lidar/data_3d_raw")
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--dense_sources", type=int, default=8)
    ap.add_argument("--sparse_sources", type=int, default=2)
    ap.add_argument("--max_points", type=int, default=4096)
    ap.add_argument("--min_target_spacing_m", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--panel_out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = UnifiedBEVDataset(
        args.manifest, lidar_root=args.lidar_root, drive=args.drive,
        min_target_spacing_m=args.min_target_spacing_m, max_samples=args.max_samples,
        dense_source_count=args.dense_sources, sparse_source_count=args.sparse_sources,
        image_size=(160, 96), max_points_per_view=args.max_points,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    a = torch.load(args.stage_a, map_location=device, weights_only=False)
    ground = GroundBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    decoder = ColumnFieldDecoder(
        hidden=a.get("config", {}).get("hidden", 128),
        samples=a.get("config", {}).get("ray_samples", 24),
    ).to(device)
    ground.load_state_dict(a["ground"])
    decoder.load_state_dict(a["decoder"])
    ground.eval(); decoder.eval()

    variants = {}
    for spec in args.stage_b:
        name, path = spec.split("=", 1)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        mode = cfg.get("fusion", "residual")
        completion = LatentCompletion(
            mode=mode, bev_height=ds.bev_size, bev_width=ds.bev_size,
            tile_size_m=ds.tile_size_m,
        ).to(device)
        completion.load_state_dict(ckpt["completion"])
        completion.eval()
        entry = {"completion": completion, "sat": None}
        if mode in ("residual", "satellite_only"):
            sat = SatelliteBEVEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
            sat.load_state_dict(ckpt["satellite_encoder"])
            sat.eval()
            entry["sat"] = sat
        variants[name] = entry

    stats: dict[str, dict[str, list[float]]] = {}
    panels = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            z_star, ref_mask = ground(
                batch["source_rgb"], batch["source_points_world"], batch["source_points_uv"],
                batch["source_points_valid"], batch["origin_xy"], ds.bev_resolution_m,
            )
            sp = slice(0, args.sparse_sources * ds.views_per_frame)
            z_sparse, sparse_mask = ground(
                batch["source_rgb"][:, sp], batch["source_points_world"][:, sp],
                batch["source_points_uv"][:, sp], batch["source_points_valid"][:, sp],
                batch["origin_xy"], ds.bev_resolution_m,
            )
            ref = satellite_bev_crop(batch["satellite"], ds.tile_size_m, SAT_M_PER_PX, ds.bev_size)

            def nadir_of(z):
                rgb, _ = decoder.render_nadir(
                    z, batch["origin_xy"], tile_size_m=ds.tile_size_m, bev_size=ds.bev_size,
                )
                return rgb

            def record(tag: str, rgb: torch.Tensor):
                row = stats.setdefault(tag, {"l1": [], "r": []})
                row["l1"].append(float(nadir_distance(rgb, ref, 1.0 - ref_mask)))
                row["r"].append(pearson(rgb.mean(dim=1), ref.mean(dim=1)))

            record("z_star", nadir_of(z_star))
            record("z_sparse", nadir_of(z_sparse))
            for name, entry in variants.items():
                if entry["sat"] is not None:
                    z_sat = entry["sat"](batch["satellite"], ds.tile_size_m, SAT_M_PER_PX)
                else:
                    z_sat = torch.zeros_like(z_sparse)
                z_hat = entry["completion"](
                    z_sat, z_sparse, sparse_mask,
                    n_sparse=args.sparse_sources, dense_sources=args.dense_sources,
                )
                record(name, nadir_of(z_hat))
                if idx < 2:
                    panels.append((name, idx, ref, nadir_of(z_hat)))
            if idx < 2:
                panels.append(("z_star", idx, ref, nadir_of(z_star)))

    print(f"tiles={args.max_samples} Ns={args.sparse_sources} mask=unobserved "
          f"(~{float((1 - ref_mask.mean())):.2f})")
    for tag, row in stats.items():
        n = len(row["l1"])
        mean_l1 = sum(row["l1"]) / n
        mean_r = sum(row["r"]) / n
        pos_r = sum(r > 0 for r in row["r"])
        print(f"{tag:>12s}: nadir_L1={mean_l1:.4f}  pearson_r={mean_r:+.4f}  (r>0 on {pos_r}/{n})")

    if args.panel_out and panels:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable; skip panel")
            return
        tile_ids = sorted({idx for _, idx, _, _ in panels})
        names = [n for n, _, _, _ in panels if n != "z_star"][:2] + ["z_star"]
        names = list(dict.fromkeys(names))
        fig, axes = plt.subplots(len(tile_ids), 1 + len(names), figsize=(3 * (1 + len(names)), 3 * len(tile_ids)))
        axes = axes.reshape(len(tile_ids), -1)
        for col, name in enumerate(names):
            axes[0, col].set_title(name if col else "satellite ref")
            for row_i, tidx in enumerate(tile_ids):
                if col == 0:
                    ref_img = next(ref for n, i, ref, _ in panels if i == tidx)
                    axes[row_i, col].imshow(ref_img[0].permute(1, 2, 0).cpu().clamp(0, 1))
                else:
                    img = next(im for n, i, _, im in panels if i == tidx and n == name)
                    axes[row_i, col].imshow(img[0].permute(1, 2, 0).cpu().clamp(0, 1))
        for ax in axes.flat:
            ax.axis("off")
        out = Path(args.panel_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"panel={out}")


if __name__ == "__main__":
    main()

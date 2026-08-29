#!/usr/bin/env python3
"""Headline QA figure: the world state "developing" as the car drives.

Columns: t=0 (satellite prior only) -> t=2 -> t=5 -> t=8 -> ground truth.
Row 1: predicted height (frozen reader) over satellite grayscale, route dots
       green=traversed / gray=ahead.  Row 2: |pred - truth| on valid cells.
Held-out scene, aligned control, frozen-VGGT measurements.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.readouts import freeze_module  # noqa: E402
from world3d.unified_bev.state_models import (  # noqa: E402
    BEVSurfaceDensityDecoder,
    BEVWorldHeightDecoder,
    EvidenceAwareUpdater,
    GroundMeasurementEncoder,
    SatelliteInitializer,
    WorldGeometryEncoder,
)
from world3d.unified_bev.world_checkpoints import (  # noqa: E402
    validate_assimilation_checkpoint,
    validate_world_interface_checkpoint,
)
from world3d.unified_bev.world_data import (  # noqa: E402
    WorldStateSceneDataset,
    collate_world_state,
    spec_from_inputs,
)
from world3d.unified_bev.world_state import ahead_mask, visited_mask  # noqa: E402
from world3d.unified_bev.world_vggt import (  # noqa: E402
    chunk_measurement_from_cache,
    load_world_vggt_cache,
)

SNAP_AT = (0, 2, 5, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="runs/world_state_targets_smoke")
    ap.add_argument("--vggt_cache", default="runs/world_state_vggt_smoke")
    ap.add_argument("--interface", default="runs/world_state_v3_formal_20260828/interface/world_interface.pt")
    ap.add_argument("--assimilation", default="runs/world_state_assim_depth_smoke/assimilation.pt")
    ap.add_argument("--out", default="runs/world_state_assim_depth_smoke/qa_filling_in.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = WorldStateSceneDataset(args.scenes)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_world_state)
    iface = torch.load(args.interface, map_location=device, weights_only=False)
    fp = validate_world_interface_checkpoint(iface)
    b = torch.load(args.assimilation, map_location=device, weights_only=False)
    validate_assimilation_checkpoint(b, fp)

    world_enc = WorldGeometryEncoder(**iface["encoder_config"]).to(device)
    height_r = BEVWorldHeightDecoder(**iface["height_reader_config"]).to(device)
    density_r = BEVSurfaceDensityDecoder(**iface["density_reader_config"]).to(device)
    world_enc.load_state_dict(iface["encoder"])
    height_r.load_state_dict(iface["height_reader"])
    density_r.load_state_dict(iface["density_reader"])
    sat_init = SatelliteInitializer(bev_height=ds.bev_size, bev_width=ds.bev_size,
                                    tile_size_m=ds.tile_size_m).to(device)
    updater = EvidenceAwareUpdater().to(device)
    sat_init.load_state_dict(b["satellite_initializer"])
    updater.load_state_dict(b["updater"])
    meas_enc = GroundMeasurementEncoder(bev_height=ds.bev_size, bev_width=ds.bev_size).to(device)
    meas_enc.load_state_dict(b["measurement_encoder"])
    for m in (world_enc, height_r, density_r, sat_init, updater, meas_enc):
        freeze_module(m)

    inputs, sup, blob = next(iter(loader))
    spec = spec_from_inputs(inputs, device=device)
    sat = inputs.satellite_bev.to(device)
    height = sup.height.to(device)
    valid = sup.world_valid.to(device)
    chunk_support = sup.chunk_lidar_support.to(device)
    cache = load_world_vggt_cache(args.vggt_cache, blob["scene_id"],
                                  str(blob["world_target_version"]), str(blob["world_target_hash"]))

    with torch.no_grad():
        state = sat_init(sat, spec)
        snapshots = {0: state}
        for t in range(chunk_support.shape[1]):
            meas = chunk_measurement_from_cache(
                meas_enc, cache["chunks"][str(t + 1)],
                origin_xy=spec.origin_xy, resolution_m=float(spec.resolution_m),
                z_datum_m=spec.z_datum_m, chunk_index=t + 1,
                query_fid=int(blob["chunk_table"][t]["core_fid"]), detach=True,
            )
            state = updater(state, meas).state
            snapshots[t + 1] = state

    # route polyline: centroid of each chunk's LiDAR support
    h_, w_ = chunk_support.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h_, dtype=torch.float32),
                            torch.arange(w_, dtype=torch.float32), indexing="ij")
    centers = []
    for t in range(chunk_support.shape[1]):
        m = chunk_support[0, t, 0].to(device)
        centers.append((float((xx.to(device) * m).sum() / m.sum()),
                        float((yy.to(device) * m).sum() / m.sum())))
    sat_np = sat[0].cpu().numpy()
    sat_gray = sat_np.mean(0) if sat_np.ndim == 3 else sat_np
    truth = height[0, 0].cpu().numpy()
    valid_np = valid[0, 0].cpu().numpy()

    snaps = {}
    for t in SNAP_AT:
        if t in snapshots:
            snaps[t] = height_r(snapshots[t].latent)[0, 0].cpu().numpy()
    snaps["GT"] = truth
    cols = list(snaps.keys())

    vmin = min(np.nanmin(truth[valid_np]), min(np.nanmin(v[valid_np]) for v in snaps.values()))
    vmax = max(np.nanmax(truth[valid_np]), max(np.nanmax(v[valid_np]) for v in snaps.values()))
    absmax = max(np.nanmax(np.abs(v[valid_np] - truth[valid_np])) for k, v in snaps.items() if k != "GT")
    err_lim = float(np.quantile([np.abs(v[valid_np] - truth[valid_np]).ravel() for k, v in snaps.items()
                                 if k != "GT"], 0.95))

    fig, axes = plt.subplots(2, len(cols), figsize=(3.1 * len(cols), 6.6))
    for j, t in enumerate(cols):
        vis = visited_mask(chunk_support, t if t != "GT" else chunk_support.shape[1])[0, 0].cpu().numpy()
        ahead = ahead_mask(sup.future_route_support.to(device),
                           visited_mask(chunk_support, t if t != "GT" else 0))[0, 0].cpu().numpy()
        for row, ax in enumerate((axes[0, j], axes[1, j])):
            ax.imshow(sat_gray, cmap="gray", alpha=0.30)
            if row == 0:
                v = snaps[t]
                im = ax.imshow(np.where(valid_np, v, np.nan), cmap="RdBu_r",
                               vmin=vmin, vmax=vmax)
                if t == "GT":
                    ax.set_title("ground truth", fontsize=11)
                else:
                    vis_mae = np.nan if vis.sum() < 50 else float(np.abs(v[vis & valid_np] - truth[vis & valid_np]).mean())
                    ah_mae = np.nan if ahead.sum() < 50 else float(np.abs(v[ahead & valid_np] - truth[ahead & valid_np]).mean())
                    ax.set_title(f"t={t}  vis {vis_mae:.2f} / ahead {ah_mae:.2f} m", fontsize=11)
            else:
                if t == "GT":
                    e = np.zeros_like(truth)
                    im = ax.imshow(np.where(valid_np, e, np.nan), cmap="magma", vmin=0, vmax=err_lim)
                    ax.set_title("|error| (m)", fontsize=11)
                else:
                    e = np.abs(v - truth)
                    im = ax.imshow(np.where(valid_np, e, np.nan), cmap="magma", vmin=0, vmax=err_lim)
                    ax.set_title(f"mean {float(e[valid_np].mean()):.2f} m", fontsize=11)
            # route dots: green = already traversed at this t, gray = ahead
            for ci, (cx, cy) in enumerate(centers):
                traversed = (t == "GT") or (ci < (t if t != "GT" else len(centers)))
                ax.plot(cx, cy, "o", ms=4,
                        color="#1a9641" if traversed else "#bbbbbb",
                        mec="k", mew=0.4)
            xs = [c[0] for c in centers]; ys = [c[1] for c in centers]
            ax.plot(xs, ys, "-", color="#1a9641", lw=1.0, alpha=0.55)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.colorbar(axes[0, 0].images[1], ax=axes[0, :], fraction=0.02, pad=0.01, label="height (m)")
    fig.colorbar(axes[1, 0].images[1], ax=axes[1, :], fraction=0.02, pad=0.01, label="abs error (m)")
    fig.suptitle(f"world state assimilation — held-out {blob['scene_id'].split('__')[0][-10:]} "
                 f"(green route = traversed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}  (absmax err {absmax:.2f} m, err color scale 0..{err_lim:.2f})")


if __name__ == "__main__":
    main()

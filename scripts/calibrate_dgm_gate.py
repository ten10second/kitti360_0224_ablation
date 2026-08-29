#!/usr/bin/env python3
"""Calibrate the dgm_gate_mad_m threshold: trusted-tier MAD distribution.

Replays every chunk measurement with the DGM anchor engaged (gate disabled)
and records the trusted-tier VGGT-vs-DGM residual MAD per chunk, across all
training scenes + the held-out scene.  The gate must sit above the normal
envelope-noise band and below the artifact band (bridges/tile edges: 2-3 m).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world3d.unified_bev.dgm import DgmAnchor, DgmTileSet, anchor_tile_tensor
from world3d.unified_bev.state_models import GroundMeasurementEncoder
from world3d.unified_bev.world_data import WorldStateSceneDataset
from world3d.unified_bev.world_vggt import chunk_measurement_from_cache, load_world_vggt_cache

DGM_TILES = "/media/shizhm/sda1/proposal/Cross-View Conditional Coding of Route-Specific Gaussian Scenes/outputs/bw_dgm_dom/tiles"
KITTI_ROOT = "/media/shizhm/sda1/KITTI-360"

PAIRS = [
    ("train", "runs/world_state_e0/targets_train_v3", "runs/world_state_e0/vggt_cache_train_v3"),
    ("heldout", "runs/world_state_targets_smoke", "runs/world_state_vggt_smoke"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiles = DgmTileSet.from_dir(Path(DGM_TILES))
    rows = []
    for split, scenes_dir, cache_dir in PAIRS:
        ds = WorldStateSceneDataset(scenes_dir)
        for inputs, sup, blob in ds:
            iface_fp = None  # encoder weights irrelevant to ground_field qa
            enc = GroundMeasurementEncoder(
                bev_height=ds.bev_size, bev_width=ds.bev_size,
                dgm_gate_mad_m=1e9,  # disable fallback: measure raw MAD
            ).to(device).eval()
            scene_id = str(blob["scene_id"])
            cache = load_world_vggt_cache(cache_dir, scene_id,
                                          str(blob["world_target_version"]), str(blob["world_target_hash"]))
            anchor = DgmAnchor.from_blob(blob, tiles, Path(KITTI_ROOT))
            z_t, v_t = anchor_tile_tensor(anchor, inputs.origin_xy[0].numpy(),
                                          ds.bev_size, float(ds.resolution_m), device)
            from world3d.unified_bev.world_data import spec_from_inputs
            spec = spec_from_inputs(inputs, device=device)
            for t in range(sup.chunk_lidar_support.shape[1]):
                meas = chunk_measurement_from_cache(
                    enc, cache["chunks"][str(t + 1)],
                    origin_xy=spec.origin_xy, resolution_m=float(spec.resolution_m),
                    z_datum_m=spec.z_datum_m, chunk_index=t + 1,
                    query_fid=int(blob["chunk_table"][t]["core_fid"]),
                    detach=True, dgm_abs_z=z_t, dgm_valid=v_t,
                )
                rows.append({"split": split, "scene_id": scene_id, "t": t + 1,
                             "mad_m": float(meas.dgm_qa["dgm_mad_m"]),
                             "delta_m": float(meas.dgm_qa["delta_m"]),
                             "n_trusted": int(meas.dgm_qa["n_trusted_dgm"]),
                             "n_tier2": int(meas.dgm_qa["n_tier2"])})
            print(f"[{split}] {scene_id}: done", flush=True)

    out = Path("runs/dgm_alignment_check/trusted_tier_mad.json")
    out.write_text(json.dumps(rows, indent=2))
    mads = np.array([r["mad_m"] for r in rows])
    tr = np.array([r["mad_m"] for r in rows if r["split"] == "train"])
    ho = np.array([r["mad_m"] for r in rows if r["split"] == "heldout"])
    t2 = np.array([r["n_tier2"] for r in rows])
    print(f"\nchunks={len(rows)}  tier2 cells median={np.median(t2):.0f}")
    print(f"MAD all:  med={np.median(mads):.3f}  p90={np.quantile(mads,.9):.3f}  p99={np.quantile(mads,.99):.3f}  max={mads.max():.3f}")
    print(f"MAD train: med={np.median(tr):.3f}  p90={np.quantile(tr,.9):.3f}  p99={np.quantile(tr,.99):.3f}  max={tr.max():.3f}")
    print(f"MAD heldout(0003): med={np.median(ho):.3f}  range=[{ho.min():.3f},{ho.max():.3f}]")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

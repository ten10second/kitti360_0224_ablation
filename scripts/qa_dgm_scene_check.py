#!/usr/bin/env python3
"""Scene-level DGM agreement check: our static road-cell heights vs DGM1.

For every existing world-target blob: fit world->UTM on the blob's own
geometry frames, resample DGM1 onto the tile grid, and compare on ROAD cells
(semantic_top in {road, sidewalk, parking} = bare ground by construction).
This is the number that gates using DGM as the measurement anchor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qa_dgm_alignment import load_dgm_rasters, sample_dgm_bilinear  # noqa: E402

ROAD_LABELS = {6, 7, 8, 9, 22}  # ground/road/sidewalk/parking/terrain = bare earth
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def load_pose_utm(sequence: Path):
    poses = {}
    for line in (sequence / "cam0_to_world.txt").read_text().splitlines():
        v = np.fromstring(line, sep=" ")
        if v.size == 17:
            poses[int(v[0])] = v[1:].reshape(4, 4)
    utm = {}
    for p in (sequence / "oxts" / "data").glob("*.txt"):
        v = p.read_text().split()
        if len(v) >= 2:
            utm[int(p.stem)] = np.asarray(TRANSFORMER.transform(float(v[1]), float(v[0])))
    return poses, utm


def fit_affine(poses, utm, frames):
    common = [f for f in frames if f in poses and f in utm]
    if len(common) < 10:
        raise RuntimeError(f"only {len(common)} frames with pose+oxts")
    src = np.asarray([poses[f][:2, 3] for f in common])
    dst = np.asarray([utm[f] for f in common])
    sm, tm = src.mean(0), dst.mean(0)
    transform, _, _, _ = np.linalg.lstsq(src - sm, dst - tm, rcond=None)
    resid = np.linalg.norm((src - sm) @ transform - (dst - tm), axis=1)
    return transform, float(np.sqrt(np.mean(resid**2)))


def main():
    dgm_root = Path("/media/shizhm/sda1/proposal/Cross-View Conditional Coding of Route-Specific Gaussian Scenes/outputs/bw_dgm_dom/tiles")
    seq_root = Path("/media/shizhm/sda1/KITTI-360")
    rasters = load_dgm_rasters(dgm_root)

    scene_dirs = [Path("runs/world_state_e0/targets_train_v3"), Path("runs/world_state_targets_smoke")]
    rows = []
    for d in scene_dirs:
        for blob_path in sorted(Path(d).glob("*.pt")):
            blob = torch.load(blob_path, map_location="cpu", weights_only=False)
            scene_id = blob["scene_id"]
            drive = scene_id.split("__")[0]
            seq = seq_root / drive
            poses, utm = load_pose_utm(seq)
            frames = [int(f) for c in blob["chunk_table"] for f in c["geometry_fids"]]
            transform, fit_rmse = fit_affine(poses, utm, frames)

            origin = blob["origin_xy"].numpy()
            res_m = float(blob["resolution_m"])
            h, w = blob["height"].shape[-2:]
            xs = origin[0] + (np.arange(w) + 0.5) * res_m
            ys = origin[1] + (np.arange(h) + 0.5) * res_m
            gx, gy = np.meshgrid(xs, ys)
            src = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
            # world -> utm: (src - sm) @ transform + tm; sm/tm folded into transform? No:
            # refit incl. translation
            pframes = [f for f in frames if f in poses and f in utm]
            smat = np.asarray([poses[f][:2, 3] for f in pframes])
            dmat = np.asarray([utm[f] for f in pframes])
            sm, tm = smat.mean(0), dmat.mean(0)
            utm_xy = (src - sm) @ transform + tm

            dgm = sample_dgm_bilinear(rasters, utm_xy).reshape(h, w)
            height = blob["height"][0].numpy().astype(np.float64)
            valid = blob["world_valid"][0].numpy().astype(bool)
            stop = blob["semantic_top"][0].numpy()
            road = np.isin(stop, list(ROAD_LABELS))
            m = valid & road & np.isfinite(dgm)
            n = int(m.sum())
            if n < 256:
                rows.append({"scene_id": scene_id, "status": "few road cells", "n": n,
                             "fit_rmse_m": fit_rmse})
                continue
            diff = height[m] - dgm[m]
            med = float(np.median(diff))
            rows.append({
                "scene_id": scene_id, "status": "ok", "road_cells": n,
                "fit_rmse_m": round(fit_rmse, 3),
                "median_diff_m": round(med, 3),
                "mad_m": round(float(np.median(np.abs(diff - med))), 3),
                "median_abs_diff_m": round(float(np.median(np.abs(diff))), 3),
                "p90_abs_diff_m": round(float(np.quantile(np.abs(diff), 0.9)), 3),
                "frac_lt_0p3": round(float(np.mean(np.abs(diff) < 0.3)), 3),
                "frac_lt_0p5": round(float(np.mean(np.abs(diff) < 0.5)), 3),
            })
            print(json.dumps(rows[-1]), flush=True)
    out = Path("runs/dgm_alignment_check/scene_agreement.json")
    out.write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        mads = [r["mad_m"] for r in ok]
        c3 = [r["frac_lt_0p3"] for r in ok]
        print(f"\nSUMMARY {len(ok)}/{len(rows)} scenes: MAD med={np.median(mads):.3f}m  "
              f"|diff|<0.3 med={np.median(c3):.1%}  after per-scene offset")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

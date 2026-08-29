#!/usr/bin/env python3
"""Windowed trajectory-vs-DGM vertical residual: does the drive-level offset
drift along the route?

For every 100-frame window: median(pose_z - sensor_h - DGM(direct oxts UTM)).
Uses the oxts UTM directly (no affine), so this isolates vertical drift from
any horizontal-fit error.  A frozen drive-level offset is estimated on the
first window; drift = how far later windows move away from it.
"""
from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from pyproj import Transformer

from qa_dgm_alignment import load_dgm_rasters, sample_dgm_bilinear  # reuse

DRIVES = ("0003", "0000", "0006", "0009", "0010")
WINDOW = 100
SENSOR_H = 0.93


def main():
    dgm_root = Path("/media/shizhm/sda1/proposal/Cross-View Conditional Coding of Route-Specific Gaussian Scenes/outputs/bw_dgm_dom/tiles")
    seq_root = Path("/media/shizhm/sda1/KITTI-360")
    rasters = load_dgm_rasters(dgm_root)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

    report = {}
    for drive in DRIVES:
        seq = seq_root / f"2013_05_28_drive_{drive}_sync"
        poses, utm = {}, {}
        for line in (seq / "cam0_to_world.txt").read_text().splitlines():
            v = np.fromstring(line, sep=" ")
            if v.size == 17:
                poses[int(v[0])] = v[1:].reshape(4, 4)
        for p in (seq / "oxts" / "data").glob("*.txt"):
            v = p.read_text().split()
            if len(v) >= 2:
                utm[int(p.stem)] = np.asarray(transformer.transform(float(v[1]), float(v[0])))
        frames = sorted(set(poses) & set(utm))
        xy = np.asarray([utm[f] for f in frames])
        z = np.asarray([poses[f][2, 3] for f in frames])
        dgm = sample_dgm_bilinear(rasters, xy)
        resid = z - SENSOR_H - dgm
        ok = np.isfinite(resid)
        frames, resid = np.asarray(frames)[ok], resid[ok]

        # frozen offset from first window (mirrors the probe's calibration)
        first = resid[:WINDOW]
        delta = float(np.median(first))
        windows = []
        for s in range(0, len(frames) - WINDOW + 1, WINDOW):
            med = float(np.median(resid[s:s + WINDOW]))
            windows.append({"frame_start": int(frames[s]), "median_m": round(med, 3),
                            "post_offset_m": round(med - delta, 3)})
        drift = [w["post_offset_m"] for w in windows]
        report[drive] = {
            "frozen_offset_m": round(delta, 3),
            "n_windows": len(windows),
            "post_offset_first": drift[0] if drift else None,
            "post_offset_last": drift[-1] if drift else None,
            "max_abs_post_offset_m": max((abs(d) for d in drift), default=None),
            "windows": windows,
        }
        w_line = " ".join(f"{d:+.2f}" for d in drift)
        print(f"[{drive}] frozen={delta:+.2f}  per-window post-offset (m): {w_line}", flush=True)

    out = Path("runs/dgm_alignment_check/vertical_drift.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

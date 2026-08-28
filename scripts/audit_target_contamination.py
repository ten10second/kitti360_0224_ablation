#!/usr/bin/env python3
"""P0 audit: classify v2-target vs official-semantics disagreements by the
cell's official semantic composition, so the v3 layer-selection rule is
grounded in measurement rather than taste.

For one scene blob: bin the official static cloud onto our 200x200 grid,
split every covered cell's points by GROUND-vs-TOP composition, and report
how often each composition explains a >1 m disagreement with v2's p90.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEMANTICS_ROOT = "/media/shizhm/sda2/KITTI360_lidar/data_3d_semantics/data_3d_semantics/train"

# v3 label policy (provisional; sources: geometry+colour inference + instance
# coding.  To be diffed against kitti360Scripts helpers/labels.py when online)
GROUND_IDS = {7, 8, 9, 22}        # road, sidewalk, parking-ish, terrain
TOP_IDS = {11, 21, 34, 6, 12}     # building, vegetation, wall/fence family
VEHICLE_CANDIDATES = {26, 24}     # dynamic person/car carriers (rare in static)


def load_ply(path):
    props, n = [], None
    with open(path, "rb") as fh:
        while True:
            line = fh.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("property"):
                p = line.split()
                props.append((p[2], {"float": "<f4", "uchar": "u1", "int": "<i4"}[p[1]]))
            if line == "end_header":
                off = fh.tell()
                break
    return np.fromfile(path, dtype=np.dtype(props), count=n, offset=off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--resolution", type=float, default=0.5)
    args = ap.parse_args()
    import torch
    from world3d.unified_bev.world_data import WorldStateSceneDataset

    ds = WorldStateSceneDataset(Path(args.blob).parent)
    idx = [i for i, r in enumerate(ds.rows) if (Path(args.blob).parent / r["file"]).name == Path(args.blob).name][0]
    _, sup, blob = ds[idx]
    origin = blob["origin_xy"].numpy().astype(np.float64)
    datum = float(blob["z_datum_m"])
    H_ours = sup.height[0, 0].numpy()
    V_ours = sup.world_valid[0, 0].numpy()

    files = sorted(glob.glob(f"{SEMANTICS_ROOT}/{blob['drive']}/static/*.ply"))
    per_cell = defaultdict(list)
    for f in files:
        d = load_ply(f)
        m = (
            (d["x"] >= origin[0]) & (d["x"] < origin[0] + 200 * args.resolution)
            & (d["y"] >= origin[1]) & (d["y"] < origin[1] + 200 * args.resolution)
            & (d["confidence"] >= 0.5) & (d["visible"] == 1)
        )
        pts = d[m]
        if len(pts) == 0:
            continue
        col = np.floor((pts["x"] - origin[0]) / args.resolution).astype(int)
        row = np.floor((pts["y"] - origin[1]) / args.resolution).astype(int)
        for r, c, z, lab, inst in zip(row, col, pts["z"], pts["semantic"], pts["instance"]):
            if 0 <= r < 200 and 0 <= c < 200:
                per_cell[(int(r), int(c))].append((float(z), int(lab), inst // 1000 != lab))

    buckets = defaultdict(lambda: [0, 0, 0, [], 0.0])  # n, ours_higher, ours_lower, diffs, share_toppts
    comp_stats = defaultdict(int)
    compared = 0
    for (r, c), zs in per_cell.items():
        if not V_ours[r, c] or len(zs) < 4:
            continue
        z_arr = np.array([z for z, _, _ in zs])
        top_mask = np.array([lab in TOP_IDS for _, lab, _ in zs])
        ground_mask = np.array([lab in GROUND_IDS for _, lab, _ in zs])
        thing_mask = np.array([is_thing for _, _, is_thing in zs])
        top_share = float(top_mask.mean())
        comp_stats[f"top>{top_share:.0%}"[:9]] += 1
        h_official = (
            np.quantile(z_arr[top_mask], 0.95) if top_share >= 0.15 and top_mask.any()
            else np.quantile(z_arr[ground_mask] if ground_mask.any() else z_arr, 0.5)
        )
        diff = H_ours[r, c] - (h_official - datum)
        key = ("ground-dominated" if top_share < 0.15 else "top-mixed")
        b = buckets[key]
        b[0] += 1
        b[1] += diff > 1.0
        b[2] += diff < -1.0
        b[3].append(diff)
        b[4] += abs(diff)
        compared += 1
        _ = thing_mask

    print(f"scene={Path(args.blob).name[:52]}")
    print(f"compared cells={compared}")
    for key, (n, hi, lo, diffs, adiff) in sorted(buckets.items()):
        print(f"{key:20s}: n={n:6d}  median_diff={np.median(diffs):+.3f}  MAD={np.median(np.abs(diffs-np.median(diffs))):.3f}"
              f"  OURS>1m={100*hi/n:5.2f}%  OURS<-1m={100*lo/n:5.2f}%")
    print("\ncomposition histogram:", dict(sorted(comp_stats.items())))


if __name__ == "__main__":
    main()

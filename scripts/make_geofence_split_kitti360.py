#!/usr/bin/env python3
"""Geofence train/test split for KITTI-360 (ICASSP27 protocol).

Replicates the split method of
  /home/shizhm/codespace/CS2S_pose_environment/dataset/kitti_raw_sat_lidar_geofence_test2_buffer30
  split_strategy = "remove_train_frames_within_gps_buffer_of_test_route", buffer_m = 30

Adapted to KITTI-360:
  - all 8 local drives share one metric world frame (poses.txt), so buffer
    distances are computed directly in that frame (no UTM conversion needed);
  - drives cluster geographically: {0002,0003} / {0005,0006} / {0009,0010} share
    streets (min cross distance 0.3-33.8 m), {0000} and {0007} are isolated
    (>1.6 km). Cluster membership therefore decides the split:
        test  = {0002, 0003}   (one contiguous held-out urban region)
        val   = {0007}         (isolated region, geographically independent)
        train = {0000, 0005, 0006, 0009, 0010}
  - buffer removal is still executed generically (train/val frames within
    buffer_m of any test-route point are dropped); expected ~0 removals given
    cluster distances, but the mechanism matches the reference method.

Output: train/val/test_manifest.jsonl + manifest_stats.json
  Schema (per frame): drive, frame_index, world_x, world_y, split,
    image_00_path, satellite_path, pose_path, cam0_to_world_path,
    nearest_test_route_distance_m, geofence_removed(false).
Only frames with an exact pose entry AND satellite file are included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BUFFER_M = 30.0
TEST_DRIVES = ["2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync"]
VAL_DRIVES = ["2013_05_28_drive_0007_sync"]
# everything else -> train


def load_poses(drive_dir: Path):
    out = {}
    for line in (drive_dir / "poses.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        T = np.eye(4)
        T[:3, :] = np.array(parts[1:], dtype=np.float64).reshape(3, 4)
        out[int(parts[0])] = T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/media/shizhm/Lenovo/KITTI-360")
    ap.add_argument("--out_dir", default="dataset_splits/kitti360_geofence_buffer30")
    ap.add_argument("--buffer_m", type=float, default=BUFFER_M)
    ap.add_argument("--test_drives", nargs="*", default=TEST_DRIVES)
    ap.add_argument("--val_drives", nargs="*", default=VAL_DRIVES)
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_drives = sorted([d.name for d in root.glob("2013_*_sync")])
    train_drives = [d for d in all_drives if d not in args.test_drives and d not in args.val_drives]
    print(f"drives: test={args.test_drives} val={args.val_drives} train={train_drives}")

    # ---- collect frames ----
    frames = {}  # drive -> {fid: (T_imu, has_sat, img_path)}
    for dname in all_drives:
        d = root / dname
        poses = load_poses(d)
        sat_dir = d / "satellite"
        per = {}
        img_dir = d / "image_00" / "data_rect"
        if not img_dir.exists():
            img_dir = d / "image_00" / "data_rgb"
        for imgp in sorted(img_dir.glob("*.png")):
            fid = int(imgp.stem)
            if fid not in poses:
                continue
            satp = None
            for ext in (".png", ".jpg"):
                c = sat_dir / f"{fid:010d}{ext}"
                if c.exists():
                    satp = c
                    break
            if satp is None:
                continue
            per[fid] = (poses[fid], imgp, satp)
        frames[dname] = per
        print(f"  {dname}: {len(per)} usable frames (exact pose + satellite)")

    # ---- test route point cloud (world xy, subsampled) ----
    test_xy = []
    for d in args.test_drives:
        for fid, (T, _, _) in frames[d].items():
            test_xy.append([T[0, 3], T[1, 3]])
    test_xy = np.array(test_xy)
    print(f"test route points: {len(test_xy)}")

    def nearest_test_dist(xy):
        # chunked to bound memory
        out = np.full(len(xy), np.inf)
        for s in range(0, len(test_xy), 4096):
            chunk = test_xy[s:s + 4096]
            d2 = ((xy[:, None, :] - chunk[None, :, :]) ** 2).sum(-1)
            out = np.minimum(out, d2.min(1))
        return np.sqrt(out)

    # ---- build manifests ----
    stats = {
        "buffer_m": args.buffer_m,
        "split_strategy": "remove_train_frames_within_world_buffer_of_test_route",
        "test_drives": args.test_drives,
        "val_drives": args.val_drives,
        "train_drives_input": train_drives,
        "frame_source": "poses.txt (exact match) + satellite file present",
    }
    removed = {"train": 0, "val": 0}
    for split, drives in (("train", train_drives), ("val", args.val_drives), ("test", args.test_drives)):
        recs = []
        xs_all = []
        for d in drives:
            for fid, (T, imgp, satp) in sorted(frames[d].items()):
                rec = {
                    "drive": d,
                    "frame_index": fid,
                    "world_x": float(T[0, 3]),
                    "world_y": float(T[1, 3]),
                    "split": split,
                    "image_00_path": str(imgp),
                    "satellite_path": str(satp),
                    "pose_path": str(root / d / "poses.txt"),
                    "cam0_to_world_path": str(root / d / "cam0_to_world.txt"),
                }
                xs_all.append([T[0, 3], T[1, 3]])
                recs.append(rec)
        xy = np.array(xs_all)
        dist = nearest_test_dist(xy)
        kept = []
        for rec, di in zip(recs, dist):
            rec["nearest_test_route_distance_m"] = float(di)
            if split in ("train", "val") and di < args.buffer_m:
                rec["geofence_removed"] = True
                removed[split] += 1
            else:
                rec["geofence_removed"] = False
                kept.append(rec)
        with open(out_dir / f"{split}_manifest.jsonl", "w") as f:
            for rec in kept:
                f.write(json.dumps(rec) + "\n")
        stats[f"num_{split}_input"] = len(recs)
        stats[f"num_{split}_kept"] = len(kept)
        stats[f"num_{split}_removed"] = len(recs) - len(kept)
        print(f"{split}: kept {len(kept)}/{len(recs)} (buffer-removed {len(recs)-len(kept)})")

    # ---- overlap sanity: min distance between kept train/val points and test points ----
    kept_train = [json.loads(l) for l in open(out_dir / "train_manifest.jsonl")]
    txy = np.array([[r["world_x"], r["world_y"]] for r in kept_train])
    dmin = nearest_test_dist(txy)
    stats["nearest_test_route_distance_m"] = {
        "min": float(dmin.min()),
        "p01": float(np.percentile(dmin, 1)),
        "p50": float(np.percentile(dmin, 50)),
        "max": float(dmin.max()),
    }
    stats["drive_overlap_after_split"] = []
    stats["out_dir"] = str(out_dir)
    with open(out_dir / "manifest_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"kept-train -> test-route min distance: {dmin.min():.1f} m (must be >= {args.buffer_m})")
    print(f"stats -> {out_dir/'manifest_stats.json'}")


if __name__ == "__main__":
    main()

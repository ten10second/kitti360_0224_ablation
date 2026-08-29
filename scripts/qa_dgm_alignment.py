#!/usr/bin/env python3
"""Can the BW DGM1 terrain model anchor our KITTI-360 world heights?

Adapted from the Cross-View project's run_kitti360_dtm_residual.py (same
world->UTM least-squares fit, same 1-DoF frozen vertical registration, same
voxel/stat machinery); only the segment selection is changed: instead of
hard-coded preferred segments, take the longest contiguous run of DGM-covered
frames per drive, calibrate the vertical offset on its first 30 frames, and
evaluate on the rest.  Verdict numbers: post-registration ground-fraction and
below-DTM fraction vs shifted/constant controls.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from pyproj import Transformer

DRIVES = ("0003", "0000", "0006", "0009", "0010")
MAX_EVAL_FRAMES = 600


def load_pose_utm(sequence: Path):
    poses = {}
    for line in (sequence / "cam0_to_world.txt").read_text().splitlines():
        v = np.fromstring(line, sep=" ")
        if v.size == 17:
            poses[int(v[0])] = v[1:].reshape(4, 4)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    utm = {}
    for path in (sequence / "oxts" / "data").glob("*.txt"):
        v = path.read_text().split()
        if len(v) >= 2:
            utm[int(path.stem)] = np.asarray(transformer.transform(float(v[1]), float(v[0])))
    return poses, utm


def _load_xyz_z(path_bytes: bytes) -> np.ndarray:
    """Fast column-2 read of a 1M-line ASCII xyz raster (~1-2 s vs ~40 s loadtxt)."""
    tokens = np.array(path_bytes.split(), dtype=np.float32)
    if tokens.size % 3:
        raise ValueError("xyz raster token count not divisible by 3")
    return tokens.reshape(-1, 3)[:, 2]


def load_dgm_rasters(root: Path):
    rasters = {}
    pattern = re.compile(r"_32_(\d+)_(\d+)_1_bw_2022\.xyz$")
    for archive in sorted(root.glob("dgm*.zip")):
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                match = pattern.search(name)
                if not match:
                    continue
                values = _load_xyz_z(handle.read(name))
                if values.size != 1_000_000:
                    raise ValueError(f"Expected 1M samples in {name}, got {values.size}")
                rasters[(int(match.group(1)), int(match.group(2)))] = values.reshape(1000, 1000)
                print(f"  loaded {name}", flush=True)
    if not rasters:
        raise FileNotFoundError(f"No DGM rasters in {root}")
    return rasters


def sample_dgm_bilinear(rasters, xy: np.ndarray) -> np.ndarray:
    out = np.full(xy.shape[0], np.nan, dtype=np.float64)
    east = np.floor(xy[:, 0] / 1000).astype(np.int64)
    north = np.floor(xy[:, 1] / 1000).astype(np.int64)
    for key, raster in rasters.items():
        mask = (east == key[0]) & (north == key[1])
        if not np.any(mask):
            continue
        x = np.clip(xy[mask, 0] - key[0] * 1000 - 0.5, 0, 999)
        y = np.clip(key[1] * 1000 + 999.5 - xy[mask, 1], 0, 999)
        c0, r0 = np.floor(x).astype(int), np.floor(y).astype(int)
        c1, r1 = np.minimum(c0 + 1, 999), np.minimum(r0 + 1, 999)
        fx, fy = x - c0, y - r0
        top = raster[r0, c0] * (1 - fx) + raster[r0, c1] * fx
        bottom = raster[r1, c0] * (1 - fx) + raster[r1, c1] * fx
        out[mask] = top * (1 - fy) + bottom * fy
    return out


def fit_alignment(poses, utm, frames):
    common = sorted(set(frames) & set(poses) & set(utm))
    if len(common) < 10:
        raise RuntimeError(f"Only {len(common)} pose/OXTS matches")
    source = np.asarray([poses[f][:2, 3] for f in common])
    target = np.asarray([utm[f] for f in common])
    sm, tm = source.mean(0), target.mean(0)
    transform, _, _, _ = np.linalg.lstsq(source - sm, target - tm, rcond=None)
    residual = np.linalg.norm((source - sm) @ transform - (target - tm), axis=1)
    return common, transform, float(np.sqrt(np.mean(residual**2))), float(residual.max())


def longest_contiguous_run(frames, step=1):
    """Longest run of consecutive frame ids (KITTI-360 frames step by 1)."""
    frames = sorted(frames)
    best_start = frames[0]
    best_len = cur_len = 1
    cur_start = frames[0]
    for a, b in zip(frames, frames[1:]):
        if b - a <= step:
            cur_len += 1
        else:
            cur_start, cur_len = b, 1
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len
    return best_start, best_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-root", type=Path, default=Path("/media/shizhm/sda1/KITTI-360"))
    ap.add_argument("--lidar-root", type=Path, default=Path("/media/shizhm/sda2/KITTI360_lidar/data_3d_raw"))
    ap.add_argument("--dgm-root", type=Path,
                    default=Path("/media/shizhm/sda1/proposal/Cross-View Conditional Coding of Route-Specific Gaussian Scenes/outputs/bw_dgm_dom/tiles"))
    ap.add_argument("--output-root", type=Path, default=Path("runs/dgm_alignment_check"))
    ap.add_argument("--sensor-height-m", type=float, default=0.93)
    ap.add_argument("--calibration-frames", type=int, default=30)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rasters = load_dgm_rasters(args.dgm_root)
    print(f"DGM tiles: {sorted(rasters)}")

    report = {}
    for drive in DRIVES:
        sequence = args.sequence_root / f"2013_05_28_drive_{drive}_sync"
        lidar_dir = args.lidar_root / f"2013_05_28_drive_{drive}_sync/velodyne_points/data"
        if not sequence.exists() or not lidar_dir.exists():
            report[drive] = {"status": "missing data"}
            continue
        poses, utm = load_pose_utm(sequence)
        common = sorted(set(poses) & set(utm))
        covered = [f for f in common
                   if (int(utm[f][0] // 1000), int(utm[f][1] // 1000)) in rasters
                   and (lidar_dir / f"{f:010d}.bin").exists()]
        print(f"[{drive}] poses={len(poses)} covered+lidar={len(covered)}", flush=True)
        if len(covered) < args.calibration_frames + 100:
            report[drive] = {"status": "insufficient DGM coverage",
                             "covered_frames": len(covered)}
            continue
        start, length = longest_contiguous_run(covered)
        length = min(length, args.calibration_frames + MAX_EVAL_FRAMES)
        cal = list(range(start, start + args.calibration_frames))
        ev = list(range(start + args.calibration_frames, start + length))

        # world -> UTM fit on the CALIBRATION window only (reference fits short
        # segments; fitting across 600 frames eats UTM nonlinearity)
        common_fit, transform, fit_rmse, fit_max = fit_alignment(poses, utm, cal)

        # 1-DoF vertical registration on calibration frames:
        # origin_z - sensor_height - DGM, median
        xy = np.asarray([utm[f] for f in cal])
        ground = sample_dgm_bilinear(rasters, xy)
        pose_z = np.asarray([poses[f][2, 3] for f in cal])
        resid = pose_z - args.sensor_height_m - ground
        resid = resid[np.isfinite(resid)]
        delta_h = float(np.median(resid))

        # evaluation: LiDAR ground voxels vs DGM
        velo_bins = [np.fromfile(lidar_dir / f"{f:010d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
                     for f in ev]
        velo = np.concatenate(velo_bins).astype(np.float64)
        # velodyne -> cam0 via calib_cam_to_velo (NOT calib_cam_to_pose), then
        # cam0 -> world via cam0_to_world — same chain as the reference probe
        values = np.fromstring((sequence / "calibration/calib_cam_to_velo.txt").read_text(), sep=" ")
        cam_to_velo = np.eye(4)
        cam_to_velo[:3] = values.reshape(3, 4)
        velo_to_cam = np.linalg.inv(cam_to_velo)
        pts_utm_list, zs_list = [], []
        for f, vb in zip(ev, velo_bins):
            w = (poses[f] @ velo_to_cam @ np.c_[vb, np.ones(len(vb))].T).T[:, :3]
            pts_utm_list.append(utm[f] + (w[:, :2] - poses[f][:2, 3]) @ transform)
            zs_list.append(w[:, 2])
        pts_utm = np.concatenate(pts_utm_list)
        zs = np.concatenate(zs_list)

        dgm = sample_dgm_bilinear(rasters, pts_utm)
        ok = np.isfinite(dgm)
        h = zs[ok] - dgm[ok] - delta_h
        # 0.25 m unique voxels (their primary resolution)
        idx = np.floor(np.c_[pts_utm[ok], zs[ok]] / 0.25).astype(np.int64)
        uid, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
        h_mean = np.zeros(len(uid))
        np.add.at(h_mean, inv, h)
        h_mean /= counts
        abs_h = np.abs(h_mean)
        entry = {
            "status": "ok",
            "covered_frames_total": len(covered),
            "eval_frames": len(ev),
            "contiguous_run_start": start,
            "world_to_utm_fit_rmse_m": fit_rmse,
            "world_to_utm_fit_max_m": fit_max,
            "delta_h_m": delta_h,
            "n_unique_voxels_025": int(len(uid)),
            "f_abs_h_lt_0p5": float(np.mean(abs_h < 0.5)),
            "f_abs_h_lt_1p0": float(np.mean(abs_h < 1.0)),
            "median_abs_h": float(np.median(abs_h)),
            "p90_abs_h": float(np.quantile(abs_h, 0.90)),
            "f_below_dtm_lt_minus_0p5": float(np.mean(h_mean <= -0.5)),
            "f_above_gt_0p5": float(np.mean(h_mean >= 0.5)),
            # controls: 5 m east-shifted DGM and constant-plane
            "ctrl": {},
        }
        shifted = sample_dgm_bilinear(rasters, pts_utm[ok] - np.asarray([5.0, 0.0]))
        hs = zs[ok] - shifted - delta_h
        _, invs, cs = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
        hs_mean = np.zeros(len(uid)); np.add.at(hs_mean, invs, hs); hs_mean /= cs
        entry["ctrl"]["shifted_5m_east_f_lt_0p5"] = float(np.mean(np.abs(hs_mean) < 0.5))
        const = float(np.median(dgm[ok]))
        hc = zs[ok] - const - delta_h
        hc_mean = np.zeros(len(uid)); np.add.at(hc_mean, inv, hc); hc_mean /= counts
        entry["ctrl"]["constant_plane_f_lt_0p5"] = float(np.mean(np.abs(hc_mean) < 0.5))
        report[drive] = entry
        print(json.dumps(entry, indent=2), flush=True)
        (args.output_root / "alignment_report.json").write_text(json.dumps(report, indent=2))

    out = args.output_root / "alignment_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

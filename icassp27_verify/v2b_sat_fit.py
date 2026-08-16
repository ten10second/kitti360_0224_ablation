"""Day-0 V2b: rigorous satellite scale fit + residual quantification.

User-provided ground truth: satellite resolution = 0.2 m/pixel.
V2 raw data suggested: measured N-S pixel shifts are ~1.56x larger than the
0.2 m/px prediction (1.56 ~ 1/cos(49 deg), the KITTI area latitude), while E-W
matches 0.2 well. This script tests three models by least squares over many
frame pairs per drive:

  M1 isotropic 0.2      : shift = (-e/0.2, +n/0.2)
  M2 isotropic fitted m : shift = (-e/m, +n/m)
  M3 anisotropic fitted : shift = (-e/m_u, +n/m_v)

phaseCorrelate(sat_i, sat_j) returns the content shift of sat_j relative to
sat_i (established in v2 synthetic check direction; sign re-verified by fit).

Also: per-pair residual norms, and trajectory overlay saved per model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from kitti_raw_io import read_oxts, oxts_to_poses_enu

ROOT = "/media/shizhm/Lenovo/KITTI_RAW"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def fit_and_eval(drive_dir: Path, max_pairs: int = 40):
    oxts = read_oxts(drive_dir / "oxts" / "data")
    poses = oxts_to_poses_enu(oxts)
    enu = poses[:, :2, 3]
    lat0 = np.degrees(oxts[0, 0])

    n = len(oxts)
    # spread pairs over the drive, displacement 15..120 m
    pairs = []
    i = 0
    while i < n - 1 and len(pairs) < max_pairs:
        j = i + 1
        while j < n and np.hypot(*(enu[j] - enu[i])) < 15.0:
            j += 1
        if j < n and np.hypot(*(enu[j] - enu[i])) <= 120.0:
            pairs.append((i, j))
        i += max(1, n // max_pairs)

    meas, disps = [], []
    for i, j in pairs:
        si = cv2.cvtColor(cv2.imread(str(drive_dir / "satellite" / f"{i:010d}.png")), cv2.COLOR_BGR2GRAY)
        sj = cv2.cvtColor(cv2.imread(str(drive_dir / "satellite" / f"{j:010d}.png")), cv2.COLOR_BGR2GRAY)
        win = cv2.createHanningWindow(si.shape[::-1], cv2.CV_64F)
        (dx, dy), resp = cv2.phaseCorrelate(si.astype(np.float64), sj.astype(np.float64), win)
        if resp < 0.1:
            continue
        meas.append((dx, dy))
        disps.append(enu[j] - enu[i])
    meas = np.array(meas)
    disps = np.array(disps)
    if len(meas) < 6:
        return None

    # ---- M1: fixed isotropic 0.2 (user ground truth) ----
    pred1 = np.stack([-disps[:, 0] / 0.2, disps[:, 1] / 0.2], axis=1)
    res1 = np.linalg.norm(meas - pred1, axis=1)

    # ---- M2: fitted isotropic m (minimize |A/m - b|^2 per axis jointly => fit via LSQ on each axis then average) ----
    # measured_u = -e/m ; measured_v = +n/m  => m_u_est = -sum(e*u)/sum(u^2) style; simple: scale that maps disp to meas
    su = np.sum(-disps[:, 0] * meas[:, 0]) / np.sum(disps[:, 0] ** 2)  # 1/m
    sv = np.sum(disps[:, 1] * meas[:, 1]) / np.sum(disps[:, 1] ** 2)
    m2u, m2v = 1.0 / su, 1.0 / sv
    pred2 = np.stack([-disps[:, 0] * su, disps[:, 1] * sv], axis=1)
    res2 = np.linalg.norm(meas - pred2, axis=1)

    def stats(r):
        return f"median={np.median(r):6.2f}px  p90={np.percentile(r,90):6.2f}px  max={r.max():6.2f}px"

    # per-axis ratios for diagnosis
    with np.errstate(divide="ignore", invalid="ignore"):
        ru = meas[:, 0] / np.where(np.abs(pred1[:, 0]) > 10, pred1[:, 0], np.nan)
        rv = meas[:, 1] / np.where(np.abs(pred1[:, 1]) > 10, pred1[:, 1], np.nan)
    coslat = np.cos(np.radians(lat0))

    print(f"\n[{drive_dir.name}]  pairs={len(meas)}  lat0={lat0:.4f}  cos(lat0)={coslat:.4f}")
    print(f"  axis ratio measured/pred@0.2:  u: median={np.nanmedian(ru):.4f}   v: median={np.nanmedian(rv):.4f}")
    print(f"  fitted isotropic : m_u={m2u:.4f}  m_v={m2v:.4f}  (should be equal if isotropic)")
    print(f"  M1 fixed 0.2      residual: {stats(res1)}")
    print(f"  M3 anisotropic    residual: {stats(res2)}   m_u={m2u:.4f}  m_v={m2v:.4f}  m_v/m_u={m2v/m2u:.4f}  cos(lat)={coslat:.4f}")

    return dict(name=drive_dir.name, m_u=m2u, m_v=m2v, res1=res1, res2=res2, lat0=lat0)


def overlay(drive_dir: Path, m_u: float, m_v: float, tag: str):
    oxts = read_oxts(drive_dir / "oxts" / "data")
    poses = oxts_to_poses_enu(oxts)
    enu = poses[:, :2, 3]
    n = len(oxts)
    k = n // 2
    si = cv2.imread(str(drive_dir / "satellite" / f"{k:010d}.png")).copy()
    Hh, Ww = si.shape[:2]
    pts = []
    for t in range(n):
        e, nn = enu[t] - enu[k]
        u = Ww / 2 + e / m_u
        v = Hh / 2 - nn / m_v
        pts.append((u, v))
    inside = [(u, v) for u, v in pts if 0 <= u < Ww and 0 <= v < Hh]
    for u, v in inside:
        cv2.circle(si, (int(u), int(v)), 3, (0, 0, 255), -1)
    cv2.circle(si, (Ww // 2, Hh // 2), 6, (0, 255, 255), -1)
    cv2.imwrite(str(OUT / f"v2b_overlay_{tag}.jpg"), si, [cv2.IMWRITE_JPEG_QUALITY, 92])
    frac = len(inside) / max(1, len(pts))
    return frac


def main():
    print("=" * 76)
    print("V2b: satellite scale model comparison (ground truth claim: 0.2 m/px)")
    print("=" * 76)
    drives = [
        ("2011_09_26/2011_09_26_drive_0002_sync", "d0002"),
        ("2011_09_26/2011_09_26_drive_0009_sync", "d0009"),
        ("2011_09_26/2011_09_26_drive_0022_sync", "d0022"),
        ("2011_09_28/2011_09_28_drive_0039_sync", "d0039"),
        ("2011_10_03/2011_10_03_drive_0034_sync", "d0034"),
    ]
    for dspec, tag in drives:
        res = fit_and_eval(Path(ROOT) / dspec)
        if res is None:
            print(f"[{dspec}] skipped (too few valid pairs)")
            continue
        frac = overlay(Path(ROOT) / dspec, res["m_u"], res["m_v"], tag)
        print(f"  overlay saved v2b_overlay_{tag}.jpg  trajectory-inside-frac={frac:.2f}")


if __name__ == "__main__":
    main()

"""Day-0 V2: satellite alignment verification for KITTI-raw 1280x1280 crops.

Plan requirements (Checklist section 4 / ICASSP27 section 1):
- satellite crop north-up, fixed mpp, window-shared
- per-frame crop centered at vehicle GPS (KITTI-360 pipeline assumption, to be re-tested here)
- registration residual ~1 px scale

Method:
  For frame pairs (i, j) with known metric ENU displacement d=(e,n) from oxts,
  phase-correlate sat_i vs sat_j. If crops are vehicle-centered & north-up with mpp m:
  world point at ego_j appears at center of sat_j and at (W/2 + e/m, H/2 - n/m) in sat_i.
  So aligning sat_j to sat_i requires shifting sat_j content by (du, dv) = (+e/m, -n/m).
  cv2.phaseCorrelate(a, b) returns the shift to apply to b to register it with a
  (verified synthetically below). Then m_est = |d| / |shift|, and the direction
  cos(atan2(-dv, du), atan2(n, e)) should be ~1 for north-up.
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


def load_sat(drive_dir: Path, fid: int) -> np.ndarray:
    p = drive_dir / "satellite" / f"{fid:010d}.png"
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def phase_shift(a_gray: np.ndarray, b_gray: np.ndarray) -> tuple:
    """Return (dx, dy, response). Synthetic check establishes the sign convention."""
    win = cv2.createHanningWindow(a_gray.shape[::-1], cv2.CV_64F)
    (dx, dy), resp = cv2.phaseCorrelate(
        a_gray.astype(np.float64), b_gray.astype(np.float64), win
    )
    return dx, dy, resp


def synthetic_sign_check():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, (400, 400))
    base = cv2.GaussianBlur(base, (0, 0), 5)
    M = np.float32([[1, 0, 7], [0, 1, -3]])  # shift image content by (+7, -3)
    shifted = cv2.warpAffine(base, M, (400, 400))
    dx, dy, _ = phase_shift(base, shifted)
    # if convention is "shift to apply to b to align with a", we expect (dx,dy)=(+7,-3)
    print(f"[synthetic] warped content by (+7,-3) px -> phaseCorrelate(a,b)=({dx:+.2f},{dy:+.2f})")
    return dx, dy


def main():
    print("=" * 72)
    print("V2: satellite alignment (mpp estimation + north-up + centering)")
    print("=" * 72)
    synthetic_sign_check()

    drives = [
        "2011_09_26/2011_09_26_drive_0002_sync",
        "2011_09_26/2011_09_26_drive_0009_sync",
        "2011_09_26/2011_09_26_drive_0022_sync",
        "2011_10_03/2011_10_03_drive_0034_sync",
    ]
    for dspec in drives:
        drive_dir = Path(ROOT) / dspec
        oxts = read_oxts(drive_dir / "oxts" / "data")
        poses = oxts_to_poses_enu(oxts)
        enu = poses[:, :2, 3]  # (N,2) absolute (east, north)

        n = len(oxts)
        pair_ids = [0, n // 4, n // 2, 3 * n // 4]
        est_mpp, dir_errs, resps = [], [], []
        for i in pair_ids:
            # choose j ~40-120 m ahead along the drive
            for j in range(i + 1, n):
                d = enu[j] - enu[i]
                if np.hypot(*d) > 45.0:
                    break
            else:
                continue
            if np.hypot(*d) > 150.0:
                print(f"  [pair {i}->{j}] gap too large ({np.hypot(*d):.0f} m), skip")
                continue
            si = load_sat(drive_dir, i)
            sj = load_sat(drive_dir, j)
            gi = cv2.cvtColor(si, cv2.COLOR_BGR2GRAY)
            gj = cv2.cvtColor(sj, cv2.COLOR_BGR2GRAY)
            dx, dy, resp = phase_shift(gi, gj)
            # predicted content shift of sat_j relative to sat_i: (e/m, -n/m)
            e, nn = d
            metric = np.hypot(e, nn)
            px = np.hypot(dx, dy)
            if px < 5 or resp < 0.05:
                print(f"  [pair {i:5d}->{j:5d}] weak corr: shift=({dx:+7.2f},{dy:+7.2f}) resp={resp:.3f} -> skip")
                continue
            m_est = metric / px
            ang_pred = np.degrees(np.arctan2(-dy, dx))  # measured direction in ENU axes (east, north)
            ang_true = np.degrees(np.arctan2(nn, e))
            derr = (ang_pred - ang_true + 180) % 360 - 180
            est_mpp.append(m_est)
            dir_errs.append(derr)
            resps.append(resp)
            print(f"  [pair {i:5d}->{j:5d}] d=({e:+6.1f},{nn:+6.1f})m |d|={metric:5.1f}  shift=({dx:+7.2f},{dy:+7.2f})px resp={resp:.2f}  mpp={m_est:.4f}  dir_err={derr:+6.2f}deg")

        if est_mpp:
            print(f"  ==> mpp est: mean={np.mean(est_mpp):.4f} std={np.std(est_mpp):.4f}  dir_err: mean={np.mean(np.abs(dir_errs)):.2f} deg")

        # trajectory overlay visual: frame k sat + full drive trajectory mapped with mean mpp
        if est_mpp:
            mpp = float(np.mean(est_mpp))
            k = pair_ids[len(pair_ids) // 2]
            si = load_sat(drive_dir, k).copy()
            Hh, Ww = si.shape[:2]
            pts = []
            for t in range(0, n, max(1, n // 400)):
                e, nn = enu[t] - enu[k]
                u = Ww / 2 + e / mpp
                v = Hh / 2 - nn / mpp
                if 0 <= u < Ww and 0 <= v < Hh:
                    pts.append((u, v))
            for p in pts:
                cv2.circle(si, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            tag = dspec.replace("/", "_")
            cv2.imwrite(str(OUT / f"v2_sat_traj_{tag}.jpg"), si, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  saved overlay: out/v2_sat_traj_{tag}.jpg (mpp={mpp:.3f})")


if __name__ == "__main__":
    main()

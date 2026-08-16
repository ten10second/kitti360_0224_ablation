#!/usr/bin/env python3
"""
Split-and-compare fisheye undistortion vs rotation.
- Combined: one-pass cv2.omnidir.initUndistortRectifyMap with R_rectify = R_rel^T
- Decoupled: undistort with R=I then apply pure rotation homography H = K_v * (R_rel^T) * K_v^{-1}
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml


def rot_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def make_Kv(out_w: int, out_h: int, hfov_deg: float) -> np.ndarray:
    hfov = np.deg2rad(hfov_deg)
    fx = (out_w * 0.5) / np.tan(hfov * 0.5)
    fy = fx
    cx, cy = out_w * 0.5, out_h * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_mei_yaml(path: Path):
    raw = path.read_text()
    if raw.lstrip().startswith("%YAML"):
        raw = "\n".join(raw.splitlines()[1:])
    y = yaml.safe_load(raw)
    xi = float(y["mirror_parameters"]["xi"])
    K = np.array([
        [float(y["projection_parameters"]["gamma1"]), 0.0, float(y["projection_parameters"]["u0"])],
        [0.0, float(y["projection_parameters"]["gamma2"]), float(y["projection_parameters"]["v0"])],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    D = np.array([
        float(y["distortion_parameters"]["k1"]),
        float(y["distortion_parameters"]["k2"]),
        float(y["distortion_parameters"]["p1"]),
        float(y["distortion_parameters"]["p2"])
    ], dtype=np.float64)
    return xi, K, D


def overlay(imgA, imgB):
    a = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    ea = cv2.Canny(a, 60, 180)
    eb = cv2.Canny(b, 60, 180)
    out = imgA.copy()
    out[ea>0] = (out[ea>0]*0.2 + np.array([0,0,255], dtype=np.float32)*0.8).astype(np.uint8)
    out[eb>0] = (out[eb>0]*0.2 + np.array([0,255,0], dtype=np.float32)*0.8).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(--data_root, required=True)
    ap.add_argument(--drive, required=True)
    ap.add_argument(--frame, type=int, required=True)
    ap.add_argument(--camera, choices=[image_02,image_03], required=True)
    ap.add_argument(--hfov, type=float, default=120.0)
    ap.add_argument(--yaw, type=float, default=0.0)
    ap.add_argument(--out_w, type=int, default=640)
    ap.add_argument(--out_h, type=int, default=256)
    ap.add_argument(--out_dir, required=True)
    ap.add_argument(--calib_yaml_override, default=None)
    ap.add_argument(--override_xi, type=float, default=None)
    ap.add_argument(--scale_k2, type=float, default=1.0)
    ap.add_argument(--zero_tangential, action=store_true)
    args = ap.parse_args()

    seq = Path(args.data_root) / args.drive
    img_path = seq / args.camera / data_rgb / f"{args.frame:010d}.png"
    if not img_path.exists():
        img_path = seq / args.camera / data / f"{args.frame:010d}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"image not found: {img_path}")

    calib_path = Path(args.calib_yaml_override) if args.calib_yaml_override else (seq / calibration / f"{args.camera}.yaml")
    xi, Kf, D = load_mei_yaml(calib_path)
    if args.override_xi is not None:
        xi = float(args.override_xi)
    if abs(args.scale_k2 - 1.0) > 1e-6:
        D = D.copy(); D[1] = float(D[1] * args.scale_k2)
    if args.zero_tangential:
        D = D.copy(); D[2:] = 0.0

    Kv = make_Kv(args.out_w, args.out_h, args.hfov)

    # Desired relative rotation around fisheye Y (OpenCV cam coords): R_rel = rot_y(-yaw)
    R_rel = rot_y(-args.yaw)
    R_rectify = R_rel.T

    # Combined one-pass
    map1c, map2c = cv2.omnidir.initUndistortRectifyMap(
        Kf, D, np.array([xi], dtype=np.float64), R_rectify, Kv, (args.out_w, args.out_h), cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE)
    view_combined = cv2.remap(img, map1c, map2c, cv2.INTER_LINEAR)

    # Decoupled: undistort only (R=I), then rotate with homography H = Kv * R_rectify * Kv^{-1}
    map1i, map2i = cv2.omnidir.initUndistortRectifyMap(
        Kf, D, np.array([xi], dtype=np.float64), np.eye(3, dtype=np.float64), Kv, (args.out_w, args.out_h), cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE)
    view_base = cv2.remap(img, map1i, map2i, cv2.INTER_LINEAR)
    H = Kv @ R_rectify @ np.linalg.inv(Kv)
    view_decoupled = cv2.warpPerspective(view_base, H, (args.out_w, args.out_h), flags=cv2.INTER_LINEAR)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"view_combined_{args.camera}_{args.frame:010d}.png"), view_combined)
    cv2.imwrite(str(out_dir / f"view_base_{args.camera}_{args.frame:010d}.png"), view_base)
    cv2.imwrite(str(out_dir / f"view_decoupled_{args.camera}_{args.frame:010d}.png"), view_decoupled)
    cv2.imwrite(str(out_dir / f"overlay_combined_vs_decoupled_{args.camera}_{args.frame:010d}.png"), overlay(view_combined, view_decoupled))

    mad = float(np.mean(np.abs(view_combined.astype(np.float32) - view_decoupled.astype(np.float32))))
    print(f"MeanAbsDiff(combined, decoupled) = {mad:.3f}")

if __name__ == "__main__":
    main()

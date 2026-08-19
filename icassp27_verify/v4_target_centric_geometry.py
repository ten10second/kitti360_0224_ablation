#!/usr/bin/env python3
"""Target-centric geometry checks and visual sanity checks for ICASSP27.

This is deliberately CPU-only: the invariant checks exercise the coordinate
contracts directly, while the optional visual pass draws real tuple poses over
their *window-shared*, north-up satellite crop.  It does not construct DINO or
the predictor, so it is suitable to run before a training job.

Run from repository root:
  python -m icassp27_verify.v4_target_centric_geometry

The numerical checks cover the contract used by the target-centric change:
  * common global translation and planar yaw leave target-relative geometry
    unchanged;
  * the target is at satellite coordinate (right, forward) = (0, 0);
  * target->source local translation preserves Euclidean distance;
  * target rays are camera-local, with left/right image tokens having the
    expected signed camera x direction.

The visual pass saves ten different (drive, window) examples under
``icassp27_verify/out/target_centric``.  Inspect them before training: the
orange forward arrow should agree with travel/view direction, and the blue
right arrow must point to camera-right rather than being mirrored.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl"
DEFAULT_OUT = REPO_ROOT / "icassp27_verify/out/target_centric"


def _normalize_xy(v: torch.Tensor) -> torch.Tensor:
    """Normalize final planar dimension and reject degenerate camera headings."""
    norm = v.norm(dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError("camera right/forward vector has no valid XY heading")
    return v / norm


def target_planar_axes(T_cam_to_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target camera planar (right, forward) axes in world XY.

    ``T_cam_to_world`` follows the project's camera convention: columns 0 and
    2 are camera right and forward respectively.  Inputs may be ``(4,4)`` or
    ``(B,4,4)``; outputs have matching leading dimensions and final size 2.
    """
    R = T_cam_to_world[..., :3, :3]
    return _normalize_xy(R[..., :2, 0]), _normalize_xy(R[..., :2, 2])


def satellite_target_xy(
    patch_world_xy: torch.Tensor,
    tgt_T_cam: torch.Tensor,
) -> torch.Tensor:
    """Express world XY points in target-centric ``(right_m, forward_m)``.

    ``patch_world_xy`` may be ``(N,2)`` or ``(B,N,2)``.  The target pose may
    be unbatched or batched.  Normal broadcasting supports the layouts used by
    patch-token geometry without hiding a world-coordinate dependency.
    """
    right, forward = target_planar_axes(tgt_T_cam)
    center = tgt_T_cam[..., :2, 3]
    while center.ndim < patch_world_xy.ndim:
        center = center.unsqueeze(-2)
        right = right.unsqueeze(-2)
        forward = forward.unsqueeze(-2)
    delta = patch_world_xy - center
    return torch.stack([(delta * right).sum(dim=-1), (delta * forward).sum(dim=-1)], dim=-1)


def target_to_source_translation(tgt_T_cam: torch.Tensor, src_T_cam: torch.Tensor) -> torch.Tensor:
    """Target-camera-local translation for the project's target -> source pose."""
    delta_world = src_T_cam[..., :3, 3] - tgt_T_cam[..., :3, 3]
    return torch.matmul(tgt_T_cam[..., :3, :3].transpose(-1, -2), delta_world.unsqueeze(-1)).squeeze(-1)


def camera_local_rays(
    K: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Camera-frame unit rays at the same raster token centers as the model."""
    device, dtype = K.device, K.dtype
    v = (torch.arange(rows, device=device, dtype=dtype) + 0.5) * (image_h / rows)
    u = (torch.arange(cols, device=device, dtype=dtype) + 0.5) * (image_w / cols)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pixels = torch.stack((uu.reshape(-1), vv.reshape(-1), torch.ones(rows * cols, device=device, dtype=dtype)), dim=-1)
    if K.ndim == 2:
        rays = torch.linalg.solve(K, pixels.T).T
    elif K.ndim == 3:
        rays = torch.einsum("bij,nj->bni", torch.linalg.inv(K), pixels)
    else:
        raise ValueError(f"K must be (3,3) or (B,3,3), received {tuple(K.shape)}")
    return rays / rays.norm(dim=-1, keepdim=True)


def _rotz(angle_rad: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=dtype)


def _apply_global_planar_transform(T: torch.Tensor, Rz: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply the same world-frame planar rigid transform to a camera pose."""
    out = T.clone()
    out[:3, :3] = Rz @ T[:3, :3]
    out[:3, 3] = Rz @ T[:3, 3] + shift
    return out


def _synthetic_pose(*, yaw: float, translation: Sequence[float]) -> torch.Tensor:
    """A pitched camera whose planar heading is still defined by camera columns."""
    T = torch.eye(4, dtype=torch.float64)
    # Keep +x/right and +z/forward orthonormal; mild pitch makes this less
    # susceptible to a test that accidentally assumes an all-planar pose.
    pitch = math.radians(-4.0)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_pitch = torch.tensor(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)), dtype=torch.float64)
    T[:3, :3] = _rotz(yaw) @ R_pitch
    T[:3, 3] = torch.tensor(translation, dtype=torch.float64)
    return T


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float) -> float:
    error = float((actual - expected).abs().max())
    if error >= atol:
        raise AssertionError(f"{name}: max_abs_error={error:.3e}, expected < {atol:.1e}")
    print(f"  PASS {name}: max_abs_error={error:.3e} < {atol:.1e}")
    return error


def run_numerical_checks() -> None:
    """Run the requested target-centric coordinate-contract checks."""
    print("== target-centric numerical checks ==")
    tgt = _synthetic_pose(yaw=math.radians(31.0), translation=(42.0, -19.0, 1.7))
    src = _synthetic_pose(yaw=math.radians(38.0), translation=(31.5, -4.0, 1.9))
    patches = torch.tensor(
        ((-31.0, -47.0), (6.0, 2.0), (18.5, -13.0), (77.0, 39.0)), dtype=torch.float64
    )

    # Exercise the model's public geometry helpers without constructing DINO.
    # The stub has precisely the scalar/grid attributes used by those helpers;
    # this catches a future refactor that changes implementation semantics while
    # leaving this reference calculation accidentally green.
    sys.path.insert(0, str(REPO_ROOT))
    from world3d.models.icassp27_predictor import ICASSP27Predictor
    sat_stub = SimpleNamespace(
        sat_grid=(37, 37), dino_sat_size=(518, 518), sat_px=512, sat_m_per_px=0.196,
    )
    sat_stub._sat_patch_world_xy = ICASSP27Predictor._sat_patch_world_xy.__get__(sat_stub, type(sat_stub))
    sat_stub._world_xy_to_target_xz = ICASSP27Predictor._world_xy_to_target_xz
    ray_stub = SimpleNamespace(target_rows=16, target_cols=40, img_h=256, img_w=640)

    # 7.1: adding a global world translation cannot alter relative geometry.
    world_shift = torch.tensor((1000.0, -500.0, 20.0), dtype=torch.float64)
    translated_tgt = tgt.clone(); translated_tgt[:3, 3] += world_shift
    translated_src = src.clone(); translated_src[:3, 3] += world_shift
    _assert_close(
        "global translation invariance / satellite",
        satellite_target_xy(patches + world_shift[:2], translated_tgt),
        satellite_target_xy(patches, tgt),
        1e-5,
    )
    _assert_close(
        "global translation invariance / source translation",
        target_to_source_translation(translated_tgt, translated_src),
        target_to_source_translation(tgt, src),
        1e-5,
    )
    origin = torch.tensor(((0.0, 0.0, 0.0),), dtype=torch.float64)
    actual_sat = ICASSP27Predictor._sat_patch_target_xy(sat_stub, origin, tgt.unsqueeze(0))
    actual_sat_translated = ICASSP27Predictor._sat_patch_target_xy(
        sat_stub, origin + world_shift.unsqueeze(0), translated_tgt.unsqueeze(0)
    )
    _assert_close("model satellite helper / global translation", actual_sat_translated, actual_sat, 1e-5)

    # 7.2: rotate every world object and both pose orientations about world Z.
    Rz = _rotz(math.radians(-113.0))
    yaw_shift = torch.tensor((-83.0, 12.0, 0.0), dtype=torch.float64)
    yaw_tgt = _apply_global_planar_transform(tgt, Rz, yaw_shift)
    yaw_src = _apply_global_planar_transform(src, Rz, yaw_shift)
    yaw_patches = (Rz[:2, :2] @ patches.T).T + yaw_shift[:2]
    _assert_close(
        "global planar yaw invariance / satellite",
        satellite_target_xy(yaw_patches, yaw_tgt),
        satellite_target_xy(patches, tgt),
        1e-4,
    )
    _assert_close(
        "global planar yaw invariance / source translation",
        target_to_source_translation(yaw_tgt, yaw_src),
        target_to_source_translation(tgt, src),
        1e-4,
    )
    actual_xy_yaw = ICASSP27Predictor._world_xy_to_target_xz(yaw_patches.unsqueeze(0), yaw_tgt.unsqueeze(0))
    actual_xy = ICASSP27Predictor._world_xy_to_target_xz(patches.unsqueeze(0), tgt.unsqueeze(0))
    _assert_close("model target-XY helper / global planar yaw", actual_xy_yaw, actual_xy, 1e-4)

    # 7.3: target center must be the target-frame origin by construction.
    target_center = tgt[:2, 3].unsqueeze(0)
    _assert_close("target center -> (right, forward)=(0,0)", satellite_target_xy(target_center, tgt), torch.zeros_like(target_center), 1e-7)

    # 7.5: rotation of the coordinate axes must preserve source distance.
    dt_world = src[:3, 3] - tgt[:3, 3]
    dt_tgt = target_to_source_translation(tgt, src)
    _assert_close("source distance preservation", dt_tgt.norm().reshape(1), dt_world.norm().reshape(1), 1e-4)
    # The inverse pair lives in the source frame, so it is not merely -dt.
    # Bringing it back through R_target<-source must recover target->source.
    dt_src = target_to_source_translation(src, tgt)
    R_target_from_source = tgt[:3, :3].T @ src[:3, :3]
    _assert_close("target->source inverse-pair convention", dt_tgt, -(R_target_from_source @ dt_src), 1e-5)

    # 7.6: target ray directions stay in camera coordinates, independent of T.
    K = torch.tensor(((700.0, 0.0, 320.0), (0.0, 700.0, 128.0), (0.0, 0.0, 1.0)), dtype=torch.float64)
    rows, cols = 16, 40
    rays = camera_local_rays(K, image_h=256, image_w=640, rows=rows, cols=cols).reshape(rows, cols, 3)
    # The predictor generates its token-center grid in float32, matching normal
    # dataloader intrinsics, so call that integration helper with float32.
    model_origins, model_rays = ICASSP27Predictor._target_rays(ray_stub, K.float().unsqueeze(0))
    _assert_close("model camera-local rays / direction", model_rays[0], rays.reshape(-1, 3).float(), 1e-6)
    _assert_close("model camera-local rays / zero origins", model_origins, torch.zeros_like(model_origins), 1e-7)
    center = rays[rows // 2, cols // 2]
    if not (center[2] > 0.99 and abs(float(center[0])) < 0.02 and abs(float(center[1])) < 0.02):
        raise AssertionError(f"center ray expected ~[0,0,1], got {center.tolist()}")
    if not (rays[rows // 2, 0, 0] < 0.0 and rays[rows // 2, -1, 0] > 0.0):
        raise AssertionError("camera-local ray x sign is mirrored: left must be x<0, right x>0")
    print(f"  PASS camera-local rays: center={center.tolist()}, left_x={rays[rows // 2, 0, 0]:.3f}, right_x={rays[rows // 2, -1, 0]:.3f}")


def _world_to_sat_px(world_xy: np.ndarray, origin_xy: np.ndarray, mpp: float, width: int, height: int) -> np.ndarray:
    """North-up satellite pixels: global x=east -> image right, y=north -> up."""
    delta = np.asarray(world_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    return np.array((width * 0.5 + delta[0] / mpp, height * 0.5 - delta[1] / mpp), dtype=np.float64)


def _draw_arrow(canvas: np.ndarray, start: np.ndarray, world_direction_xy: np.ndarray, mpp: float, color: tuple[int, int, int], label: str) -> None:
    """Draw a metre-valued world direction after converting it to north-up pixels."""
    pixel_delta = np.array((world_direction_xy[0] / mpp, -world_direction_xy[1] / mpp))
    end = start + pixel_delta
    p0, p1 = tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int))
    cv2.arrowedLine(canvas, p0, p1, color, 3, tipLength=0.18)
    cv2.putText(canvas, label, (p1[0] + 5, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _label(canvas: np.ndarray, text: str, xy: Iterable[int], color: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, tuple(xy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def save_real_tuple_visualizations(manifest: Path, out_dir: Path, *, count: int) -> list[Path]:
    """Save one visualisation from each of up to ``count`` distinct windows."""
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    sys.path.insert(0, str(REPO_ROOT))
    from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset

    dataset = Kitti360TupleDataset(str(manifest), mode="eval", seed=0)
    if not dataset.tuples:
        raise RuntimeError(f"no valid tuples in {manifest}")
    candidates: list[int] = []
    seen: set[tuple[str, int]] = set()
    for tuple_index, spec in enumerate(dataset.tuples):
        key = (spec.drive, spec.window_id)
        if key not in seen:
            candidates.append(tuple_index * len(dataset.eval_k))
            seen.add(key)
    if len(candidates) < count:
        raise RuntimeError(f"only found {len(candidates)} distinct windows, requested {count}")
    # A fixed seed makes the requested random sample reproducible in logs and
    # avoids accidentally checking only the first contiguous route segment.
    selected = sorted(np.random.default_rng(0).choice(candidates, size=count, replace=False).tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for n, item_index in enumerate(selected):
        sample = dataset[item_index]
        meta = sample["meta"]
        sat = sample["sat"].permute(1, 2, 0).numpy()
        # OpenCV expects BGR, whereas the tuple is RGB in [0,1].
        canvas = cv2.cvtColor(np.rint(sat * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        height, width = canvas.shape[:2]
        mpp = float(sample["sat_m_per_px"])
        origin_xy = sample["window_origin_xyz"][:2].numpy()
        tgt = sample["tgt_T_cam"].numpy()
        sources = sample["src_Ts"].numpy()
        target_px = _world_to_sat_px(tgt[:2, 3], origin_xy, mpp, width, height)
        right_xy, forward_xy = tgt[:2, 0], tgt[:2, 2]
        forward_xy = forward_xy / np.linalg.norm(forward_xy)
        right_xy = right_xy / np.linalg.norm(right_xy)

        # Target-local axes are 12 m long so direction and mirroring are easy
        # to inspect; their coordinate origin is exactly the target camera.
        _draw_arrow(canvas, target_px, forward_xy * 12.0, mpp, (0, 140, 255), "+forward")
        _draw_arrow(canvas, target_px, right_xy * 9.0, mpp, (255, 120, 0), "+right")
        cv2.circle(canvas, tuple(np.rint(target_px).astype(int)), 7, (0, 0, 255), -1)
        _label(canvas, "target (0,0)", (int(target_px[0]) + 8, int(target_px[1]) + 19), (0, 0, 255))
        for source_index, source_T in enumerate(sources):
            source_px = _world_to_sat_px(source_T[:2, 3], origin_xy, mpp, width, height)
            cv2.circle(canvas, tuple(np.rint(source_px).astype(int)), 5, (80, 255, 80), -1)
            _label(canvas, f"src{source_index}", (int(source_px[0]) + 6, int(source_px[1]) - 6), (80, 255, 80))
        window_center = np.array((width * 0.5, height * 0.5))
        cv2.drawMarker(canvas, tuple(np.rint(window_center).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        header = f"{meta['drive'].split('/')[-1]}  window={meta['window_id']}  target={meta['target_fid']}  mpp={mpp:.3f}"
        _label(canvas, header, (12, 24), (255, 255, 255))
        _label(canvas, "orange: +forward   blue: +right   green: sources   red: target", (12, 48), (255, 255, 255))
        name = f"{n:02d}_{meta['drive'].split('/')[-1]}_window{meta['window_id']:03d}_target{meta['target_fid']:010d}.jpg"
        output = out_dir / name
        if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {output}")
        outputs.append(output)
        print(f"  saved {output.relative_to(REPO_ROOT)}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="tuple manifest used for visual checks")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="directory for satellite overlays")
    parser.add_argument("--count", type=int, default=10, help="number of distinct windows to visualise (default: 10)")
    parser.add_argument("--no-visuals", action="store_true", help="run only deterministic numerical tests")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    run_numerical_checks()
    if not args.no_visuals:
        print("== real tuple visual checks ==")
        save_real_tuple_visualizations(args.manifest, args.out_dir, count=args.count)
    print("TARGET-CENTRIC GEOMETRY CHECKS OK")


if __name__ == "__main__":
    main()

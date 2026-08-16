from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from world3d.data.deterministic import make_rng
from world3d.io.kitti360d_dataloader import _sample_random_signed_yaw
from world3d.train.conditioning_ar import build_condition_tokens_with_coords
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.pose_ar import build_pose_vec

IMU_TO_GROUND_HEIGHT = 0.93  # consistent with bev_to_camera_warp.py


def compute_bev_visibility_mask(
    K: torch.Tensor,
    T_cam_to_world: torch.Tensor,
    T_imu_to_world: torch.Tensor,
    bev_size: int = 64,
    sat_pixels: int = 512,
    sat_resolution: float = 0.2,
    cam_h: int = 256,
    cam_w: int = 640,
) -> torch.Tensor:
    """Compute which BEV cells are visible from the camera's FOV.

    Forward-projects the 64x64 BEV grid to the camera image plane.
    A cell is visible if z_cam > 0 and the projected pixel is within image bounds.

    Args:
        K: (3,3) camera intrinsics
        T_cam_to_world: (4,4) camera-to-world transform
        T_imu_to_world: (4,4) IMU-to-world transform
        bev_size: BEV grid size (default 64)
        sat_pixels: satellite image pixel size (default 512)
        sat_resolution: meters per satellite pixel (default 0.2)
        cam_h: camera image height in pixels
        cam_w: camera image width in pixels

    Returns:
        mask: (1, bev_size, bev_size) binary float tensor
    """
    device = K.device
    scale = sat_pixels / bev_size  # 8.0 for 512/64

    # BEV cell centers in satellite pixel coords
    cell_idx = torch.arange(bev_size, device=device, dtype=torch.float32)
    cell_centers = cell_idx * scale + scale / 2.0  # [4, 12, 20, ..., 508]

    # Meshgrid: v_sat = row (top-to-bottom), u_sat = col (left-to-right)
    v_sat, u_sat = torch.meshgrid(cell_centers, cell_centers, indexing='ij')  # (64, 64)

    # Satellite pixel → world coords (satellite is north-up, centered at IMU)
    imu_x = T_imu_to_world[0, 3]
    imu_y = T_imu_to_world[1, 3]
    half_pix = sat_pixels / 2.0  # 256.0

    X_world = imu_x + (u_sat - half_pix) * sat_resolution
    Y_world = imu_y - (v_sat - half_pix) * sat_resolution
    Z_world = T_imu_to_world[2, 3] - IMU_TO_GROUND_HEIGHT

    ones = torch.ones_like(X_world)
    pts_world = torch.stack([X_world, Y_world, ones * Z_world, ones], dim=-1)  # (64, 64, 4)

    # World → camera
    T_world_to_cam = torch.inverse(T_cam_to_world)  # (4, 4)
    pts_cam = (T_world_to_cam @ pts_world.reshape(-1, 4).T).T  # (4096, 4)
    pts_cam = pts_cam.reshape(bev_size, bev_size, 4)

    x_cam = pts_cam[..., 0]
    y_cam = pts_cam[..., 1]
    z_cam = pts_cam[..., 2]

    # Project to image plane
    u_pix = K[0, 0] * x_cam / z_cam + K[0, 2]
    v_pix = K[1, 1] * y_cam / z_cam + K[1, 2]

    # Visibility: in front of camera AND within image bounds
    visible = (
        (z_cam > 0.1)
        & (u_pix >= 0) & (u_pix < cam_w)
        & (v_pix >= 0) & (v_pix < cam_h)
    )

    return visible.float().unsqueeze(0)  # (1, 64, 64)


@dataclass
class ArSample:
    input_tokens: torch.Tensor  # (L,) long
    target_tokens: torch.Tensor  # (L,) long
    semantic_fine: torch.Tensor  # (4,R,C) float
    coords_fine: torch.Tensor  # (2,R,C) float
    pose: torch.Tensor  # (13,) float
    K: torch.Tensor  # (3,3) float
    bev: Optional[torch.Tensor]  # (1,C,H,W) float, may be None
    bev_vis_mask: Optional[torch.Tensor]  # (1,64,64) float, may be None
    sup_valid: torch.Tensor  # (L,) float
    frame_id: Optional[int] = None  # Original frame ID from dataset
    drive: Optional[str] = None
    view_name: Optional[str] = None
    view_index: Optional[int] = None
    vis_rgb: Optional[torch.Tensor] = None  # (3,H,W) float in [-1,1]
    vis_ipm: Optional[torch.Tensor] = None  # (3,H,W) float in [0,1]
    vis_ipm_valid: Optional[torch.Tensor] = None  # (1,H,W) float in [0,1] {0,1}
    vis_sat: Optional[torch.Tensor] = None  # (3,H_sat,W_sat) float in [0,1]
    vis_T_cam_to_world: Optional[torch.Tensor] = None  # (4,4) float
    vis_T_imu_to_world: Optional[torch.Tensor] = None  # (4,4) float


class MixedViewIndexDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds_front: torch.utils.data.Dataset,
        ds_virtual: torch.utils.data.Dataset,
        *,
        p_front: float,
        strict_ddp: bool,
        seed: int,
    ):
        self.ds_front = ds_front
        self.ds_virtual = ds_virtual
        self.p_front = float(p_front)
        self.strict_ddp = bool(strict_ddp)
        self.seed = int(seed)
        self.epoch = 0
        assert len(self.ds_front) == len(self.ds_virtual)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.ds_front)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.strict_ddp:
            rng = make_rng(self.seed, self.epoch, idx, salt=1)
            use_front = rng.random() < self.p_front
        else:
            use_front = random.random() < self.p_front
        return self.ds_front[idx] if use_front else self.ds_virtual[idx]


class DeterministicYawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base: torch.utils.data.Dataset,
        *,
        enable: bool,
        seed: int,
        yaw_min_abs: float,
        yaw_max_abs: float,
    ):
        self.base = base
        self.enable = bool(enable)
        self.seed = int(seed)
        self.yaw_min_abs = float(yaw_min_abs)
        self.yaw_max_abs = float(yaw_max_abs)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        if not self.enable:
            return sample

        # Deterministic camera and yaw for this idx/epoch
        rng = make_rng(self.seed, self.epoch, idx, salt=2)
        rs = np.random.RandomState(rng.randrange(0, 2**31 - 1))

        # 1. Choose a fisheye camera
        cam_idx = rs.randint(0, 2)
        camera_override = "image_02" if cam_idx == 0 else "image_03"

        # 2. Choose a yaw relative to that camera's optical axis
        yaw = _sample_random_signed_yaw(self.yaw_min_abs, self.yaw_max_abs, rng=rs)

        meta = dict(sample.get("meta", {}))
        meta["fisheye_camera_override"] = camera_override
        meta["fisheye_relative_yaw_deg_override"] = float(yaw)
        sample = dict(sample)
        sample["meta"] = meta
        return sample


@dataclass(frozen=True)
class FixedViewSpec:
    name: str
    source: str  # "front" or "virtual"
    fisheye_camera_override: Optional[str] = None
    fisheye_relative_yaw_deg_override: Optional[float] = None


class FixedFiveViewDataset(torch.utils.data.Dataset):
    """Expand each frame into five deterministic viewpoints."""

    def __init__(
        self,
        ds_front: torch.utils.data.Dataset,
        ds_virtual: torch.utils.data.Dataset,
        *,
        turn_to_front_deg: float = 30.0,
    ):
        self.ds_front = ds_front
        self.ds_virtual = ds_virtual
        self.turn_to_front_deg = float(abs(turn_to_front_deg))
        self.epoch = 0
        assert len(self.ds_front) == len(self.ds_virtual)

        # image_02 is left fisheye, image_03 is right fisheye.
        # Positive/negative signs are chosen so both virtual views rotate
        # toward the vehicle front direction.
        self.view_specs: List[FixedViewSpec] = [
            FixedViewSpec(name="front", source="front"),
            FixedViewSpec(
                name="left_to_front_30",
                source="virtual",
                fisheye_camera_override="image_02",
                fisheye_relative_yaw_deg_override=self.turn_to_front_deg,
            ),
            FixedViewSpec(
                name="right_to_front_30",
                source="virtual",
                fisheye_camera_override="image_03",
                fisheye_relative_yaw_deg_override=-self.turn_to_front_deg,
            ),
            FixedViewSpec(
                name="left_axis",
                source="virtual",
                fisheye_camera_override="image_02",
                fisheye_relative_yaw_deg_override=0.0,
            ),
            FixedViewSpec(
                name="right_axis",
                source="virtual",
                fisheye_camera_override="image_03",
                fisheye_relative_yaw_deg_override=0.0,
            ),
        ]

    def set_epoch(self, epoch: int):
        # Kept for trainer compatibility.
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.ds_front) * len(self.view_specs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n_views = len(self.view_specs)
        frame_idx = int(idx) // n_views
        view_idx = int(idx) % n_views
        spec = self.view_specs[view_idx]

        # Prepare override metadata BEFORE calling the base dataset
        override_meta = {}
        if spec.fisheye_camera_override is not None:
            override_meta["fisheye_camera_override"] = spec.fisheye_camera_override
        if spec.fisheye_relative_yaw_deg_override is not None:
            override_meta["fisheye_relative_yaw_deg_override"] = float(spec.fisheye_relative_yaw_deg_override)

        # Inject override into the SampleIndex object so Kitti360dDataset can read it
        ds = self.ds_front if spec.source == "front" else self.ds_virtual
        if hasattr(ds, 'samples') and frame_idx < len(ds.samples):
            sample_idx = ds.samples[frame_idx]
            if override_meta:
                # Store original meta to restore later
                original_meta = sample_idx.meta
                sample_idx.meta = override_meta
                try:
                    sample = ds[frame_idx]
                finally:
                    # Restore original meta
                    sample_idx.meta = original_meta
            else:
                sample = ds[frame_idx]
        else:
            sample = ds[frame_idx]

        # Add fixed view metadata to the returned sample
        sample = dict(sample)
        meta = dict(sample.get("meta", {}))
        # Preserve override information for debugging/visualization
        if spec.fisheye_camera_override is not None:
            meta["fisheye_camera_override"] = spec.fisheye_camera_override
        if spec.fisheye_relative_yaw_deg_override is not None:
            meta["fisheye_relative_yaw_deg_override"] = float(spec.fisheye_relative_yaw_deg_override)
        meta["fixed_view_name"] = spec.name
        meta["fixed_view_index"] = int(view_idx)
        sample["meta"] = meta
        return sample


class ArTransformDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base: torch.utils.data.Dataset,
        *,
        bos_token: int,
        target_rows: int,
        target_cols: int,
        bev_feature_dim: int,
    ):
        self.base = base
        self.bos_token = int(bos_token)
        self.target_rows = int(target_rows)
        self.target_cols = int(target_cols)
        self.target_h = self.target_rows * 16
        self.target_w = self.target_cols * 16
        self.seq_len = self.target_rows * self.target_cols
        self.bev_feature_dim = int(bev_feature_dim)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> ArSample:
        try:
            sample = self.base[idx]

            # Extract tensors - keep them on CPU!
            rgb = sample["image"].cpu()  # Keep on CPU
            rgb = rgb * 2.0 - 1.0
            K = sample["K"].cpu()
            T_cam_to_world = sample["T_cam_to_world"].cpu()
            T_imu_to_world = sample["T_imu_to_world"].cpu()
            sat_available = bool(sample.get("sat_available", False))
            sat_tensor = sample["sat"].cpu()

            # Extract frame_id if available
            frame_id = sample.get("frame_id", None)
            drive = sample.get("drive", None)
            sample_meta = sample.get("meta", {})
            if not isinstance(sample_meta, dict):
                sample_meta = {}
            view_name = sample_meta.get("fixed_view_name")
            view_index = sample_meta.get("fixed_view_index")
            if view_index is not None:
                try:
                    view_index = int(view_index)
                except Exception:
                    view_index = None

            # Return raw data without any GPU computations - all tensors are on CPU!
            return ArSample(
                input_tokens=torch.zeros(self.seq_len, dtype=torch.long),  # Will be filled in training loop
                target_tokens=torch.zeros(self.seq_len, dtype=torch.long),  # Will be filled in training loop
                semantic_fine=torch.zeros(4, self.target_rows, self.target_cols),  # Will be filled in training loop
                coords_fine=torch.zeros(2, self.target_rows, self.target_cols),  # Will be filled in training loop
                pose=torch.zeros(13),  # Will be filled in training loop
                K=K,
                bev=None,  # Will be filled in training loop
                bev_vis_mask=None,  # Will be filled in training loop
                sup_valid=torch.ones(self.seq_len),  # Will be filled in training loop
                frame_id=frame_id,
                drive=drive,
                view_name=view_name,
                view_index=view_index,
                vis_rgb=rgb,
                vis_ipm=None,  # Will be filled in training loop
                vis_ipm_valid=None,  # Will be filled in training loop
                vis_sat=sat_tensor,
                vis_T_cam_to_world=T_cam_to_world,
                vis_T_imu_to_world=T_imu_to_world,
            )

        except Exception as e:
            print(f"[Error ArTransform] Error in __getitem__ for idx={idx}: {str(e)}")
            print(f"[Error ArTransform] Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return ArSample(
                input_tokens=torch.zeros(self.seq_len, dtype=torch.long),
                target_tokens=torch.zeros(self.seq_len, dtype=torch.long),
                semantic_fine=torch.zeros(4, self.target_rows, self.target_cols),
                coords_fine=torch.zeros(2, self.target_rows, self.target_cols),
                pose=torch.zeros(13),
                K=torch.eye(3),
                bev=None,
                bev_vis_mask=torch.zeros(1, 64, 64),
                sup_valid=torch.zeros(self.seq_len),
            )


def collate_ar_samples(samples: list[ArSample], *, device: torch.device = None, bev_feature_dim: int = 256):
    # Collect CPU tensors
    input_tokens = torch.stack([s.input_tokens for s in samples], dim=0)
    target_tokens = torch.stack([s.target_tokens for s in samples], dim=0)
    semantic = torch.stack([s.semantic_fine for s in samples], dim=0)
    coords = torch.stack([s.coords_fine for s in samples], dim=0)
    pose = torch.stack([s.pose for s in samples], dim=0)
    K = torch.stack([s.K for s in samples], dim=0)
    sup_valid = torch.stack([s.sup_valid for s in samples], dim=0)

    # Collect frame_ids
    frame_ids = [s.frame_id for s in samples if s.frame_id is not None]
    if frame_ids:
        frame_ids = torch.tensor(frame_ids, dtype=torch.long)
    else:
        frame_ids = None
    drive_names = [s.drive if s.drive is not None else "unknown_drive" for s in samples]
    view_names = [s.view_name if s.view_name is not None else "unknown" for s in samples]
    view_indices = torch.tensor(
        [int(s.view_index) if s.view_index is not None else -1 for s in samples],
        dtype=torch.long,
    )

    # Stack BEV visibility masks (keep on CPU)
    bev_vis_mask = torch.stack([
        s.bev_vis_mask if s.bev_vis_mask is not None
        else torch.zeros(1, 64, 64)
        for s in samples
    ], dim=0)  # (B, 1, 64, 64)

    # Robustly stack visualization tensors, handling None for invalid samples
    has_vis_data = any(s.vis_rgb is not None for s in samples)
    if has_vis_data:
        ref_sample = next(s for s in samples if s.vis_rgb is not None)
        vis_rgb = torch.stack([s.vis_rgb if s.vis_rgb is not None else torch.zeros_like(ref_sample.vis_rgb) for s in samples])
        # Handle vis_ipm and vis_ipm_valid which may be None
        if ref_sample.vis_ipm is not None:
            vis_ipm = torch.stack([s.vis_ipm if s.vis_ipm is not None else torch.zeros_like(ref_sample.vis_ipm) for s in samples])
        else:
            vis_ipm = None
        if ref_sample.vis_ipm_valid is not None:
            vis_ipm_valid = torch.stack([s.vis_ipm_valid if s.vis_ipm_valid is not None else torch.zeros_like(ref_sample.vis_ipm_valid) for s in samples])
        else:
            vis_ipm_valid = None
    else:
        vis_rgb, vis_ipm, vis_ipm_valid = None, None, None

    # Stack satellite images and camera poses for visualization
    has_sat_vis = any(s.vis_sat is not None for s in samples)
    if has_sat_vis:
        ref_sat = next(s for s in samples if s.vis_sat is not None)
        vis_sat = torch.stack([s.vis_sat if s.vis_sat is not None else torch.zeros_like(ref_sat.vis_sat) for s in samples])
        vis_T_cam = torch.stack([s.vis_T_cam_to_world if s.vis_T_cam_to_world is not None else torch.eye(4) for s in samples])
        vis_T_imu = torch.stack([s.vis_T_imu_to_world if s.vis_T_imu_to_world is not None else torch.eye(4) for s in samples])
    else:
        vis_sat, vis_T_cam, vis_T_imu = None, None, None

    valid_mask = sup_valid.sum(dim=1) > 0

    # BEV shape normalization (B,C,H,W)
    bev_hw = None
    for s in samples:
        if s.bev is not None:
            bev_hw = (int(s.bev.shape[-2]), int(s.bev.shape[-1]))
            break
    if bev_hw is None:
        bev_hw = (64, 64)

    bev_list = []
    for s in samples:
        if s.bev is None:
            bev_list.append(torch.zeros(bev_feature_dim, bev_hw[0], bev_hw[1], dtype=torch.float32))
        else:
            bev_list.append(s.bev.squeeze(0))
    bev = torch.stack(bev_list, dim=0)

    # filter invalid samples
    if valid_mask.numel() > 0 and (not bool(valid_mask.all())):
        input_tokens = input_tokens[valid_mask]
        target_tokens = target_tokens[valid_mask]
        semantic = semantic[valid_mask]
        coords = coords[valid_mask]
        pose = pose[valid_mask]
        K = K[valid_mask]
        sup_valid = sup_valid[valid_mask]
        bev = bev[valid_mask]
        bev_vis_mask = bev_vis_mask[valid_mask]
        vis_rgb = vis_rgb[valid_mask] if vis_rgb is not None else None
        vis_ipm = vis_ipm[valid_mask] if vis_ipm is not None else None
        vis_ipm_valid = vis_ipm_valid[valid_mask] if vis_ipm_valid is not None else None
        vis_sat = vis_sat[valid_mask] if vis_sat is not None else None
        vis_T_cam = vis_T_cam[valid_mask] if vis_T_cam is not None else None
        vis_T_imu = vis_T_imu[valid_mask] if vis_T_imu is not None else None
        if frame_ids is not None:
            frame_ids = frame_ids[valid_mask]
        view_indices = view_indices[valid_mask]
        keep_flags = valid_mask.detach().cpu().tolist()
        drive_names = [name for name, keep in zip(drive_names, keep_flags) if keep]
        view_names = [name for name, keep in zip(view_names, keep_flags) if keep]

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "condition": {
            "semantic": semantic,
            "coords": coords,
            "pose": pose,
            "K": K,
        },
        "bev": bev,
        "bev_vis_mask": bev_vis_mask,
        "sup_valid": sup_valid,
        "frame_id": frame_ids,
        "drive": drive_names,
        "view_name": view_names,
        "view_index": view_indices,
        "vis": {
            "rgb": vis_rgb,
            "ipm": vis_ipm,
            "ipm_valid": vis_ipm_valid,
            "sat": vis_sat,
            "T_cam_to_world": vis_T_cam,
            "T_imu_to_world": vis_T_imu,
            "K": K,  # (B, 3, 3) — for pair consistency loss
        },
    }

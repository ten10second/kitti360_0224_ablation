"""
Diffusion Model Data Pipeline.

This module provides data loading and preprocessing utilities for the
diffusion-based view synthesis model. Reuses existing Kitti360dDataset
and view sampling logic from the AR model pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.data.deterministic import make_rng
from world3d.data.ar_pipeline import (
    DeterministicYawDataset,
    FixedFiveViewDataset,
    MixedViewIndexDataset,
)
from world3d.train.pose_ar import build_pose_vec
from world3d.data.ar_pipeline import compute_bev_visibility_mask


@dataclass
class DiffusionSample:
    """Sample container for diffusion model training/inference."""

    rgb: torch.Tensor                 # Target image [-1, 1], (3, H, W)
    sat: torch.Tensor                 # Satellite image [0, 1], (3, H_sat, W_sat)
    pose: torch.Tensor                # Pose vector (13,)
    K: torch.Tensor                   # Camera intrinsics (3, 3)
    T_cam_to_world: torch.Tensor     # Camera-to-world transform (4, 4)
    T_imu_to_world: torch.Tensor     # IMU-to-world transform (4, 4)
    frame_id: Optional[int] = None
    view_name: Optional[str] = None
    view_index: Optional[int] = None
    bev_vis_mask: Optional[torch.Tensor] = None  # BEV visibility mask (1, 64, 64)


class DiffusionTransformDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper to convert Kitti360dDataset samples to DiffusionSample format.

    Reuses the existing dataset but transforms the outputs to be compatible
    with the diffusion model pipeline.
    """

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        target_size: Tuple[int, int] = (512, 512),
        sat_size: int = 512,
        img_h: int = 256,
        img_w: int = 640,
        compute_pose_vec: bool = True,
        compute_bev_mask: bool = True,
    ):
        super().__init__()

        self.base = base
        self.target_size = target_size
        self.sat_size = sat_size
        self.img_h = img_h
        self.img_w = img_w
        self.compute_pose_vec = compute_pose_vec
        self.compute_bev_mask = compute_bev_mask

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> DiffusionSample:
        sample = self.base[idx]

        # Extract basic data
        rgb = sample["image"]  # (3, H, W) in [0, 1]
        sat = sample["sat"]  # (3, H_sat, W_sat) in [0, 1]
        K = sample["K"]  # (3, 3)
        T_cam_to_world = sample["T_cam_to_world"]  # (4, 4)
        T_imu_to_world = sample["T_imu_to_world"]  # (4, 4)

        # Convert RGB to [-1, 1] range
        rgb = rgb * 2.0 - 1.0

        # Resize RGB to target size for SD
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Resize satellite image to consistent size
        sat = F.interpolate(
            sat.unsqueeze(0),
            size=(self.sat_size, self.sat_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Compute pose vector (13-dim)
        pose = torch.zeros(13)
        if self.compute_pose_vec:
            pose = build_pose_vec(
                K,
                T_cam_to_world,
                T_imu_to_world,
                self.img_h,
                self.img_w,
                device=K.device,
            )

        # Compute BEV visibility mask
        bev_vis_mask = None
        if self.compute_bev_mask:
            bev_vis_mask = compute_bev_visibility_mask(
                K=K,
                T_cam_to_world=T_cam_to_world,
                T_imu_to_world=T_imu_to_world,
                cam_h=self.img_h,
                cam_w=self.img_w,
            )

        # Extract metadata
        frame_id = sample.get("frame_id", None)
        sample_meta = sample.get("meta", {})
        view_name = sample_meta.get("fixed_view_name")
        view_index = sample_meta.get("fixed_view_index")
        if view_index is not None:
            try:
                view_index = int(view_index)
            except Exception:
                view_index = None

        return DiffusionSample(
            rgb=rgb,
            sat=sat,
            pose=pose,
            K=K,
            T_cam_to_world=T_cam_to_world,
            T_imu_to_world=T_imu_to_world,
            frame_id=frame_id,
            view_name=view_name,
            view_index=view_index,
            bev_vis_mask=bev_vis_mask,
        )


def collate_diffusion_samples(
    samples: List[DiffusionSample],
) -> Dict[str, torch.Tensor]:
    """
    Collate a list of DiffusionSample into a batch dictionary.

    Args:
        samples: list of DiffusionSample objects

    Returns:
        batch: dictionary with batched tensors
    """
    # Collect tensors
    rgb = torch.stack([s.rgb for s in samples])
    sat = torch.stack([s.sat for s in samples])
    pose = torch.stack([s.pose for s in samples])
    K = torch.stack([s.K for s in samples])
    T_cam_to_world = torch.stack([s.T_cam_to_world for s in samples])
    T_imu_to_world = torch.stack([s.T_imu_to_world for s in samples])

    # Collect BEV visibility masks
    bev_vis_masks = []
    for s in samples:
        if s.bev_vis_mask is not None:
            bev_vis_masks.append(s.bev_vis_mask)
        else:
            bev_vis_masks.append(torch.zeros(1, 64, 64))
    bev_vis_mask = torch.stack(bev_vis_masks)

    # Collect metadata
    frame_ids = []
    view_names = []
    view_indices = []
    for s in samples:
        frame_ids.append(s.frame_id if s.frame_id is not None else -1)
        view_names.append(s.view_name if s.view_name is not None else "unknown")
        view_indices.append(s.view_index if s.view_index is not None else -1)

    frame_ids_tensor = torch.tensor(frame_ids, dtype=torch.long)
    view_indices_tensor = torch.tensor(view_indices, dtype=torch.long)

    return {
        "rgb": rgb,
        "sat": sat,
        "pose": pose,
        "K": K,
        "T_cam_to_world": T_cam_to_world,
        "T_imu_to_world": T_imu_to_world,
        "bev_vis_mask": bev_vis_mask,
        "frame_ids": frame_ids_tensor,
        "view_names": view_names,
        "view_indices": view_indices_tensor,
    }


def build_diffusion_data_pipeline(
    cfg,
    mode: str = "train",
    frames: Optional[List[int]] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Build complete data pipeline for diffusion model training/inference.

    Args:
        cfg: configuration object
        mode: "train" or "val"
        frames: optional list of frame IDs to use

    Returns:
        train_ds, val_ds: training and validation datasets
    """
    data_root = Path(cfg.data_root)

    # Support both single-drive and multi-drive configurations
    drives_config = getattr(cfg, 'drives', None)

    if drives_config is not None:
        # Multi-drive mode
        all_drive_dirs = []
        all_frame_ids = []

        for drive_spec in drives_config:
            # Handle both dict and object types
            if isinstance(drive_spec, dict):
                drive_name = drive_spec.get('name')
                # 根据 mode 自动选择使用 train_frames.txt 或 test_frames.txt
                if mode == "train":
                    frames_file = drive_spec.get('frames_file', 'train_frames.txt')
                else:
                    frames_file = drive_spec.get('frames_file', 'test_frames.txt').replace('train', 'test')
            else:
                drive_name = getattr(drive_spec, 'name', None)
                if mode == "train":
                    frames_file = getattr(drive_spec, 'frames_file', 'train_frames.txt')
                else:
                    frames_file = getattr(drive_spec, 'frames_file', 'test_frames.txt').replace('train', 'test')

            if drive_name is None:
                continue

            drive_dir = data_root / drive_name

            if frames_file:
                # Load frames from specified file (train_frames.txt or test_frames.txt)
                frames_path = drive_dir / frames_file
                if not frames_path.exists():
                    # Try poses.txt as fallback
                    frames_path = drive_dir / "poses.txt"
            else:
                frames_path = drive_dir / "poses.txt"

            frame_ids = None
            if frames_path.exists():
                # Read frame IDs
                frame_ids = []
                with open(frames_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            frame_ids.append(int(parts[0]))
                        except Exception:
                            continue
                frame_ids = sorted(list(set(frame_ids)))

            all_drive_dirs.append(drive_dir)
            all_frame_ids.append(frame_ids)

        drives_for_ds = all_drive_dirs
        frames_for_ds = all_frame_ids if any(all_frame_ids) else None
    else:
        # Single-drive mode (backward compatibility)
        drives_for_ds = [str(data_root / cfg.drive)] if hasattr(cfg, 'drive') else [data_root]

        # 单驱动器模式下，根据 mode 选择对应的 frames 文件
        if mode == "train":
            frames_file = "train_frames.txt"
        else:
            frames_file = "test_frames.txt"

        frames_path = data_root / cfg.drive / frames_file
        if frames_path.exists():
            # 从文件读取 frame IDs
            frame_ids = []
            with open(frames_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        frame_ids.append(int(parts[0]))
                    except Exception:
                        continue
            frames_for_ds = sorted(list(set(frame_ids)))
        else:
            # 使用 poses.txt 作为备用（获取所有 frames）
            frames_for_ds = frames

    if not drives_for_ds:
        raise ValueError(f"No valid drives found at {data_root}")

    # Create base datasets
    ds_front = Kitti360dDataset(
        drives=drives_for_ds,
        frames=frames_for_ds,
        mode="front",
        front_resize=(cfg.virtual_w, cfg.virtual_h),
    )

    ds_virtual = Kitti360dDataset(
        drives=drives_for_ds,
        frames=frames_for_ds,
        mode="fisheye_virtual",
        virtual_size=(cfg.virtual_w, cfg.virtual_h),
    )

    # Use fixed five views (same as AR model)
    if cfg.use_fixed_five_views:
        full_ds = FixedFiveViewDataset(
            ds_front,
            ds_virtual,
            turn_to_front_deg=cfg.fixed_view_turn_deg,
        )
    else:
        # Mixed view sampling
        full_ds = MixedViewIndexDataset(
            ds_front,
            ds_virtual,
            p_front=cfg.p_front,
            strict_ddp=getattr(cfg, "ddp_strict_view", True),
            seed=getattr(cfg, "seed", 42),
        )

    # Add deterministic yaw sampling if enabled
    if getattr(cfg, "yaw_min_abs", 0.0) > 0 or getattr(cfg, "yaw_max_abs", 0.0) > 0:
        full_ds = DeterministicYawDataset(
            full_ds,
            enable=True,
            seed=getattr(cfg, "data_seed", 42),
            yaw_min_abs=getattr(cfg, "yaw_min_abs", 0.0),
            yaw_max_abs=getattr(cfg, "yaw_max_abs", 40.0),
        )

    # Transform to diffusion sample format
    diffusion_ds = DiffusionTransformDataset(
        full_ds,
        target_size=(512, 512),
        sat_size=512,
        img_h=cfg.virtual_h,
        img_w=cfg.virtual_w,
        compute_pose_vec=True,
        compute_bev_mask=True,
    )

    if mode == "train":
        # 训练模式：直接返回整个数据集作为训练集，不进行随机分割
        # 验证集将在 build_train_val_loaders 中单独构建（使用 mode="val" 调用）
        return diffusion_ds, None
    else:
        # 验证/测试模式
        return None, diffusion_ds


def build_train_val_loaders(
    cfg,
    train_ds: torch.utils.data.Dataset,
    val_ds: Optional[torch.utils.data.Dataset] = None,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Build DataLoaders for training and validation.

    Args:
        cfg: configuration object
        train_ds: training dataset
        val_ds: optional validation dataset

    Returns:
        train_loader, val_loader: data loaders
    """
    from torch.utils.data import DataLoader

    # Training loader
    num_workers = getattr(cfg, "loader", {}).get("num_workers", 8)
    prefetch_factor = getattr(cfg, "loader", {}).get("prefetch_factor", 2) if num_workers > 0 else None
    persistent_workers = getattr(cfg, "loader", {}).get("persistent_workers", True) if num_workers > 0 else False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=getattr(cfg, "shuffle", True),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_diffusion_samples,
        drop_last=True,
    )

    # Validation loader
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=getattr(cfg, "eval_batch_size", cfg.batch_size),
            shuffle=False,
            num_workers=getattr(cfg, "loader", {}).get("num_workers", 4),
            collate_fn=collate_diffusion_samples,
            drop_last=False,
        )

    return train_loader, val_loader

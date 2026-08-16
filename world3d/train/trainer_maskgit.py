"""MaskGIT-style iterative parallel token predictor trainer."""
from __future__ import annotations

import math
import os
import random
import functools
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from models.stage2.simplified_token_predictor import MaskGITTokenPredictor
from utils.distributed import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
)
from world3d.config import ArTrainConfig
from world3d.data.ar_pipeline import (
    ArTransformDataset,
    DeterministicYawDataset,
    MixedViewIndexDataset,
    compute_bev_visibility_mask,
    collate_ar_samples,
)
from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.conditioning_ar import get_condition_scale_sizes, build_condition_tokens_with_coords
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.pose_ar import build_pose_vec
from world3d.train.vis_utils import (
    plot_loss_curve,
    render_bev_attn_heatmap,
    render_sat_with_frustum,
    save_samples_grid_step_debug,
)


def get_model_attr(model, attr: str):
    if hasattr(model, "module"):
        return getattr(model.module, attr)
    return getattr(model, attr)


class MaskGITTrainer:
    """Trainer for MaskGIT-style iterative parallel token prediction."""

    def __init__(self, cfg: ArTrainConfig, repo_root: str):
        self.cfg = cfg
        self.repo_root = repo_root

        self.device = init_distributed_mode()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.is_main = is_main_process()

        if not is_dist_avail_and_initialized() and self.cfg.device:
            self.device = torch.device(self.cfg.device)

        if self.is_main:
            print(f"[Distributed] World size: {self.world_size}, Rank: {self.rank}, Device: {self.device}")

        if os.environ.get("TORCH_ANOMALY_DETECT", "0") == "1":
            torch.autograd.set_detect_anomaly(True)

        # --- Numerical optimizations (same as ARTrainer) ---
        # Enable TF32 if configured
        if getattr(self.cfg.perf, "enable_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if self.is_main:
                print("[Perf] TF32 enabled")

        # Enable SDPA if configured
        if getattr(self.cfg.perf, "enable_sdpa", True):
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                if self.is_main:
                    print("[Perf] SDPA enabled (Flash/MemEfficient/Math)")
            except Exception as e:
                if self.is_main:
                    print(f"[Perf] Failed to enable SDPA: {e}")

        self._seed_everything(self.cfg.seed)
        self._build_tokenizer()
        self._build_model()
        self._build_optim()
        self._maybe_resume()
        self._build_datasets()

        self.loss_history = []
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    def _trainer_name(self) -> str:
        return "MaskGIT"

    def _checkpoint_model_type(self) -> str:
        return "maskgit"

    # ── Setup helpers (same as ARTrainer) ──────────────────────────

    def _seed_everything(self, seed: int):
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if bool(getattr(self.cfg, "full_determinism", False)):
            try:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    def _build_tokenizer(self):
        if self.is_main:
            print(f"[Tokenizer] Using VQGAN from {self.cfg.vq_ckpt}")
        self.vq = PretrainedTokenizer(self.cfg.vq_ckpt).to(self.device)
        self.vocab_size = 1024
        self.vq.eval()
        self.vq.requires_grad_(False)

    def _build_model(self):
        self.model_vocab_size = max(self.vocab_size, self.cfg.bos_token + 1)
        self.target_cols = max(1, int(self.cfg.grid_cols))
        self.target_rows = max(1, int(self.cfg.grid_rows))
        self.target_w = self.target_cols * 16
        self.target_h = self.target_rows * 16
        self.seq_len = self.target_rows * self.target_cols

        self.predictor = MaskGITTokenPredictor(
            d_model=self.cfg.d_model,
            vocab_size=self.model_vocab_size,
            num_layers=self.cfg.num_layers,
            nhead=self.cfg.nhead,
            dropout=0.1,
            max_seq_len=self.seq_len,
            target_rows=self.target_rows,
            target_cols=self.target_cols,
            semantic_dim=4,
            fourier_freqs=self.cfg.fourier_freqs,
            train_bev_encoder=self.cfg.train_bev_encoder,
            no_bev_pretrain=self.cfg.no_bev_pretrain,
            pose_dim=13,
            use_pose_token=self.cfg.use_pose_token,
            n_pose_queries=self.cfg.n_pose_queries,
            mode=self.cfg.mode,  # Use mode from config (vanilla/direct/hybrid)
            use_ipm_semantic=self.cfg.use_ipm_semantic,  # Whether to use IPM semantic features
            hybrid_memory_source=self.cfg.hybrid_memory_source,
            use_explicit_token_pos=self.cfg.use_explicit_token_pos,
        ).to(self.device)
        self.predictor.train()

        if self.is_main:
            trainer_name = self._trainer_name()
            num_params = sum(p.numel() for p in self.predictor.parameters())
            print(f"[{trainer_name}TokenPredictor] Parameters: {num_params:,}")
            print(f"[{trainer_name}TokenPredictor] use_ipm_semantic={self.cfg.use_ipm_semantic}")
            print(f"[{trainer_name}TokenPredictor] use_pose_token={self.cfg.use_pose_token}")

        if self.world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.predictor = DDP(
                self.predictor, device_ids=[local_rank],
                output_device=local_rank, find_unused_parameters=True,
            )

        self.bev_encoder_module = get_model_attr(self.predictor, "bev_encoder")
        self.bev_feature_dim = get_model_attr(self.predictor, "bev_feature_dim")
        self.condition_scale_specs = get_condition_scale_sizes(self.target_rows, self.target_cols)

    def _build_optim(self):
        self.optim = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.use_amp = self.device.type == "cuda"
        self.amp_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.scheduler = None
        if self.cfg.use_warmup_cosine:
            total_updates = math.ceil(self.cfg.steps / max(1, int(self.cfg.accum_steps)))
            warmup = max(0, int(self.cfg.warmup_updates))
            min_lr_ratio = max(0.0, float(self.cfg.min_lr) / max(1e-12, float(self.cfg.lr)))

            def lr_lambda(current_update: int):
                if current_update < warmup:
                    return float(current_update + 1) / float(max(1, warmup))
                progress = float(current_update - warmup) / float(max(1, total_updates - warmup))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            from torch.optim.lr_scheduler import LambdaLR
            self.scheduler = LambdaLR(self.optim, lr_lambda=lr_lambda)

        self.start_step = 1

    def _maybe_resume(self):
        if self.cfg.resume_ckpt and getattr(self.cfg, "warm_start_ckpt", None):
            raise ValueError("Specify only one of resume_ckpt and warm_start_ckpt.")

        warm_start_ckpt = getattr(self.cfg, "warm_start_ckpt", None)
        if warm_start_ckpt:
            if not os.path.exists(warm_start_ckpt):
                if self.is_main:
                    print(f"[Warning] Warm-start checkpoint not found: {warm_start_ckpt}")
                return

            if self.is_main:
                print(f"[WarmStart] Loading model weights from {warm_start_ckpt}")
            ckpt = torch.load(warm_start_ckpt, map_location=self.device)

            if self.world_size > 1:
                self.predictor.module.load_state_dict(ckpt["model"], strict=False)
            else:
                self.predictor.load_state_dict(ckpt["model"], strict=False)

            self.start_step = 1
            if self.is_main:
                print("[WarmStart] Loaded model weights only; optimizer and step were reset.")
            return

        if not self.cfg.resume_ckpt:
            return
        if not os.path.exists(self.cfg.resume_ckpt):
            if self.is_main:
                print(f"[Warning] Checkpoint not found: {self.cfg.resume_ckpt}")
            return

        if self.is_main:
            print(f"[Resume] Loading checkpoint from {self.cfg.resume_ckpt}")
        ckpt = torch.load(self.cfg.resume_ckpt, map_location=self.device)

        if self.world_size > 1:
            self.predictor.module.load_state_dict(ckpt["model"], strict=False)
        else:
            self.predictor.load_state_dict(ckpt["model"], strict=False)

        if "optimizer" in ckpt:
            try:
                self.optim.load_state_dict(ckpt["optimizer"])
            except Exception:
                if self.is_main:
                    print("[Resume] Skipping optimizer state due to mismatch.")

        self.start_step = ckpt.get("step", 1) + 1
        if self.is_main:
            print(f"[Resume] Resuming from step {self.start_step}")

    def _build_datasets(self):
        drives_config = getattr(self.cfg, "drives", None)

        if drives_config is not None:
            all_drive_dirs = []
            all_frame_ids = []
            all_weights = []

            data_root = Path(self.cfg.data_root) if self.cfg.data_root else Path(self.repo_root)

            for drive_spec in drives_config:
                drive_name = drive_spec["name"]
                frames_file = drive_spec.get("frames_file", None)
                weight = float(drive_spec.get("weight", 1.0))

                drive_dir = data_root / drive_name

                if frames_file:
                    frames_path = drive_dir / frames_file
                    if not frames_path.exists():
                        if self.is_main:
                            print(f"[Warning] {frames_file} not found in {drive_name}, using poses.txt")
                        frames_path = drive_dir / "poses.txt"
                else:
                    frames_path = drive_dir / "poses.txt"

                if not frames_path.exists():
                    if self.is_main:
                        print(f"[Warning] Skipping {drive_name}: no poses file found")
                    continue

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

                if len(frame_ids) == 0:
                    if self.is_main:
                        print(f"[Warning] Skipping {drive_name}: no valid frames")
                    continue

                all_drive_dirs.append(drive_dir)
                all_frame_ids.append(frame_ids)
                all_weights.append(weight)

                if self.is_main:
                    print(f"[Dataset] {drive_name}: {len(frame_ids)} frames (weight={weight})")

            if len(all_drive_dirs) == 0:
                raise SystemExit("No valid drives found in multi-drive configuration")

            self.frame_count = sum(len(fids) for fids in all_frame_ids)
            if self.is_main:
                print(f"[Dataset] Multi-drive mode: {len(all_drive_dirs)} drives, {self.frame_count} total frames")
        else:
            drive_dir = Path(self.cfg.data_root) / self.cfg.drive if self.cfg.data_root else (Path(self.repo_root) / self.cfg.drive)
            poses_path = drive_dir / "poses.txt"
            if not poses_path.exists():
                raise SystemExit(f"poses.txt not found: {poses_path}")

            pose_frame_ids = []
            with open(poses_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        pose_frame_ids.append(int(parts[0]))
                    except Exception:
                        continue
            pose_frame_ids = sorted(list(set(pose_frame_ids)))

            if self.cfg.shuffle:
                random.shuffle(pose_frame_ids)
            if self.cfg.subset is not None:
                pose_frame_ids = pose_frame_ids[: max(0, int(self.cfg.subset))]
            if len(pose_frame_ids) == 0:
                raise SystemExit("No pose frames found in poses.txt")

            self.frame_count = len(pose_frame_ids)
            all_drive_dirs = [drive_dir]
            all_frame_ids = [pose_frame_ids]

            if self.is_main:
                print(f"[Dataset] Single-drive mode: {drive_dir.name}, {self.frame_count} frames")

        ds_front = Kitti360dDataset(
            drives=all_drive_dirs, frames=all_frame_ids,
            exclude_frames=getattr(self.cfg, 'exclude_frames', None),
            require_exact_pose=True,
            mode="front", front_resize=(self.cfg.virtual_w, self.cfg.virtual_h),
            seed=self.cfg.data_seed if self.cfg.deterministic_data else None,
        )
        ds_virtual = Kitti360dDataset(
            drives=all_drive_dirs, frames=all_frame_ids,
            exclude_frames=getattr(self.cfg, 'exclude_frames', None),
            require_exact_pose=True,
            mode="fisheye_virtual", virtual_hfov_deg=self.cfg.virtual_hfov,
            virtual_size=(self.cfg.virtual_w, self.cfg.virtual_h),
            random_fisheye_relative_yaw=True,
            yaw_min_abs=self.cfg.yaw_min_abs, yaw_max_abs=self.cfg.yaw_max_abs,
            seed=self.cfg.data_seed if self.cfg.deterministic_data else None,
        )
        mixed = MixedViewIndexDataset(
            ds_front, ds_virtual, p_front=self.cfg.p_front,
            strict_ddp=self.cfg.ddp_strict_view, seed=self.cfg.data_seed,
        )
        yaw_wrapped = DeterministicYawDataset(
            mixed, enable=self.cfg.ddp_strict_view, seed=self.cfg.data_seed,
            yaw_min_abs=self.cfg.yaw_min_abs, yaw_max_abs=self.cfg.yaw_max_abs,
        )

        transformed = ArTransformDataset(
            yaw_wrapped,
            bos_token=self.cfg.bos_token,
            target_rows=self.target_rows,
            target_cols=self.target_cols,
            bev_feature_dim=self.bev_feature_dim,
        )

        if self.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                transformed, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, seed=int(self.cfg.data_seed), drop_last=True,
            )
        else:
            sampler = None

        # DataLoader parameters from config (defaults if not specified)
        num_workers = getattr(self.cfg.loader, "num_workers", getattr(self.cfg, "num_workers", 8))
        prefetch_factor = getattr(self.cfg.loader, "prefetch_factor", 2)
        persistent_workers = getattr(self.cfg.loader, "persistent_workers", True)

        if int(num_workers) <= 0:
            prefetch_factor = None
            persistent_workers = False

        self.train_loader = torch.utils.data.DataLoader(
            transformed, batch_size=self.cfg.batch_size,
            shuffle=(sampler is None), sampler=sampler, drop_last=True,
            num_workers=int(num_workers),
            collate_fn=functools.partial(collate_ar_samples, bev_feature_dim=self.bev_feature_dim),
            pin_memory=True,
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else None,
        )
        self.train_sampler = sampler
        self._mixed_view_dataset = mixed
        self._yaw_dataset = yaw_wrapped
        self.dataset_size = len(yaw_wrapped)
        self._data_epoch = 0

        if self.is_main:
            print(f"[Dataset] Training samples: {self.dataset_size}")

    def _set_data_epoch(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self._data_epoch)
        if hasattr(self._mixed_view_dataset, "set_epoch"):
            self._mixed_view_dataset.set_epoch(self._data_epoch)
        if hasattr(self._yaw_dataset, "set_epoch"):
            self._yaw_dataset.set_epoch(self._data_epoch)
        self._data_epoch += 1

    # ── Masking utilities ───────────────────────────────────────────

    def _sample_mask(self, B: int, L: int, device: torch.device, step: int | None = None) -> torch.Tensor:
        """Sample random masks with cosine-scheduled mask ratio per sample.

        Returns:
            mask: (B, L) bool tensor, True = masked
        """
        # Sample uniform r in [0, 1) per sample, then cosine schedule
        r = torch.rand(B, device=device)
        mask_ratio = torch.cos(r * math.pi / 2)  # maps [0,1) -> [1, ~0)
        num_masked = (mask_ratio * L).clamp(min=1).long()  # at least 1 masked

        # Create masks by sorting random noise
        noise = torch.rand(B, L, device=device)
        sorted_indices = noise.argsort(dim=1)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            mask[i, sorted_indices[i, :num_masked[i]]] = True
        return mask

    # ── Training loop ─────────────────────────────────────────────

    def train(self):
        if self.is_main:
            trainer_name = self._trainer_name()
            print(f"[{trainer_name} Training] Steps: {self.start_step} -> {self.cfg.steps}")
            print(f"[{trainer_name} Training] Seq len: {self.seq_len} ({self.target_rows}x{self.target_cols})")
            print(f"[{trainer_name} Training] Batch size per GPU: {self.cfg.batch_size}")

        self._set_data_epoch()
        data_iter = iter(self.train_loader)

        for step in range(self.start_step, self.cfg.steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                self._set_data_epoch()
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Rebuild training targets/conditions on GPU (same pattern as AR trainer)
            rgb_batch = batch["vis"]["rgb"].to(self.device, memory_format=torch.channels_last, non_blocking=True)
            K_batch = batch["condition"]["K"].to(self.device, non_blocking=True)
            T_cam_to_world_batch = batch["vis"]["T_cam_to_world"].to(self.device, non_blocking=True)
            T_imu_to_world_batch = batch["vis"]["T_imu_to_world"].to(self.device, non_blocking=True)
            sat_tensor_batch = batch["vis"]["sat"].to(self.device, memory_format=torch.channels_last, non_blocking=True)

            with torch.no_grad():
                idx_grid = self.vq.encode(rgb_batch)
                if idx_grid.dim() == 4:
                    idx_grid = idx_grid.squeeze(1)
                target_tokens = idx_grid.view(-1, self.target_rows, self.target_cols)

            if target_tokens.size(0) == 0:
                continue

            B = target_tokens.size(0)

            use_ipm_image = bool(getattr(self.cfg, "use_ipm_semantic", False))
            warped_front_batch = []
            warped_valid_batch = []
            warped_coords_batch = []
            for b in range(B):
                warped_front, warped_valid, warped_coords = compute_inverse_projection_view(
                    sat_tensor=sat_tensor_batch[b],
                    K=K_batch[b],
                    T_cam_to_world=T_cam_to_world_batch[b],
                    T_imu_to_world=T_imu_to_world_batch[b],
                    target_h=self.target_h,
                    target_w=self.target_w,
                    device=self.device,
                    return_ipm_image=use_ipm_image,
                )
                warped_front_batch.append(warped_front.squeeze(0))
                warped_valid_batch.append(warped_valid.squeeze(0))
                warped_coords_batch.append(warped_coords.squeeze(0))

            warped_front_batch = torch.stack(warped_front_batch, dim=0)
            warped_valid_batch = torch.stack(warped_valid_batch, dim=0)
            warped_coords_batch = torch.stack(warped_coords_batch, dim=0)

            semantic_tokens_batch, coord_tokens_batch = build_condition_tokens_with_coords(
                warped_front_batch,
                warped_coords_batch,
                warped_valid_batch,
                self.target_rows,
                self.target_cols,
                self.device,
            )

            pose_batch = []
            for b in range(B):
                pv = build_pose_vec(
                    K=K_batch[b],
                    T_cam_to_world=T_cam_to_world_batch[b],
                    T_imu_to_world=T_imu_to_world_batch[b],
                    img_h=self.target_h,
                    img_w=self.target_w,
                    device=self.device,
                )
                pose_batch.append(pv)
            pose_batch = torch.stack(pose_batch, dim=0)

            bev_feats = self.bev_encoder_module(sat_tensor_batch)

            bev_vis_mask = []
            for b in range(B):
                mask_bev = compute_bev_visibility_mask(
                    K=K_batch[b],
                    T_cam_to_world=T_cam_to_world_batch[b],
                    T_imu_to_world=T_imu_to_world_batch[b],
                    bev_size=64,
                    cam_h=self.target_h,
                    cam_w=self.target_w,
                )
                bev_vis_mask.append(mask_bev)
            bev_vis_mask = torch.stack(bev_vis_mask, dim=0)

            sup_valid = torch.ones(B, self.target_rows, self.target_cols, device=self.device)
            tokens_flat = target_tokens.view(B, -1)  # (B, L)

            # Random masking with cosine schedule
            mask = self._sample_mask(B, self.seq_len, tokens_flat.device, step=step)

            condition_tokens = {
                "pose": pose_batch,
                "K": K_batch,
                "coords": coord_tokens_batch["fine"],  # For Direct/hybrid mode
                "T_cam_to_world": T_cam_to_world_batch,  # For RayRoPE world coordinate encoding
            }
            if self.cfg.use_ipm_semantic:
                condition_tokens["semantic"] = semantic_tokens_batch["fine"]

            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                logits = self.predictor(
                    tokens=tokens_flat,
                    mask=mask,
                    condition_tokens=condition_tokens,
                    aligned_bev_feature_map=bev_feats,
                    bev_vis_mask=bev_vis_mask,
                )

                # CE loss only on masked positions
                ce_per_token = F.cross_entropy(
                    logits.view(-1, self.model_vocab_size),
                    tokens_flat.view(-1),
                    reduction="none",
                    label_smoothing=self.cfg.label_smoothing,
                ).view(B, -1)

                # Combine mask with supervision validity
                sup_valid_flat = sup_valid.clamp(0.0, 1.0).view(B, -1)
                weight = mask.float() * sup_valid_flat
                denom = weight.sum().clamp(min=1e-6)
                ce_loss = (ce_per_token * weight).sum() / denom
                total_loss = self.cfg.ce_weight * ce_loss

            if self.is_main:
                self.loss_history.append((step, total_loss.item()))

            accum_steps = max(1, int(self.cfg.accum_steps))
            if (step - self.start_step) % accum_steps == 0:
                self.optim.zero_grad(set_to_none=True)

            is_last_micro = ((step - self.start_step) % accum_steps == accum_steps - 1) or (step == self.cfg.steps)
            ddp_ctx = (
                self.predictor.no_sync()
                if (self.world_size > 1 and hasattr(self.predictor, "no_sync") and not is_last_micro)
                else nullcontext()
            )
            with ddp_ctx:
                self.amp_scaler.scale(total_loss / accum_steps).backward()

            if is_last_micro:
                if self.cfg.grad_clip > 0:
                    self.amp_scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.cfg.grad_clip)
                self.amp_scaler.step(self.optim)
                self.amp_scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()

            if self.is_main and step % self.cfg.print_every == 0:
                mask_pct = mask.float().mean().item() * 100
                msg = f"[Step {step}] CE Loss: {ce_loss.item():.4f} | Mask%: {mask_pct:.1f}%"
                if self.scheduler is not None:
                    try:
                        msg += f" | LR={self.scheduler.get_last_lr()[0]:.2e}"
                    except Exception:
                        pass
                print(msg)

            if self.is_main and int(getattr(self.cfg, "plot_every", 0)) > 0 and step % int(self.cfg.plot_every) == 0:
                try:
                    plot_loss_curve(self.loss_history, out_dir=self.cfg.out_dir, current_step=step)
                except Exception:
                    pass

            # ── Visualization ──
            if self.is_main and int(getattr(self.cfg, "vis_every", 0)) > 0 and step % int(self.cfg.vis_every) == 0:
                try:
                    vis_data = batch.get("vis", {})
                    if isinstance(vis_data, dict) and vis_data:
                        gt_img_vis = vis_data.get("rgb")
                        if gt_img_vis is not None and torch.is_tensor(gt_img_vis) and gt_img_vis.numel() > 0:
                            gt_img_vis = gt_img_vis[0]
                        else:
                            gt_img_vis = None

                        inverse_proj_vis = None
                        inverse_valid_mask_vis = None

                        with torch.no_grad():
                            gt_tokens_1 = target_tokens[0:1]
                            vq_recon = self.vq.decode(gt_tokens_1)
                            vq_recon_vis = vq_recon[0]

                            # Generate via MaskGIT iterative decoding
                            predictor_module = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
                            predictor_module.eval()
                            cond_1 = {k: v[0:1] for k, v in condition_tokens.items()}
                            bev_1 = bev_feats[0:1] if bev_feats is not None else None
                            bvm_1 = bev_vis_mask[0:1] if bev_vis_mask is not None else None
                            gen_grid = predictor_module.generate(
                                condition_tokens=cond_1,
                                aligned_bev_feature_map=bev_1,
                                bev_vis_mask=bvm_1,
                                num_steps=self.cfg.maskgit_num_steps,
                                temperature=self.cfg.maskgit_temperature,
                                top_k=self.cfg.maskgit_top_k,
                            )
                            gen_decoded = self.vq.decode(gen_grid)
                            generated_vis = gen_decoded[0]
                            predictor_module.train()

                        bev_attn_heatmap_vis = None
                        sat_frustum_vis = None
                        try:
                            attn_source = getattr(predictor_module, "pose_aware_anchor_query", None)
                            attn_weights = None
                            if attn_source is not None:
                                attn_weights = getattr(attn_source, "_last_cross_attn_weights", None)

                            if attn_source is not None and attn_weights is not None:
                                if attn_weights.dim() >= 3:
                                    aw_np = attn_weights[0].detach().cpu().numpy()
                                else:
                                    aw_np = attn_weights.detach().cpu().numpy()

                                anchor_pts_np = None
                                if hasattr(attn_source, "_last_anchors") and getattr(attn_source, "_last_anchors", None) is not None:
                                    anchor_pts_np = attn_source._last_anchors[0].detach().cpu().numpy()
                                else:
                                    anchor_cache = getattr(attn_source, "_last_anchor_positions", None)
                                    if anchor_cache is not None:
                                        anchor_pts_np = anchor_cache[0].detach().cpu().numpy()

                                sat_vis = vis_data.get("sat")
                                sat_bg = None
                                if sat_vis is not None and torch.is_tensor(sat_vis) and sat_vis.numel() > 0:
                                    sat_t = sat_vis[0]
                                    if sat_t.dim() == 3 and sat_t.shape[0] == 3:
                                        sat_t = sat_t.permute(1, 2, 0)
                                    sat_bg = (sat_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                                bev_attn_heatmap_vis = render_bev_attn_heatmap(
                                    aw_np, sat_img=sat_bg, anchor_points=anchor_pts_np,
                                )
                        except Exception:
                            pass

                        save_samples_grid_step_debug(
                            step=step, out_dir=self.cfg.out_dir,
                            gt_img_vis=gt_img_vis,
                            inverse_proj_vis=inverse_proj_vis,
                            inverse_valid_mask_vis=inverse_valid_mask_vis,
                            vq_recon_vis=vq_recon_vis,
                            generated_vis=generated_vis,
                            bev_attn_heatmap_vis=bev_attn_heatmap_vis,
                            sat_frustum_vis=sat_frustum_vis,
                            is_main=True, subdir="samples",
                        )
                except Exception as e:
                    import traceback
                    print(f"[Vis] Failed: {e}")
                    print(traceback.format_exc())

            if self.is_main and step % self.cfg.save_every == 0:
                self._save_ckpt(step)

    def _save_ckpt(self, step: int):
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"ckpt_step_{step:07d}.pt"

        model_state = self.predictor.module.state_dict() if hasattr(self.predictor, "module") else self.predictor.state_dict()
        ckpt = {
            "step": step,
            "model": model_state,
            "optimizer": self.optim.state_dict(),
            "model_vocab_size": self.model_vocab_size,
            "model_type": self._checkpoint_model_type(),
        }
        torch.save(ckpt, str(ckpt_path))
        if self.is_main:
            print(f"[Checkpoint] Saved: {ckpt_path}")

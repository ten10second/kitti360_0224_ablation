from __future__ import annotations

import functools
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from models.stage2.simplified_token_predictor import (
    BottomUpSimplifiedTokenPredictor,
    SimplifiedTokenPredictor,
)
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
    FixedFiveViewDataset,
    MixedViewIndexDataset,
    collate_ar_samples,
)
from world3d.data.grouped_sampler import GroupedFrameSampler
from world3d.data.distributed_grouped_sampler import DistributedGroupedFrameSampler

def _top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Sample from logits using top-k filtering and temperature scaling.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        k: Number of top tokens to consider
        temperature: Softmax temperature for sampling.

    Returns:
        Sampled token indices of shape (batch_size, seq_len)
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Ensure k is not larger than vocab size
    k = min(k, logits.size(-1))

    # Get top-k logits and indices
    top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)

    # Convert to probabilities
    probs = F.softmax(top_k_values, dim=-1)

    # Sample from the top-k distribution
    sampled_indices = torch.multinomial(probs.view(-1, k), num_samples=1)
    sampled_indices = sampled_indices.view(logits.shape[0], -1)

    # Gather the actual token indices
    sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    return sampled_tokens
from world3d.io.kitti360d_dataloader import Kitti360dDataset
from world3d.train.conditioning_ar import get_condition_scale_sizes, build_condition_tokens_with_coords
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.pose_ar import build_pose_vec
from world3d.data.ar_pipeline import compute_bev_visibility_mask
from world3d.train.consistency_loss import compute_multi_view_consistency_loss
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


class ArTrainer:
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
            if self.is_main:
                print("[Debug] Anomaly detection enabled")

        # --- Numerical optimizations ---
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

        if self.cfg.ntp_order == "bottomup":
            ModelClass = BottomUpSimplifiedTokenPredictor
        else:
            ModelClass = SimplifiedTokenPredictor
        if self.is_main:
            print(f"[Model] Using predictor class: {ModelClass.__name__} (ntp_order={self.cfg.ntp_order})")

        self.predictor = ModelClass(
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
            mode=self.cfg.mode,
            use_ipm_semantic=self.cfg.use_ipm_semantic,
            hybrid_memory_source=self.cfg.hybrid_memory_source,
            use_explicit_token_pos=self.cfg.use_explicit_token_pos,
        ).to(self.device)
        self.predictor.train()

        if self.is_main:
            print(f"[Model] use_ipm_semantic={self.cfg.use_ipm_semantic}")
            print(f"[Model] use_pose_token={self.cfg.use_pose_token}")

        # Apply channels_last memory format if configured
        if getattr(self.cfg.perf, "channels_last", False):
            try:
                self.predictor.to(memory_format=torch.channels_last)
                if self.is_main:
                    print("[Perf] Channels last memory format enabled")
            except Exception as e:
                if self.is_main:
                    print(f"[Perf] Failed to enable channels last: {e}")

        # Apply torch.compile if configured
        if getattr(self.cfg.perf, "torch_compile", False):
            try:
                self.predictor = torch.compile(
                    self.predictor,
                    mode="reduce-overhead",
                    dynamic=False
                )
                if self.is_main:
                    print("[Perf] torch.compile enabled (reduce-overhead)")
            except Exception as e:
                if self.is_main:
                    print(f"[Perf] Failed to enable torch.compile: {e}")

        if self.is_main:
            num_params = sum(p.numel() for p in self.predictor.parameters())
            print(f"[SimplifiedTokenPredictor] Parameters: {num_params:,}")

        if self.world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.predictor = DDP(
                self.predictor,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=getattr(self.cfg.dist, "find_unused_parameters", True),
                static_graph=getattr(self.cfg.dist, "static_graph", False),
            )

        self.bev_encoder_module = get_model_attr(self.predictor, "bev_encoder")
        self.bev_feature_dim = get_model_attr(self.predictor, "bev_feature_dim")

        self.condition_scale_specs = get_condition_scale_sizes(self.target_rows, self.target_cols)
        self.condition_scale_names = ["fine"]

    def _build_optim(self):
        self.optim = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # AMP mixed precision
        self.use_amp = self.device.type == "cuda"
        self.amp_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.is_main:
            print(f"[AMP] Mixed precision: {'enabled' if self.use_amp else 'disabled'}")

        self.scheduler = None
        if self.cfg.use_warmup_cosine:
            import math as _math

            total_updates = _math.ceil(self.cfg.steps / max(1, int(self.cfg.accum_steps)))
            warmup = max(0, int(self.cfg.warmup_updates))
            min_lr_ratio = max(0.0, float(self.cfg.min_lr) / max(1e-12, float(self.cfg.lr)))

            def lr_lambda(current_update: int):
                if current_update < warmup:
                    return float(current_update + 1) / float(max(1, warmup))
                progress = float(current_update - warmup) / float(max(1, total_updates - warmup))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            from torch.optim.lr_scheduler import LambdaLR

            self.scheduler = LambdaLR(self.optim, lr_lambda=lr_lambda)
            if self.is_main:
                print(
                    f"[Scheduler] Warmup+Cosine enabled. Updates: {total_updates}, Warmup: {warmup}, min_lr: {self.cfg.min_lr}"
                )

        self.start_step = 1

    def _maybe_resume(self):
        if not self.cfg.resume_ckpt:
            return
        if not os.path.exists(self.cfg.resume_ckpt):
            if self.is_main:
                print(f"[Warning] Checkpoint not found: {self.cfg.resume_ckpt}")
            return

        if self.is_main:
            print(f"[Resume] Loading checkpoint from {self.cfg.resume_ckpt}")
        ckpt = torch.load(self.cfg.resume_ckpt, map_location=self.device)

        predictor_module = self.predictor.module if self.world_size > 1 else self.predictor
        try:
            predictor_module.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as err:
            missing_anchor_gate = {
                "anchor_condition_gate.weight",
                "anchor_condition_gate.bias",
            }
            load_result = predictor_module.load_state_dict(ckpt["model"], strict=False)
            missing_keys = set(load_result.missing_keys)
            unexpected_keys = set(load_result.unexpected_keys)
            if unexpected_keys or (missing_keys - missing_anchor_gate):
                raise err
            if self.is_main:
                print(
                    "[Resume] Checkpoint predates anchor condition gate; "
                    "using freshly initialized gate parameters."
                )

        if "optimizer" in ckpt:
            try:
                self.optim.load_state_dict(ckpt["optimizer"])
            except Exception as opt_err:
                if self.is_main:
                    print(f"[Resume] Skipping optimizer state due to mismatch: {opt_err}")
                    print("[Resume] Continuing with freshly initialized optimizer.")

        self.start_step = ckpt.get("step", 1) + 1
        if self.is_main:
            print(f"[Resume] Resuming from step {self.start_step}")

    def _build_datasets(self):
        # Support both single-drive and multi-drive configurations
        drives_config = getattr(self.cfg, 'drives', None)

        if drives_config is not None:
            # Multi-drive mode
            all_drive_dirs = []
            all_frame_ids = []
            all_weights = []

            data_root = Path(self.cfg.data_root) if self.cfg.data_root else Path(self.repo_root)

            for drive_spec in drives_config:
                drive_name = drive_spec['name']
                frames_file = drive_spec.get('frames_file', None)
                weight = float(drive_spec.get('weight', 1.0))

                drive_dir = data_root / drive_name

                if frames_file:
                    # Load frames from specified file (e.g., train_frames.txt)
                    frames_path = drive_dir / frames_file
                    if not frames_path.exists():
                        if self.is_main:
                            print(f"[Warning] {frames_file} not found in {drive_name}, using all frames from poses.txt")
                        frames_path = drive_dir / "poses.txt"
                else:
                    frames_path = drive_dir / "poses.txt"

                if not frames_path.exists():
                    if self.is_main:
                        print(f"[Warning] Skipping {drive_name}: no poses file found")
                    continue

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
            # Single-drive mode (backward compatibility)
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
                if self.is_main:
                    print("[Dataset] Shuffled pose_frame_ids")

            if self.cfg.subset is not None:
                pose_frame_ids = pose_frame_ids[: max(0, int(self.cfg.subset))]
                if self.is_main:
                    print(f"[Dataset] Subset: limited to {len(pose_frame_ids)} pose frames")

            if len(pose_frame_ids) == 0:
                raise SystemExit("No pose frames found in poses.txt")

            self.frame_count = len(pose_frame_ids)

            all_drive_dirs = [drive_dir]
            all_frame_ids = [pose_frame_ids]
            all_weights = [1.0]

        use_fixed_five_views = bool(getattr(self.cfg, "use_fixed_five_views", True))
        fixed_view_turn_deg = float(getattr(self.cfg, "fixed_view_turn_deg", 30.0))

        ds_front = Kitti360dDataset(
            drives=all_drive_dirs,
            frames=all_frame_ids,
            exclude_frames=getattr(self.cfg, 'exclude_frames', None),
            require_exact_pose=True,
            mode="front",
            front_resize=(self.cfg.virtual_w, self.cfg.virtual_h),
            seed=self.cfg.data_seed if self.cfg.deterministic_data else None,
        )
        ds_virtual = Kitti360dDataset(
            drives=all_drive_dirs,
            frames=all_frame_ids,
            exclude_frames=getattr(self.cfg, 'exclude_frames', None),
            require_exact_pose=True,
            mode="fisheye_virtual",
            virtual_hfov_deg=self.cfg.virtual_hfov,
            virtual_size=(self.cfg.virtual_w, self.cfg.virtual_h),
            random_fisheye_relative_yaw=(not use_fixed_five_views),
            yaw_min_abs=self.cfg.yaw_min_abs,
            yaw_max_abs=self.cfg.yaw_max_abs,
            seed=self.cfg.data_seed if self.cfg.deterministic_data else None,
        )

        self._mixed_view_dataset = None
        self._yaw_dataset = None
        if use_fixed_five_views:
            train_index_dataset = FixedFiveViewDataset(
                ds_front,
                ds_virtual,
                turn_to_front_deg=fixed_view_turn_deg,
            )
            if self.is_main:
                print("[Dataset] Using fixed five-view expansion per frame")
                print(
                    "[Dataset] Fixed views: front, left_to_front_30, right_to_front_30, left_axis, right_axis"
                )
        else:
            mixed = MixedViewIndexDataset(
                ds_front,
                ds_virtual,
                p_front=self.cfg.p_front,
                strict_ddp=self.cfg.ddp_strict_view,
                seed=self.cfg.data_seed,
            )
            yaw_wrapped = DeterministicYawDataset(
                mixed,
                enable=self.cfg.ddp_strict_view,
                seed=self.cfg.data_seed,
                yaw_min_abs=self.cfg.yaw_min_abs,
                yaw_max_abs=self.cfg.yaw_max_abs,
            )
            self._mixed_view_dataset = mixed
            self._yaw_dataset = yaw_wrapped
            train_index_dataset = yaw_wrapped

        transformed = ArTransformDataset(
            train_index_dataset,
            bos_token=self.cfg.bos_token,
            target_rows=self.target_rows,
            target_cols=self.target_cols,
            bev_feature_dim=self.bev_feature_dim,
        )

        if use_fixed_five_views:
            if self.world_size > 1:
                sampler = DistributedGroupedFrameSampler(
                    transformed,
                    batch_size=self.cfg.batch_size,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=int(self.cfg.data_seed),
                    drop_last=True,
                )
            else:
                sampler = GroupedFrameSampler(
                    transformed,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    seed=int(self.cfg.data_seed),
                )
        else:
            if self.world_size > 1:
                sampler = torch.utils.data.DistributedSampler(
                    transformed,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=int(self.cfg.data_seed),
                    drop_last=True,
                )
            else:
                sampler = None

        # DataLoader parameters from config (defaults if not specified)
        num_workers = getattr(self.cfg.loader, "num_workers", getattr(self.cfg, "num_workers", 8))
        prefetch_factor = getattr(self.cfg.loader, "prefetch_factor", 2)
        persistent_workers = getattr(self.cfg.loader, "persistent_workers", True)

        # Only use prefetch_factor if num_workers > 0 (multiprocessing mode)
        if int(num_workers) <= 0:
            prefetch_factor = None
            persistent_workers = False  # Can't use persistent workers with num_workers=0

        self.train_loader = torch.utils.data.DataLoader(
            transformed,
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            num_workers=int(num_workers),
            collate_fn=functools.partial(collate_ar_samples, bev_feature_dim=self.bev_feature_dim),
            pin_memory=True,
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else None,
        )
        self.train_sampler = sampler
        self._train_index_dataset = train_index_dataset
        self.dataset_size = len(train_index_dataset)
        self._data_epoch = 0

        if self.is_main:
            print(f"[Dataset] Drive: {drive_dir}")
            print(f"[Dataset] Pose frames: {self.frame_count}")
            print(f"[Dataset] Training samples (all views): {self.dataset_size}")

    def _set_data_epoch(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self._data_epoch)
        if hasattr(self._train_index_dataset, "set_epoch"):
            self._train_index_dataset.set_epoch(self._data_epoch)
        if hasattr(self._mixed_view_dataset, "set_epoch"):
            self._mixed_view_dataset.set_epoch(self._data_epoch)
        if hasattr(self._yaw_dataset, "set_epoch"):
            self._yaw_dataset.set_epoch(self._data_epoch)
        self._data_epoch += 1

    def train(self):
        if self.is_main:
            if self.start_step > 1:
                print(f"[Training] Resuming from step {self.start_step} to {self.cfg.steps}")
            else:
                print(f"[Training] Starting training for {self.cfg.steps} steps")
            print(f"[Training] Dataset size: {self.dataset_size}")
            print(f"[Training] Batch size per GPU: {self.cfg.batch_size}")
            print(f"[Training] Grad accumulation steps: {max(1, int(self.cfg.accum_steps))}")
            print(f"[Training] Effective global batch: {self.cfg.batch_size * self.world_size * max(1, int(self.cfg.accum_steps))}")
            print(f"[Training] BOS token: {self.cfg.bos_token}")
            print(f"[Training] Target size: {self.target_h}x{self.target_w} ({self.target_rows}x{self.target_cols} tokens)")

        self._set_data_epoch()
        data_iter = iter(self.train_loader)

        # Metrics tracking
        import time
        start_time = time.time()
        total_samples_processed = 0

        for step in range(self.start_step, self.cfg.steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                self._set_data_epoch()
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Track samples processed
            total_samples_processed += batch["vis"]["rgb"].size(0)

            # --- Batch processing: Move computations from Dataset to training loop ---

            # 1. Move all tensors to GPU first
            rgb_batch = batch["vis"]["rgb"].to(self.device, memory_format=torch.channels_last, non_blocking=True)
            K_batch = batch["condition"]["K"].to(self.device, non_blocking=True)
            T_cam_to_world_batch = batch["vis"]["T_cam_to_world"].to(self.device, non_blocking=True)
            T_imu_to_world_batch = batch["vis"]["T_imu_to_world"].to(self.device, non_blocking=True)
            sat_tensor_batch = batch["vis"]["sat"].to(self.device, memory_format=torch.channels_last, non_blocking=True)

            # 2. Batch VQ encoding
            with torch.no_grad():
                idx_grid = self.vq.encode(rgb_batch)
                if idx_grid.dim() == 4:
                    idx_grid = idx_grid.squeeze(1)
                idx_grid = idx_grid.view(-1, self.target_rows, self.target_cols)

            # 3. Batch teacher forcing
            predictor_module = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
            input_tokens_batch, target_tokens_batch = predictor_module.make_teacher_forcing(idx_grid, bos_token=self.cfg.bos_token)

            # 4. Batch geometric projection (IPM image disabled when semantic conditioning is off)
            use_ipm_image = bool(getattr(self.cfg, "use_ipm_semantic", False))
            warped_front_batch = []
            warped_valid_batch = []
            warped_coords_batch = []

            B = rgb_batch.size(0)
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
                # 挤压掉批次维度，从 (1, C, H, W) 变为 (C, H, W)
                warped_front_batch.append(warped_front.squeeze(0))
                warped_valid_batch.append(warped_valid.squeeze(0))
                warped_coords_batch.append(warped_coords.squeeze(0))

            warped_front_batch = torch.stack(warped_front_batch, dim=0)
            warped_valid_batch = torch.stack(warped_valid_batch, dim=0)
            warped_coords_batch = torch.stack(warped_coords_batch, dim=0)

            # 添加到 batch['vis'] 字典中，用于可视化
            if use_ipm_image:
                batch['vis']['ipm'] = warped_front_batch
                batch['vis']['ipm_valid'] = warped_valid_batch
            else:
                batch['vis'].pop('ipm', None)
                batch['vis'].pop('ipm_valid', None)

            # 5. Batch condition token building
            semantic_tokens_batch = {}
            coord_tokens_batch = {}
            semantic_tokens, coord_tokens = build_condition_tokens_with_coords(
                warped_front_batch,
                warped_coords_batch,
                warped_valid_batch,
                self.target_rows,
                self.target_cols,
                self.device,
            )
            semantic_tokens_batch["fine"] = semantic_tokens["fine"]
            coord_tokens_batch["fine"] = coord_tokens["fine"]

            # 6. Batch pose vector construction
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

            # 7. Batch BEV encoding
            # DDP fix: BEV encoder参数需要参与计算图，即使 requires_grad=False
            # 移除 torch.no_grad() 上下文，让 frozen params 也能被 DDP reducer 追踪
            bev_feats_input = self.bev_encoder_module(sat_tensor_batch)

            # 8. Batch BEV visibility mask computation
            bev_vis_mask = []
            for b in range(B):
                mask = compute_bev_visibility_mask(
                    K=K_batch[b],
                    T_cam_to_world=T_cam_to_world_batch[b],
                    T_imu_to_world=T_imu_to_world_batch[b],
                    bev_size=64,
                    cam_h=self.target_h,
                    cam_w=self.target_w,
                )
                bev_vis_mask.append(mask)
            bev_vis_mask = torch.stack(bev_vis_mask, dim=0)

            # 9. Supervision validity mask
            supervision_valid_batch = torch.ones(B, self.target_rows, self.target_cols, device=self.device)

            if input_tokens_batch.size(0) == 0:
                if self.is_main:
                    print(f"[Warn] All samples invalid after filtering at step {step}, skipping...")
                continue

            condition_tokens_input = {
                "coords": coord_tokens_batch["fine"],
                "pose": pose_batch,
                "K": K_batch,
                "T_cam_to_world": T_cam_to_world_batch,
            }
            if self.cfg.use_ipm_semantic:
                condition_tokens_input["semantic"] = semantic_tokens_batch["fine"]

            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                logits = self.predictor(
                    generated_tokens=input_tokens_batch.view(input_tokens_batch.size(0), -1),
                    condition_tokens=condition_tokens_input,
                    aligned_bev_feature_map=bev_feats_input,
                    bev_vis_mask=bev_vis_mask,
                )

                ce_per_token = F.cross_entropy(
                    logits.view(-1, self.model_vocab_size),
                    target_tokens_batch.view(-1),
                    reduction="none",
                    label_smoothing=self.cfg.label_smoothing,
                ).view_as(target_tokens_batch)

                sup_valid = supervision_valid_batch.clamp(0.0, 1.0).view_as(ce_per_token)
                denom = sup_valid.sum().clamp(min=1e-6)
                ce_loss = (ce_per_token * sup_valid).sum() / denom
                total_loss = self.cfg.ce_weight * ce_loss

                # Multi-view consistency loss (direct mode only)
                consistency_loss = torch.tensor(0.0, device=self.device)
                consistency_cosine_loss = torch.tensor(0.0, device=self.device)
                consistency_nce_loss = torch.tensor(0.0, device=self.device)
                consistency_matched_pairs = torch.tensor(0.0, device=self.device)
                consistency_valid_view_pairs = torch.tensor(0.0, device=self.device)
                consistency_overlap_ratio = torch.tensor(0.0, device=self.device)
                consistency_nce_weight = 0.0
                if getattr(self.cfg, "use_consistency_loss", False):
                    predictor_module = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
                    consistency_rel_step = max(0, int(step) - int(self.start_step))
                    consistency_nce_weight_max = float(getattr(self.cfg, "consistency_nce_weight_max", 0.1))
                    consistency_nce_ramp_start = int(getattr(self.cfg, "consistency_nce_ramp_start", 10000))
                    consistency_nce_ramp_end = int(getattr(self.cfg, "consistency_nce_ramp_end", 20000))
                    if consistency_rel_step < consistency_nce_ramp_start:
                        consistency_nce_weight = 0.0
                    elif consistency_rel_step >= consistency_nce_ramp_end:
                        consistency_nce_weight = consistency_nce_weight_max
                    else:
                        ramp_span = max(1, consistency_nce_ramp_end - consistency_nce_ramp_start)
                        ramp_ratio = float(consistency_rel_step - consistency_nce_ramp_start) / float(ramp_span)
                        consistency_nce_weight = consistency_nce_weight_max * ramp_ratio

                    # Get intermediate features from model
                    sampled_bev_feat = getattr(predictor_module, "last_sampled_bev_feat", None)
                    semantic_feat = getattr(predictor_module, "last_semantic_feat", None)
                    coords_map = getattr(predictor_module, "last_coords_map", None)
                    consistency_overlap_mask = F.adaptive_avg_pool2d(
                        warped_valid_batch.float(),
                        (self.target_rows, self.target_cols),
                    ) > 0.2

                    if sampled_bev_feat is not None and coords_map is not None:
                        consistency_dict = compute_multi_view_consistency_loss(
                            sampled_bev_feat=sampled_bev_feat,
                            semantic_feat=semantic_feat,
                            coords_map=coords_map,
                            frame_ids=batch.get('frame_id'),
                            view_indices=batch.get('view_index'),
                            overlap_mask=consistency_overlap_mask,
                            temperature=getattr(self.cfg, "consistency_temperature", 0.5),
                            use_semantic=False,  # Use BEV feature consistency by default
                            use_cosine=bool(getattr(self.cfg, "consistency_use_cosine", True)),
                            nce_weight=consistency_nce_weight,
                        )
                        consistency_loss = consistency_dict["loss"]
                        consistency_cosine_loss = consistency_dict["cosine_loss"]
                        consistency_nce_loss = consistency_dict["nce_loss"]
                        consistency_matched_pairs = consistency_dict["matched_pairs"]
                        consistency_valid_view_pairs = consistency_dict["valid_view_pairs"]
                        consistency_overlap_ratio = consistency_dict["overlap_ratio"]
                        total_loss = total_loss + getattr(self.cfg, "consistency_weight", 0.1) * consistency_loss

            if self.is_main:
                ce_val = float(ce_loss.item())
                consistency_val = float(consistency_loss.item()) if torch.is_tensor(consistency_loss) else float(consistency_loss)
                self.loss_history.append((
                    step,
                    float(total_loss.item()),
                    ce_val,
                    consistency_val,
                    float(consistency_cosine_loss.item()),
                    float(consistency_nce_loss.item()),
                    float(consistency_nce_weight),
                    float(consistency_matched_pairs.item()),
                    float(consistency_valid_view_pairs.item()),
                    float(consistency_overlap_ratio.item()),
                ))

            accum_steps = max(1, int(self.cfg.accum_steps))
            if (step - self.start_step) % accum_steps == 0:
                self.optim.zero_grad(set_to_none=True)

            is_last_micro = ((step - self.start_step) % accum_steps == accum_steps - 1) or (step == self.cfg.steps)
            ddp_no_sync_ctx_pred = (
                self.predictor.no_sync()
                if (self.world_size > 1 and hasattr(self.predictor, "no_sync") and not is_last_micro)
                else nullcontext()
            )
            with ddp_no_sync_ctx_pred:
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
                # Calculate throughput
                elapsed_time = time.time() - start_time
                samples_per_sec = total_samples_processed / elapsed_time
                tokens_per_sec = total_samples_processed * self.seq_len / elapsed_time

                # GPU memory usage
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_cached = torch.cuda.memory_reserved() / 1024**3
                mem_peak = torch.cuda.max_memory_allocated() / 1024**3

                msg = f"[Step {step}] CE Loss: {ce_loss.item():.4f}"
                if consistency_loss > 0.0001:
                    msg += f" | Consistency Loss: {consistency_loss.item():.4f}"
                    msg += f" (cos={consistency_cosine_loss.item():.4f}, nce={consistency_nce_loss.item():.4f}, nce_w={consistency_nce_weight:.3f})"
                    msg += f" | Match={int(consistency_matched_pairs.item())}"
                    msg += f" | Pair={int(consistency_valid_view_pairs.item())}"
                    msg += f" | Overlap={consistency_overlap_ratio.item():.3f}"
                msg += f" | Total: {total_loss.item():.4f}"
                if self.scheduler is not None:
                    try:
                        msg += f" | LR={self.scheduler.get_last_lr()[0]:.2e}"
                    except Exception:
                        pass

                # Add performance metrics
                msg += (
                    f" | Throughput: {samples_per_sec:.1f} samples/s, "
                    f"{tokens_per_sec:.0f} tokens/s"
                )
                msg += (
                    f" | GPU Mem: {mem_used:.1f}GB used, "
                    f"{mem_cached:.1f}GB cached, "
                    f"{mem_peak:.1f}GB peak"
                )
                msg += (
                    f" | Batch: {self.cfg.batch_size} (per GPU), "
                    f"Global: {self.cfg.batch_size * self.world_size * max(1, int(self.cfg.accum_steps))}"
                )

                print(msg)

            if self.is_main and int(getattr(self.cfg, "plot_every", 0)) > 0 and step % int(self.cfg.plot_every) == 0:
                try:
                    plot_loss_curve(self.loss_history, out_dir=self.cfg.out_dir, current_step=step)
                except Exception as e:
                    print(f"[Plot] Failed to save loss curve: {e}")


            if self.is_main and int(getattr(self.cfg, "vis_every", 0)) > 0 and step % int(self.cfg.vis_every) == 0:
                try:
                    vis_data = batch.get("vis", {})
                    if not isinstance(vis_data, dict) or not vis_data:
                        continue

                    # Get ground truth image from the 'vis' dictionary
                    gt_img_vis = vis_data.get("rgb")
                    if gt_img_vis is not None and torch.is_tensor(gt_img_vis) and gt_img_vis.numel() > 0:
                        gt_img_vis = gt_img_vis[0]  # Take first sample
                    else:
                        gt_img_vis = None

                    inverse_proj_vis = None
                    inverse_valid_mask_vis = None
                    if bool(getattr(self.cfg, "use_ipm_semantic", False)):
                        inverse_proj_vis = vis_data.get("ipm")
                        if inverse_proj_vis is not None and torch.is_tensor(inverse_proj_vis) and inverse_proj_vis.numel() > 0:
                            inverse_proj_vis = inverse_proj_vis[0]  # Take first sample
                        else:
                            inverse_proj_vis = None

                        inverse_valid_mask_vis = vis_data.get("ipm_valid")
                        if inverse_valid_mask_vis is not None and torch.is_tensor(inverse_valid_mask_vis) and inverse_valid_mask_vis.numel() > 0:
                            inverse_valid_mask_vis = inverse_valid_mask_vis[0]  # Take first sample
                        else:
                            inverse_valid_mask_vis = None

                    with torch.no_grad():
                        # Get VQ reconstruction from ground truth tokens
                        gt_tokens_1 = target_tokens_batch[0:1]  # (1, rows, cols)
                        vq_recon = self.vq.decode(gt_tokens_1)  # (1, 3, H, W) in [-1, 1]
                        vq_recon_vis = vq_recon[0]  # (3, H, W)

                        # Generate predicted tokens and decode
                        pred_tokens = _top_k_sampling(
                            logits[:, :, :self.vocab_size],
                            k=self.cfg.vis_top_k,
                            temperature=self.cfg.vis_temperature
                        )
                        pred_grid = pred_tokens.view(-1, self.target_rows, self.target_cols)[0:1]  # (1, rows, cols)
                        gen_decoded = self.vq.decode(pred_grid)  # (1, 3, H, W) in [-1, 1]
                        generated_vis = gen_decoded[0]  # (3, H, W)

                    # --- BEV Attention Heatmap ---
                    bev_attn_heatmap_vis = None
                    sat_frustum_vis = None
                    try:
                        predictor_module = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
                        attn_source = getattr(predictor_module, "pose_aware_anchor_query", None)
                        attn_weights = None
                        if attn_source is not None:
                            attn_weights = getattr(attn_source, "_last_cross_attn_weights", None)

                        if attn_source is not None and attn_weights is not None:
                            # Use first sample in batch
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
                                sat_t = sat_vis[0]  # (3, H, W) in [0,1]
                                if sat_t.dim() == 3 and sat_t.shape[0] == 3:
                                    sat_t = sat_t.permute(1, 2, 0)
                                sat_bg = (sat_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                            bev_attn_heatmap_vis = render_bev_attn_heatmap(
                                aw_np, sat_img=sat_bg, anchor_points=anchor_pts_np,
                            )
                    except Exception as e_attn:
                        print(f"[Vis] BEV attention heatmap failed: {e_attn}")

                    # --- Satellite + FOV Frustum ---
                    try:
                        sat_vis = vis_data.get("sat")
                        T_cam_vis = vis_data.get("T_cam_to_world")
                        T_imu_vis = vis_data.get("T_imu_to_world")
                        K_vis = batch["condition"]["K"]
                        if (sat_vis is not None and T_cam_vis is not None
                                and T_imu_vis is not None and K_vis is not None):
                            sat_t = sat_vis[0]
                            if sat_t.dim() == 3 and sat_t.shape[0] == 3:
                                sat_t = sat_t.permute(1, 2, 0)
                            sat_np = (sat_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                            sat_frustum_vis = render_sat_with_frustum(
                                sat_img=sat_np,
                                K=K_vis[0].cpu().numpy(),
                                T_cam_to_world=T_cam_vis[0].cpu().numpy(),
                                T_imu_to_world=T_imu_vis[0].cpu().numpy(),
                                cam_h=self.target_h,
                                cam_w=self.target_w,
                            )
                    except Exception as e_frust:
                        print(f"[Vis] Satellite frustum failed: {e_frust}")

                    # Save visualization
                    save_samples_grid_step_debug(
                        step=step,
                        out_dir=self.cfg.out_dir,
                        gt_img_vis=gt_img_vis,
                        inverse_proj_vis=inverse_proj_vis,
                        inverse_valid_mask_vis=inverse_valid_mask_vis,
                        vq_recon_vis=vq_recon_vis,
                        generated_vis=generated_vis,
                        bev_attn_heatmap_vis=bev_attn_heatmap_vis,
                        sat_frustum_vis=sat_frustum_vis,
                        is_main=True,
                        subdir="samples",
                    )

                except Exception as e:
                    import traceback
                    print(f"[Vis] Failed to save samples grid: {e}")
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
        }
        torch.save(ckpt, str(ckpt_path))
        if self.is_main:
            print(f"[Checkpoint] Saved: {ckpt_path}")

    def _save_test_frames(self, n_frames: int = 8):
        """Save a fixed set of test frames for later evaluation.

        Saves raw data (GT image, satellite, poses, intrinsics, tokens, etc.)
        that can be used to evaluate generation quality and geometric consistency
        across different checkpoints without re-running the data pipeline.

        The frames are saved once and never used for training.
        """
        test_dir = Path(self.cfg.out_dir) / "test_frames"
        if test_dir.exists() and len(list(test_dir.glob("frame_*.pt"))) >= n_frames:
            print(f"[TestFrames] Already saved {n_frames} test frames in {test_dir}")
            return

        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TestFrames] Saving {n_frames} test frames to {test_dir} ...")

        # Use a separate data iterator so we don't disturb training
        data_iter = iter(self.train_loader)
        saved = 0
        for batch in data_iter:
            if saved >= n_frames:
                break

            vis_data = batch.get("vis", {})
            if not isinstance(vis_data, dict) or not vis_data:
                continue

            B = batch["input_tokens"].size(0)
            for b_idx in range(min(B, n_frames - saved)):
                frame_data = {
                    "input_tokens": batch["input_tokens"][b_idx].cpu(),
                    "target_tokens": batch["target_tokens"][b_idx].cpu(),
                    "pose": batch["condition"]["pose"][b_idx].cpu(),
                    "K": batch["condition"]["K"][b_idx].cpu(),
                    "semantic": batch["condition"]["semantic"][b_idx].cpu(),
                    "sup_valid": batch["sup_valid"][b_idx].cpu(),
                }

                # Save frame_id if available
                if "frame_id" in batch:
                    frame_data["frame_id"] = int(batch["frame_id"][b_idx].item()) if torch.is_tensor(batch["frame_id"]) else int(batch["frame_id"][b_idx])
                if "view_index" in batch and torch.is_tensor(batch["view_index"]):
                    frame_data["view_index"] = int(batch["view_index"][b_idx].item())
                if "view_name" in batch and isinstance(batch["view_name"], list):
                    frame_data["view_name"] = str(batch["view_name"][b_idx])

                # BEV features
                if batch["bev"] is not None:
                    frame_data["bev"] = batch["bev"][b_idx].cpu()
                if batch["bev_vis_mask"] is not None:
                    frame_data["bev_vis_mask"] = batch["bev_vis_mask"][b_idx].cpu()

                # Visualization data
                for vis_key in ["rgb", "ipm", "ipm_valid", "sat", "T_cam_to_world", "T_imu_to_world"]:
                    val = vis_data.get(vis_key)
                    if val is not None and torch.is_tensor(val) and val.numel() > 0:
                        frame_data[f"vis_{vis_key}"] = val[b_idx].cpu()

                save_path = test_dir / f"frame_{saved:04d}.pt"
                torch.save(frame_data, str(save_path))
                saved += 1

        print(f"[TestFrames] Saved {saved} test frames to {test_dir}")

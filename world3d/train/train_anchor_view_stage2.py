"""Trainer for stage2 anchor-view conditioned generation."""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from collections import OrderedDict

# 先把仓库根目录加入Python路径，必须在导入models之前执行
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import functools
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from models.stage2.simplified_token_predictor import SimplifiedTokenPredictor
from utils.distributed import (
    get_rank, get_world_size, init_distributed_mode,
    is_dist_avail_and_initialized, is_main_process
)
from world3d.config import ArTrainConfig
from world3d.data.ar_pipeline import (
    ArTransformDataset, FixedFiveViewDataset, collate_ar_samples
)
from world3d.data.distributed_grouped_sampler import DistributedGroupedFrameSampler
from world3d.train.conditioning_ar import build_condition_tokens_with_coords
from world3d.train.geometry_ar import compute_inverse_projection_view
from world3d.train.pose_ar import build_pose_vec
from world3d.train.view_pairing import group_and_pair_views
from world3d.train.anchor_view_conditioning import AnchorViewConditioner
from world3d.train.anchor_view_consistency_loss import AnchorViewConsistencyLoss
from world3d.train.vis_utils import plot_anchor_stage2_loss_curve


def get_model_attr(model, attr_name):
    if hasattr(model, 'module'):
        return getattr(model.module, attr_name)
    return getattr(model, attr_name)


class ArAnchorViewStage2Trainer:
    def __init__(self, cfg: ArTrainConfig, repo_root: str):
        self.cfg = cfg
        self.repo_root = repo_root
        self.device = init_distributed_mode()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.is_main = is_main_process()

        if not is_dist_avail_and_initialized() and self.cfg.device:
            self.device = torch.device(self.cfg.device)

        self._seed_everything(self.cfg.seed)
        self._build_tokenizer()
        self._build_model()
        self._build_anchor_components()  # 在 optimizer 之前构建，以便包含参数
        self._build_optim()
        self._maybe_resume()
        self._build_anchor_generator()
        self._build_datasets()

        self.anchor_mem_cache = OrderedDict()
        self.anchor_cache_dir = None
        cache_dir = getattr(self.cfg, "anchor_view_cache_dir", None)
        if cache_dir:
            namespace = self._anchor_cache_namespace()
            self.anchor_cache_dir = Path(cache_dir) / namespace
            self.anchor_cache_dir.mkdir(parents=True, exist_ok=True)

        self.loss_history = []
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    def _trace_step(self, step: int, tag: str):
        if os.environ.get("WORLD3D_TRACE_FIRST_STEP", "0") != "1":
            return
        if step != self.start_step:
            return
        print(f"[Trace][rank{self.rank}][step{step}] {tag}", flush=True)

    def _seed_everything(self, seed: int):
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

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

        if self.is_main:
            print(f"[Model] Target size: {self.target_h}x{self.target_w}")

        self.predictor = SimplifiedTokenPredictor(
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
            use_pose_token=True,
            n_pose_queries=self.cfg.n_pose_queries,
            mode=self.cfg.mode,
            use_ipm_semantic=self.cfg.use_ipm_semantic,
        ).to(self.device)
        self.predictor.train()

        if self.world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.predictor = DDP(
                self.predictor,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=getattr(self.cfg.dist, "find_unused_parameters", True),
                static_graph=getattr(self.cfg.dist, "static_graph", False),
                broadcast_buffers=False,
            )

        self.bev_encoder_module = get_model_attr(self.predictor, "bev_encoder")
        self.bev_feature_dim = get_model_attr(self.predictor, "bev_feature_dim")

        if self.is_main:
            num_params = sum(p.numel() for p in self.predictor.parameters())
            print(f"[Model] Parameters: {num_params:,}")

    def _build_optim(self):
        # 收集所有需要优化的参数
        params = list(self.predictor.parameters())
        if self.anchor_conditioner is not None:
            params.extend(self.anchor_conditioner.parameters())
        if self.anchor_consistency_loss is not None:
            params.extend(self.anchor_consistency_loss.parameters())
        self.trainable_params = [param for param in params if param.requires_grad]

        self.optim = torch.optim.AdamW(
            self.trainable_params,
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
            if self.is_main:
                print(
                    f"[Scheduler] Warmup+Cosine enabled. Updates: {total_updates}, Warmup: {warmup}, min_lr: {self.cfg.min_lr}"
                )

        self.start_step = 1

    def _build_anchor_generator(self):
        self.anchor_generator = None
        if not getattr(self.cfg, "use_anchor_view_training", False):
            return
        if str(getattr(self.cfg, "anchor_view_source", "generated")).lower() != "generated":
            return
        if bool(getattr(self.cfg, "anchor_view_cache_only", False)):
            if self.is_main:
                print("[Anchor] Cache-only mode enabled; skip building GPU anchor generator")
            return

        source_model = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
        self.anchor_generator = copy.deepcopy(source_model).to(self.device)
        self.anchor_generator.eval()
        self.anchor_generator.requires_grad_(False)

        if self.is_main:
            print("[Anchor] Using frozen generated anchor-view source")

    def _anchor_cache_namespace(self) -> str:
        ckpt = str(getattr(self.cfg, "resume_ckpt", "") or "no_resume")
        ckpt_name = Path(ckpt).stem if ckpt else "no_resume"
        return f"{ckpt_name}_topk{int(getattr(self.cfg, 'anchor_view_top_k', 1))}_temp{str(getattr(self.cfg, 'anchor_view_temperature', 1.0)).replace('.', 'p')}"

    def _anchor_cache_key(self, drive: str | None, frame_id: int | None, view_index: int | None) -> str | None:
        if drive is None or frame_id is None or view_index is None:
            return None
        return f"{drive}__{int(frame_id):010d}__view{int(view_index)}"

    def _anchor_cache_path(self, cache_key: str) -> Path | None:
        if self.anchor_cache_dir is None:
            return None
        return self.anchor_cache_dir / f"{cache_key}.pt"

    def _get_anchor_from_mem_cache(self, cache_key: str) -> torch.Tensor | None:
        tensor = self.anchor_mem_cache.get(cache_key)
        if tensor is None:
            return None
        self.anchor_mem_cache.move_to_end(cache_key)
        return tensor

    def _put_anchor_into_mem_cache(self, cache_key: str, tensor_cpu: torch.Tensor):
        max_items = max(0, int(getattr(self.cfg, "anchor_view_mem_cache_size", 2048)))
        if max_items <= 0:
            return
        self.anchor_mem_cache[cache_key] = tensor_cpu
        self.anchor_mem_cache.move_to_end(cache_key)
        while len(self.anchor_mem_cache) > max_items:
            self.anchor_mem_cache.popitem(last=False)

    def _load_anchor_from_disk_cache(self, cache_key: str) -> torch.Tensor | None:
        cache_path = self._anchor_cache_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if isinstance(payload, dict):
                tensor = payload.get("image")
            else:
                tensor = payload
            if tensor is None:
                return None
            tensor = tensor.to(dtype=torch.float16, device="cpu").contiguous()
            self._put_anchor_into_mem_cache(cache_key, tensor)
            return tensor
        except Exception:
            return None

    def _save_anchor_to_disk_cache(self, cache_key: str, tensor_cpu: torch.Tensor):
        cache_path = self._anchor_cache_path(cache_key)
        if cache_path is None:
            return
        tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}.pt")
        try:
            torch.save({"image": tensor_cpu}, tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _anchor_tensor_to_device(self, tensor_cpu: torch.Tensor) -> torch.Tensor:
        return tensor_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True)

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

        # Load model
        if self.world_size > 1:
            load_result = self.predictor.module.load_state_dict(ckpt["model"], strict=False)
        else:
            load_result = self.predictor.load_state_dict(ckpt["model"], strict=False)
        if self.is_main:
            missing = list(getattr(load_result, "missing_keys", []))
            unexpected = list(getattr(load_result, "unexpected_keys", []))
            if missing:
                print(f"[Resume] Missing model keys: {missing}")
            if unexpected:
                print(f"[Resume] Unexpected model keys: {unexpected}")

        # Load anchor_conditioner (may not exist in old checkpoints)
        if self.anchor_conditioner is not None and "anchor_conditioner" in ckpt:
            try:
                # 和保存逻辑保持一致，自动判断是否有module属性
                if hasattr(self.anchor_conditioner, "module"):
                    self.anchor_conditioner.module.load_state_dict(ckpt["anchor_conditioner"], strict=True)
                else:
                    self.anchor_conditioner.load_state_dict(ckpt["anchor_conditioner"], strict=True)
                if self.is_main:
                    print("[Resume] Loaded anchor_conditioner")
            except Exception as e:
                if self.is_main:
                    print(f"[Resume] Failed to load anchor_conditioner: {e}")
        elif self.anchor_conditioner is not None:
            if self.is_main:
                print("[Resume] anchor_conditioner not in checkpoint, training from scratch")

        # Load anchor_consistency_loss (may not exist in old checkpoints)
        if self.anchor_consistency_loss is not None and "anchor_consistency_loss" in ckpt:
            try:
                # 和保存逻辑保持一致，自动判断是否有module属性
                if hasattr(self.anchor_consistency_loss, "module"):
                    self.anchor_consistency_loss.module.load_state_dict(ckpt["anchor_consistency_loss"], strict=True)
                else:
                    self.anchor_consistency_loss.load_state_dict(ckpt["anchor_consistency_loss"], strict=True)
                if self.is_main:
                    print("[Resume] Loaded anchor_consistency_loss")
            except Exception as e:
                if self.is_main:
                    print(f"[Resume] Failed to load anchor_consistency_loss: {e}")
        elif self.anchor_consistency_loss is not None:
            if self.is_main:
                print("[Resume] anchor_consistency_loss not in checkpoint, training from scratch")

        # Load optimizer (may fail if parameter count changed)
        if "optimizer" in ckpt:
            try:
                self.optim.load_state_dict(ckpt["optimizer"])
                if self.is_main:
                    print("[Resume] Loaded optimizer state")
            except Exception as e:
                if self.is_main:
                    print(f"[Resume] Failed to load optimizer state (will continue without): {e}")

        # Load AMP GradScaler state
        if "amp_scaler" in ckpt and self.use_amp:
            try:
                self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                if self.is_main:
                    print("[Resume] Loaded AMP GradScaler state")
            except Exception as e:
                if self.is_main:
                    print(f"[Resume] Failed to load AMP GradScaler state (will continue without): {e}")

        if self.scheduler is not None and "scheduler" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
                if self.is_main:
                    print("[Resume] Loaded scheduler state")
            except Exception as e:
                if self.is_main:
                    print(f"[Resume] Failed to load scheduler state (will continue without): {e}")

        self.start_step = ckpt.get("step", 1) + 1
        if self.is_main:
            print(f"[Resume] Resuming from step {self.start_step}")

    def _build_datasets(self):
        drives_config = getattr(self.cfg, 'drives', None)

        if drives_config is None:
            data_root = Path(self.cfg.data_root) if self.cfg.data_root else Path(self.repo_root)
            drive_dir = data_root / self.cfg.drive
            poses_path = drive_dir / "poses.txt"

            pose_frame_ids = []
            with open(poses_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pose_frame_ids.append(int(line.split()[0]))
                    except:
                        continue

            pose_frame_ids = sorted(list(set(pose_frame_ids)))

            all_drive_dirs = [drive_dir]
            all_frame_ids = [pose_frame_ids]
        else:
            all_drive_dirs = []
            all_frame_ids = []

            data_root = Path(self.cfg.data_root) if self.cfg.data_root else Path(self.repo_root)

            for drive_spec in drives_config:
                drive_name = drive_spec['name']
                frames_file = drive_spec.get('frames_file', None)

                drive_dir = data_root / drive_name
                frames_path = drive_dir / (frames_file if frames_file else "poses.txt")

                if not frames_path.exists():
                    continue

                frame_ids = []
                with open(frames_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            frame_ids.append(int(line.split()[0]))
                        except:
                            continue

                frame_ids = sorted(list(set(frame_ids)))
                all_drive_dirs.append(drive_dir)
                all_frame_ids.append(frame_ids)

        from world3d.io.kitti360d_dataloader import Kitti360dDataset

        ds_front = Kitti360dDataset(
            drives=all_drive_dirs,
            frames=all_frame_ids,
            mode="front",
            front_resize=(self.cfg.virtual_w, self.cfg.virtual_h),
            seed=self.cfg.data_seed if self.cfg.deterministic_data else None,
        )
        ds_virtual = Kitti360dDataset(
            drives=all_drive_dirs,
            frames=all_frame_ids,
            mode="fisheye_virtual",
            virtual_hfov_deg=self.cfg.virtual_hfov,
            virtual_size=(self.cfg.virtual_w, self.cfg.virtual_h),
            random_fisheye_relative_yaw=False,
            seed=self.cfg.data_seed if self.cfg.deterministic_data else None,
        )

        fixed_view_turn_deg = float(getattr(self.cfg, "fixed_view_turn_deg", 30.0))
        train_index_dataset = FixedFiveViewDataset(
            ds_front, ds_virtual, turn_to_front_deg=fixed_view_turn_deg,
        )

        transformed = ArTransformDataset(
            train_index_dataset,
            bos_token=self.cfg.bos_token,
            target_rows=self.target_rows,
            target_cols=self.target_cols,
            bev_feature_dim=self.bev_feature_dim,
        )

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
            from world3d.data.grouped_sampler import GroupedFrameSampler
            sampler = GroupedFrameSampler(
                transformed,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                seed=int(self.cfg.data_seed),
            )

        self.train_loader = torch.utils.data.DataLoader(
            transformed,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            num_workers=int(getattr(self.cfg, "num_workers", 8)),
            collate_fn=functools.partial(collate_ar_samples, bev_feature_dim=self.bev_feature_dim),
            pin_memory=True,
        )

        self.train_sampler = sampler
        self.dataset_size = len(train_index_dataset)
        self.steps_per_epoch = len(self.train_loader)

        if self.is_main:
            print(f"[Dataset] Training samples: {self.dataset_size}")

    def _build_anchor_components(self):
        if not getattr(self.cfg, "use_anchor_view_training", False):
            self.anchor_conditioner = None
            self.anchor_consistency_loss = None
            return

        if self.is_main:
            print("[Anchor] Building anchor-view conditioning components")

        self.anchor_conditioner = AnchorViewConditioner(
            feature_channels=self.cfg.d_model,
            image_channels=3,
            hidden_dim=128,
        ).to(self.device)

        if self.world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.anchor_conditioner = DDP(
                self.anchor_conditioner,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                static_graph=False,
                broadcast_buffers=False,
            )

        if bool(getattr(self.cfg, "anchor_view_use_consistency", False)):
            self.anchor_consistency_loss = AnchorViewConsistencyLoss(
                feature_loss_weight=getattr(self.cfg, "anchor_feature_loss_weight", 1.0),
                temperature=getattr(self.cfg, "anchor_consistency_temperature", 0.5),
                distance_threshold=getattr(self.cfg, "anchor_distance_threshold", 0.05),
            ).to(self.device)
        else:
            self.anchor_consistency_loss = None

        # Use DDP for anchor_conditioner as well so gradient synchronization stays
        # in the reducer and avoids manual all_reduce deadlocks across ranks.

    def _sync_anchor_conditioner_grads(self):
        if self.world_size <= 1 or self.anchor_conditioner is None:
            return
        if hasattr(self.anchor_conditioner, "no_sync"):
            return
        if not dist.is_available() or not dist.is_initialized():
            return

        for param in self.anchor_conditioner.parameters():
            if not param.requires_grad:
                continue

            grad_buf = param.grad
            if grad_buf is None:
                grad_buf = torch.zeros_like(param)
            else:
                grad_buf = grad_buf.detach()

            dist.all_reduce(grad_buf, op=dist.ReduceOp.SUM)
            grad_buf.div_(float(self.world_size))

            if param.grad is None:
                param.grad = grad_buf
            else:
                param.grad.copy_(grad_buf)

    def train(self):
        if self.is_main:
            if self.start_step > 1:
                print(f"[Training] Resuming from step {self.start_step} to {self.cfg.steps}")
            else:
                print(f"[Training] Starting training for {self.cfg.steps} steps")

            use_anchor = getattr(self.cfg, "use_anchor_view_training", False)
            use_anchor_consistency = bool(getattr(self.cfg, "anchor_view_use_consistency", False))
            print(f"[Training] Anchor-view training: {use_anchor}")
            print(f"[Training] Anchor-view consistency: {use_anchor_consistency}")

        # 计算初始epoch
        current_epoch = (self.start_step - 1) // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
        # 初始化时设置一次epoch
        if hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(current_epoch)
        data_iter = iter(self.train_loader)

        for step in range(self.start_step, self.cfg.steps + 1):
            self._trace_step(step, "loop_enter")
            try:
                self._trace_step(step, "before_next_batch")
                batch = next(data_iter)
                self._trace_step(step, "after_next_batch")
            except StopIteration:
                # 新epoch开始，更新epoch并重新shuffle
                current_epoch += 1
                if hasattr(self.train_sampler, "set_epoch"):
                    self.train_sampler.set_epoch(current_epoch)
                data_iter = iter(self.train_loader)
                self._trace_step(step, "after_reset_dataloader")
                batch = next(data_iter)
                self._trace_step(step, "after_next_batch_reset")

            accum_steps = max(1, int(self.cfg.accum_steps))
            if (step - self.start_step) % accum_steps == 0:
                self.optim.zero_grad(set_to_none=True)

            is_last_micro = ((step - self.start_step) % accum_steps == accum_steps - 1) or (step == self.cfg.steps)
            ddp_no_sync = self.predictor.no_sync() if (self.world_size > 1 and hasattr(self.predictor, "no_sync") and not is_last_micro) else nullcontext()
            conditioner_no_sync = self.anchor_conditioner.no_sync() if (
                self.world_size > 1 and self.anchor_conditioner is not None and hasattr(self.anchor_conditioner, "no_sync") and not is_last_micro
            ) else nullcontext()
            with ddp_no_sync, conditioner_no_sync:
                # For DDP gradient accumulation, no_sync must wrap the full
                # forward + backward region instead of backward only.
                self._trace_step(step, "before_compute_loss")
                t0 = time.perf_counter()
                loss_dict = self._compute_loss(batch, step)
                self._trace_step(step, f"after_compute_loss dt={time.perf_counter() - t0:.3f}s")
                self._trace_step(step, "before_backward")
                t1 = time.perf_counter()
                self.amp_scaler.scale(loss_dict["total_loss"] / accum_steps).backward()
                self._trace_step(step, f"after_backward dt={time.perf_counter() - t1:.3f}s")

            if is_last_micro:
                if self.cfg.grad_clip > 0:
                    self._trace_step(step, "before_unscale")
                    self.amp_scaler.unscale_(self.optim)
                    self._sync_anchor_conditioner_grads()
                    self._trace_step(step, "before_grad_clip")
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.grad_clip)
                else:
                    self._sync_anchor_conditioner_grads()

                self._trace_step(step, "before_optim_step")
                t2 = time.perf_counter()
                self.amp_scaler.step(self.optim)
                self.amp_scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self._trace_step(step, f"after_optim_step dt={time.perf_counter() - t2:.3f}s")

            # Logging
            if self.is_main and step % self.cfg.print_every == 0:
                msg = f"[Step {step}] Baseline CE: {loss_dict['ce_loss']:.4f}"
                msg += f" | Target Cond CE: {loss_dict['anchor_ce_loss']:.4f}"
                msg += f" | Consistency: {loss_dict['consistency_loss']:.6f}"  # 强制显示，保留6位小数
                msg += f" | Overlap: {loss_dict.get('overlap_ratio', 0.0):.3f}"
                msg += f" | AnchorSrc: {loss_dict.get('anchor_source', 'n/a')}"
                msg += f" | Total: {loss_dict['total_loss'].item():.4f}"
                if self.scheduler is not None:
                    try:
                        msg += f" | LR={self.scheduler.get_last_lr()[0]:.2e}"
                    except Exception:
                        pass
                print(msg)

                # 记录 loss 到 TXT 文件
                self.loss_history.append({
                    'step': step,
                    'ce_loss': loss_dict['ce_loss'],
                    'anchor_ce_loss': loss_dict['anchor_ce_loss'],
                    'consistency_loss': loss_dict['consistency_loss'],
                    'overlap_ratio': loss_dict.get('overlap_ratio', 0.0),
                    'anchor_source': loss_dict.get('anchor_source', 'n/a'),
                    'total_loss': loss_dict['total_loss'].item()
                })

                # 保存到 TXT 文件（简单格式，直接追加）
                loss_txt_path = Path(self.cfg.out_dir) / 'loss_history.txt'
                file_exists = loss_txt_path.exists()
                with open(loss_txt_path, 'a') as f:
                    if not file_exists:
                        # 写入表头
                        f.write(f"{'step':>10} {'ce_loss':>10} {'anchor_ce_loss':>10} {'consistency_loss':>12} {'overlap_ratio':>10} {'anchor_source':>10} {'total_loss':>10}\n")
                        f.write("-" * 80 + "\n")
                    # 写入数据行
                    f.write(f"{step:10d} {loss_dict['ce_loss']:10.4f} {loss_dict['anchor_ce_loss']:10.4f} {loss_dict['consistency_loss']:12.6f} {loss_dict.get('overlap_ratio', 0.0):10.3f} {loss_dict.get('anchor_source', 'n/a'):10} {loss_dict['total_loss'].item():10.4f}\n")

            if self.is_main and int(getattr(self.cfg, "plot_every", 0)) > 0 and step % int(self.cfg.plot_every) == 0:
                try:
                    plot_anchor_stage2_loss_curve(self.loss_history, out_dir=self.cfg.out_dir, current_step=step)
                except Exception as plot_err:
                    print(f"[Plot] Failed to save anchor CE curve: {plot_err}")

            # Save checkpoint
            if self.is_main and step % self.cfg.save_every == 0:
                self._save_ckpt(step)

    def _sample_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        top_k = max(1, int(getattr(self.cfg, "anchor_view_top_k", 1)))
        temperature = float(getattr(self.cfg, "anchor_view_temperature", 1.0))
        logits_sample = logits[:, :self.vocab_size]
        if temperature != 1.0:
            logits_sample = logits_sample / temperature
        k = min(top_k, logits_sample.size(-1))
        if k == 1:
            return torch.argmax(logits_sample, dim=-1)
        top_vals, top_idx = torch.topk(logits_sample, k, dim=-1)
        probs = F.softmax(top_vals, dim=-1)
        sampled = torch.multinomial(probs, 1)
        return torch.gather(top_idx, -1, sampled).squeeze(-1)

    @torch.no_grad()
    def _generate_anchor_images_for_indices(
        self,
        anchor_indices,
        condition_tokens,
        bev_feats,
        bev_vis_mask,
        K_batch,
        T_cam_batch,
        frame_ids=None,
        view_indices=None,
        drive_names=None,
    ):
        if self.anchor_generator is None or len(anchor_indices) == 0:
            return {}

        anchor_indices = sorted(set(int(idx) for idx in anchor_indices))
        result = {}
        missing_indices = []

        frame_id_list = frame_ids.tolist() if isinstance(frame_ids, torch.Tensor) else frame_ids
        view_index_list = view_indices.tolist() if isinstance(view_indices, torch.Tensor) else view_indices

        for idx in anchor_indices:
            cache_key = self._anchor_cache_key(
                drive_names[idx] if drive_names is not None and idx < len(drive_names) else None,
                frame_id_list[idx] if frame_id_list is not None and idx < len(frame_id_list) else None,
                view_index_list[idx] if view_index_list is not None and idx < len(view_index_list) else None,
            )
            if cache_key is None:
                missing_indices.append(idx)
                continue

            cached = self._get_anchor_from_mem_cache(cache_key)
            if cached is None:
                cached = self._load_anchor_from_disk_cache(cache_key)

            if cached is None:
                missing_indices.append(idx)
                continue

            result[idx] = self._anchor_tensor_to_device(cached)

        if len(missing_indices) == 0:
            return result

        if bool(getattr(self.cfg, "anchor_view_cache_only", False)):
            missing_keys = []
            for idx in missing_indices[:8]:
                cache_key = self._anchor_cache_key(
                    drive_names[idx] if drive_names is not None and idx < len(drive_names) else None,
                    frame_id_list[idx] if frame_id_list is not None and idx < len(frame_id_list) else None,
                    view_index_list[idx] if view_index_list is not None and idx < len(view_index_list) else None,
                )
                if cache_key is not None:
                    missing_keys.append(cache_key)
            preview = ", ".join(missing_keys) if missing_keys else "<unknown keys>"
            raise RuntimeError(
                "anchor_view_cache_only=True but pseudo-anchor cache miss occurred. "
                f"Missing {len(missing_indices)} entries, e.g. {preview}"
            )

        if self.anchor_generator is None:
            raise RuntimeError(
                "anchor_view_source='generated' requires either a built anchor_generator "
                "or fully populated anchor cache"
            )

        anchor_indices = missing_indices
        cond = {
            "coords": condition_tokens["coords"][anchor_indices],
            "pose": condition_tokens["pose"][anchor_indices],
            "K": K_batch[anchor_indices],
            "T_cam_to_world": T_cam_batch[anchor_indices],
        }
        if "semantic" in condition_tokens:
            cond["semantic"] = condition_tokens["semantic"][anchor_indices]

        bev_anchor = bev_feats[anchor_indices]
        bev_vis_anchor = bev_vis_mask[anchor_indices]

        generated = torch.full(
            (len(anchor_indices), 1),
            int(self.cfg.bos_token),
            dtype=torch.long,
            device=self.device,
        )
        past_kv = None

        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
            for _ in range(self.seq_len):
                inp = generated if past_kv is None else generated[:, -1:]
                logits, past_kv = self.anchor_generator(
                    generated_tokens=inp,
                    condition_tokens=cond,
                    aligned_bev_feature_map=bev_anchor,
                    bev_vis_mask=bev_vis_anchor,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                next_tok = self._sample_next_token(logits[:, -1, :])
                generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)

        token_seq = generated[:, 1:]
        token_grid = self.anchor_generator.seq_to_grid(token_seq)
        anchor_images = self.vq.decode(token_grid)
        for pos, idx in enumerate(anchor_indices):
            image = anchor_images[pos:pos+1]
            result[idx] = image

            cache_key = self._anchor_cache_key(
                drive_names[idx] if drive_names is not None and idx < len(drive_names) else None,
                frame_id_list[idx] if frame_id_list is not None and idx < len(frame_id_list) else None,
                view_index_list[idx] if view_index_list is not None and idx < len(view_index_list) else None,
            )
            if cache_key is None:
                continue

            image_cpu = image.detach().to(device="cpu", dtype=torch.float16).contiguous()
            self._put_anchor_into_mem_cache(cache_key, image_cpu)
            self._save_anchor_to_disk_cache(cache_key, image_cpu)

        return result

    def _compute_loss(self, batch: dict, step: int) -> dict:
        B = batch["vis"]["rgb"].size(0)
        device = batch["vis"]["rgb"].device

        # Move to GPU
        rgb_batch = batch["vis"]["rgb"].to(self.device)
        K_batch = batch["condition"]["K"].to(self.device)
        T_cam_batch = batch["vis"]["T_cam_to_world"].to(self.device)
        T_imu_batch = batch["vis"]["T_imu_to_world"].to(self.device)
        sat_batch = batch["vis"]["sat"].to(self.device)

        # VQ encoding
        with torch.no_grad():
            idx_grid = self.vq.encode(rgb_batch)
            if idx_grid.dim() == 4:
                idx_grid = idx_grid.squeeze(1)
            idx_grid = idx_grid.view(-1, self.target_rows, self.target_cols)

        # Teacher forcing
        predictor_module = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
        input_tokens, target_tokens = predictor_module.make_teacher_forcing(idx_grid, bos_token=self.cfg.bos_token)

        # IPM projection
        use_ipm_image = bool(getattr(self.cfg, "use_ipm_semantic", False))
        warped_front, warped_valid, warped_coords = [], [], []
        for b in range(B):
            wf, wv, wc = compute_inverse_projection_view(
                sat_tensor=sat_batch[b],
                K=K_batch[b],
                T_cam_to_world=T_cam_batch[b],
                T_imu_to_world=T_imu_batch[b],
                target_h=self.target_h,
                target_w=self.target_w,
                device=self.device,
                return_ipm_image=use_ipm_image,
            )
            warped_front.append(wf.squeeze(0))
            warped_valid.append(wv.squeeze(0))
            warped_coords.append(wc.squeeze(0))

        warped_front = torch.stack(warped_front, dim=0)
        warped_valid = torch.stack(warped_valid, dim=0)
        warped_coords = torch.stack(warped_coords, dim=0)

        # Condition tokens
        semantic_tokens, coord_tokens = build_condition_tokens_with_coords(
            warped_front, warped_coords, warped_valid,
            self.target_rows, self.target_cols, self.device,
        )

        # Pose vectors
        pose_batch = []
        for b in range(B):
            pv = build_pose_vec(
                K=K_batch[b], T_cam_to_world=T_cam_batch[b],
                T_imu_to_world=T_imu_batch[b],
                img_h=self.target_h, img_w=self.target_w, device=self.device,
            )
            pose_batch.append(pv)
        pose_batch = torch.stack(pose_batch, dim=0)

        # BEV encoding
        bev_feats = self.bev_encoder_module(sat_batch)

        # BEV visibility mask
        from world3d.data.ar_pipeline import compute_bev_visibility_mask
        bev_vis_mask = []
        for b in range(B):
            mask = compute_bev_visibility_mask(
                K=K_batch[b], T_cam_to_world=T_cam_batch[b],
                T_imu_to_world=T_imu_batch[b], bev_size=64,
                cam_h=self.target_h, cam_w=self.target_w,
            )
            bev_vis_mask.append(mask)
        bev_vis_mask = torch.stack(bev_vis_mask, dim=0)

        condition_tokens = {
            "coords": coord_tokens["fine"],
            "pose": pose_batch,
            "K": K_batch,
            "T_cam_to_world": T_cam_batch,
        }
        if bool(getattr(self.cfg, "use_ipm_semantic", False)):
            condition_tokens["semantic"] = semantic_tokens["fine"]

        # Supervision mask
        sup_valid = torch.ones(B, self.target_rows, self.target_cols, device=self.device).flatten(1)  # (B, R*C) = (B, 640) 展平成和token序列相同形状

        # Compute loss
        use_anchor = getattr(self.cfg, "use_anchor_view_training", False)
        frame_ids = batch.get("frame_id")
        drive_names = batch.get("drive")
        view_indices = batch.get("view_index")

        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
            if use_anchor and frame_ids is not None and view_indices is not None:
                loss_dict = self._compute_anchor_loss(
                    rgb_batch, input_tokens, target_tokens, condition_tokens,
                    bev_feats, bev_vis_mask, sup_valid,
                    frame_ids, drive_names, view_indices, K_batch, T_cam_batch, T_imu_batch,
                    warped_coords, warped_valid, step,
                )
            else:
                # Baseline CE only
                logits = self.predictor(
                    generated_tokens=input_tokens.view(input_tokens.size(0), -1),
                    condition_tokens=condition_tokens,
                    aligned_bev_feature_map=bev_feats,
                    bev_vis_mask=bev_vis_mask,
                )

                ce_per_token = F.cross_entropy(
                    logits.view(-1, self.model_vocab_size),
                    target_tokens.view(-1), reduction="none",
                    label_smoothing=self.cfg.label_smoothing,
                ).view_as(target_tokens)

                denom = sup_valid.sum().clamp(min=1e-6)
                ce_loss = (ce_per_token * sup_valid).sum() / denom
                total_loss = self.cfg.ce_weight * ce_loss

                loss_dict = {
                    "ce_loss": ce_loss.item(),
                    "anchor_ce_loss": 0.0,
                    "consistency_loss": 0.0,
                    "overlap_ratio": 0.0,
                    "total_loss": total_loss,  # 保持tensor用于backward
                }

        return loss_dict

    def _select_target_condition_tokens(
        self,
        condition_tokens: dict,
        K_batch: torch.Tensor,
        T_cam_batch: torch.Tensor,
        indices: torch.Tensor,
    ) -> dict:
        selected = {
            "coords": condition_tokens["coords"].index_select(0, indices),
            "pose": condition_tokens["pose"].index_select(0, indices),
            "K": K_batch.index_select(0, indices),
            "T_cam_to_world": T_cam_batch.index_select(0, indices),
        }
        if "semantic" in condition_tokens:
            selected["semantic"] = condition_tokens["semantic"].index_select(0, indices)
        return selected

    def _compute_pair_batch_ce(
        self,
        pair_target_indices: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        condition_tokens: dict,
        bev_feats: torch.Tensor,
        bev_vis_mask: torch.Tensor,
        sup_valid: torch.Tensor,
        K_batch: torch.Tensor,
        T_cam_batch: torch.Tensor,
        anchor_memory: torch.Tensor | None,
    ) -> torch.Tensor:
        target_condition_tokens = self._select_target_condition_tokens(
            condition_tokens=condition_tokens,
            K_batch=K_batch,
            T_cam_batch=T_cam_batch,
            indices=pair_target_indices,
        )

        logits_cond = self.predictor(
            generated_tokens=input_tokens.index_select(0, pair_target_indices).view(pair_target_indices.numel(), -1),
            condition_tokens=target_condition_tokens,
            aligned_bev_feature_map=bev_feats.index_select(0, pair_target_indices),
            bev_vis_mask=bev_vis_mask.index_select(0, pair_target_indices),
            anchor_memory=anchor_memory,
        )

        ce_cond = F.cross_entropy(
            logits_cond.view(-1, self.model_vocab_size),
            target_tokens.index_select(0, pair_target_indices).view(-1),
            reduction="none",
            label_smoothing=self.cfg.label_smoothing,
        ).view(pair_target_indices.numel(), -1)

        sup_valid_t = sup_valid.index_select(0, pair_target_indices).view_as(ce_cond)
        ce_per_pair = (ce_cond * sup_valid_t).sum(dim=1) / sup_valid_t.sum(dim=1).clamp(min=1e-6)
        return ce_per_pair.sum()

    def _compute_anchor_loss_loop(
        self,
        pairs,
        rgb_batch,
        input_tokens,
        target_tokens,
        condition_tokens,
        bev_feats,
        bev_vis_mask,
        sup_valid,
        K_batch,
        T_cam_batch,
        T_imu_batch,
        warped_coords,
        warped_valid,
        step,
        use_anchor_consistency,
        predictor_module,
        anchor_view_source,
        generated_anchor_cache,
    ):
        device = rgb_batch.device
        anchor_ce_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)
        overlap_ratio_sum = torch.tensor(0.0, device=device)
        anchor_source = anchor_view_source
        pair_count = 0

        for pair in pairs:
            anchor_idx = pair.anchor_idx
            target_idx = pair.target_idx

            use_dropout = False
            anchor_dropout_prob = float(getattr(self.cfg, "anchor_view_dropout_prob", 0.3))
            if torch.rand(1).item() < anchor_dropout_prob:
                use_dropout = True
                anchor_cond_feat = None
            else:
                anchor_image = rgb_batch[anchor_idx:anchor_idx+1]
                anchor_source = "gt"

                if anchor_view_source == "generated":
                    anchor_source = "generated"
                    if anchor_idx in generated_anchor_cache:
                        anchor_image = generated_anchor_cache[anchor_idx]
                    else:
                        anchor_source = "gt_fallback"

                anchor_cond_feat, valid_mask = self.anchor_conditioner(
                    anchor_image=anchor_image,
                    anchor_T_cam_to_world=T_cam_batch[anchor_idx:anchor_idx+1],
                    anchor_T_imu_to_world=T_imu_batch[anchor_idx:anchor_idx+1],
                    anchor_K=K_batch[anchor_idx:anchor_idx+1],
                    target_T_cam_to_world=T_cam_batch[target_idx:target_idx+1],
                    target_T_imu_to_world=T_imu_batch[target_idx:target_idx+1],
                    target_K=K_batch[target_idx:target_idx+1],
                )

                overlap_ratio = valid_mask.sum() / valid_mask.numel()
                if self.is_main and step % 100 == 0:
                    print(f"[Debug] Pair overlap: {overlap_ratio:.3f}, consistency={use_anchor_consistency}, source={anchor_source}")

                anchor_cond_feat = F.adaptive_avg_pool2d(anchor_cond_feat, (self.target_rows, self.target_cols))
                valid_mask_down = F.adaptive_avg_pool2d(valid_mask, (self.target_rows, self.target_cols))
                anchor_cond_feat = anchor_cond_feat * valid_mask_down

            pair_target_indices = torch.tensor([target_idx], dtype=torch.long, device=device)
            anchor_ce_loss = anchor_ce_loss + self._compute_pair_batch_ce(
                pair_target_indices=pair_target_indices,
                input_tokens=input_tokens,
                target_tokens=target_tokens,
                condition_tokens=condition_tokens,
                bev_feats=bev_feats,
                bev_vis_mask=bev_vis_mask,
                sup_valid=sup_valid,
                K_batch=K_batch,
                T_cam_batch=T_cam_batch,
                anchor_memory=anchor_cond_feat,
            )

            if not use_dropout and use_anchor_consistency and self.anchor_consistency_loss is not None and overlap_ratio >= 0.01:
                target_sampled_bev = getattr(predictor_module, "last_sampled_bev_feat", None)
                target_semantic_feat = getattr(predictor_module, "last_semantic_feat", None)
                anchor_sampled_bev = None
                anchor_semantic_feat = None
                anchor_coords_map = None
                with torch.no_grad():
                    anchor_condition_tokens = {
                        "coords": condition_tokens["coords"][anchor_idx:anchor_idx+1],
                        "pose": condition_tokens["pose"][anchor_idx:anchor_idx+1],
                        "K": K_batch[anchor_idx:anchor_idx+1],
                        "T_cam_to_world": T_cam_batch[anchor_idx:anchor_idx+1],
                    }
                    if "semantic" in condition_tokens:
                        anchor_condition_tokens["semantic"] = condition_tokens["semantic"][anchor_idx:anchor_idx+1]

                    self.predictor(
                        generated_tokens=input_tokens[anchor_idx:anchor_idx+1].view(1, -1),
                        condition_tokens=anchor_condition_tokens,
                        aligned_bev_feature_map=bev_feats[anchor_idx:anchor_idx+1],
                        bev_vis_mask=bev_vis_mask[anchor_idx:anchor_idx+1],
                    )

                anchor_sampled_bev = getattr(predictor_module, "last_sampled_bev_feat", None)
                anchor_semantic_feat = getattr(predictor_module, "last_semantic_feat", None)
                anchor_coords_map = getattr(predictor_module, "last_coords_map", None)

                anchor_consistency_feat = anchor_sampled_bev
                target_consistency_feat = target_sampled_bev
                if bool(getattr(self.cfg, "use_ipm_semantic", False)):
                    if anchor_semantic_feat is not None and target_semantic_feat is not None:
                        anchor_consistency_feat = anchor_semantic_feat
                        target_consistency_feat = target_semantic_feat

                if anchor_consistency_feat is not None and target_consistency_feat is not None and anchor_coords_map is not None:
                    target_valid_raw = warped_valid[target_idx:target_idx+1]
                    target_coords_raw = warped_coords[target_idx:target_idx+1]

                    target_valid_down = F.adaptive_avg_pool2d(target_valid_raw, (self.target_rows, self.target_cols))
                    target_valid_mask = target_valid_down > 0.2

                    target_coords_masked = target_coords_raw * target_valid_raw
                    target_coords_sum = F.adaptive_avg_pool2d(target_coords_masked, (self.target_rows, self.target_cols)) * (256*640) / (16*40)
                    target_coords = target_coords_sum / (target_valid_down * (256*640) / (16*40) + 1e-8)
                    target_coords = torch.where(target_valid_mask.expand(-1, 2, -1, -1), target_coords, torch.full_like(target_coords, -2.0))

                    consistency_dict = self.anchor_consistency_loss(
                        anchor_feat=anchor_consistency_feat,
                        target_feat=target_consistency_feat,
                        anchor_coords=anchor_coords_map,
                        target_coords=target_coords,
                        overlap_mask=target_valid_mask.float(),
                        use_feature_loss=True,
                    )
                    consistency_loss = consistency_loss + consistency_dict["total_loss"]
                    overlap_ratio_sum = overlap_ratio_sum + consistency_dict["overlap_ratio"]

            pair_count += 1

        return anchor_ce_loss, consistency_loss, overlap_ratio_sum, anchor_source, pair_count

    def _compute_anchor_loss_batched(
        self,
        pairs,
        rgb_batch,
        input_tokens,
        target_tokens,
        condition_tokens,
        bev_feats,
        bev_vis_mask,
        sup_valid,
        K_batch,
        T_cam_batch,
        T_imu_batch,
        step,
        anchor_view_source,
        generated_anchor_cache,
    ):
        device = rgb_batch.device
        anchor_ce_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)
        overlap_ratio_sum = torch.tensor(0.0, device=device)
        anchor_source = anchor_view_source

        pair_count = len(pairs)
        if pair_count == 0:
            return anchor_ce_loss, consistency_loss, overlap_ratio_sum, anchor_source, pair_count

        anchor_dropout_prob = float(getattr(self.cfg, "anchor_view_dropout_prob", 0.3))
        dropout_mask = torch.rand(pair_count, device=device) < anchor_dropout_prob

        target_indices_all = torch.tensor([pair.target_idx for pair in pairs], dtype=torch.long, device=device)
        anchor_indices_all = torch.tensor([pair.anchor_idx for pair in pairs], dtype=torch.long, device=device)

        dropout_target_indices = target_indices_all[dropout_mask]
        if dropout_target_indices.numel() > 0:
            anchor_ce_loss = anchor_ce_loss + self._compute_pair_batch_ce(
                pair_target_indices=dropout_target_indices,
                input_tokens=input_tokens,
                target_tokens=target_tokens,
                condition_tokens=condition_tokens,
                bev_feats=bev_feats,
                bev_vis_mask=bev_vis_mask,
                sup_valid=sup_valid,
                K_batch=K_batch,
                T_cam_batch=T_cam_batch,
                anchor_memory=None,
            )

        conditioned_mask = ~dropout_mask
        if conditioned_mask.any():
            conditioned_anchor_indices = anchor_indices_all[conditioned_mask]
            conditioned_target_indices = target_indices_all[conditioned_mask]

            anchor_images = []
            used_generated = False
            used_fallback = False
            for anchor_idx in conditioned_anchor_indices.tolist():
                if anchor_view_source == "generated" and anchor_idx in generated_anchor_cache:
                    anchor_images.append(generated_anchor_cache[anchor_idx])
                    used_generated = True
                else:
                    anchor_images.append(rgb_batch[anchor_idx:anchor_idx+1])
                    used_fallback = used_fallback or (anchor_view_source == "generated")

            anchor_image_batch = torch.cat(anchor_images, dim=0)
            anchor_cond_feat, valid_mask = self.anchor_conditioner(
                anchor_image=anchor_image_batch,
                anchor_T_cam_to_world=T_cam_batch.index_select(0, conditioned_anchor_indices),
                anchor_T_imu_to_world=T_imu_batch.index_select(0, conditioned_anchor_indices),
                anchor_K=K_batch.index_select(0, conditioned_anchor_indices),
                target_T_cam_to_world=T_cam_batch.index_select(0, conditioned_target_indices),
                target_T_imu_to_world=T_imu_batch.index_select(0, conditioned_target_indices),
                target_K=K_batch.index_select(0, conditioned_target_indices),
            )

            overlap_ratio = valid_mask.mean(dim=(1, 2, 3))
            overlap_ratio_sum = overlap_ratio_sum + overlap_ratio.sum()

            if self.is_main and step % 100 == 0 and overlap_ratio.numel() > 0:
                debug_source = "generated" if used_generated and not used_fallback else ("gt_fallback" if used_fallback else "gt")
                print(f"[Debug] Pair overlap mean: {overlap_ratio.mean().item():.3f}, count={int(overlap_ratio.numel())}, source={debug_source}")

            anchor_cond_feat = F.adaptive_avg_pool2d(anchor_cond_feat, (self.target_rows, self.target_cols))
            valid_mask_down = F.adaptive_avg_pool2d(valid_mask, (self.target_rows, self.target_cols))
            anchor_cond_feat = anchor_cond_feat * valid_mask_down

            anchor_ce_loss = anchor_ce_loss + self._compute_pair_batch_ce(
                pair_target_indices=conditioned_target_indices,
                input_tokens=input_tokens,
                target_tokens=target_tokens,
                condition_tokens=condition_tokens,
                bev_feats=bev_feats,
                bev_vis_mask=bev_vis_mask,
                sup_valid=sup_valid,
                K_batch=K_batch,
                T_cam_batch=T_cam_batch,
                anchor_memory=anchor_cond_feat,
            )

            pair_count = int(dropout_target_indices.numel() + conditioned_target_indices.numel())

            if used_fallback:
                anchor_source = "gt_fallback"
            elif used_generated:
                anchor_source = "generated"
            else:
                anchor_source = "gt"
        else:
            pair_count = int(dropout_target_indices.numel())

        return anchor_ce_loss, consistency_loss, overlap_ratio_sum, anchor_source, pair_count

    def _compute_anchor_loss(
        self, rgb_batch, input_tokens, target_tokens, condition_tokens,
        bev_feats, bev_vis_mask, sup_valid,
        frame_ids, drive_names, view_indices, K_batch, T_cam_batch, T_imu_batch,
        warped_coords, warped_valid, step,
    ) -> dict:
        B = rgb_batch.size(0)
        device = rgb_batch.device

        # 1. Prepare anchor-target pairs and optional generated anchors first.
        # Do this before any gradient-tracked forward on self.predictor so the
        # frozen anchor generator cannot perturb runtime state needed by autograd.
        pairs = group_and_pair_views(frame_ids, view_indices)
        use_anchor_consistency = bool(getattr(self.cfg, "anchor_view_use_consistency", False))
        predictor_module = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
        anchor_view_source = str(getattr(self.cfg, "anchor_view_source", "generated")).lower()
        generated_anchor_cache = {}
        if anchor_view_source == "generated" and len(pairs) > 0:
            generated_anchor_cache = self._generate_anchor_images_for_indices(
                [pair.anchor_idx for pair in pairs],
                condition_tokens,
                bev_feats,
                bev_vis_mask,
                K_batch,
                T_cam_batch,
                frame_ids=frame_ids,
                view_indices=view_indices,
                drive_names=drive_names,
            )

        # 2. Baseline CE
        logits_baseline = self.predictor(
            generated_tokens=input_tokens.view(input_tokens.size(0), -1),
            condition_tokens=condition_tokens,
            aligned_bev_feature_map=bev_feats,
            bev_vis_mask=bev_vis_mask,
        )

        ce_per_token = F.cross_entropy(
            logits_baseline.view(-1, self.model_vocab_size),
            target_tokens.view(-1), reduction="none",
            label_smoothing=self.cfg.label_smoothing,
        ).view_as(target_tokens)

        denom = sup_valid.sum().clamp(min=1e-6)
        ce_loss_baseline = (ce_per_token * sup_valid).sum() / denom

        anchor_ce_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)
        overlap_ratio_sum = torch.tensor(0.0, device=device)
        anchor_source = anchor_view_source

        if len(pairs) > 0:
            if use_anchor_consistency:
                anchor_ce_loss, consistency_loss, overlap_ratio_sum, anchor_source, pair_count = self._compute_anchor_loss_loop(
                    pairs=pairs,
                    rgb_batch=rgb_batch,
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,
                    condition_tokens=condition_tokens,
                    bev_feats=bev_feats,
                    bev_vis_mask=bev_vis_mask,
                    sup_valid=sup_valid,
                    K_batch=K_batch,
                    T_cam_batch=T_cam_batch,
                    T_imu_batch=T_imu_batch,
                    warped_coords=warped_coords,
                    warped_valid=warped_valid,
                    step=step,
                    use_anchor_consistency=use_anchor_consistency,
                    predictor_module=predictor_module,
                    anchor_view_source=anchor_view_source,
                    generated_anchor_cache=generated_anchor_cache,
                )
            else:
                anchor_ce_loss, consistency_loss, overlap_ratio_sum, anchor_source, pair_count = self._compute_anchor_loss_batched(
                    pairs=pairs,
                    rgb_batch=rgb_batch,
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,
                    condition_tokens=condition_tokens,
                    bev_feats=bev_feats,
                    bev_vis_mask=bev_vis_mask,
                    sup_valid=sup_valid,
                    K_batch=K_batch,
                    T_cam_batch=T_cam_batch,
                    T_imu_batch=T_imu_batch,
                    step=step,
                    anchor_view_source=anchor_view_source,
                    generated_anchor_cache=generated_anchor_cache,
                )

            if pair_count > 0:
                anchor_ce_loss = anchor_ce_loss / pair_count
                consistency_loss = consistency_loss / pair_count
                overlap_ratio_sum = overlap_ratio_sum / pair_count

        # Total loss
        baseline_ce_weight = getattr(self.cfg, "anchor_view_baseline_ce_weight", 0.3)
        anchor_ce_weight = getattr(self.cfg, "anchor_view_ce_weight", 1.0)
        consistency_weight = getattr(self.cfg, "anchor_view_consistency_weight", 0.2)

        total_loss = (
            baseline_ce_weight * ce_loss_baseline +
            anchor_ce_weight * anchor_ce_loss +
            consistency_weight * consistency_loss
        )

        return {
            "ce_loss": ce_loss_baseline.item(),
            "anchor_ce_loss": anchor_ce_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "overlap_ratio": float(overlap_ratio_sum.item()),
            "anchor_source": anchor_source,
            "total_loss": total_loss,  # 保持tensor用于backward
        }

    def _save_ckpt(self, step: int):
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"ckpt_step_{step:07d}.pt"

        model_state = self.predictor.module.state_dict() if hasattr(self.predictor, "module") else self.predictor.state_dict()
        ckpt = {
            "step": step,
            "model": model_state,
            "optimizer": self.optim.state_dict(),
            "amp_scaler": self.amp_scaler.state_dict(),  # 保存AMP GradScaler状态
            "model_vocab_size": self.model_vocab_size,
        }

        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()

        if self.anchor_conditioner is not None:
            anchor_state = self.anchor_conditioner.module.state_dict() if hasattr(self.anchor_conditioner, "module") else self.anchor_conditioner.state_dict()
            ckpt["anchor_conditioner"] = anchor_state

        if self.anchor_consistency_loss is not None:
            loss_state = self.anchor_consistency_loss.module.state_dict() if hasattr(self.anchor_consistency_loss, "module") else self.anchor_consistency_loss.state_dict()
            ckpt["anchor_consistency_loss"] = loss_state

        torch.save(ckpt, str(ckpt_path))
        if self.is_main:
            print(f"[Checkpoint] Saved: {ckpt_path}")


if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp

    # Must be called before any CUDA init so DataLoader workers can use GPU.
    mp.set_start_method("spawn", force=True)

    import torch

    from world3d.config import load_cfg


    def parse_args():
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=str, default="configs/ar_anchor_view.yaml")

        ap.add_argument("--device", type=str, default=None)
        ap.add_argument("--out_dir", type=str, default=None)
        ap.add_argument("--resume_ckpt", type=str, default=None)
        ap.add_argument("--data_root", type=str, default=None)

        ap.add_argument("--steps", type=int, default=None)
        ap.add_argument("--batch_size", type=int, default=None)
        ap.add_argument("--accum_steps", type=int, default=None)
        ap.add_argument("--lr", type=float, default=None)

        return ap.parse_args()


    def main():
        args = parse_args()
        overrides = {
            "device": args.device,
            "out_dir": args.out_dir,
            "resume_ckpt": args.resume_ckpt,
            "data_root": args.data_root,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "lr": args.lr,
        }

        cfg = load_cfg(args.config, overrides=overrides)

        if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
            cfg.device = "cpu"

        trainer = ArAnchorViewStage2Trainer(cfg, repo_root=REPO_ROOT)
        trainer.train()


    main()

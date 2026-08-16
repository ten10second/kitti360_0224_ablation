"""
Diffusion Model Trainer.

This module implements the training loop for the diffusion-based view synthesis model,
including evaluation metrics integration and visualization.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Add parent path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.stage2.diffusion_model import DiffusionPoseModel
from world3d.data.diffusion_pipeline import (
    build_diffusion_data_pipeline,
    build_train_val_loaders,
    collate_diffusion_samples,
)
from world3d.train.vis_utils import visualize_diffusion_results
from utils.distributed import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
)


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Build optimizer for diffusion model.

    Args:
        model: diffusion model
        cfg: configuration object

    Returns:
        optimizer
    """
    # Only optimize trainable parameters
    trainable_params = model.get_trainable_parameters()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """
    Build learning rate scheduler.

    Args:
        optimizer: optimizer
        cfg: configuration object

    Returns:
        scheduler
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

    if not getattr(cfg, "use_warmup_cosine", True):
        # Simple constant LR
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
    else:
        # Warmup + cosine annealing
        warmup_updates = getattr(cfg, "warmup_updates", 4000)
        total_steps = getattr(cfg, "steps", 80000)
        min_lr_factor = getattr(cfg, "min_lr", 1.0e-6) / cfg.lr

        def lr_lambda(step):
            if step < warmup_updates:
                return float(step) / float(max(1, warmup_updates))
            else:
                progress = float(step - warmup_updates) / float(
                    max(1, total_steps - warmup_updates)
                )
                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
                return (
                    min_lr_factor + (1.0 - min_lr_factor) * cosine_decay
                )

        scheduler = LambdaLR(optimizer, lr_lambda)

    return scheduler


class DiffusionTrainer:
    """
    Trainer for diffusion-based view synthesis model.

    Manages training loop, validation, logging, and checkpointing.
    """

    def __init__(
        self,
        cfg,
        repo_root: Union[str, Path],
        device: Optional[torch.device] = None,
        resume_ckpt: Optional[str] = None,
    ):
        self.cfg = cfg
        self.repo_root = Path(repo_root)
        self.device = (
            device
            if device is not None
            else torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        )
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()

        # Initialize distributed training
        if is_dist_avail_and_initialized():
            torch.cuda.set_device(self.rank)
            torch.distributed.barrier()

        # Set up output directories
        self.out_dir = Path(self.cfg.out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log_dir = self.out_dir / "logs"
        self.vis_dir = self.out_dir / "visualizations"

        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Set up TensorBoard logging
        self.writer = None
        if self.is_main:
            self.writer = SummaryWriter(self.log_dir)

        # Build model
        self.model = self._build_model()

        # Build data loaders
        self.train_loader, self.val_loader = self._build_data_loaders()

        # Build optimizer and scheduler
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.optimizer.zero_grad(set_to_none=True)

        # Initialize training state
        self.step = 0
        self.epoch = 0
        self.best_fid = float("inf")

        # Resume from checkpoint if requested
        if resume_ckpt is not None:
            self._load_checkpoint(resume_ckpt)

        # Performance optimizations
        self._setup_performance()

    def _build_model(self) -> DiffusionPoseModel:
        """
        Build and wrap the diffusion model.

        Returns:
            diffusion model (possibly wrapped in DDP)
        """
        model = DiffusionPoseModel(
            sd_model_id=self.cfg.model.sd_model_id,
            bev_encoder_ckpt=self.repo_root / self.cfg.model.bev_encoder_ckpt,
            freeze_sd=self.cfg.model.freeze_sd,
            freeze_bev_encoder=self.cfg.model.freeze_bev_encoder,
            device=self.device,
        )

        # Wrap with DDP if distributed training
        if is_dist_avail_and_initialized():
            model = DDP(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=self.cfg.dist['find_unused_parameters'],
                static_graph=self.cfg.dist['static_graph'],
            )

        return model

    def _build_data_loaders(
        self,
    ) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """
        Build training and validation data loaders.

        Returns:
            train_loader, val_loader
        """
        # 分别构建训练集和验证集，使用各自的 frames 文件
        train_ds, _ = build_diffusion_data_pipeline(
            self.cfg,
            mode="train",
        )

        _, val_ds = build_diffusion_data_pipeline(
            self.cfg,
            mode="val",
        )

        train_loader, val_loader = build_train_val_loaders(
            self.cfg,
            train_ds,
            val_ds,
        )

        if self.is_main:
            print(f"[Trainer] Training samples: {len(train_ds)}")
            if val_ds is not None:
                print(f"[Trainer] Validation samples: {len(val_ds)}")

        return train_loader, val_loader

    def _setup_performance(self):
        """Set up performance optimizations."""
        perf_cfg = getattr(self.cfg, "perf", {})

        # Enable TF32
        if perf_cfg.get("enable_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Enable SDPA
        if perf_cfg.get("enable_sdpa", True):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        # Channels last
        if perf_cfg.get("channels_last", False):
            self.model = self.model.to(memory_format=torch.channels_last)

    def _save_checkpoint(self, step: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            step: current training step
            is_best: whether this is the best checkpoint so far
        """
        if not self.is_main:
            return

        ckpt_dict = {
            "step": step,
            "epoch": self.epoch,
            "best_fid": self.best_fid,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "cfg": self.cfg,
        }

        # Save regular checkpoint
        ckpt_path = self.ckpt_dir / f"ckpt_step_{step:07d}.pt"
        torch.save(ckpt_dict, ckpt_path)

        # Save best checkpoint
        if is_best:
            best_path = self.ckpt_dir / "ckpt_best.pt"
            torch.save(ckpt_dict, best_path)
            print(f"[Trainer] Saved best checkpoint to {best_path}")

        print(f"[Trainer] Saved checkpoint to {ckpt_path}")

    def _load_checkpoint(self, ckpt_path: str):
        """
        Load model from checkpoint.

        Args:
            ckpt_path: path to checkpoint
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"[Trainer] Warning: Checkpoint {ckpt_path} not found, starting fresh")
            return

        print(f"[Trainer] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.step = ckpt["step"]
        self.epoch = ckpt.get("epoch", 0)
        self.best_fid = ckpt.get("best_fid", float("inf"))

        print(f"[Trainer] Resumed at step {self.step}, epoch {self.epoch}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.

        Args:
            batch: batch data dictionary

        Returns:
            dict with loss and other metrics
        """
        self.model.train()

        # Move data to device
        sat_images = batch["sat"].to(self.device, non_blocking=True)
        pose_vec = batch["pose"].to(self.device, non_blocking=True)
        target_images = batch["rgb"].to(self.device, non_blocking=True)

        # Forward pass
        outputs = self.model(
            sat_images=sat_images,
            pose_vec=pose_vec,
            target_images=target_images,
        )

        loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / getattr(self.cfg, "accum_steps", 1)

        # Backward pass
        loss.backward()

        # Gradient clipping and optimizer step only when accumulation steps are met
        if (self.step + 1) % getattr(self.cfg, "accum_steps", 1) == 0:
            if getattr(self.cfg, "grad_clip", 0.0) > 0:
                trainable_params = self.model.get_trainable_parameters()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    self.cfg.grad_clip,
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": outputs["loss"].item(),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation and compute metrics.

        Returns:
            dict with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Move data to device
            sat_images = batch["sat"].to(self.device, non_blocking=True)
            pose_vec = batch["pose"].to(self.device, non_blocking=True)
            target_images = batch["rgb"].to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(
                sat_images=sat_images,
                pose_vec=pose_vec,
                target_images=target_images,
            )

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "val_loss": avg_loss,
        }

    @torch.no_grad()
    def generate_samples(
        self,
        batch: Dict[str, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
    ) -> torch.Tensor:
        """
        Generate samples from a batch.

        Args:
            batch: batch data dictionary
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale

        Returns:
            generated_images: (B, 3, 512, 512) generated images
        """
        self.model.eval()

        sat_images = batch["sat"].to(self.device, non_blocking=True)
        pose_vec = batch["pose"].to(self.device, non_blocking=True)

        generated = self.model.generate(
            sat_images=sat_images,
            pose_vec=pose_vec,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        return generated

    def train(self):
        """Main training loop."""
        print(f"[Trainer] Starting training for {self.cfg.steps} steps")
        print(f"[Trainer] Device: {self.device}")

        start_time = time.time()
        last_log_time = start_time

        # Training loop
        while self.step < self.cfg.steps:
            self.epoch += 1

            # Set epoch for samplers
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)

            for batch in self.train_loader:
                if self.step >= self.cfg.steps:
                    break

                self.step += 1

                # Training step
                metrics = self.train_step(batch)

                # Logging
                if self.step % self.cfg.print_every == 0 and self.is_main:
                    current_lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - last_log_time
                    steps_per_sec = self.cfg.print_every / elapsed

                    print(
                        f"[Trainer] Step {self.step:6d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )

                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", metrics["loss"], self.step)
                        self.writer.add_scalar("train/lr", current_lr, self.step)

                    last_log_time = time.time()

                # Visualization
                if self.step % self.cfg.vis_every == 0 and self.is_main:
                    print(f"[Trainer] Generating visualization at step {self.step}")
                    try:
                        generated = self.generate_samples(batch)
                        vis_path = self.vis_dir / f"vis_step_{self.step:07d}.png"
                        visualize_diffusion_results(
                            generated,
                            batch["rgb"],
                            batch["sat"],
                            save_path=vis_path,
                        )
                    except Exception as e:
                        print(f"[Trainer] Visualization failed: {e}")

                # Validation
                if self.step % getattr(self.cfg, "eval_every", self.cfg.save_every) == 0:
                    val_metrics = self.validate()
                    if self.is_main and val_metrics:
                        print(
                            f"[Trainer] Validation at step {self.step}: "
                            f"val_loss={val_metrics.get('val_loss', 0):.4f}"
                        )
                        if self.writer is not None:
                            for key, value in val_metrics.items():
                                self.writer.add_scalar(f"val/{key}", value, self.step)

                # Checkpointing
                if self.step % self.cfg.save_every == 0:
                    is_best = False
                    self._save_checkpoint(self.step, is_best=is_best)

        # Final checkpoint
        if self.is_main:
            self._save_checkpoint(self.step, is_best=False)
            if self.writer is not None:
                self.writer.close()

        total_time = time.time() - start_time
        print(f"[Trainer] Training complete in {total_time/3600:.1f} hours")

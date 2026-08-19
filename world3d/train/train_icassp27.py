#!/usr/bin/env python3
"""Single-stage teacher-forcing trainer for the ICASSP27 framework.

Kept from legacy (checklist §6): VQ tokenizer (frozen), teacher-forcing CE
loop structure, e_pose 13-dim pipeline, checkpoint/visual conventions.
Removed by design: IPM warp, BEV encoder, anchor routing, RayRoPE,
consistency/stage-2 losses — nothing here imports them.

Usage:
  python -m world3d.train.train_icassp27 --config configs/icassp27_pilot.yaml
  # ablations:
  ... --use_sat false            # B1 (source only)
  ... --use_src false            # B0 (satellite only)
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples
from world3d.models.icassp27_predictor import ICASSP27Predictor


def git_commit_sha() -> str:
    """Record the source revision in every checkpoint without making git required."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def decode_and_save(vq, model, batch, device, path: Path, max_show: int = 3):
    """Teacher-forced argmax reconstruction vs VQ-GT decode, saved as a grid."""
    import numpy as np
    rgb = batch["tgt_rgb"][:max_show].to(device) * 2 - 1
    with torch.no_grad():
        tokens = vq.encode(rgb)
        if tokens.dim() == 4:
            tokens = tokens.squeeze(1)
        B = tokens.shape[0]
        tokens = tokens.view(B, 16, 40).flatten(1)
        inp, _ = model.make_teacher_forcing(tokens)
        logits = model(
            inp,
            batch["pose_vec"][:max_show].to(device),
            sat=batch["sat"][:max_show].to(device),
            window_origin_xyz=batch["window_origin_xyz"][:max_show].to(device),
            src_rgbs=batch["src_rgbs"][:max_show].to(device),
            rel_poses=batch["rel_poses"][:max_show].to(device),
            src_mask=batch["src_mask"][:max_show].to(device),
            tgt_K=batch["tgt_K"][:max_show].to(device),
            tgt_T_cam=batch["tgt_T_cam"][:max_show].to(device),
        )
        pred = logits.argmax(-1)
        rec = vq.decode(pred).clamp(-1, 1)
        gt = vq.decode(tokens).clamp(-1, 1)
    imgs = []
    for i in range(B):
        row = torch.cat([rgb[i], gt[i], rec[i]], dim=-1)  # input | VQ-GT | teacher-forced pred
        imgs.append((row + 1) / 2)
    grid = torch.cat(imgs, dim=1).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(path), (grid[:, :, ::-1] * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/icassp27_pilot.yaml")
    ap.add_argument("--use_sat", type=str, default=None)
    ap.add_argument("--use_src", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.use_sat is not None: cfg.model.use_sat = args.use_sat.lower() in ("1", "true", "yes")
    if args.use_src is not None: cfg.model.use_src = args.use_src.lower() in ("1", "true", "yes")
    if args.steps is not None: cfg.train.steps = args.steps
    if args.out_dir is not None: cfg.out_dir = args.out_dir
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.get("device", "cuda"))
    out_dir = Path(cfg.out_dir)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.get("seed", 42))

    # ---- data ----
    train_ds = Kitti360TupleDataset(
        cfg.data.train_manifest,
        mode="train",
        img_size=tuple(cfg.data.img_size),
        window_m=cfg.data.window_m,
        anchor_spacing_m=cfg.data.anchor_spacing_m,
        anchor_stride_m=cfg.data.anchor_stride_m,
        k_min=cfg.data.k_min,
        k_max=cfg.data.k_max,
        dist_min_m=cfg.data.dist_min_m,
        dist_max_m=cfg.data.dist_max_m,
        dyaw_max_deg=cfg.data.dyaw_max_deg,
        seed=cfg.get("seed", 42),
    )
    print(f"[data] train tuples base (K=1, per bin): {len(train_ds)}  "
          f"(dyaw-rejected {train_ds.n_reject_yaw}, pose-missing {train_ds.n_reject_pose})")
    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.loader.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_tuples,
        persistent_workers=cfg.loader.get("num_workers", 8) > 0,
    )

    # ---- model ----
    vq = PretrainedTokenizer(cfg.vq_ckpt).to(device).eval()
    model = ICASSP27Predictor(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
        pose_dim=cfg.model.pose_dim,
        dino_arch=cfg.model.dino_arch,
        sat_encoder=cfg.model.sat_encoder,
        geo=cfg.model.get("geo", "raymap"),
        use_sat=cfg.model.use_sat,
        use_src=cfg.model.use_src,
        fourier_freqs=cfg.model.get("fourier_freqs", 10),
        sat_pe_mode=cfg.model.get("sat_pe_mode", "legacy_fourier"),
        sat_coord_scale_m=cfg.model.get("sat_coord_scale_m", None),
        sat_px=cfg.model.sat_px,
        sat_m_per_px=cfg.model.sat_m_per_px,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {n_train/1e6:.1f}M / total {n_total/1e6:.1f}M  "
          f"sat={cfg.model.use_sat} src={cfg.model.use_src} "
          f"sat_pe={cfg.model.get('sat_pe_mode', 'legacy_fourier')} geo={cfg.model.get('geo', 'raymap')}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    total_steps = cfg.train.steps
    warmup = cfg.train.get("warmup_steps", 1000)
    def lr_at(step):
        if step < warmup:
            return cfg.train.lr * step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return cfg.train.get("min_lr", 1e-5) + 0.5 * (cfg.train.lr - cfg.train.get("min_lr", 1e-5)) * (1 + __import__("math").cos(__import__("math").pi * min(1.0, t)))

    step = 0
    t0 = time.time()
    loss_acc, n_acc = 0.0, 0
    model.train()
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            rgb = batch["tgt_rgb"].to(device, non_blocking=True) * 2 - 1
            with torch.no_grad():
                tokens = vq.encode(rgb)
                if tokens.dim() == 4:
                    tokens = tokens.squeeze(1)
                tokens = tokens.view(tokens.shape[0], 16, 40).flatten(1)  # row-major raster
            inp, label = model.make_teacher_forcing(tokens)

            logits = model(
                inp,
                batch["pose_vec"].to(device, non_blocking=True),
                sat=batch["sat"].to(device, non_blocking=True),
                window_origin_xyz=batch["window_origin_xyz"].to(device, non_blocking=True),
                src_rgbs=batch["src_rgbs"].to(device, non_blocking=True),
                rel_poses=batch["rel_poses"].to(device, non_blocking=True),
                src_mask=batch["src_mask"].to(device, non_blocking=True),
                tgt_K=batch["tgt_K"].to(device, non_blocking=True),
                tgt_T_cam=batch["tgt_T_cam"].to(device, non_blocking=True),
            )
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), label.reshape(-1),
                                   label_smoothing=cfg.train.get("label_smoothing", 0.0))
            loss.backward()
            if cfg.train.get("grad_clip", 1.0) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

            loss_acc += loss.item(); n_acc += 1
            if step % cfg.train.print_every == 0:
                el = time.time() - t0
                print(f"step {step}/{total_steps}  loss {loss_acc/max(1,n_acc):.4f}  ppl {torch.exp(torch.tensor(loss_acc/max(1,n_acc))):.2f}  "
                      f"lr {lr_at(step):.2e}  {el/step:.2f}s/it", flush=True)
                loss_acc, n_acc = 0.0, 0
            if step % cfg.train.vis_every == 0:
                model.eval()
                decode_and_save(vq, model, batch, device, out_dir / "vis" / f"tf_{step:07d}.jpg")
                model.train()
            if step % cfg.train.save_every == 0 or step == total_steps:
                torch.save({
                    "model": model.state_dict(),
                    "config": OmegaConf.to_container(cfg),
                    "step": step,
                    "git_commit": git_commit_sha(),
                }, out_dir / "ckpt.pt")
        # loader exhausted -> new epoch, fresh stochastic sampling
        train_ds.epoch += 1

    print("done")


if __name__ == "__main__":
    main()

"""Smoke test: tuple dataset + ICASSP27Predictor forward/backward on real data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/shizhm/Lenovo/kitti360_0224_ablation")

from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples
from world3d.models.icassp27_predictor import ICASSP27Predictor
from world3d.train.pose_ar import build_pose_vec

MANIFEST = "/media/shizhm/Lenovo/kitti360_0224_ablation/dataset_splits/kitti360_geofence_buffer30/train_manifest.jsonl"


def main():
    print("== dataset ==")
    ds = Kitti360TupleDataset(MANIFEST, mode="train", seed=0)
    print(f"tuples: {len(ds)}  dyaw-rejected: {ds.n_reject_yaw}  pose-missing: {ds.n_reject_pose}")
    # determinism
    s0 = ds[0]; s0b = ds[0]
    assert torch.allclose(s0["tgt_rgb"], s0b["tgt_rgb"]), "not deterministic"
    # sample a few and sanity-check geometry
    import random
    rng = random.Random(1)
    for i in rng.sample(range(len(ds)), 5):
        s = ds[i]
        m = s["meta"]
        T_imu = ds._imu_poses[m["drive"]]
        tgt = T_imu[m["target_fid"]]; last_src = T_imu[m["source_fids"][-1]]
        real_d = float(np.hypot(*(tgt[:2, 3] - last_src[:2, 3])))
        inter_src_d = float(np.hypot(*(T_imu[m["source_fids"][0]][:2, 3] - T_imu[m["source_fids"][-1]][:2, 3]))) if len(m["source_fids"]) > 1 else 0.0
        print(f"  [{m['drive'][-10:]}] K={s['n_src']} srcs={m['source_fids']} tgt={m['target_fid']} "
              f"req_d={m['dist_m']:.1f} real_d={real_d:.1f} span={inter_src_d:.1f} dyaw={m['dyaw_deg']:.1f} "
              f"bin={m['bin']} sat={tuple(s['sat'].shape)} rgb={tuple(s['tgt_rgb'].shape)} rel={tuple(s['rel_poses'].shape)}")
        assert real_d >= 1.0, "target too close"
        assert s["sat"].shape == (3, 512, 512)
    batch = collate_tuples([ds[i] for i in range(4)])
    print(f"batch: tgt_rgb {tuple(batch['tgt_rgb'].shape)} src_rgbs {tuple(batch['src_rgbs'].shape)} "
          f"sat {tuple(batch['sat'].shape)} n_src {batch['n_src'].tolist()} mask {batch['src_mask'].sum(1).tolist()}")
    print(f"window_origin_xyz: {batch['window_origin_xyz'][0].tolist()}")

    print("\n== model ==")
    dev = torch.device("cuda")
    model = ICASSP27Predictor(
        vocab_size=1024, d_model=256, nhead=8, num_layers=2, dim_feedforward=512,
        max_seq_len=1080, dino_arch="vitb14", geo="raymap",
    ).to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_tr/1e6:.2f}M")
    B, L = 4, 640
    inp = torch.randint(0, 1024, (B, L), device=dev)
    pose = batch["pose_vec"].to(dev)
    sat = batch["sat"].to(dev)
    origin = batch["window_origin_xyz"].to(dev)
    src = batch["src_rgbs"].to(dev)
    rel = batch["rel_poses"].to(dev)
    mask = batch["src_mask"].to(dev)
    tK = batch["tgt_K"].to(dev)
    tT = batch["tgt_T_cam"].to(dev)
    logits = model(inp, pose, sat=sat, window_origin_xyz=origin, src_rgbs=src, rel_poses=rel,
                   src_mask=mask, tgt_K=tK, tgt_T_cam=tT)
    print(f"logits: {tuple(logits.shape)}")
    assert logits.shape == (B, L, 1024)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 1024), inp.reshape(-1))
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"loss {loss.item():.4f}  grad-mass {gnorm:.3f}  (grads flowing)")

    # --- raymap semantics: translating the tuple within the window MUST change conditioning ---
    tT2 = tT.clone(); tT2[:, 0, 3] += 30.0  # same window, target 30 m further east
    o1, d1 = model._target_rays(tK, tT, origin)
    o2, _ = model._target_rays(tK, tT2, origin)
    ray_shift = (o2 - o1).norm(dim=-1).mean().item()
    print(f"ray origin shift when target moves +30m in-window: {ray_shift:.1f} m (expect ~30)")
    assert abs(ray_shift - 30.0) < 0.5
    pe1 = model._token_ray_pe(tK, tT, origin, L)
    pe2 = model._token_ray_pe(tK, tT2, origin, L)
    pe_diff = (pe1 - pe2).abs().sum(-1)[:, 1:].mean().item()  # skip BOS position
    print(f"ray PE per-token |diff| after +30m shift: {pe_diff:.3f} (expect >> 0; B0 no longer translation-invariant)")
    assert pe_diff > 0.1

    # memory token accounting
    mem, pad = model.build_memory(pose, sat, origin, src, rel, mask)
    n_sat = model.sat_grid[0] * model.sat_grid[1]
    exp = 1 + n_sat + int(mask.shape[1]) * model.src_tokens_per_view
    print(f"memory: {tuple(mem.shape)} (expected {exp})  pad tokens {int(pad.sum())}")

    # B0/B1 ablation switches (B0 uses raymap -> position-aware even without sources)
    m0 = ICASSP27Predictor(d_model=128, num_layers=1, use_src=False, geo="raymap").to(dev)
    l0 = m0(inp, pose, sat=sat, window_origin_xyz=origin, tgt_K=tK, tgt_T_cam=tT)
    m1 = ICASSP27Predictor(d_model=128, num_layers=1, use_sat=False, geo="raymap").to(dev)
    l1 = m1(inp, pose, src_rgbs=src, rel_poses=rel, src_mask=mask, tgt_K=tK, tgt_T_cam=tT)
    print(f"B0 (sat-only) logits {tuple(l0.shape)}; B1 (src-only) logits {tuple(l1.shape)}")

    # geo=pose_add ablation row (no ray inputs needed)
    mp = ICASSP27Predictor(d_model=128, num_layers=1, geo="pose_add").to(dev)
    lp = mp(inp, pose, sat=sat, window_origin_xyz=origin, src_rgbs=src, rel_poses=rel, src_mask=mask)
    print(f"geo=pose_add logits {tuple(lp.shape)}")

    # generate smoke (few tokens)
    gen = model.generate(pose, max_len=8, sat=sat, window_origin_xyz=origin, src_rgbs=src, rel_poses=rel,
                         src_mask=mask, tgt_K=tK, tgt_T_cam=tT)
    print(f"generate: {tuple(gen.shape)} values in [0,1024]: {bool((gen >= 0).all() and (gen < 1024).all())}")
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()

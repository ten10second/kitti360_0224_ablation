#!/usr/bin/env python3
"""Paired target-centric satellite-grounding generation diagnostic.

The intervention is applied after satellite projection plus metric positional
encoding. Source memory, target-ray positional encoding, tuple selection, and
the AR sampling policy are held fixed across modes.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from metrics import LPIPS, PSNR
from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples
from world3d.models.icassp27_predictor import ICASSP27Predictor


MODES = ("real", "zero", "shuffle", "pe_permute", "rot90", "b1")


def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    model = ICASSP27Predictor(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], max_seq_len=cfg["max_seq_len"], pose_dim=cfg["pose_dim"],
        dino_arch=cfg["dino_arch"], sat_encoder=cfg["sat_encoder"],
        geo=cfg.get("geo", "raymap"), use_sat=cfg.get("use_sat", True),
        use_src=cfg.get("use_src", True), fourier_freqs=cfg.get("fourier_freqs", 10),
        sat_pe_mode=cfg.get("sat_pe_mode", "legacy_fourier"),
        sat_coord_scale_m=cfg.get("sat_coord_scale_m", None),
        sat_px=cfg.get("sat_px", 512), sat_m_per_px=cfg.get("sat_m_per_px", 0.196),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    return model, ckpt.get("step", -1)


def select_eval_indices(ds: Kitti360TupleDataset, num_tuples: int):
    """Balance (distance bin, K) while spreading recipients across sat windows."""
    groups = defaultdict(dict)
    window_keys = set()
    for i in range(len(ds)):
        k_slot = i % len(ds.eval_k)
        spec = ds.tuples[i // len(ds.eval_k)]
        bin_id = next(j for j, (lo, hi) in enumerate(ds.bins) if lo <= spec.dist_m < hi)
        window_key = (spec.drive, spec.window_id, spec.window_center_fid)
        groups[(bin_id, k_slot)].setdefault(window_key, i)
        window_keys.add(window_key)

    group_keys = sorted(groups)
    window_keys = sorted(window_keys)
    target = min(num_tuples, len(window_keys))
    idxs = []
    used_windows = set()
    for slot in range(target):
        group = group_keys[slot % len(group_keys)]
        start = slot * len(window_keys) // target
        for offset in range(len(window_keys)):
            window_key = window_keys[(start + offset) % len(window_keys)]
            if window_key not in used_windows and window_key in groups[group]:
                idxs.append(groups[group][window_key])
                used_windows.add(window_key)
                break
        else:
            raise RuntimeError(f"could not find a distinct satellite window for group {group}")
    return idxs


def deranged_batch_perm(window_keys, batch_index: int, seed: int, device: torch.device):
    """Deterministically pair every recipient with a different satellite window."""
    batch_size = len(window_keys)
    base = torch.arange(batch_size, device=device)
    if batch_size < 2:
        raise ValueError("shuffle requires at least two samples per batch")
    for offset in range(batch_size - 1):
        shift = 1 + ((seed + batch_index + offset) % (batch_size - 1))
        perm = torch.roll(base, shifts=shift)
        if all(window_keys[i] != window_keys[int(perm[i])] for i in range(batch_size)):
            return perm
    raise ValueError("shuffle batch does not contain a valid cross-window derangement")


def mean_metrics(records):
    return {
        "n": len(records),
        "psnr": float(np.mean([r["psnr"] for r in records])),
        "lpips": float(np.mean([r["lpips"] for r in records])),
    }


def paired_bootstrap(records, seed: int, samples: int = 10_000):
    """Counterfactual effect sizes with deterministic paired-bootstrap CIs."""
    by_tuple = defaultdict(dict)
    for record in records:
        by_tuple[record["tuple_index"]][record["sat_memory_mode"]] = record
    out = {}
    rng = np.random.default_rng(seed)
    for mode in MODES:
        if mode == "real":
            continue
        pairs = [row for row in by_tuple.values() if "real" in row and mode in row]
        psnr = np.asarray([row["real"]["psnr"] - row[mode]["psnr"] for row in pairs])
        lpips = np.asarray([row[mode]["lpips"] - row["real"]["lpips"] for row in pairs])
        if not len(pairs):
            continue
        idx = rng.integers(0, len(pairs), size=(samples, len(pairs)))
        out[mode] = {
            "n": len(pairs),
            "real_minus_condition_psnr": float(psnr.mean()),
            "condition_minus_real_lpips": float(lpips.mean()),
            "psnr_win_rate": float((psnr > 0).mean()),
            "lpips_win_rate": float((lpips > 0).mean()),
            "psnr_bootstrap_95ci": [float(x) for x in np.quantile(psnr[idx].mean(axis=1), [0.025, 0.975])],
            "lpips_bootstrap_95ci": [float(x) for x in np.quantile(lpips[idx].mean(axis=1), [0.025, 0.975])],
        }
    return out


def build_summary(records, args, step: int, selected_count: int):
    summary = {
        "checkpoint": args.ckpt,
        "b1_checkpoint": args.b1_ckpt,
        "checkpoint_step": step,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sat_pe_mode": getattr(args, "sat_pe_mode", "unknown"),
        "selected_tuples": selected_count,
        "modes": {},
    }
    for mode in MODES:
        mode_records = [r for r in records if r["sat_memory_mode"] == mode]
        by_bin = {}
        for bin_id in sorted({r["bin"] for r in mode_records}):
            by_bin[str(bin_id)] = mean_metrics([r for r in mode_records if r["bin"] == bin_id])
        summary["modes"][mode] = {
            "overall": mean_metrics(mode_records),
            "by_bin": by_bin,
        }
    summary["paired_effects_vs_real"] = paired_bootstrap(records, args.seed)
    return summary


def to_bgr(image: torch.Tensor):
    rgb = (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def labeled(image: np.ndarray, label: str):
    canvas = np.zeros((image.shape[0] + 24, image.shape[1], 3), dtype=np.uint8)
    canvas[24:] = image
    cv2.putText(canvas, label, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_comparisons(out: Path, vis_cache):
    for batch_index, images in sorted(vis_cache.items()):
        rows = []
        batch_size = images["gt"].shape[0]
        for b in range(min(batch_size, 3)):
            cells = [
                labeled(to_bgr(images["src0"][b]), "src0"),
                labeled(to_bgr(images["gt"][b]), "gt"),
            ]
            cells.extend(labeled(to_bgr(images[mode][b]), mode) for mode in MODES)
            rows.append(np.concatenate(cells, axis=1))
        grid = np.concatenate(rows, axis=0)
        cv2.imwrite(
            str(out / f"comparison_{batch_index:02d}.jpg"),
            grid,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/icassp27_b2_pilot/ckpt.pt")
    ap.add_argument("--b1_ckpt", default="runs/icassp27_b1_pilot/ckpt.pt")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--num_tuples", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vis_batches", type=int, default=2)
    ap.add_argument("--out", default="runs/eval_b2_sat_ablate")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, step = build_model_from_ckpt(args.ckpt, device)
    if not model.use_sat or not model.use_src:
        raise ValueError("--ckpt must be a B2 checkpoint with use_sat=True and use_src=True")
    b1_model, b1_step = build_model_from_ckpt(args.b1_ckpt, device)
    if b1_model.use_sat or not b1_model.use_src:
        raise ValueError("--b1_ckpt must be a B1 checkpoint with use_sat=False and use_src=True")
    vq = PretrainedTokenizer("ckpts/maskgit-vqgan-imagenet-f16-256.bin").to(device).eval()
    ds = Kitti360TupleDataset(args.manifest, mode="eval", seed=args.seed)
    args.sat_pe_mode = model.sat_pe_mode
    idxs = select_eval_indices(ds, args.num_tuples)
    print(f"[eval] ckpt={args.ckpt} (step {step})  device={device}")
    selected_windows = {
        (ds.tuples[i // len(ds.eval_k)].drive,
         ds.tuples[i // len(ds.eval_k)].window_id,
         ds.tuples[i // len(ds.eval_k)].window_center_fid)
        for i in idxs
    }
    print(f"[eval] selected {len(idxs)} tuples from {len(selected_windows)} satellite windows "
          f"over {len(ds.bins) * len(ds.eval_k)} (bin,K) groups")

    psnr_fn = PSNR(reduction="none")
    lpips_fn = LPIPS(net="alex", reduction="none").to(device).eval()
    records = []
    vis_cache = {}
    max_len = ds.img_w // 16 * (ds.img_h // 16)

    for mode in MODES:
        torch.manual_seed(args.seed)
        t0 = time.time()
        current_model = b1_model if mode == "b1" else model
        print(f"[eval] mode={mode} step={b1_step if mode == 'b1' else step}", flush=True)
        for s in range(0, len(idxs), args.batch_size):
            batch_index = s // args.batch_size
            selected = idxs[s : s + args.batch_size]
            batch = collate_tuples([ds[i] for i in selected])
            batch_size = batch["tgt_rgb"].shape[0]
            sat_perm = None
            if mode == "shuffle":
                window_keys = [(m["drive"], m["window_id"]) for m in batch["meta"]]
                sat_perm = deranged_batch_perm(window_keys, batch_index, args.seed, device)
            sat_token_perm = None
            if mode == "pe_permute":
                token_generator = torch.Generator(device="cpu").manual_seed(args.seed + batch_index)
                sat_token_perm = torch.randperm(
                    model.sat_grid[0] * model.sat_grid[1], generator=token_generator
                ).to(device)

            with torch.no_grad():
                tokens = current_model.generate(
                    batch["pose_vec"].to(device), max_len=max_len,
                    temperature=args.temperature, top_p=args.top_p,
                    sat=batch["sat"].to(device),
                    window_origin_xyz=batch["window_origin_xyz"].to(device),
                    src_rgbs=batch["src_rgbs"].to(device),
                    rel_poses=batch["rel_poses"].to(device),
                    src_mask=batch["src_mask"].to(device),
                    tgt_K=batch["tgt_K"].to(device),
                    tgt_T_cam=batch["tgt_T_cam"].to(device),
                    sat_memory_mode="real" if mode == "b1" else mode,
                    sat_memory_perm=sat_perm,
                    sat_token_perm=sat_token_perm,
                )
                gen = vq.decode(tokens.view(batch_size, ds.img_h // 16, ds.img_w // 16)).clamp(-1, 1)
                gt = batch["tgt_rgb"].to(device) * 2 - 1
                gen01 = (gen + 1) / 2
                gt01 = (gt + 1) / 2
                psnr = psnr_fn(gen01, gt01).detach().cpu().numpy()
                lpips = lpips_fn(gen01, gt01).detach().cpu().numpy()

            for b in range(batch_size):
                meta = batch["meta"][b]
                records.append({
                    "sat_memory_mode": mode,
                    "ckpt_step": b1_step if mode == "b1" else step,
                    "tuple_index": int(selected[b]),
                    "bin": meta["actual_bin"],
                    "K": int(batch["n_src"][b]),
                    "dist_m": round(meta["actual_source_target_dist_m"], 2),
                    "requested_dist_m": round(meta["dist_m"], 2),
                    "dyaw_deg": round(meta["dyaw_deg"], 2),
                    "drive": meta["drive"],
                    "target_fid": meta["target_fid"],
                    "source_fids": meta["source_fids"],
                    "shuffle_window_id": (
                        batch["meta"][int(sat_perm[b])]["window_id"] if mode == "shuffle" else None
                    ),
                    "psnr": float(psnr[b]),
                    "lpips": float(lpips[b]),
                })

            if batch_index < args.vis_batches:
                cached = vis_cache.setdefault(batch_index, {})
                if mode == "real":
                    cached["src0"] = batch["src_rgbs"][:, 0].clone()
                    cached["gt"] = gt01.detach().cpu()
                cached[mode] = gen01.detach().cpu()

            elapsed = time.time() - t0
            done = s + batch_size
            print(f"  {done}/{len(idxs)}  {elapsed / done:.1f}s/tuple", flush=True)

    with open(out / "records.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    summary = build_summary(records, args, step, len(idxs))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    save_comparisons(out, vis_cache)

    print(f"\n===== satellite-memory ablation (ckpt step {step}) =====")
    print(f"{'mode':>7} {'bin':>4} {'n':>4} {'PSNR':>7} {'LPIPS':>7}")
    for mode in MODES:
        mode_summary = summary["modes"][mode]
        overall = mode_summary["overall"]
        print(f"{mode:>7} {'all':>4} {overall['n']:>4} {overall['psnr']:>7.2f} {overall['lpips']:>7.3f}")
        for bin_id, metrics in mode_summary["by_bin"].items():
            print(f"{mode:>7} {bin_id:>4} {metrics['n']:>4} {metrics['psnr']:>7.2f} {metrics['lpips']:>7.3f}")
    print(f"[eval] records -> {out / 'records.jsonl'}")
    print(f"[eval] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()

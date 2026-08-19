#!/usr/bin/env python3
"""Dense-distance, paired B2 satellite-sensitivity evaluation.

Uses completed B2/B1 checkpoints only.  At every requested extrapolation
distance it selects the same number of K=1 and K=3 tuples, with each selected
tuple coming from a distinct satellite window.  B2(real), B2(cross-window
visual-shuffle), and B1 are generated with reset AR RNG so the output supports
both SCS(d) = B2(real) - B2(shuffle) and fusion(d) = B2(real) - B1.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metrics import LPIPS, PSNR
from models.stage1.maskgit.tokenizer import PretrainedTokenizer
from scripts.eval_icassp27_sat_ablate import build_model_from_ckpt, deranged_batch_perm
from world3d.data.kitti360_tuple_dataset import Kitti360TupleDataset, collate_tuples


MODES = ("real", "shuffle", "b1")


def parse_distances(text: str) -> tuple[float, ...]:
    distances = tuple(float(value) for value in text.split(","))
    if not distances or len(set(distances)) != len(distances):
        raise ValueError("--distances must be a non-empty comma-separated list without duplicates")
    if any(distance < 2.0 or distance > 20.0 for distance in distances):
        raise ValueError("--distances must lie in [2, 20] m")
    return distances


def select_dense_indices(
    dataset: Kitti360TupleDataset,
    distances: tuple[float, ...],
    per_distance: int,
    seed: int,
) -> list[int]:
    """Balance requested distance and K while enforcing unique sat windows.

    The unique-window constraint makes every in-batch shuffle donor genuinely
    different from its recipient.  It also avoids accidentally measuring a
    repeated window's easy/hard appearance rather than a distance effect.
    """
    if per_distance < len(dataset.eval_k) or per_distance % len(dataset.eval_k):
        raise ValueError("--per_distance must be a positive multiple of the number of eval K values")
    groups: dict[tuple[float, int], dict[tuple, int]] = defaultdict(dict)
    for index in range(len(dataset)):
        spec = dataset.tuples[index // len(dataset.eval_k)]
        distance = float(spec.dist_m)
        if distance not in distances:
            continue
        k_slot = index % len(dataset.eval_k)
        window_key = (spec.drive, spec.window_id, spec.window_center_fid)
        groups[(distance, k_slot)].setdefault(window_key, index)

    all_windows = sorted(set().union(*(set(group) for group in groups.values())))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_windows)
    selected, used_windows = [], set()
    cursor = 0
    for distance in distances:
        for slot_number in range(per_distance):
            k_slot = slot_number % len(dataset.eval_k)
            candidates = groups[(distance, k_slot)]
            for offset in range(len(all_windows)):
                window_key = all_windows[(cursor + offset) % len(all_windows)]
                if window_key not in used_windows and window_key in candidates:
                    selected.append(candidates[window_key])
                    used_windows.add(window_key)
                    cursor = (cursor + offset + 1) % len(all_windows)
                    break
            else:
                raise RuntimeError(
                    f"not enough distinct satellite windows for {distance:g}m, K slot {k_slot}; "
                    f"requested {len(distances) * per_distance}, available {len(all_windows)}"
                )
    return selected


def mean_metrics(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "lpips": float(np.mean([row["lpips"] for row in rows])),
    }


def paired_effects(rows: list[dict], *, seed: int, samples: int = 10_000) -> dict:
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["tuple_index"])][row["condition"]] = row
    rng = np.random.default_rng(seed)
    output = {}
    for condition, title in (("shuffle", "scs_b2_real_minus_shuffle"), ("b1", "fusion_b2_real_minus_b1")):
        pairs = [modes for modes in grouped.values() if "real" in modes and condition in modes]
        psnr = np.asarray([modes["real"]["psnr"] - modes[condition]["psnr"] for modes in pairs])
        # Lower LPIPS is better, so the positive direction remains B2(real).
        lpips = np.asarray([modes[condition]["lpips"] - modes["real"]["lpips"] for modes in pairs])
        if not len(pairs):
            continue
        indices = rng.integers(0, len(pairs), size=(samples, len(pairs)))
        output[title] = {
            "n": len(pairs),
            "positive_direction": "favours B2(real)",
            "psnr_mean": float(psnr.mean()),
            "lpips_mean": float(lpips.mean()),
            "psnr_win_rate": float((psnr > 0).mean()),
            "lpips_win_rate": float((lpips > 0).mean()),
            "psnr_bootstrap_95ci": [float(value) for value in np.quantile(psnr[indices].mean(axis=1), [0.025, 0.975])],
            "lpips_bootstrap_95ci": [float(value) for value in np.quantile(lpips[indices].mean(axis=1), [0.025, 0.975])],
        }
    return output


def save_curve(summary: dict, path: Path) -> None:
    """Save a compact SCS(d) figure without making plotting a runtime dependency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[dense] matplotlib unavailable; plot not saved: {exc}")
        return
    distances, means, low, high = [], [], [], []
    for key, section in sorted(summary["by_requested_distance"].items(), key=lambda item: float(item[0])):
        effect = section["paired_effects"]["scs_b2_real_minus_shuffle"]
        distances.append(float(key))
        means.append(effect["psnr_mean"])
        low.append(effect["psnr_mean"] - effect["psnr_bootstrap_95ci"][0])
        high.append(effect["psnr_bootstrap_95ci"][1] - effect["psnr_mean"])
    fig, axis = plt.subplots(figsize=(6.2, 3.7), constrained_layout=True)
    axis.axhline(0.0, color="0.45", linewidth=1, linestyle="--")
    axis.errorbar(distances, means, yerr=np.asarray([low, high]), marker="o", capsize=3, color="#1f77b4")
    axis.set(xlabel="requested source-to-target distance (m)", ylabel="SCS = B2(real) − B2(shuffle) PSNR", title="Satellite scene-specific sensitivity")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/icassp27_tc_b2/ckpt.pt")
    ap.add_argument("--b1_ckpt", default="runs/icassp27_tc_b1/ckpt.pt")
    ap.add_argument("--manifest", default="dataset_splits/kitti360_geofence_buffer30/val_manifest.jsonl")
    ap.add_argument("--distances", default="2,4,6,8,10,12,15,18,20")
    ap.add_argument("--per_distance", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--selection_seed", type=int, default=0,
        help="fixed tuple-selection seed; keep this constant across AR seeds for paired aggregation",
    )
    ap.add_argument("--out", default="runs/eval_tc_dense_seed0")
    args = ap.parse_args()
    distances = parse_distances(args.distances)
    if args.batch_size < 2:
        raise ValueError("--batch_size must be at least two for a valid cross-window shuffle")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b2, b2_step = build_model_from_ckpt(args.ckpt, device)
    b1, b1_step = build_model_from_ckpt(args.b1_ckpt, device)
    if not (b2.use_sat and b2.use_src and not b1.use_sat and b1.use_src):
        raise ValueError("--ckpt must be B2 and --b1_ckpt must be B1")
    dataset = Kitti360TupleDataset(
        args.manifest, mode="eval", seed=args.seed, eval_distances=distances,
    )
    indexes = select_dense_indices(dataset, distances, args.per_distance, args.selection_seed)
    if len(indexes) % args.batch_size == 1:
        raise ValueError("final shuffle batch would contain one item; choose another --batch_size")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vq = PretrainedTokenizer("ckpts/maskgit-vqgan-imagenet-f16-256.bin").to(device).eval()
    psnr_fn = PSNR(reduction="none")
    lpips_fn = LPIPS(net="alex", reduction="none").to(device).eval()
    max_len = (dataset.img_h // 16) * (dataset.img_w // 16)
    records = []

    print(
        f"[dense] B2 step={b2_step}; B1 step={b1_step}; {len(indexes)} tuples "
        f"({args.per_distance}/distance, distances={distances}), AR seed={args.seed}, "
        f"selection seed={args.selection_seed}",
        flush=True,
    )
    for condition in MODES:
        current_model = b1 if condition == "b1" else b2
        torch.manual_seed(args.seed)
        started = time.time()
        print(f"[dense] condition={condition}", flush=True)
        for start in range(0, len(indexes), args.batch_size):
            batch_index = start // args.batch_size
            selected = indexes[start : start + args.batch_size]
            batch = collate_tuples([dataset[index] for index in selected])
            B = batch["tgt_rgb"].shape[0]
            sat_permutation = None
            if condition == "shuffle":
                window_keys = [(meta["drive"], meta["window_id"]) for meta in batch["meta"]]
                sat_permutation = deranged_batch_perm(window_keys, batch_index, args.seed, device)
            with torch.no_grad():
                tokens = current_model.generate(
                    batch["pose_vec"].to(device), max_len=max_len,
                    temperature=args.temperature, top_p=args.top_p,
                    sat=batch["sat"].to(device), window_origin_xyz=batch["window_origin_xyz"].to(device),
                    src_rgbs=batch["src_rgbs"].to(device), rel_poses=batch["rel_poses"].to(device),
                    src_mask=batch["src_mask"].to(device), tgt_K=batch["tgt_K"].to(device),
                    tgt_T_cam=batch["tgt_T_cam"].to(device),
                    sat_memory_mode="real" if condition == "b1" else condition,
                    sat_memory_perm=sat_permutation,
                )
                gen01 = ((vq.decode(tokens.view(B, dataset.img_h // 16, dataset.img_w // 16)) + 1) / 2).clamp(0, 1)
                gt01 = batch["tgt_rgb"].to(device)
                psnr = psnr_fn(gen01, gt01).detach().cpu().numpy()
                lpips = lpips_fn(gen01, gt01).detach().cpu().numpy()
            for local in range(B):
                meta = batch["meta"][local]
                record = {
                    "condition": condition,
                    "ar_seed": args.seed,
                    "ckpt_step": b1_step if condition == "b1" else b2_step,
                    "tuple_index": int(selected[local]),
                    "requested_dist_m": float(meta["dist_m"]),
                    "actual_dist_m": float(meta["actual_source_target_dist_m"]),
                    "K": int(batch["n_src"][local]),
                    "dyaw_deg": float(meta["dyaw_deg"]),
                    "drive": meta["drive"], "window_id": int(meta["window_id"]),
                    "target_fid": int(meta["target_fid"]), "source_fids": meta["source_fids"],
                    "shuffle_donor_window_id": (
                        int(batch["meta"][int(sat_permutation[local])]["window_id"])
                        if condition == "shuffle" else None
                    ),
                    "psnr": float(psnr[local]), "lpips": float(lpips[local]),
                }
                records.append(record)
            done = start + B
            print(f"  {done}/{len(indexes)} {(time.time() - started) / done:.1f}s/tuple", flush=True)

    with (out / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "purpose": "inference-only dense-distance paired satellite diagnostic",
        "b2_checkpoint": args.ckpt, "b1_checkpoint": args.b1_ckpt,
        "b2_step": b2_step, "b1_step": b1_step, "ar_seed": args.seed,
        "selection_seed": args.selection_seed,
        "requested_distances_m": distances, "tuples_per_distance": args.per_distance,
        "selected_tuples": len(indexes), "by_requested_distance": {},
    }
    for distance in distances:
        rows = [row for row in records if row["requested_dist_m"] == distance]
        by_condition = {
            condition: mean_metrics([row for row in rows if row["condition"] == condition])
            for condition in MODES
        }
        summary["by_requested_distance"][str(distance)] = {
            "n_unique_tuples": len({row["tuple_index"] for row in rows}),
            "conditions": by_condition,
            "paired_effects": paired_effects(rows, seed=args.seed + int(distance * 100)),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_curve(summary, out / "scs_psnr_by_distance.png")
    print(f"[dense] records -> {out / 'records.jsonl'}")
    print(f"[dense] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate dense paired B2/B1 diagnostics across AR sampling seeds.

Tuples are the bootstrap unit: conditions are averaged within tuple across
sampling seeds before estimating SCS(d) and B2(real)-B1(d).  This preserves the
counterfactual pairing and prevents duplicated AR draws from acting as
independent scenes.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODES = ("real", "shuffle", "b1")


def paired_effects(rows: list[dict], *, seed: int, samples: int = 20_000) -> dict:
    rng = np.random.default_rng(seed)
    output = {}
    for other, key in (("shuffle", "scs_b2_real_minus_shuffle"), ("b1", "fusion_b2_real_minus_b1")):
        psnr = np.asarray([row["real_psnr"] - row[f"{other}_psnr"] for row in rows])
        lpips = np.asarray([row[f"{other}_lpips"] - row["real_lpips"] for row in rows])
        picks = rng.integers(0, len(rows), size=(samples, len(rows)))
        output[key] = {
            "n_tuples": len(rows),
            "positive_direction": "favours B2(real)",
            "psnr_mean": float(psnr.mean()), "lpips_mean": float(lpips.mean()),
            "psnr_win_rate": float((psnr > 0).mean()), "lpips_win_rate": float((lpips > 0).mean()),
            "psnr_bootstrap_95ci": [float(value) for value in np.quantile(psnr[picks].mean(axis=1), [0.025, 0.975])],
            "lpips_bootstrap_95ci": [float(value) for value in np.quantile(lpips[picks].mean(axis=1), [0.025, 0.975])],
        }
    return output


def save_curve(summary: dict, output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[dense-analysis] matplotlib unavailable: {exc}")
        return
    ds, means, lows, highs = [], [], [], []
    for distance, row in sorted(summary["by_requested_distance"].items(), key=lambda item: float(item[0])):
        effect = row["paired_effects"]["scs_b2_real_minus_shuffle"]
        ds.append(float(distance)); means.append(effect["psnr_mean"])
        lows.append(effect["psnr_mean"] - effect["psnr_bootstrap_95ci"][0])
        highs.append(effect["psnr_bootstrap_95ci"][1] - effect["psnr_mean"])
    fig, axis = plt.subplots(figsize=(6.2, 3.7), constrained_layout=True)
    axis.axhline(0, color="0.45", linestyle="--", linewidth=1)
    axis.errorbar(ds, means, yerr=np.asarray([lows, highs]), marker="o", capsize=3, color="#1f77b4")
    axis.set(xlabel="requested source-to-target distance (m)", ylabel="SCS = B2(real) − B2(shuffle) PSNR", title="Satellite scene-specific sensitivity")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", default=[
        "runs/eval_tc_dense_seed0/records.jsonl", "runs/eval_tc_dense_seed1/records.jsonl",
    ])
    ap.add_argument("--out", default="runs/analysis_tc_final/dense")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    raw = []
    for name in args.records:
        path = Path(name)
        raw.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    by_tuple: dict[tuple[float, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in raw:
        by_tuple[(float(row["requested_dist_m"]), int(row["tuple_index"]))][row["condition"]].append(row)
    aggregate = []
    for (distance, tuple_index), modes in sorted(by_tuple.items()):
        if any(mode not in modes for mode in MODES):
            raise RuntimeError(f"missing condition for {distance}m tuple {tuple_index}")
        seed_sets = [sorted(row["ar_seed"] for row in modes[mode]) for mode in MODES]
        if not all(seeds == seed_sets[0] for seeds in seed_sets[1:]):
            raise RuntimeError(f"unpaired AR seeds for {distance}m tuple {tuple_index}")
        first = modes["real"][0]
        aggregate.append({
            "requested_dist_m": distance, "tuple_index": tuple_index, "n_ar_seeds": len(seed_sets[0]),
            "actual_dist_m": float(np.mean([row["actual_dist_m"] for row in modes["real"]])),
            "K": int(first["K"]), "dyaw_deg": float(first["dyaw_deg"]),
            **{
                f"{mode}_{metric}": float(np.mean([row[metric] for row in modes[mode]]))
                for mode in MODES for metric in ("psnr", "lpips")
            },
        })
    for row in aggregate:
        row["scs_psnr"] = row["real_psnr"] - row["shuffle_psnr"]
        row["scs_lpips"] = row["shuffle_lpips"] - row["real_lpips"]
        row["fusion_b2_minus_b1_psnr"] = row["real_psnr"] - row["b1_psnr"]
        row["fusion_b2_minus_b1_lpips"] = row["b1_lpips"] - row["real_lpips"]

    summary = {"records": args.records, "n_raw_seeded_records": len(raw), "n_unique_tuples": len(aggregate), "by_requested_distance": {}}
    for distance in sorted({row["requested_dist_m"] for row in aggregate}):
        rows = [row for row in aggregate if row["requested_dist_m"] == distance]
        summary["by_requested_distance"][str(distance)] = {
            "n_tuples": len(rows), "paired_effects": paired_effects(rows, seed=args.seed + int(distance * 100)),
        }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fields = list(aggregate[0])
    with (out / "per_tuple_seed_mean.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(aggregate)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_curve(summary, out / "scs_psnr_by_distance.png")
    print("[dense-analysis] distance  SCS PSNR [95% CI]  fusion PSNR [95% CI]")
    for distance, row in summary["by_requested_distance"].items():
        scs = row["paired_effects"]["scs_b2_real_minus_shuffle"]
        fusion = row["paired_effects"]["fusion_b2_real_minus_b1"]
        print(f"{float(distance):>5.1f}m {scs['psnr_mean']:+.3f} {scs['psnr_bootstrap_95ci']}  {fusion['psnr_mean']:+.3f} {fusion['psnr_bootstrap_95ci']}")
    print(f"[dense-analysis] {len(aggregate)} tuples -> {out}")


if __name__ == "__main__":
    main()

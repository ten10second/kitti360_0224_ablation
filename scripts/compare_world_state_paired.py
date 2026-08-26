#!/usr/bin/env python3
"""Scene-level paired bootstrap of world-state trajectory records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def load(path: str) -> Dict[tuple, dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {(r["scene_id"], int(r["version"])): r for r in rows}


def mean_finite(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def bootstrap_ci(deltas: np.ndarray, n: int = 10000, seed: int = 0):
    if deltas.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n)])
    return float(deltas.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="treatment JSONL (aligned)")
    ap.add_argument("--b", required=True, help="control JSONL")
    ap.add_argument("--metrics", default="height_ahead_mae,density_ahead_mae,height_visited_mae,g_update_height")
    ap.add_argument("--version", type=int, default=0, help="state version to compare; 0 = Z0 prior")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    keys = [k for k in keys if k[1] == args.version]
    if not keys:
        raise SystemExit("no paired scene/version rows")
    report = {"paired_scenes": len({k[0] for k in keys}), "version": args.version}
    for metric in args.metrics.split(","):
        va = np.array([a[k][metric] for k in keys if a[k].get(metric) is not None], dtype=np.float64)
        vb = np.array([b[k][metric] for k in keys if b[k].get(metric) is not None], dtype=np.float64)
        n = min(len(va), len(vb))
        va, vb = va[:n], vb[:n]
        # lower-is-better for mae
        delta = vb - va
        mean, lo, hi = bootstrap_ci(delta)
        report[metric] = {
            "A": mean_finite(va.tolist()),
            "B": mean_finite(vb.tolist()),
            "B_minus_A": mean,
            "ci95": [lo, hi],
            "wins": int((delta > 0).sum()),
            "n": int(n),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

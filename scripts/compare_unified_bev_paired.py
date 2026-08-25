#!/usr/bin/env python3
"""Paired tile bootstrap comparison between two eval JSONL record sets."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load(path: str) -> dict:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return {(r["drive"], r["target_fid"]): r for r in rows}


def higher_is_better(key: str) -> bool:
    """Metric direction for both legacy and claim-aligned evaluator fields."""
    name = key.lower()
    return any(token in name for token in ("psnr", "ssim", "delta1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="records A (treatment)")
    ap.add_argument("--b", required=True, help="records B (control)")
    ap.add_argument(
        "--keys",
        default=(
            "full_height_fill_mae,full_height_fill_rmse,"
            "full_rgb_lowfreq_psnr,full_rgb_supported_psnr,full_psnr,full_absrel"
        ),
    )
    ap.add_argument(
        "--allow_missing", action="store_true",
        help="legacy compatibility only: warn and skip requested metrics absent from records",
    )
    ap.add_argument("--bootstraps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    keys = [(key, "up" if higher_is_better(key) else "down")
            for key in args.keys.split(",") if key]
    paired = [(a[k], b[k]) for k in sorted(set(a) & set(b))]
    print(f"a={args.a}")
    print(f"b={args.b}")
    print(f"paired_tiles={len(paired)}")
    if not paired:
        raise SystemExit("no common (drive,target_fid) pairs")
    rng = random.Random(args.seed)
    for key, better in keys:
        missing = sum(key not in row for pair in paired for row in pair)
        if missing:
            message = f"requested metric {key!r} is absent from {missing} paired records"
            if args.allow_missing:
                print(f"WARNING: {message}; skipped")
                continue
            raise SystemExit(message)
        diffs = [
            float(ra[key]) - float(rb[key])
            for ra, rb in paired
            if ra[key] is not None and rb[key] is not None
            and math.isfinite(float(ra[key])) and math.isfinite(float(rb[key]))
        ]
        if not diffs:
            message = f"requested metric {key!r} has no finite paired values"
            if args.allow_missing:
                print(f"WARNING: {message}; skipped")
                continue
            raise SystemExit(message)
        n = len(diffs)
        mean = sum(diffs) / n
        samples = []
        for _ in range(args.bootstraps):
            picks = [diffs[rng.randrange(n)] for _ in range(n)]
            samples.append(sum(picks) / n)
        samples.sort()
        lo, hi = samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]
        wins = sum(d > 0 for d in diffs) if better == "up" else sum(d < 0 for d in diffs)
        significant = (lo > 0) if better == "up" else (hi < 0)
        print(f"{key:>30s}: mean={mean:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  "
              f"A_better={wins}/{n}  direction={better}  "
              f"{'SIGNIFICANT' if significant else 'ns'}")


if __name__ == "__main__":
    main()

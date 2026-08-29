#!/usr/bin/env python3
"""Fig 4: the obvious idea, done naively vs done correctly.

Same held-out scene, same protocol, two formal trainings: v3 (naive: no depth
consistency, retention locks the write region, camera-rig-only anchoring) vs
v4 (depth consistency + retention mask + DGM two-tier anchor).  Three panels:
per-step write gain, persistent-vs-one-shot, held-out depth readout.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def _series(rows, key):
    ts, ys = [], []
    for r in rows:
        if r.get(key) is not None:
            ts.append(r["version"])
            ys.append(r[key])
    return ts, ys


def main():
    v3 = Path("runs/world_state_v3_formal_20260828")
    v4 = Path("runs/world_state_v4_formal_20260829")
    out = Path("runs/advisor_briefing/fig4_what_fixed_it.png")

    a3, a4 = _load(v3 / "eval_aligned.jsonl"), _load(v4 / "eval_aligned.jsonl")
    o3, o4 = _load(v3 / "eval_one_shot.jsonl"), _load(v4 / "eval_one_shot.jsonl")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # panel 1: per-step write gain
    ax = axes[0]
    for rows, color, label in ((a3, "#b2182b", "naive (v3)"), (a4, "#1a6fae", "fixed (v4)")):
        ts, ys = _series(rows, "g_update_height")
        ts, ys = ts[1:], ys[1:]  # drop t=1 (first write has no stale prior)
        ax.plot(ts, ys, "o-", color=color, lw=2, ms=5, label=label)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("chunk index")
    ax.set_ylabel("write gain $g_{update}$ (m)")
    ax.set_title("arrival writes: noise → consistently positive", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # panel 2: final visited MAE, persistent vs one-shot
    ax = axes[1]
    import numpy as np
    labels = ["naive\n(v3)", "fixed\n(v4)"]
    pers = [max(r["version"] for r in a3 if r.get("height_visited_mae") is not None
                and r["version"] > 0) and
            [r["height_visited_mae"] for r in a3 if r["version"] == max(
                x["version"] for x in a3)][0],
            [r["height_visited_mae"] for r in a4 if r["version"] == max(
                x["version"] for x in a4)][0]]
    ones = [[r["height_visited_mae"] for r in o3 if r["version"] == max(
        x["version"] for x in o3)][0],
        [r["height_visited_mae"] for r in o4 if r["version"] == max(
            x["version"] for x in o4)][0]]
    x = np.arange(2)
    ax.bar(x - 0.18, pers, width=0.36, color=["#b2182b", "#1a6fae"],
           label="persistent (queryable anytime)")
    ax.bar(x + 0.18, ones, width=0.36, color=["#b2182b", "#1a6fae"],
           alpha=0.45, hatch="//", label="one-shot (map only at end)")
    for xi, v in zip(x - 0.18, pers):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.18, ones):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("final visited MAE (m)")
    ax.set_ylim(0, max(max(pers), max(ones)) * 1.25)
    ax.set_title("persistence: losing to one-shot → beating it", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # panel 3: held-out depth readout over time
    ax = axes[2]
    for rows, color, label in ((a3, "#b2182b", "naive (v3)"), (a4, "#1a6fae", "fixed (v4)")):
        ts, ys = _series(rows, "depth_absrel")
        ax.plot(ts, ys, "o-", color=color, lw=2, ms=5, label=label)
    ax.set_xlabel("chunk index")
    ax.set_ylabel("held-out depth AbsRel")
    ax.set_title("secondary readout: drifting → stable", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("The obvious idea done naively does not work — three failure modes, three fixes "
                 "(same held-out scene, same protocol, formal trainings)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

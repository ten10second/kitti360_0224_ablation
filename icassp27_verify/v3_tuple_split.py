"""Day-0 V3: tuple sampler feasibility + geographic split check on KITTI-raw.

Plan requirements tested (ICASSP27 sec.4 / Checklist sec.4):
 [T1] anchors spaced >=2 m along each sequence; target in [2,20] m ahead, metric
 [T2] dyaw(source_heading -> target_heading) <= 20 deg filter (drop turns)
 [T3] distance bins [2,5)/[5,10)/[10,20] m; interpolation vs extrapolation counts
 [T4] ~60 m route windows: how many windows per drive, tuples per window
 [T5] geographic split: drive GPS bbox overlap graph (anti-leak, MM26 killer 2)
      - KITTI-raw drives on the same date often share streets; must split by
        connected components of overlapping footprints, not by drive id alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kitti_raw_io import read_oxts, oxts_to_poses_enu, effective_drives

ROOT = "/media/shizhm/Lenovo/KITTI_RAW"


def drive_stats(drive_dir: Path):
    oxts = read_oxts(drive_dir / "oxts" / "data")
    poses_abs = oxts_to_poses_enu(oxts)
    xy = poses_abs[:, :2, 3]
    yaw = oxts[:, 5]
    n = len(oxts)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])

    def frame_at(i_from: int, dist: float):
        """first frame index at/after >=dist meters ahead of i_from along cumulative arc."""
        j = np.searchsorted(cum, cum[i_from] + dist, side="left")
        return j if j < n else -1

    # [T1-T3] enumerate anchors every ~2m, K=3 sources, targets by bin
    bins = [(2, 5), (5, 10), (10, 20)]
    counts = {("extra", b): 0 for b in bins}
    counts.update({("inter", b): 0 for b in bins})
    dropped_yaw = 0
    total_cand = 0
    anchors = []
    i = 0
    while i < n - 1:
        anchors.append(i)
        i = frame_at(i, 2.0)
        if i < 0:
            break
    for a in anchors:
        # K=3 sources: a and next two 2m-spaced anchors, all behind target
        s3 = []
        j = a
        for _ in range(3):
            if j < 0:
                break
            s3.append(j)
            j = frame_at(j, 2.0)
        if len(s3) < 3:
            continue
        for lo, hi in bins:
            # target at midpoint of bin (deterministic feasibility count)
            t = frame_at(s3[-1], (lo + hi) / 2.0)
            if t < 0:
                continue
            total_cand += 1
            d_yaw = np.degrees(np.abs((yaw[t] - yaw[a] + np.pi) % (2 * np.pi) - np.pi))
            if d_yaw > 20.0:
                dropped_yaw += 1
                continue
            counts[("extra", (lo, hi))] += 1
        # interpolation: sources before and after target
        for lo, hi in bins:
            t = frame_at(s3[-1], (lo + hi) / 2.0)
            if t < 0:
                continue
            fwd = frame_at(t, 2.0)
            if fwd < 0:
                continue
            d_yaw = np.degrees(np.abs((yaw[t] - yaw[a] + np.pi) % (2 * np.pi) - np.pi))
            if d_yaw > 20.0:
                continue
            counts[("inter", (lo, hi))] += 1  # sources s3 + fwd straddle t

    # [T4] 60m windows
    n_windows = int(np.floor(cum[-1] / 60.0))

    return dict(
        name=drive_dir.name,
        n=n,
        length=cum[-1],
        n_windows=n_windows,
        counts=counts,
        dropped_yaw=dropped_yaw,
        total_cand=total_cand,
        bbox=(xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()),
    )


def bbox_overlap(b1, b2, margin=0.0):
    return not (b1[2] < b2[0] - margin or b2[2] < b1[0] - margin
                or b1[3] < b2[1] - margin or b2[3] < b1[1] - margin)


def main():
    drives = [Path(p) for p in effective_drives(ROOT)]
    print(f"effective drives: {len(drives)}")

    all_stats = []
    for d in drives:
        st = drive_stats(d)
        all_stats.append(st)

    tot_extra = sum(s["counts"][("extra", b)] for s in all_stats for b in [(2, 5), (5, 10), (10, 20)])
    tot_inter = sum(s["counts"][("inter", b)] for s in all_stats for b in [(2, 5), (5, 10), (10, 20)])
    tot_drop = sum(s["dropped_yaw"] for s in all_stats)
    tot_cand = sum(s["total_cand"] for s in all_stats)
    tot_win = sum(s["n_windows"] for s in all_stats)
    tot_len = sum(s["length"] for s in all_stats)
    print(f"\n[T1-T3] K=3 tuples (feasibility count, 1 target/bin/anchor):")
    for b in [(2, 5), (5, 10), (10, 20)]:
        e = sum(s["counts"][("extra", b)] for s in all_stats)
        i_ = sum(s["counts"][("inter", b)] for s in all_stats)
        print(f"  bin {b}: extrapolation={e}  interpolation={i_}")
    print(f"  dropped by dyaw>20deg: {tot_drop}/{tot_cand} ({100*tot_drop/max(1,tot_cand):.1f}%)")
    print(f"[T4] 60m windows total: {tot_win}  (trajectory {tot_len/1000:.1f} km)")
    print(f"  windows/drive: median={np.median([s['n_windows'] for s in all_stats]):.0f}  max={max(s['n_windows'] for s in all_stats)}")

    # per-date totals
    from collections import defaultdict
    by_date = defaultdict(lambda: [0, 0])
    for s in all_stats:
        date = s["name"][:10]
        by_date[date][0] += 1
        by_date[date][1] += sum(s["counts"].values())
    print("[split-by-date]")
    for date, (nd, nt) in sorted(by_date.items()):
        print(f"  {date}: {nd} drives, {nt} tuples")

    # [T5] geographic overlap graph
    n = len(all_stats)
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_overlap(all_stats[i]["bbox"], all_stats[j]["bbox"], margin=100.0):
                adj[i][j] = adj[j][i] = 1
    # connected components
    seen = set()
    comps = []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if adj[u][v] and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    print(f"\n[T5] geographic overlap components (bbox+100m margin): {len(comps)}")
    comps_sorted = sorted(comps, key=len, reverse=True)
    for k, comp in enumerate(comps_sorted[:8]):
        dates = sorted({all_stats[i]['name'][:10] for i in comp})
        print(f"  comp{k}: {len(comp)} drives, {dates}")
    print(f"  ... {len(comps)-8} more components" if len(comps) > 8 else "")
    sizes = [len(c) for c in comps]
    print(f"  component sizes: {sizes[:20]}{'...' if len(sizes)>20 else ''}")
    print(f"  tuples per component: " + ", ".join(
        f"c{k}={sum(sum(all_stats[i]['counts'].values()) for i in c)}" for k, c in enumerate(comps_sorted)))


if __name__ == "__main__":
    main()

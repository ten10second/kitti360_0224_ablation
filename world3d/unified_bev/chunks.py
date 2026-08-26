"""Route chunks: a short contiguous trajectory arc as one ground measurement.

Chunks are cut by cumulative arc length and hard-split at trajectory jumps
(fid gaps / revisits).  They are measurement packets for world-state
assimilation, not a sparsity axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RouteChunk:
    index: int
    fids: List[int]
    positions: np.ndarray           # (n, 2) IMU xy in trajectory order
    arc_start: float
    arc_end: float
    segment: int

    @property
    def arc_length(self) -> float:
        return self.arc_end - self.arc_start

    @property
    def center_xy(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    def member_arcs(self) -> np.ndarray:
        if len(self.positions) == 1:
            return np.asarray([self.arc_start])
        steps = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return self.arc_start + np.concatenate([[0.0], np.cumsum(steps)])


def build_route_chunks(
    positions: np.ndarray,
    fids: Sequence[int],
    *,
    chunk_arc_m: float = 12.0,
    max_step_m: float = 5.0,
) -> List[RouteChunk]:
    """Cut one fid-ordered trajectory into consecutive fixed-arc chunks."""
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return []
    arc = np.zeros(n)
    segment = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        step = float(np.linalg.norm(positions[i] - positions[i - 1]))
        jump = step > max_step_m
        segment[i] = segment[i - 1] + (1 if jump else 0)
        arc[i] = 0.0 if jump else arc[i - 1] + step
    chunks: List[RouteChunk] = []
    i = 0
    while i < n:
        j = i
        while (
            j + 1 < n
            and segment[j + 1] == segment[i]
            and arc[j + 1] - arc[i] <= chunk_arc_m
        ):
            j += 1
        chunks.append(RouteChunk(
            index=len(chunks),
            fids=[int(fids[k]) for k in range(i, j + 1)],
            positions=positions[i:j + 1].copy(),
            arc_start=float(arc[i]),
            arc_end=float(arc[j]),
            segment=int(segment[i]),
        ))
        i = j + 1
    return chunks


def core_member_index(chunk: RouteChunk, guard_m: float = 0.0) -> int:
    """Index of the chunk's arc-midpoint frame (query pose)."""
    arcs = chunk.member_arcs()
    mid = 0.5 * (chunk.arc_start + chunk.arc_end)
    i = int(np.argmin(np.abs(arcs - mid)))
    if min(abs(arcs[i] - chunk.arc_start), abs(arcs[i] - chunk.arc_end)) < guard_m:
        raise ValueError(
            f"chunk {chunk.index} (arc {chunk.arc_length:.1f} m) too short for "
            f"a core query at guard {guard_m} m"
        )
    return i


def select_chunk_frames(
    chunk: RouteChunk,
    frames_per_chunk: int,
    guard_m: float = 0.0,
    max_geometry_frames: int = 8,
    *,
    guard_left: bool = False,
    guard_right_arc: Optional[float] = None,
) -> Tuple[List[int], List[int]]:
    """Pick lift frames and a capped joint-forward geometry set.

    ``lift`` is spread over frames at least ``guard_m`` inside the requested
    sides.  ``geometry`` is lift plus evenly-spread context, capped so a slow
    chunk cannot explode the VGGT view budget.
    """
    n = len(chunk.fids)
    if frames_per_chunk < 1 or frames_per_chunk > n:
        raise ValueError(f"frames_per_chunk must be in [1,{n}]")
    max_geometry_frames = min(int(max_geometry_frames), n)
    if max_geometry_frames < frames_per_chunk:
        raise ValueError(
            f"max_geometry_frames must be >= frames_per_chunk={frames_per_chunk}"
        )
    arcs = chunk.member_arcs()
    ok = np.ones(n, dtype=bool)
    if guard_left:
        ok &= arcs - chunk.arc_start >= guard_m
    if guard_right_arc is not None:
        ok &= float(guard_right_arc) - arcs >= guard_m
    safe = np.where(ok)[0]
    if len(safe) < frames_per_chunk:
        raise ValueError(
            f"chunk {chunk.index} (arc {chunk.arc_length:.1f} m) has only "
            f"{len(safe)} guard-safe frames, need {frames_per_chunk}"
        )
    if frames_per_chunk == 1:
        lift = [int(safe[len(safe) // 2])]
    else:
        pick = np.linspace(0, len(safe) - 1, frames_per_chunk).round().astype(int)
        lift = [int(safe[k]) for k in pick]
    rest = [i for i in range(n) if i not in set(lift)]
    budget = max_geometry_frames - len(lift)
    if budget == 0 or not rest:
        return lift, list(lift)
    if budget >= len(rest):
        ctx = rest
    elif budget == 1:
        ctx = [rest[len(rest) // 2]]
    else:
        sel = np.linspace(0, len(rest) - 1, budget).round().astype(int)
        ctx = [rest[k] for k in sel]
    return lift, sorted(lift + ctx)

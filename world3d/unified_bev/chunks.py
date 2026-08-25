"""Route chunks: the unit of ground evidence is a short contiguous arc.

Chunks are cut by cumulative arc length, hard-splitting at trajectory jumps
(fid gaps / revisits) so a chunk only ever contains consecutive frames.
A window is ``N_c`` consecutive chunks; the hole experiment drops the middle
``N_c - K`` chunks, drops kept-chunk frames within ``guard_m`` of the hole
boundary, and queries the midpoint frame of each missing chunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RouteChunk:
    index: int                      # position along the drive
    fids: List[int]
    positions: np.ndarray           # (n, 2) IMU xy in trajectory order
    arc_start: float                # absolute arc within the segment [m]
    arc_end: float
    segment: int                    # incremented at every trajectory jump

    @property
    def arc_length(self) -> float:
        return self.arc_end - self.arc_start

    @property
    def center_xy(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    def member_arcs(self) -> np.ndarray:
        """Absolute arc coordinate of each member frame."""
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
    """Cut one fid-ordered trajectory into consecutive fixed-arc chunks.

    A step longer than ``max_step_m`` starts a new segment (arc resets).
    A chunk closes when the next step would push its span past ``chunk_arc_m``;
    the trailing partial chunk is kept as-is.
    """
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


def build_chunk_windows(
    chunks: List[RouteChunk],
    *,
    chunks_per_window: int = 4,
    min_frames_per_chunk: int = 6,
    max_window_span_m: float = 48.0,
) -> List[List[RouteChunk]]:
    """Sliding windows of consecutive same-segment chunks with enough frames."""
    windows: List[List[RouteChunk]] = []
    for i in range(len(chunks) - chunks_per_window + 1):
        w = chunks[i:i + chunks_per_window]
        if len({c.segment for c in w}) != 1:
            continue
        if any(len(c.fids) < min_frames_per_chunk for c in w):
            continue
        if w[-1].arc_end - w[0].arc_start > max_window_span_m + 1e-6:
            continue
        windows.append(w)
    return windows


def missing_chunks(window: Sequence[RouteChunk], kept_count: int) -> List[RouteChunk]:
    """The hole: the middle ``N_c - kept_count`` consecutive chunks.

    ``kept=3`` of 4 drops c1; ``kept=2`` drops c1,c2; ``kept=1`` drops
    c1,c2,c3.  Kept chunks always include both ends when possible, so the
    hole is an interior block with ground on its borders.
    """
    n = len(window)
    if not 1 <= kept_count < n:
        raise ValueError(f"kept_count must be in [1,{n - 1}], got {kept_count}")
    return list(window)[1:1 + (n - kept_count)]


def guard_keep_mask(chunk: RouteChunk, hole_arc: Tuple[float, float],
                    guard_m: float) -> np.ndarray:
    """Which frames of a kept chunk stay as sparse observations.

    Frames whose arc lies inside the hole interval or within ``guard_m`` of
    it are dropped (distance-to-interval < guard_m), so boundary frames
    cannot see directly into the hole.
    """
    a, b = hole_arc
    arcs = chunk.member_arcs()
    dist = np.maximum(np.maximum(a - arcs, arcs - b), 0.0)
    return dist >= guard_m


def core_member_index(chunk: RouteChunk, guard_m: float) -> int:
    """Index of the chunk's arc-midpoint frame: the frozen-query location.

    The midpoint is guaranteed deeper than ``guard_m`` inside both chunk
    boundaries, matching the guard dropped at the hole borders.
    """
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
    guard_m: float,
    max_geometry_frames: int = 8,
    *,
    guard_left: bool = True,
    guard_right_arc: Optional[float] = None,
) -> Tuple[List[int], List[int]]:
    """Pick the chunk's lift frames and its joint-forward geometry frames.

    Guard is one-sided, facing the hole: holes are always interior blocks
    starting at chunk 1, so chunk 0 only ever needs a right-side guard
    (``guard_left=False, guard_right_arc=<first hole edge>``) while every
    other kept chunk only needs a left-side guard.  The same selection serves
    every hole pattern — a sparse condition that keeps this chunk observes
    frames that cannot see into the neighbouring hole.

    ``geometry``: the lift frames plus evenly-spread context frames from the
    rest of the chunk, capped at ``max_geometry_frames``.  This bounds the
    per-chunk VGGT view budget while guaranteeing every lift frame has
    cache depth (the forward covers it by construction).
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

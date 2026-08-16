"""Utilities for pairing anchor and target views within a batch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

VIEW_INDEX_TO_NAME: Dict[int, str] = {
    0: "front",
    1: "left_to_front_30",
    2: "right_to_front_30",
    3: "left_axis",
    4: "right_axis",
}

VIEW_NAME_TO_INDEX: Dict[str, int] = {v: k for k, v in VIEW_INDEX_TO_NAME.items()}

ANCHOR_VIEW_MAP: Dict[str, Optional[str]] = {
    "front": None,
    "left_to_front_30": "front",
    "right_to_front_30": "front",
    "left_axis": "left_to_front_30",
    "right_axis": "right_to_front_30",
}


@dataclass
class ViewPair:
    anchor_idx: int
    target_idx: int
    anchor_view_name: str
    target_view_name: str
    frame_id: int


def _group_indices_by_frame(frame_ids: torch.Tensor) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    if frame_ids is None:
        return groups
    for idx, fid in enumerate(frame_ids.tolist()):
        groups.setdefault(int(fid), []).append(idx)
    return groups


def group_and_pair_views(
    frame_ids: Optional[torch.Tensor],
    view_indices: Optional[torch.Tensor],
    view_names: Optional[Sequence[str]] = None,
) -> List[ViewPair]:
    """Return valid anchor-target pairs within the batch."""
    if frame_ids is None or view_indices is None:
        return []

    groups = _group_indices_by_frame(frame_ids)
    pairs: List[ViewPair] = []

    for fid, indices in groups.items():
        name_by_idx: Dict[int, str] = {}
        for batch_idx in indices:
            if view_names is not None:
                name = view_names[batch_idx]
            else:
                name = VIEW_INDEX_TO_NAME.get(int(view_indices[batch_idx].item()), "unknown")
            name_by_idx[batch_idx] = name

        # Reverse lookup: view name -> batch idx (prefer first occurrence)
        idx_by_view: Dict[str, int] = {}
        for batch_idx, name in name_by_idx.items():
            if name not in idx_by_view:
                idx_by_view[name] = batch_idx

        for target_name, anchor_name in ANCHOR_VIEW_MAP.items():
            if anchor_name is None:
                continue
            target_idx = idx_by_view.get(target_name)
            anchor_idx = idx_by_view.get(anchor_name)
            if target_idx is None or anchor_idx is None:
                continue
            pairs.append(
                ViewPair(
                    anchor_idx=anchor_idx,
                    target_idx=target_idx,
                    anchor_view_name=anchor_name,
                    target_view_name=target_name,
                    frame_id=fid,
                )
            )

    return pairs

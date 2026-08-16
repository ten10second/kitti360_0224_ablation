from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def get_condition_scale_sizes(rows: int, cols: int):
    return [("fine", rows, cols)]


def build_condition_tokens_with_coords(
    warped_front: Optional[torch.Tensor],
    warped_coords: Optional[torch.Tensor],
    warped_valid: Optional[torch.Tensor],
    target_rows: int,
    target_cols: int,
    device: torch.device,
    scale_sizes=None,
):
    if scale_sizes is None:
        scale_sizes = get_condition_scale_sizes(target_rows, target_cols)
    token_h = max(1, int(target_rows))
    token_w = max(1, int(target_cols))
    full_h = token_h * 16
    full_w = token_w * 16

    if warped_front is None:
        cond_img = torch.zeros((1, 3, full_h, full_w), device=device)
    else:
        cond_img = warped_front.to(device)
    cond_img = torch.clamp(cond_img, 0.0, 1.0)
    cond_img = cond_img * 2.0 - 1.0

    if warped_coords is None:
        coords = torch.zeros((1, 2, full_h, full_w), device=device)
    else:
        coords = warped_coords.to(device)
        coords = torch.where(coords < -1.1, torch.full_like(coords, -1.5), coords)

    if warped_valid is None:
        cond_mask = torch.zeros((1, 1, full_h, full_w), device=device)
    else:
        cond_mask = warped_valid.to(device).float()
        if cond_mask.shape[-2:] != (full_h, full_w):
            cond_mask = F.interpolate(cond_mask, size=(full_h, full_w), mode="nearest")

    pooled_rgb = F.avg_pool2d(cond_img, kernel_size=16, stride=16)
    pooled_coords = F.interpolate(coords, size=(token_h, token_w), mode="nearest")
    pooled_mask = F.avg_pool2d(cond_mask, kernel_size=16, stride=16)

    semantic_outputs = {}
    coord_outputs = {}
    for name, rows_s, cols_s in scale_sizes:
        rows_s = max(1, rows_s)
        cols_s = max(1, cols_s)
        semantic_s = torch.cat([pooled_rgb, pooled_mask], dim=1)
        semantic_s = F.adaptive_avg_pool2d(semantic_s, (rows_s, cols_s))
        coords_s = F.interpolate(pooled_coords, size=(rows_s, cols_s), mode="nearest")
        semantic_outputs[name] = semantic_s.squeeze(0)
        coord_outputs[name] = coords_s.squeeze(0)
    return semantic_outputs, coord_outputs


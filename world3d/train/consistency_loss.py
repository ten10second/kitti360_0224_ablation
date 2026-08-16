"""
Multi-view semantic consistency loss for Direct mode.

Computes consistency loss between different views of the same frame.
Uses BEV coordinate correspondences to find matching positions across views.
"""

from typing import Optional

import torch
import torch.nn.functional as F


FIRST_ORDER_VIEW_NEIGHBORS = {
    frozenset((0, 1)),  # front <-> left_to_front_30
    frozenset((0, 2)),  # front <-> right_to_front_30
    frozenset((1, 3)),  # left_to_front_30 <-> left_axis
    frozenset((2, 4)),  # right_to_front_30 <-> right_axis
}


def compute_multi_view_consistency_loss(
    sampled_bev_feat: torch.Tensor,
    semantic_feat: Optional[torch.Tensor],
    coords_map: torch.Tensor,
    frame_ids: torch.Tensor,
    view_indices: torch.Tensor,
    overlap_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.5,
    use_semantic: bool = False,  # False = BEV feature consistency (recommended)
    use_cosine: bool = True,
    nce_weight: float = 1.0,
) -> dict:
    """
    Compute multi-view semantic consistency loss.

    Args:
        sampled_bev_feat: (B, D, R, C) — Features sampled from BEV
        semantic_feat: (B, D, R, C) — IPM semantic features
        coords_map: (B, 2, R, C) — BEV coordinate mapping
        frame_ids: (B,) — Frame IDs (same frame has same ID)
        view_indices: (B,) — View indices (0-4 for five views)
        overlap_mask: (B, 1, R, C) or (B, R, C) — valid / overlap-only token mask
        temperature: float — Contrastive loss temperature
        use_semantic: bool — Use semantic features for consistency (else use sampled_bev)
        use_cosine: bool — Use positive-pair cosine alignment term
        nce_weight: float — Weight for InfoNCE term inside consistency loss

    Returns:
        dict:
            - loss: total consistency loss value (scalar)
            - cosine_loss: positive-pair cosine term
            - nce_loss: InfoNCE term before outer weighting
            - matched_pairs: total matched token count across valid view pairs
            - valid_view_pairs: number of view pairs contributing loss
            - overlap_ratio: matched tokens / valid overlap tokens
    """
    B, D, R, C = sampled_bev_feat.shape
    device = sampled_bev_feat.device

    # Choose features to use for consistency
    # Default: BEV feature consistency (Scheme A)
    if use_semantic:
        if semantic_feat is None:
            return {
                "loss": torch.tensor(0.0, device=device),
                "cosine_loss": torch.tensor(0.0, device=device),
                "nce_loss": torch.tensor(0.0, device=device),
                "matched_pairs": torch.tensor(0.0, device=device),
                "valid_view_pairs": torch.tensor(0.0, device=device),
                "overlap_ratio": torch.tensor(0.0, device=device),
            }
        feat = semantic_feat  # Scheme B: IPM semantic consistency
    else:
        feat = sampled_bev_feat  # Scheme A: BEV feature consistency (recommended)

    # Flatten features
    feat_flat = feat.permute(0, 2, 3, 1).reshape(B, R * C, D)  # (B, L, D)
    coords_flat = coords_map.permute(0, 2, 3, 1).reshape(B, R * C, 2)  # (B, L, 2)

    if overlap_mask is not None:
        if overlap_mask.dim() == 4:
            overlap_mask = overlap_mask[:, 0]
        overlap_mask = overlap_mask.reshape(B, R * C).bool()

    coord_valid = (coords_flat[..., 0] > -1.1) & (coords_flat[..., 1] > -1.1)
    if overlap_mask is not None:
        coord_valid = coord_valid & overlap_mask

    total_loss = torch.tensor(0.0, device=device)
    total_cosine_loss = torch.tensor(0.0, device=device)
    total_nce_loss = torch.tensor(0.0, device=device)
    num_pairs = 0
    total_matched_pairs = 0
    total_overlap_tokens = 0

    # Find unique frames in the batch
    if frame_ids is not None and frame_ids.numel() > 0:
        unique_frames = torch.unique(frame_ids)

        for frame_id in unique_frames:
            # Find all samples from this frame
            frame_mask = (frame_ids == frame_id)
            frame_indices = torch.where(frame_mask)[0]

            # Need at least 2 views to compute consistency
            if len(frame_indices) < 2:
                continue

            # Iterate all view pairs
            for i in range(len(frame_indices)):
                for j in range(i + 1, len(frame_indices)):
                    idx1 = frame_indices[i]
                    idx2 = frame_indices[j]

                    if view_indices is not None:
                        view_pair = frozenset((
                            int(view_indices[idx1].item()),
                            int(view_indices[idx2].item()),
                        ))
                        if view_pair not in FIRST_ORDER_VIEW_NEIGHBORS:
                            continue

                    valid_idx1 = torch.where(coord_valid[idx1])[0]
                    valid_idx2 = torch.where(coord_valid[idx2])[0]

                    if valid_idx1.numel() == 0 or valid_idx2.numel() == 0:
                        continue

                    total_overlap_tokens += int(min(valid_idx1.numel(), valid_idx2.numel()))

                    # Get features and coordinates for both views (valid/overlap-only tokens)
                    feat1 = feat_flat[idx1, valid_idx1]  # (L1, D)
                    feat2 = feat_flat[idx2, valid_idx2]  # (L2, D)
                    coords1 = coords_flat[idx1, valid_idx1]  # (L1, 2)
                    coords2 = coords_flat[idx2, valid_idx2]  # (L2, 2)

                    # Compute coordinate distance matrix
                    dist_matrix = torch.cdist(coords1, coords2, p=2)  # (L, L)

                    # Find nearest neighbor for each token
                    min_dist1, nn_idx1 = dist_matrix.min(dim=1)  # (L,)
                    min_dist2, nn_idx2 = dist_matrix.min(dim=0)  # (L,)

                    # Only keep matches that are close enough
                    # 0.05 normalized distance ≈ 2.5 meters in physical space
                    valid1 = (min_dist1 < 0.05)
                    valid2 = (min_dist2 < 0.05)

                    if not valid1.any() or not valid2.any():
                        continue

                    # Compute bidirectional valid matches (both tokens must match each other)
                    bidirectional_valid = []
                    for k in range(len(valid1)):
                        if valid1[k]:
                            m = nn_idx1[k]
                            if valid2[m] and nn_idx2[m] == k:
                                bidirectional_valid.append(k)
                    bidirectional_valid = torch.tensor(bidirectional_valid, device=device, dtype=torch.long)

                    if len(bidirectional_valid) == 0:
                        continue

                    total_matched_pairs += int(len(bidirectional_valid))

                    # Extract bidirectional valid features
                    feat1_bid = feat1[bidirectional_valid]
                    feat2_bid = feat2[nn_idx1[bidirectional_valid]]

                    # Normalize features
                    feat1_norm = F.normalize(feat1_bid, dim=-1)
                    feat2_norm = F.normalize(feat2_bid, dim=-1)

                    pair_loss = torch.tensor(0.0, device=device)
                    has_loss = False

                    if use_cosine:
                        loss_cos = 1.0 - (feat1_norm * feat2_norm).sum(dim=-1)
                        loss_cos_mean = loss_cos.mean()
                        pair_loss = pair_loss + loss_cos_mean
                        total_cosine_loss = total_cosine_loss + loss_cos_mean
                        has_loss = True

                    if nce_weight > 0.0 and len(bidirectional_valid) >= 2:
                        # Compute cosine similarities
                        sim_matrix = torch.matmul(feat1_norm, feat2_norm.T) / temperature

                        # Create labels for contrastive learning (positive pairs: diagonal elements)
                        labels = torch.arange(len(bidirectional_valid), device=device)

                        # InfoNCE loss
                        loss1 = F.cross_entropy(sim_matrix, labels)
                        loss2 = F.cross_entropy(sim_matrix.T, labels)
                        loss_nce = (loss1 + loss2) / 2
                        pair_loss = pair_loss + nce_weight * loss_nce
                        total_nce_loss = total_nce_loss + loss_nce
                        has_loss = True

                    if not has_loss:
                        continue

                    total_loss = total_loss + pair_loss
                    num_pairs += 1

    if num_pairs > 0:
        total_loss = total_loss / num_pairs
        total_cosine_loss = total_cosine_loss / num_pairs
        total_nce_loss = total_nce_loss / num_pairs

    overlap_ratio = 0.0
    if total_overlap_tokens > 0:
        overlap_ratio = float(total_matched_pairs) / float(total_overlap_tokens)

    return {
        "loss": total_loss,
        "cosine_loss": total_cosine_loss,
        "nce_loss": total_nce_loss,
        "matched_pairs": torch.tensor(float(total_matched_pairs), device=device),
        "valid_view_pairs": torch.tensor(float(num_pairs), device=device),
        "overlap_ratio": torch.tensor(float(overlap_ratio), device=device),
    }

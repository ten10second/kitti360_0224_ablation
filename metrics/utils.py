import torch
from torch import Tensor
from typing import Optional


def reduce_tensor(x: Tensor, reduction: str):
    assert reduction in ['mean', 'sum', 'none'], f"Reduction {reduction} not implemented"
    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()
    return x


def check_range(image: Tensor, low: float, high: float):
    return torch.ge(image, low).all() and torch.le(image, high).all()


def image_float_to_uint8(image: torch.Tensor):
    """ [0, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert check_range(image, 0, 1)
    return (image * 255).to(dtype=torch.uint8)


def align_depth_scale(depth_pred: Tensor, depth_gt: Tensor, mask: Optional[Tensor] = None):
    """
    Align predicted depth to ground truth depth using median scaling (for relative depth)
    Args:
        depth_pred: (B, 1, H, W) predicted relative depth
        depth_gt: (B, 1, H, W) ground truth depth (or reference depth)
        mask: (B, 1, H, W) optional mask of valid pixels

    Returns:
        depth_pred_aligned: (B, 1, H, W) scale-aligned depth
    """
    B = depth_pred.shape[0]
    depth_pred_aligned = torch.zeros_like(depth_pred)

    for i in range(B):
        pred = depth_pred[i, 0]
        gt = depth_gt[i, 0]

        if mask is not None:
            valid = mask[i, 0] > 0
        else:
            valid = torch.ones_like(pred, dtype=torch.bool)

        valid = valid & torch.isfinite(pred) & torch.isfinite(gt)
        valid = valid & (torch.abs(pred) > 1e-6) & (torch.abs(gt) > 1e-6)

        pred_valid = pred[valid]
        gt_valid = gt[valid]

        if pred_valid.numel() < 10 or gt_valid.numel() < 10:
            depth_pred_aligned[i, 0] = pred
            continue

        # Median scaling factor
        pred_median = torch.median(pred_valid)
        gt_median = torch.median(gt_valid)
        if (not torch.isfinite(pred_median)) or (not torch.isfinite(gt_median)) or torch.abs(pred_median) < 1e-6:
            depth_pred_aligned[i, 0] = pred
            continue
        scale = gt_median / pred_median
        depth_pred_aligned[i, 0] = pred * scale

    return depth_pred_aligned

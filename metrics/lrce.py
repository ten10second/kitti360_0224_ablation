import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Literal
from .utils import reduce_tensor, check_range


class LRCE:
    """
    Low Resolution Correlation Error metric.
    Calculates MSE or normalized cross-correlation between downsampled versions of two images.

    Args:
        reduction: 'mean', 'sum', or 'none'
        downsample_scales: list of downsampling scales to use
        metric_type: 'mse' (lower = more similar) or 'ncc' (normalized cross correlation, higher = more similar)
        downsample_mode: 'bilinear', 'bicubic', or 'area'
    """
    def __init__(
        self,
        reduction: str = 'mean',
        downsample_scales: List[float] = [2.0, 4.0, 8.0],
        metric_type: Literal['mse', 'ncc'] = 'mse',
        downsample_mode: str = 'bilinear',
    ):
        self.reduction = reduction
        self.downsample_scales = downsample_scales
        self.metric_type = metric_type
        self.downsample_mode = downsample_mode

    def _downsample(self, img: Tensor, scale: float) -> Tensor:
        """ Downsample image by given scale """
        B, C, H, W = img.shape
        new_h, new_w = int(H / scale), int(W / scale)
        return F.interpolate(
            img,
            size=(new_h, new_w),
            mode=self.downsample_mode,
            align_corners=False
        )

    def _compute_metric(self, img1: Tensor, img2: Tensor) -> Tensor:
        """ Compute metric between two images """
        if self.metric_type == 'mse':
            return F.mse_loss(img1, img2, reduction='none').mean(dim=[1, 2, 3])
        elif self.metric_type == 'ncc':
            # Normalized cross correlation
            mean1 = img1.mean(dim=[1, 2, 3], keepdim=True)
            mean2 = img2.mean(dim=[1, 2, 3], keepdim=True)
            var1 = ((img1 - mean1) ** 2).mean(dim=[1, 2, 3])
            var2 = ((img2 - mean2) ** 2).mean(dim=[1, 2, 3])
            covar = ((img1 - mean1) * (img2 - mean2)).mean(dim=[1, 2, 3])
            ncc = covar / (torch.sqrt(var1 * var2) + 1e-8)
            return ncc
        else:
            raise ValueError(f"Unknown metric type {self.metric_type}")

    def __call__(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        Args:
            img1: Tensor of shape (B, 3, H, W) and dtype float32 in range [0, 1]
            img2: Tensor of shape (B, 3, H, W) and dtype float32 in range [0, 1]

        Returns:
            if reduction is 'none', returns a Tensor of shape (B, )
            else, returns a scalar Tensor
        """
        assert img1.shape == img2.shape
        assert img1.device == img2.device
        assert check_range(img1, 0, 1) and check_range(img2, 0, 1)

        B = img1.shape[0]
        scores = torch.zeros(B, len(self.downsample_scales), device=img1.device)

        for i, scale in enumerate(self.downsample_scales):
            img1_ds = self._downsample(img1, scale)
            img2_ds = self._downsample(img2, scale)
            scores[:, i] = self._compute_metric(img1_ds, img2_ds)

        # Average across all scales
        avg_scores = scores.mean(dim=1)

        return reduce_tensor(avg_scores, self.reduction)

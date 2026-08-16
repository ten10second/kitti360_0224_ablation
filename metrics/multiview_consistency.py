import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal

from .utils import reduce_tensor, check_range, align_depth_scale
from .lpips import LPIPS
from .dino_similarity import DINOSimilarity
from .depth_consistency import DepthConsistency


def get_overlap_mask(
    depth_src: Tensor,
    K_src: Tensor,
    K_tgt: Tensor,
    T_src_to_tgt: Tensor,
    img_shape: tuple
) -> Tensor:
    """
    Calculate overlap mask between source and target view using depth and camera parameters.
    """
    B, _, H, W = depth_src.shape
    device = depth_src.device

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    pixels = pixels.view(1, H * W, 3).repeat(B, 1, 1)

    K_src_inv = torch.inverse(K_src)
    cam_points = torch.bmm(K_src_inv, pixels.transpose(1, 2))
    cam_points = cam_points * depth_src.view(B, 1, H * W)
    cam_points_h = torch.cat([cam_points, torch.ones(B, 1, H * W, device=device)], dim=1)

    cam_points_tgt = torch.bmm(T_src_to_tgt, cam_points_h)[:, :3, :]
    points_tgt = torch.bmm(K_tgt, cam_points_tgt)
    z_tgt = points_tgt[:, 2, :].clamp(min=1e-6)
    points_tgt = points_tgt[:, :2, :] / z_tgt.unsqueeze(1)
    points_tgt = points_tgt.transpose(1, 2)

    x_tgt = points_tgt[..., 0]
    y_tgt = points_tgt[..., 1]
    valid_x = (x_tgt >= 0) & (x_tgt < W)
    valid_y = (y_tgt >= 0) & (y_tgt < H)
    valid_z = z_tgt > 0
    valid = valid_x & valid_y & valid_z

    return valid.view(B, 1, H, W)


def warp_image(
    img_src: Tensor,
    depth_src: Tensor,
    K_src: Tensor,
    K_tgt: Tensor,
    T_src_to_tgt: Tensor,
    mode: Literal['bilinear', 'nearest'] = 'bilinear',
) -> Tensor:
    """
    Warp source image to target view using depth and camera parameters.
    """
    B, _, H, W = img_src.shape
    device = img_src.device

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels_tgt = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    pixels_tgt = pixels_tgt.view(1, H * W, 3).repeat(B, 1, 1)

    K_tgt_inv = torch.inverse(K_tgt)
    cam_points_tgt = torch.bmm(K_tgt_inv, pixels_tgt.transpose(1, 2))

    T_tgt_to_src = torch.inverse(T_src_to_tgt)
    cam_points_tgt_h = torch.cat([cam_points_tgt, torch.ones(B, 1, H * W, device=device)], dim=1)
    cam_points_src = torch.bmm(T_tgt_to_src, cam_points_tgt_h)[:, :3, :]

    points_src = torch.bmm(K_src, cam_points_src)
    z_src = points_src[:, 2, :].clamp(min=1e-6)
    points_src = points_src[:, :2, :] / z_src.unsqueeze(1)

    points_src[:, 0, :] = 2 * points_src[:, 0, :] / (W - 1) - 1
    points_src[:, 1, :] = 2 * points_src[:, 1, :] / (H - 1) - 1
    grid = points_src.transpose(1, 2).view(B, H, W, 2)

    img_warped = F.grid_sample(
        img_src,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=False,
    )

    valid = (
        (grid[..., 0] >= -1) & (grid[..., 0] <= 1) &
        (grid[..., 1] >= -1) & (grid[..., 1] <= 1) &
        (z_src.view(B, H, W) > 0)
    )
    valid_mask = valid.unsqueeze(1)

    return img_warped, valid_mask


class MultiViewConsistency:
    """
    Multi-view consistency metric for overlapping regions between two views.
    Supports depth consistency and warping-based semantic/perceptual metrics.
    """
    def __init__(
        self,
        reduction: str = 'mean',
        metric_type: Literal['depth_consistency', 'warp_lpips', 'warp_dino', 'warp_seg_miou'] = 'depth_consistency',
        model_type: str = 'MiDaS_small',
        dino_model_name: str = 'facebook/dinov2-small',
        sam_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.reduction = reduction
        self.metric_type = metric_type
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        if metric_type == 'depth_consistency':
            self.depth_model = DepthConsistency(reduction='none', model_type=model_type, device=device)
        elif metric_type == 'warp_lpips':
            self.lpips = LPIPS(reduction='none').to(self.device)
            self.depth_model = DepthConsistency(reduction='none', model_type=model_type, device=device)
        elif metric_type == 'warp_dino':
            self.dino = DINOSimilarity(
                reduction='none',
                similarity_type='cosine',
                model_name=dino_model_name,
                device=device,
            )
            self.depth_model = DepthConsistency(reduction='none', model_type=model_type, device=device)
        elif metric_type == 'warp_seg_miou':
            try:
                from .segany_consistency import SegAnyConsistency
            except ImportError as e:
                raise ImportError(
                    'warp_seg_miou requires segment_anything; please install it and provide --sam-checkpoint if needed.'
                ) from e

            self.segany = SegAnyConsistency(
                reduction='none',
                metric_type='miou',
                sam_checkpoint=sam_checkpoint,
                device=device,
            )
            self.depth_model = DepthConsistency(reduction='none', model_type=model_type, device=device)
        else:
            raise ValueError(f'Unknown metric type {metric_type}')

    @staticmethod
    def _compute_masked_miou(seg1: Tensor, seg2: Tensor, valid_mask: Tensor) -> Tensor:
        labels1 = seg1[valid_mask]
        labels2 = seg2[valid_mask]

        classes = torch.unique(torch.cat([labels1, labels2], dim=0))
        ious = []
        for cls in classes:
            if cls.item() == 0:
                continue
            cls1 = labels1 == cls
            cls2 = labels2 == cls
            union = (cls1 | cls2).sum()
            if union.item() == 0:
                continue
            inter = (cls1 & cls2).sum()
            ious.append(inter.float() / union.float())

        if len(ious) == 0:
            return torch.tensor(float('nan'), device=seg1.device)
        return torch.stack(ious).mean()

    @staticmethod
    def _compute_masked_local_cosine(feat1: Tensor, feat2: Tensor, valid_mask: Tensor) -> Tensor:
        """Compute cosine similarity only on valid local feature positions."""
        if feat1.shape != feat2.shape:
            raise ValueError(f'Feature shapes must match, got {tuple(feat1.shape)} vs {tuple(feat2.shape)}')
        if valid_mask.dim() != 4 or valid_mask.size(1) != 1:
            raise ValueError(f'valid_mask must be (B,1,H,W), got {tuple(valid_mask.shape)}')

        mask_feat = F.interpolate(valid_mask.float(), size=feat1.shape[-2:], mode='nearest') > 0.5
        if mask_feat.sum() < 1:
            return torch.tensor(float('nan'), device=feat1.device)

        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        cos_map = (feat1 * feat2).sum(dim=1, keepdim=True)
        cos_valid = cos_map[mask_feat]
        return ((cos_valid.mean()) + 1.0) * 0.5

    def __call__(
        self,
        img_src: Tensor,
        img_tgt: Tensor,
        K_src: Tensor,
        K_tgt: Tensor,
        T_src_to_tgt: Tensor,
        depth_src: Optional[Tensor] = None,
        depth_tgt: Optional[Tensor] = None,
        overlap_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert img_src.shape == img_tgt.shape
        assert img_src.device == img_tgt.device
        assert check_range(img_src, 0, 1) and check_range(img_tgt, 0, 1)

        B, _, H, W = img_src.shape

        if depth_src is None:
            depth_src = self.depth_model.predict_depth(img_src)
        if depth_tgt is None:
            depth_tgt = self.depth_model.predict_depth(img_tgt)

        if overlap_mask is None:
            overlap_mask = get_overlap_mask(depth_src, K_src, K_tgt, T_src_to_tgt, (H, W))

        scores = []
        for i in range(B):
            mask = overlap_mask[i:i + 1]
            if mask.sum() < 10:
                scores.append(torch.tensor(float('nan'), device=img_src.device))
                continue

            if self.metric_type == 'depth_consistency':
                d1_aligned = align_depth_scale(depth_src[i:i + 1], depth_tgt[i:i + 1], mask)
                rmse = torch.sqrt(((d1_aligned[mask] - depth_tgt[i:i + 1][mask]) ** 2).mean())
                scores.append(rmse)

            elif self.metric_type == 'warp_lpips':
                img_warped, warp_mask = warp_image(
                    img_src[i:i + 1], depth_src[i:i + 1],
                    K_src[i:i + 1], K_tgt[i:i + 1], T_src_to_tgt[i:i + 1],
                )
                combined_mask = mask & warp_mask
                if combined_mask.sum() < 10:
                    scores.append(torch.tensor(float('nan'), device=img_src.device))
                    continue
                lpips_val = self.lpips(
                    img_warped * combined_mask.float(),
                    img_tgt[i:i + 1] * combined_mask.float(),
                )
                scores.append(lpips_val[0])

            elif self.metric_type == 'warp_dino':
                img_warped, warp_mask = warp_image(
                    img_src[i:i + 1], depth_src[i:i + 1],
                    K_src[i:i + 1], K_tgt[i:i + 1], T_src_to_tgt[i:i + 1],
                )
                combined_mask = mask & warp_mask
                if combined_mask.sum() < 10:
                    scores.append(torch.tensor(float('nan'), device=img_src.device))
                    continue

                feat_warped = self.dino.extract_patch_features(img_warped)
                feat_tgt = self.dino.extract_patch_features(img_tgt[i:i + 1])
                dino_val = self._compute_masked_local_cosine(feat_warped, feat_tgt, combined_mask)
                scores.append(dino_val)

            elif self.metric_type == 'warp_seg_miou':
                seg_src_np = self.segany.generate_masks(img_src[i:i + 1])
                seg_tgt_np = self.segany.generate_masks(img_tgt[i:i + 1])

                seg_src = torch.from_numpy(seg_src_np).to(img_src.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                seg_tgt = torch.from_numpy(seg_tgt_np).to(img_src.device, dtype=torch.long)

                seg_warped, warp_mask = warp_image(
                    seg_src,
                    depth_src[i:i + 1],
                    K_src[i:i + 1],
                    K_tgt[i:i + 1],
                    T_src_to_tgt[i:i + 1],
                    mode='nearest',
                )

                combined_mask = (mask & warp_mask).squeeze(0).squeeze(0)
                if combined_mask.sum() < 10:
                    scores.append(torch.tensor(float('nan'), device=img_src.device))
                    continue

                seg_warped_label = seg_warped.squeeze(0).squeeze(0).round().to(torch.long)
                miou = self._compute_masked_miou(seg_warped_label, seg_tgt, combined_mask)
                scores.append(miou)

        scores = torch.stack(scores)
        return reduce_tensor(scores, self.reduction)

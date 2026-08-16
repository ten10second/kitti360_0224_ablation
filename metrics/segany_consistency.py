import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from .utils import reduce_tensor, check_range, image_float_to_uint8


class SegAnyConsistency:
    """
    Semantic segmentation consistency metric using SAM (Segment Anything Model).
    Calculates mIoU between segmentation masks generated from two input images.

    Args:
        reduction: 'mean', 'sum', or 'none'
        metric_type: 'miou' (mean Intersection over Union) or 'pixel_acc' (pixel accuracy)
        model_type: SAM model type, 'vit_b', 'vit_l', 'vit_h'
        sam_checkpoint: path to SAM checkpoint file, if None will attempt to download
        device: device to run model on, defaults to 'cuda' if available
    """
    def __init__(
        self,
        reduction: str = 'mean',
        metric_type: Literal['miou', 'pixel_acc'] = 'miou',
        model_type: str = 'vit_b',
        sam_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.reduction = reduction
        self.metric_type = metric_type
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load SAM model
        if sam_checkpoint is None:
            # Default checkpoint paths, you may need to adjust these
            checkpoint_paths = {
                'vit_b': 'sam_vit_b_01ec64.pth',
                'vit_l': 'sam_vit_l_0b3195.pth',
                'vit_h': 'sam_vit_h_4b8939.pth',
            }
            sam_checkpoint = checkpoint_paths[model_type]

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)

        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    @torch.no_grad()
    def generate_masks(self, img: Tensor) -> np.ndarray:
        """ Generate segmentation masks from input image """
        # Convert to uint8 numpy array
        img_np = image_float_to_uint8(img.squeeze(0)).permute(1, 2, 0).cpu().numpy()

        # Generate masks
        masks = self.mask_generator.generate(img_np)

        # Combine masks into a single segmentation map
        h, w = img_np.shape[:2]
        seg_map = np.zeros((h, w), dtype=np.int32)

        for i, mask in enumerate(masks, 1):
            seg_map[mask['segmentation']] = i

        return seg_map

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
        assert img1.shape[0] == 1, "Batch processing not supported for SegAnyConsistency yet"

        # Generate segmentation masks
        seg1 = self.generate_masks(img1)  # (H, W)
        seg2 = self.generate_masks(img2)  # (H, W)

        # Calculate metric
        if self.metric_type == 'miou':
            # Compute IoU for each class and average
            classes = np.unique(np.concatenate([seg1, seg2]))
            ious = []
            for cls in classes:
                if cls == 0:
                    continue  # skip background
                mask1 = seg1 == cls
                mask2 = seg2 == cls
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                if union > 0:
                    ious.append(intersection / union)
            score = np.mean(ious) if ious else 0.0
        elif self.metric_type == 'pixel_acc':
            # Pixel accuracy
            score = np.mean(seg1 == seg2)
        else:
            raise ValueError(f"Unknown metric type {self.metric_type}")

        score = torch.tensor(score, device=img1.device).unsqueeze(0)
        return reduce_tensor(score, self.reduction)

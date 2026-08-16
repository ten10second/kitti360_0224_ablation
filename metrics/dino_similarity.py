import math
from typing import Optional, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

from .utils import reduce_tensor, check_range


class DINOSimilarity:
    """
    DINOv2 feature similarity metric.

    By default, __call__ compares global CLS features. For multi-view consistency,
    extract_patch_features() exposes local patch descriptors so callers can compute
    masked local feature similarity instead of relying on whole-image embeddings.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        similarity_type: Literal['cosine', 'l2'] = 'cosine',
        model_name: str = 'facebook/dinov2-small',
        device: Optional[str] = None,
    ):
        self.reduction = reduction
        self.similarity_type = similarity_type
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.patch_size = getattr(self.model.config, 'patch_size', None)

    def _preprocess(self, img: Tensor) -> Tensor:
        if img.dim() != 4 or img.size(1) != 3:
            raise ValueError(f"Expected image tensor of shape (B, 3, H, W), got {tuple(img.shape)}")
        pixel_values = self.processor(
            images=img,
            return_tensors='pt',
            do_rescale=False,
        ).pixel_values.to(self.device)
        return pixel_values

    @staticmethod
    def _infer_patch_grid(num_tokens: int, pixel_values: Tensor, patch_size: Optional[int]) -> Tuple[int, int]:
        if patch_size is not None and patch_size > 0:
            grid_h = pixel_values.shape[-2] // patch_size
            grid_w = pixel_values.shape[-1] // patch_size
            if grid_h * grid_w == num_tokens:
                return int(grid_h), int(grid_w)

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Cannot infer patch grid from {num_tokens} patch tokens")
        return side, side

    @torch.no_grad()
    def extract_features(self, img: Tensor) -> Tensor:
        """Extract global CLS features from input image."""
        assert check_range(img, 0, 1)
        pixel_values = self._preprocess(img)
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def extract_patch_features(self, img: Tensor) -> Tensor:
        """Extract local patch features as a spatial feature map (B, C, Hf, Wf)."""
        assert check_range(img, 0, 1)
        pixel_values = self._preprocess(img)
        outputs = self.model(pixel_values)
        tokens = outputs.last_hidden_state[:, 1:, :]  # drop CLS token
        grid_h, grid_w = self._infer_patch_grid(tokens.shape[1], pixel_values, self.patch_size)
        patch_feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w)
        return patch_feat

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

        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        if self.similarity_type == 'cosine':
            sim = F.cosine_similarity(feat1, feat2, dim=1)
            sim = (sim + 1) / 2
        elif self.similarity_type == 'l2':
            sim = torch.norm(feat1 - feat2, p=2, dim=1)
        else:
            raise ValueError(f"Unknown similarity type {self.similarity_type}")

        return reduce_tensor(sim, self.reduction)

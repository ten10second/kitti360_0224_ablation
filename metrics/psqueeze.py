import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import squeezenet1_1
from typing import Optional
from .utils import reduce_tensor, check_range


class P_Squeeze:
    """
    SqueezeNet perceptual feature distance metric.
    Extracts features from SqueezeNet conv5 layer and calculates L2 distance.

    Args:
        reduction: 'mean', 'sum', or 'none'
        device: device to run model on, defaults to 'cuda' if available
    """
    def __init__(
        self,
        reduction: str = 'mean',
        device: Optional[str] = None,
    ):
        self.reduction = reduction
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load SqueezeNet and extract conv5 features
        self.model = squeezenet1_1(pretrained=True).features
        self.model.to(self.device)
        self.model.eval()

        # Normalization parameters for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @torch.no_grad()
    def extract_features(self, img: Tensor) -> Tensor:
        """ Extract conv5 features from SqueezeNet """
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        # Normalize image for ImageNet
        img = (img - self.mean) / self.std

        # Extract features up to conv5 layer
        x = img
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == 12:  # conv5 layer output
                break

        # Global average pooling
        feat = F.adaptive_avg_pool2d(x, (1, 1)).view(B, -1)
        return feat

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

        # Extract features
        feat1 = self.extract_features(img1)  # (B, D)
        feat2 = self.extract_features(img2)  # (B, D)

        # Calculate L2 distance
        dist = torch.norm(feat1 - feat2, p=2, dim=1)

        return reduce_tensor(dist, self.reduction)

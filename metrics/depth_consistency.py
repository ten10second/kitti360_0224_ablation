from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal
from .utils import reduce_tensor, check_range, align_depth_scale


def _find_cached_torch_hub_repo(repo_name: str) -> Optional[Path]:
    """Return local torch hub cache dir for a GitHub repo if available."""
    hub_dir = Path(torch.hub.get_dir())
    repo_key = repo_name.replace("/", "_")
    candidates = sorted(hub_dir.glob(f"{repo_key}_*"))
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "hubconf.py").exists():
            return candidate
    return None


@contextmanager
def _prefer_local_torch_hub_cache():
    """Monkeypatch torch.hub.load to use local cached repos before GitHub."""
    original_load = torch.hub.load

    def cached_first_load(repo_or_dir, model, *args, **kwargs):
        if isinstance(repo_or_dir, str) and kwargs.get("source") != "local":
            cached_repo = _find_cached_torch_hub_repo(repo_or_dir)
            if cached_repo is not None:
                kwargs = dict(kwargs)
                kwargs["source"] = "local"
                repo_or_dir = str(cached_repo)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = cached_first_load
    try:
        yield
    finally:
        torch.hub.load = original_load


class DepthConsistency:
    """
    Depth consistency metric using MiDaS pre-trained relative depth model.
    Calculates error between depth maps predicted from two input images after scale alignment.

    Args:
        reduction: 'mean', 'sum', or 'none'
        metric_type: 'rmse', 'abs_rel', 'sq_rel', 'delta1', 'delta2', 'delta3'
        model_type: MiDaS model type, 'DPT_Large', 'DPT_BEiT_L_384', 'MiDaS_small'
        device: device to run model on, defaults to 'cuda' if available
    """
    def __init__(
        self,
        reduction: str = 'mean',
        metric_type: Literal['rmse', 'abs_rel', 'sq_rel', 'delta1', 'delta2', 'delta3'] = 'rmse',
        model_type: str = 'MiDaS_small',
        device: Optional[str] = None,
    ):
        self.reduction = reduction
        self.metric_type = metric_type
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_eps = 1e-6
        self.outlier_quantile = 0.01

        # Load MiDaS model and transforms, preferring local torch hub cache to avoid
        # per-process GitHub resolution when cache is already available.
        with _prefer_local_torch_hub_cache():
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type, verbose=False)
            self.midas.to(self.device)
            self.midas.eval()

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_BEiT_L_384":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def predict_depth(self, img: Tensor) -> Tensor:
        """ Predict relative depth from input image """
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        # Convert to numpy for MiDaS transforms
        img_np = img.permute(0, 2, 3, 1).cpu().numpy() * 255  # [0, 1] -> [0, 255]
        img_np = img_np.astype('uint8')

        depth_maps = []
        for im in img_np:
            input_batch = self.transform(im).to(self.device)
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                prediction = self._sanitize_depth_map(prediction)
            depth_maps.append(prediction)

        depth = torch.stack(depth_maps).unsqueeze(1)  # (B, 1, H, W)
        return depth.to(self.device)

    def _sanitize_depth_map(self, depth: Tensor) -> Tensor:
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        finite = torch.isfinite(depth)
        if finite.sum() < 10:
            return torch.full_like(depth, self.depth_eps)

        valid_vals = depth[finite]
        if valid_vals.numel() >= 32:
            q = float(self.outlier_quantile)
            lower = torch.quantile(valid_vals, q)
            upper = torch.quantile(valid_vals, 1.0 - q)
            depth = depth.clamp(min=lower, max=upper)

        min_val = depth.min()
        if not torch.isfinite(min_val):
            return torch.full_like(depth, self.depth_eps)
        if min_val <= 0:
            depth = depth - min_val + self.depth_eps

        depth = torch.nan_to_num(depth, nan=self.depth_eps, posinf=self.depth_eps, neginf=self.depth_eps)
        return depth

    def _build_valid_mask(self, d1: Tensor, d2: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            valid = mask > 0
        else:
            valid = torch.ones_like(d1, dtype=torch.bool)

        valid = valid & torch.isfinite(d1) & torch.isfinite(d2)
        valid = valid & (torch.abs(d1) > self.depth_eps) & (torch.abs(d2) > self.depth_eps)

        if valid.sum() < 10:
            return valid

        d1_valid = d1[valid]
        d2_valid = d2[valid]
        if d1_valid.numel() >= 32:
            q = float(self.outlier_quantile)
            d1_low = torch.quantile(d1_valid, q)
            d1_high = torch.quantile(d1_valid, 1.0 - q)
            d2_low = torch.quantile(d2_valid, q)
            d2_high = torch.quantile(d2_valid, 1.0 - q)
            valid = valid & (d1 >= d1_low) & (d1 <= d1_high) & (d2 >= d2_low) & (d2 <= d2_high)

        return valid

    def __call__(self, img1: Tensor, img2: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            img1: Tensor of shape (B, 3, H, W) and dtype float32 in range [0, 1]
            img2: Tensor of shape (B, 3, H, W) and dtype float32 in range [0, 1]
            mask: Optional Tensor of shape (B, 1, H, W) with boolean values indicating valid pixels

        Returns:
            if reduction is 'none', returns a Tensor of shape (B, )
            else, returns a scalar Tensor
        """
        assert img1.shape == img2.shape
        assert img1.device == img2.device
        assert check_range(img1, 0, 1) and check_range(img2, 0, 1)

        # Predict depth maps
        depth1 = self.predict_depth(img1)  # (B, 1, H, W)
        depth2 = self.predict_depth(img2)  # (B, 1, H, W)

        # Align depth scales (relative depth to relative depth alignment)
        depth1_aligned = align_depth_scale(depth1, depth2, mask)

        # Compute error
        error_list = []
        for i in range(img1.shape[0]):
            d1 = depth1_aligned[i, 0]
            d2 = depth2[i, 0]

            if mask is not None:
                valid = self._build_valid_mask(d1, d2, mask[i, 0])
            else:
                valid = self._build_valid_mask(d1, d2)

            if valid.sum() < 10:
                error_list.append(torch.tensor(float('nan'), device=img1.device))
                continue

            d1_valid = d1[valid]
            d2_valid = d2[valid]

            if self.metric_type == 'rmse':
                rmse = torch.sqrt(torch.mean((d1_valid - d2_valid) ** 2))
                error_list.append(rmse)
            elif self.metric_type == 'abs_rel':
                abs_rel = torch.mean(torch.abs(d1_valid - d2_valid) / d2_valid.clamp(min=self.depth_eps))
                error_list.append(abs_rel)
            elif self.metric_type == 'sq_rel':
                sq_rel = torch.mean(((d1_valid - d2_valid) ** 2) / d2_valid.clamp(min=self.depth_eps))
                error_list.append(sq_rel)
            elif self.metric_type.startswith('delta'):
                delta = float(self.metric_type[5:])
                thresh = torch.max((d2_valid / d1_valid.clamp(min=self.depth_eps)), (d1_valid / d2_valid.clamp(min=self.depth_eps)))
                delta_acc = (thresh < delta).float().mean()
                error_list.append(1 - delta_acc)  # return error instead of accuracy
            else:
                raise ValueError(f"Unknown metric type {self.metric_type}")

        errors = torch.tensor(error_list, device=img1.device)
        return reduce_tensor(errors, self.reduction)

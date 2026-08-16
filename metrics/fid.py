import torch
from torch import Tensor
from pytorch_fid.fid_score import calculate_fid_given_paths, calculate_frechet_distance, compute_statistics_of_path
from pathlib import Path
import os
import tempfile
from PIL import Image
import numpy as np
import shutil
import gc


def tensor_to_pil(img_tensor: Tensor) -> Image.Image:
    """Convert tensor in [0, 1] range to PIL Image."""
    arr = img_tensor.cpu().float().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_images_to_temp_dir(images: Tensor, temp_dir: str, start_idx: int = 0):
    """Save images to temporary directory for FID calculation with starting index."""
    os.makedirs(temp_dir, exist_ok=True)
    for i, img in enumerate(images):
        pil_img = tensor_to_pil(img)
        pil_img.save(os.path.join(temp_dir, f"{start_idx + i:06d}.png"))


class FID:
    def __init__(self, device: str = "cuda", chunk_size: int = 100):
        self.device = device
        self.chunk_size = chunk_size  # Number of images per chunk
        self.temp_dir1 = tempfile.mkdtemp()
        self.temp_dir2 = tempfile.mkdtemp()
        self.inception_model = None
        self.mu1 = None
        self.sigma1 = None
        self.mu2 = None
        self.sigma2 = None

    def __del__(self):
        if os.path.exists(self.temp_dir1):
            shutil.rmtree(self.temp_dir1)
        if os.path.exists(self.temp_dir2):
            shutil.rmtree(self.temp_dir2)
        if self.inception_model is not None:
            del self.inception_model
            gc.collect()
            torch.cuda.empty_cache()

    def _compute_statistics_in_chunks(self, images: Tensor):
        """Compute FID statistics by processing images in chunks to reduce memory usage."""
        # Initialize statistics
        mu_total = torch.zeros(2048, device=self.device)
        sigma_total = torch.zeros(2048, 2048, device=self.device)
        n_total = 0

        # Process images in chunks
        for i in range(0, images.shape[0], self.chunk_size):
            # Get chunk of images
            chunk = images[i:i + self.chunk_size]

            # Clear temporary directories for this chunk
            if os.path.exists(self.temp_dir1):
                shutil.rmtree(self.temp_dir1)
            os.makedirs(self.temp_dir1)

            # Save images to temporary directory
            save_images_to_temp_dir(chunk, self.temp_dir1)

            # Calculate statistics for this chunk
            from pytorch_fid.inception import InceptionV3
            if self.inception_model is None:
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                self.inception_model = InceptionV3([block_idx]).to(self.device)

            mu, sigma = compute_statistics_of_path(
                self.temp_dir1,
                self.inception_model,
                self.chunk_size,  # Use chunk size for batch
                self.device,
                dims=2048,
                num_workers=2,
            )

            # Accumulate statistics
            mu = torch.tensor(mu, device=self.device)
            sigma = torch.tensor(sigma, device=self.device)
            n = chunk.shape[0]

            if n_total == 0:
                mu_total = mu * n
                sigma_total = sigma * n
            else:
                mu_total = mu_total + mu * n
                sigma_total = sigma_total + sigma * n

            n_total += n

            # Clear cache
            torch.cuda.empty_cache()

        # Compute final statistics
        mu_total /= n_total
        sigma_total /= n_total

        # Add epsilon to diagonal for numerical stability
        sigma_total += torch.eye(sigma_total.shape[0], device=self.device) * 1e-6

        return mu_total.cpu().numpy(), sigma_total.cpu().numpy()

    def __call__(self, img1: Tensor, img2: Tensor):
        """
        Calculate FID between two sets of images using chunked processing.

        Args:
            img1: Tensor of shape (B, C, H, W) and dtype float32 in range [0, 1]
            img2: Tensor of shape (B, C, H, W) and dtype float32 in range [0, 1]

        Returns:
            FID score (scalar)
        """
        assert img1.shape == img2.shape
        assert img1.device == img2.device
        assert img1.dtype == torch.float32
        assert img2.dtype == torch.float32
        assert img1.min() >= 0 and img1.max() <= 1
        assert img2.min() >= 0 and img2.max() <= 1

        print(f"Processing {img1.shape[0]} images in chunks of {self.chunk_size}")

        # Compute statistics for both image sets in chunks
        mu1, sigma1 = self._compute_statistics_in_chunks(img1)
        mu2, sigma2 = self._compute_statistics_in_chunks(img2)

        # Calculate FID from precomputed statistics
        fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        # Clear cache again
        torch.cuda.empty_cache()

        return fid_score

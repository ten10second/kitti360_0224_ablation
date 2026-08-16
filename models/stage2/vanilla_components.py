import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaPositionEncoder(nn.Module):
    """Standard 2D Sine/Cosine positional encoding for tokens."""

    def __init__(self, d_model: int, max_h: int = 256, max_w: int = 512, temperature: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        self.temperature = temperature

    def forward(self, y_coords: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_coords: (B, L) normalized y coordinates [0, 1]
            x_coords: (B, L) normalized x coordinates [0, 1]
        Returns:
            pe: (B, L, d_model) position encoding
        """
        B, L = y_coords.shape
        assert x_coords.shape == (B, L)

        pe = torch.zeros(B, L, self.d_model, device=y_coords.device)

        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2, device=y_coords.device) *
                           (-math.log(self.temperature) / (self.d_model // 2)))

        for i in range(self.d_model // 2):
            if i % 2 == 0:
                pe[:, :, i] = torch.sin(y_coords * self.max_h * div_term[i // 2])
                pe[:, :, i + self.d_model // 2] = torch.cos(y_coords * self.max_h * div_term[i // 2])
            else:
                pe[:, :, i] = torch.sin(x_coords * self.max_w * div_term[i // 2])
                pe[:, :, i + self.d_model // 2] = torch.cos(x_coords * self.max_w * div_term[i // 2])

        return pe


class VanillaPoseProjector(nn.Module):
    """Project camera pose to 1D token for vanilla AR."""

    def __init__(self, pose_dim: int, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.pose_dim = pose_dim
        self.d_model = d_model

        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (B, pose_dim) camera pose
        Returns:
            pose_token: (B, 1, d_model) projected pose token
        """
        B = pose.shape[0]
        pose_token = self.mlp(pose).unsqueeze(1)  # (B, 1, d_model)
        return pose_token

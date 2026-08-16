import math
import torch
import torch.nn as nn


class RayCoordinateEncoder(nn.Module):
    """
    Ray/Coordinate encoder with Fourier Feature Mapping (FFM) + MLP.

    Inputs:
      - normalized_coords: (B, H, W, 2), in [-1, 1], where [..., 0] = u_norm, [..., 1] = v_norm
    Outputs:
      - coord_embeds: (B, H, W, out_dim)

    Args:
      out_dim: output embedding dimension (e.g., D_model)
      fourier_bands: number of frequency bands for FFM (2^k, k=0..fourier_bands-1)
      include_input: whether to concatenate the raw (u,v) with FFM features
      mlp_layers: number of linear layers after FFM (>=1)
      hidden_dim: hidden size for MLP
    """

    def __init__(
        self,
        out_dim: int = 512,
        fourier_bands: int = 8,
        include_input: bool = True,
        mlp_layers: int = 2,
        hidden_dim: int = 512,
    ):
        super().__init__()
        assert mlp_layers >= 1, "mlp_layers must be >= 1"
        self.fourier_bands = int(fourier_bands)
        self.include_input = bool(include_input)

        ffm_dim = 4 * self.fourier_bands  # [sin/cos u, sin/cos v] per band
        in_dim = ffm_dim + (2 if self.include_input else 0)

        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

        # 输出归一化：让 E_coord 的数值范围稳定在 std≈1
        # 配合 coord_pe_scale=0.02 使用，使 E_coord 和 E_pos (~0.02 std) 同量级
        self.output_norm = nn.LayerNorm(out_dim)

    @torch.no_grad()
    def _ffm(self, coords_uv: torch.Tensor) -> torch.Tensor:
        """
        Fourier feature mapping for (u,v) in [-1,1].
        coords_uv: (B, H, W, 2)
        returns: (B, H, W, 4*fourier_bands)  -> [sin(w*u), cos(w*u), sin(w*v), cos(w*v)] for w=2^k*pi
        """
        B, H, W, _ = coords_uv.shape
        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        outs = []
        for k in range(self.fourier_bands):
            w = (2.0 ** k) * math.pi
            outs.append(torch.sin(w * u))
            outs.append(torch.cos(w * u))
            outs.append(torch.sin(w * v))
            outs.append(torch.cos(w * v))
        return torch.stack(outs, dim=-1)  # (B,H,W,4*bands)

    def forward(self, normalized_coords: torch.Tensor) -> torch.Tensor:
        """
        normalized_coords: (B, H, W, 2) in [-1,1]
        returns: (B, H, W, out_dim)
        """
        assert normalized_coords.dim() == 4 and normalized_coords.size(-1) == 2, \
            "normalized_coords must be (B,H,W,2)"
        ffm = self._ffm(normalized_coords)
        if self.include_input:
            x = torch.cat([normalized_coords, ffm], dim=-1)
        else:
            x = ffm
        B, H, W, C = x.shape
        x = x.view(B * H * W, C)
        x = self.net(x)
        x = self.output_norm(x)  # 归一化输出，控制数值范围
        x = x.view(B, H, W, -1)
        return x


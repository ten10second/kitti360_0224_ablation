"""
Simplified Single-Scale Token Predictor for BEV-conditioned Generation

Architecture:
    Input: Token indices (B, L)
    ↓
    Token Embedding + Positional Embedding + Viewpoint Embedding
    ↓
    Conditions:
    - Semantic: IPM RGB + mask → ResNet → (B, D, H, W)
    - Geometric: BEV coords → Fourier → (B, D, H, W)
    - BEV: Satellite → CNN → grid_sample → (B, D, H', W')
    ↓
    Fused Condition Memory: (B, H*W + H'*W', D)
    ↓
    Transformer Decoder (8 layers, all use same memory):
    - Self-Attention (causal)
    - Cross-Attention (to condition memory)
    - FFN
    ↓
    Output: Logits (B, L, vocab_size)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ResNet50 BEV Encoder
from ..multiscale_vit_encoder import MultiScaleViTBEVEncoder

def grid_to_seq_bottomup(grid: torch.Tensor) -> torch.Tensor:
    """
    Bottom-up raster order:
      rows: bottom -> top
      cols: left -> right within each row

    Args:
        grid: (B, R, C) or (R, C)

    Returns:
        seq: (B, R*C) or (R*C,)
    """
    if grid.dim() == 2:
        # (R,C) -> (R*C,)
        return grid.flip(0).reshape(-1)
    if grid.dim() == 3:
        # (B,R,C) -> (B,R*C)
        return grid.flip(1).reshape(grid.shape[0], -1)
    raise ValueError(f"grid_to_seq_bottomup expects (R,C) or (B,R,C), got {tuple(grid.shape)}")


def seq_to_grid_bottomup(seq: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Inverse of grid_to_seq_bottomup.
    Given a sequence in bottom-up order, reshape back to a NORMAL top-down grid.

    Args:
        seq: (B, L) or (L,)
        rows, cols: target grid size

    Returns:
        grid_topdown: (B, R, C) or (R, C)
    """
    L = rows * cols
    if seq.dim() == 1:
        if seq.numel() != L:
            raise ValueError(f"seq length {seq.numel()} != rows*cols {L}")
        return seq.view(rows, cols).flip(0)
    if seq.dim() == 2:
        if seq.shape[1] != L:
            raise ValueError(f"seq length {seq.shape[1]} != rows*cols {L}")
        return seq.view(seq.shape[0], rows, cols).flip(1)
    raise ValueError(f"seq_to_grid_bottomup expects (L,) or (B,L), got {tuple(seq.shape)}")


class CacheableAttention(nn.Module):
    """Wrapper for nn.MultiheadAttention with KV caching and RoPE support for fast inference.

    RoPE is applied AFTER Q/K projection, inside this module, to ensure it works correctly
    with PyTorch's multihead attention.
    """

    def __init__(self, attn_module: nn.MultiheadAttention, apply_rope: bool = False):
        super().__init__()
        self.attn = attn_module
        self.head_dim = attn_module.embed_dim // attn_module.num_heads
        self.num_heads = attn_module.num_heads
        self.apply_rope = apply_rope

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        rope_q_cos: Optional[torch.Tensor] = None,
        rope_q_sin: Optional[torch.Tensor] = None,
        rope_k_cos: Optional[torch.Tensor] = None,
        rope_k_sin: Optional[torch.Tensor] = None,
        need_weights: bool = True,
    ):
        is_cross_attention = key.shape[1] != query.shape[1]
        B, L_q, D = query.shape
        present_kv = None

        use_self_rope = self.apply_rope and (not is_cross_attention) and rope_cos is not None and rope_sin is not None
        use_cross_rope = is_cross_attention and rope_q_cos is not None and rope_q_sin is not None and rope_k_cos is not None and rope_k_sin is not None

        # Apply RoPE AFTER projection, before attention computation
        # This is the correct way - RoPE should rotate the projected Q/K, not raw hidden states
        if use_self_rope or use_cross_rope:
            if self.attn.in_proj_weight is not None:
                w_q, w_k, w_v = self.attn.in_proj_weight.chunk(3)
                if self.attn.in_proj_bias is not None:
                    b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3)
                else:
                    b_q = b_k = b_v = None
            else:
                w_q = self.attn.q_proj_weight
                w_k = self.attn.k_proj_weight
                w_v = self.attn.v_proj_weight
                if self.attn.in_proj_bias is not None:
                    b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3)
                else:
                    b_q = b_k = b_v = None

            q = F.linear(query, w_q, b_q)
            q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)

            if is_cross_attention and use_cache and past_kv is not None:
                k, v = past_kv
            else:
                k_current = F.linear(key, w_k, b_k)
                v_current = F.linear(value, w_v, b_v)
                k_current = k_current.view(B, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
                v_current = v_current.view(B, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

                if use_self_rope:
                    # Self-attention RoPE: use same rope for query/key
                    cos_exp = rope_cos.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
                    sin_exp = rope_sin.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
                    q = q * cos_exp + rotate_half(q) * sin_exp
                    k_current = k_current * cos_exp + rotate_half(k_current) * sin_exp
                elif use_cross_rope:
                    # Cross-attention RoPE: query/key can use different geometry sources
                    L_k_current = k_current.shape[-2]
                    cos_k_exp = rope_k_cos.reshape(B, L_k_current, self.num_heads, self.head_dim).transpose(1, 2)
                    sin_k_exp = rope_k_sin.reshape(B, L_k_current, self.num_heads, self.head_dim).transpose(1, 2)
                    k_current = k_current * cos_k_exp + rotate_half(k_current) * sin_k_exp

                if use_cache and past_kv is not None:
                    past_k, past_v = past_kv
                    k = torch.cat([past_k, k_current], dim=2)
                    v = torch.cat([past_v, v_current], dim=2)
                else:
                    k = k_current
                    v = v_current

            if use_cross_rope:
                # If using cached cross K/V, query still needs RoPE every step
                cos_q_exp = rope_q_cos.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
                sin_q_exp = rope_q_sin.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
                q = q * cos_q_exp + rotate_half(q) * sin_q_exp

            if use_cache:
                present_kv = (k, v)

            L_k = k.shape[-2]
            attn_bias = None

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    mask_tensor = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    if attn_mask.size(0) == B * self.num_heads:
                        mask_tensor = attn_mask.view(B, self.num_heads, L_q, L_k)
                    elif attn_mask.size(0) == B:
                        mask_tensor = attn_mask.unsqueeze(1)
                    else:
                        mask_tensor = attn_mask.unsqueeze(0)
                else:
                    mask_tensor = attn_mask

                if mask_tensor.dtype == torch.bool:
                    mask_bias = torch.zeros_like(mask_tensor, dtype=q.dtype)
                    mask_bias = mask_bias.masked_fill(mask_tensor, torch.finfo(q.dtype).min)
                else:
                    mask_bias = mask_tensor.to(dtype=q.dtype)
                attn_bias = mask_bias

            if key_padding_mask is not None:
                if key_padding_mask.dtype == torch.bool:
                    kpm_bias = torch.zeros((B, 1, 1, L_k), device=q.device, dtype=q.dtype)
                    kpm_bias = kpm_bias.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(q.dtype).min)
                else:
                    kpm_bias = key_padding_mask[:, None, None, :].to(dtype=q.dtype)
                attn_bias = kpm_bias if attn_bias is None else (attn_bias + kpm_bias)

            dropout_p = self.attn.dropout if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).reshape(B, L_q, D)
            attn_output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)

            return attn_output, present_kv

        # Handle caching
        if use_cache:
            if past_kv is not None:
                if is_cross_attention:
                    key, value = past_kv
                else:
                    past_k, past_v = past_kv
                    key = torch.cat([past_k, key], dim=1)
                    value = torch.cat([past_v, value], dim=1)
            present_kv = (key, value)

        # Standard path without RoPE or for cross-attention
        attn_output, attn_weights = self.attn(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights
        )

        # Save attention weights for visualization (cross-attention only)
        if is_cross_attention:
            self._last_attn_weights = attn_weights.detach()

        return attn_output, present_kv


class ResidualBlock(nn.Module):
    """Residual block for semantic encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(16, channels)  # Use GroupNorm instead of BatchNorm for DDP
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, channels)
        self.relu = nn.ReLU(inplace=False)  # Changed to False for DDP compatibility

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class SemanticEncoder(nn.Module):
    """Lightweight ResNet-style encoder for semantic (RGB+mask) inputs."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128, num_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(16, hidden_channels),  # Use GroupNorm instead of BatchNorm for DDP
            nn.ReLU(inplace=False),  # Changed to False for DDP compatibility
        )
        blocks = []
        for _ in range(max(1, num_blocks)):
            blocks.append(ResidualBlock(hidden_channels))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class MultiModalFusionGate(nn.Module):
    """Gated fusion module to balance multi-modal features.

    Problem: Different modalities have very different magnitude:
    Solution: Learn adaptive gates to balance their contribution.
    """

    def __init__(self, d_model: int, num_modalities: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities

        # Per-modality normalization (similar to LayerNorm but per-modality)
        self.modality_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_modalities)
        ])

        # Gating network: learn importance weights for each modality
        # Input: concatenated features from all modalities
        # Output: num_modalities weights (will be softmax-ed)
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.ReLU(inplace=False),
            nn.Linear(d_model, num_modalities),
        )

        # Initialize gate to equal weights
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.zeros_(self.gate_network[-1].bias)

    def forward(self, *features):
        """
        Args:
            *features: Variable number of feature tensors, each (B, L, D) or (B, D, H, W)
        Returns:
            fused: (B, L, D) or (B, D, H, W) - same shape as input
        """
        assert len(features) == self.num_modalities, \
            f"Expected {self.num_modalities} features, got {len(features)}"

        # Determine if input is spatial (B, D, H, W) or sequential (B, L, D)
        is_spatial = features[0].dim() == 4

        if is_spatial:
            # Convert to (B, L, D) for processing
            B, D, H, W = features[0].shape
            features = [f.flatten(2).transpose(1, 2) for f in features]  # (B, H*W, D)

        B, L, D = features[0].shape

        # Step 1: Normalize each modality independently
        normalized_features = []
        for i, feat in enumerate(features):
            # Reshape for LayerNorm: (B, L, D) -> (B*L, D) -> LayerNorm -> (B, L, D)
            feat_flat = feat.reshape(B * L, D)
            feat_normed = self.modality_norms[i](feat_flat).reshape(B, L, D)
            normalized_features.append(feat_normed)

        # Step 2: Compute gating weights
        # Pool each feature to (B, D) for gate computation
        pooled_features = [f.mean(dim=1) for f in normalized_features]  # each (B, D)
        concat_pooled = torch.cat(pooled_features, dim=1)  # (B, D * num_modalities)

        gate_logits = self.gate_network(concat_pooled)  # (B, num_modalities)
        gates = torch.softmax(gate_logits, dim=1)  # (B, num_modalities)

        # Step 3: Weighted sum with learned gates
        fused = torch.zeros_like(normalized_features[0])  # (B, L, D)
        for i, feat in enumerate(normalized_features):
            weight = gates[:, i].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            fused = fused + weight * feat

        # Convert back to spatial if needed
        if is_spatial:
            fused = fused.transpose(1, 2).reshape(B, D, H, W)

        return fused, gates  # Return gates for monitoring


class FourierCoordEncoder(nn.Module):
    """Fourier feature encoding for BEV coordinates.

    Encodes normalized BEV coordinates (u, v) ∈ [-1, 1] into high-dimensional features
    using sinusoidal functions with multiple frequency bands.

    Inspired by NeRF's positional encoding:
        γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^(L-1) π p), cos(2^(L-1) π p)]

    This provides:
    - Multi-scale position information (low freq = coarse, high freq = fine)
    - Smooth interpolation between positions
    - Better representation than simple linear projection
    """

    def __init__(self, d_model: int = 512, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model

        # Frequency bands: [2^0, 2^1, ..., 2^(num_freqs-1)]
        # Each coordinate (u, v) will be encoded with all these frequencies
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freq_bands', freq_bands)

        # Input dimension after Fourier encoding:
        # 2 coords * 2 (sin/cos) * num_freqs = 4 * num_freqs
        fourier_dim = 4 * num_freqs

        # MLP to project Fourier features to d_model
        # Use multiple layers to capture non-linear patterns
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(inplace=False),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=False),
        )

        # Initialize MLP weights
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2, H, W) normalized BEV coordinates in [-1, 1]
                    coords[:, 0, :, :] = u (horizontal)
                    coords[:, 1, :, :] = v (vertical)
        Returns:
            features: (B, d_model, H, W) encoded coordinate features
        """
        B, C, H, W = coords.shape
        assert C == 2, "Coords should have 2 channels (u, v)"

        # Reshape: (B, 2, H, W) -> (B, H, W, 2)
        coords = coords.permute(0, 2, 3, 1)

        # Apply frequency bands: (B, H, W, 2, 1) * (num_freqs,) -> (B, H, W, 2, num_freqs)
        coords_expanded = coords.unsqueeze(-1)  # (B, H, W, 2, 1)
        freqs = self.freq_bands.view(1, 1, 1, 1, -1)  # (1, 1, 1, 1, num_freqs)
        coords_freq = coords_expanded * freqs * math.pi  # (B, H, W, 2, num_freqs)

        # Apply sin and cos
        coords_sin = torch.sin(coords_freq)  # (B, H, W, 2, num_freqs)
        coords_cos = torch.cos(coords_freq)  # (B, H, W, 2, num_freqs)

        # Concatenate: (B, H, W, 2, num_freqs, 2) -> (B, H, W, 4*num_freqs)
        coords_encoded = torch.cat([coords_sin, coords_cos], dim=-1)  # (B, H, W, 2, 2*num_freqs)
        coords_encoded = coords_encoded.reshape(B, H, W, -1)  # (B, H, W, 4*num_freqs)

        # Apply MLP: (B, H, W, 4*num_freqs) -> (B, H, W, d_model)
        features = self.mlp(coords_encoded)

        # Permute back: (B, H, W, d_model) -> (B, d_model, H, W)
        features = features.permute(0, 3, 1, 2)

        return features


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions, for RoPE."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


class RayRoPEEncoder(nn.Module):
    """Ray-aware Rotary Position Encoding (RayRoPE) using world coordinate ray angles.

    Encodes horizontal and vertical ray directions in world coordinate system
    using rotary position encoding, providing geometric invariance across views.
    """

    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model % 4 != 0:
            raise ValueError(f"RayRoPEEncoder requires d_model divisible by 4, got {self.d_model}")
        # Split dimension equally between horizontal (alpha) and vertical (beta) angles
        self.angle_dim = self.d_model // 2
        freqs = base ** (-2.0 * torch.arange(self.angle_dim // 2, dtype=torch.float32) / float(self.angle_dim))
        self.register_buffer("freqs", freqs, persistent=False)

        # Reuse the same index to rc conversion logic
        self._seq_index_to_rc = RayDirectionEncoder._seq_index_to_rc

    def forward(
        self,
        K: torch.Tensor,
        T_cam_to_world: torch.Tensor,
        past_length: int,
        L_gen: int,
        rows: int,
        cols: int,
        ntp_order: str,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            K: (B, 3, 3) camera intrinsics
            T_cam_to_world: (B, 4, 4) camera to world transformation matrix
            past_length: int, length of already generated sequence
            L_gen: int, number of tokens to generate
            rows: int, target grid rows
            cols: int, target grid cols
            ntp_order: str, token ordering
            device: torch device

        Returns:
            cos: (B, L_gen, D) cosine terms for RoPE rotation
            sin: (B, L_gen, D) sine terms for RoPE rotation
        """
        if K.dim() == 2:
            K = K.unsqueeze(0)
        if T_cam_to_world.dim() == 2:
            T_cam_to_world = T_cam_to_world.unsqueeze(0)

        B = K.size(0)
        idx = torch.arange(past_length, past_length + L_gen, device=device)
        r, c = self._seq_index_to_rc(idx, rows=rows, cols=cols, ntp_order=ntp_order)
        r = r.float()
        c = c.float()

        # Step 1: Compute pixel coordinates in actual image pixel coordinates
        # Each token corresponds to 16x16 pixels in the rendered image, compute center pixel
        # Same logic as original RayDirectionEncoder for correct ray calculation
        img_h = rows * 16  # total image height
        img_w = cols * 16  # total image width

        # Convert token grid coordinate -> center pixel of each 16x16 token
        # Keep explicit batch axis to avoid fragile implicit broadcasting.
        u = (c * 16 + 8.0).unsqueeze(0).expand(B, -1)  # (B, L_gen) - center pixel x
        v = (r * 16 + 8.0).unsqueeze(0).expand(B, -1)  # (B, L_gen) - center pixel y

        # Step 2: Camera coordinate ray direction (unproject)
        fx = K[:, 0, 0].view(B, 1)  # (B, 1)
        fy = K[:, 1, 1].view(B, 1)
        cx = K[:, 0, 2].view(B, 1)
        cy = K[:, 1, 2].view(B, 1)

        # Convert to camera coordinate directions
        x_cam = (u - cx) / fx  # (B, L_gen)
        y_cam = (v - cy) / fy  # (B, L_gen)
        z_cam = torch.ones_like(x_cam)

        d_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (B, L_gen, 3)
        d_cam = F.normalize(d_cam, dim=-1)

        # Step 3: Transform to world coordinate system
        R = T_cam_to_world[:, :3, :3]  # (B, 3, 3)
        d_world = torch.bmm(R, d_cam.transpose(1, 2)).transpose(1, 2)  # (B, L_gen, 3)
        d_world = F.normalize(d_world, dim=-1)

        # Step 4: Compute world coordinate angles
        alpha = torch.atan2(d_world[..., 0], d_world[..., 2])  # (B, L_gen) horizontal yaw
        beta = torch.asin(torch.clamp(d_world[..., 1], -1.0, 1.0))  # (B, L_gen) vertical pitch

        # Step 5: Compute RoPE frequencies
        freqs = self.freqs.to(device=device, dtype=alpha.dtype).clone()  # (angle_dim//2,)

        # Compute rotation terms for alpha and beta
        alpha_freqs = alpha[..., None] * freqs[None, None, :]  # (B, L_gen, angle_dim//2)
        beta_freqs = beta[..., None] * freqs[None, None, :]    # (B, L_gen, angle_dim//2)

        # Interleave cos and sin terms
        alpha_cos = alpha_freqs.cos().repeat_interleave(2, dim=-1)  # (B, L_gen, angle_dim)
        alpha_sin = alpha_freqs.sin().repeat_interleave(2, dim=-1)  # (B, L_gen, angle_dim)

        beta_cos = beta_freqs.cos().repeat_interleave(2, dim=-1)  # (B, L_gen, angle_dim)
        beta_sin = beta_freqs.sin().repeat_interleave(2, dim=-1)  # (B, L_gen, angle_dim)

        # Combine into full rotation terms
        cos = torch.cat([alpha_cos, beta_cos], dim=-1)  # (B, L_gen, D)
        sin = torch.cat([alpha_sin, beta_sin], dim=-1)  # (B, L_gen, D)

        return cos, sin


# Keep old RayDirectionEncoder for backward compatibility during transition
class RayDirectionEncoder(nn.Module):
    """Encode per-token ray directions (in camera frame) into d_model features.
    DEPRECATED: Use RayRoPEEncoder instead.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=False),
            nn.Linear(d_model, d_model),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _seq_index_to_rc(idx: torch.Tensor, rows: int, cols: int, ntp_order: str) -> Tuple[torch.Tensor, torch.Tensor]:
        c = idx % cols
        if ntp_order == "bottomup":
            r = (rows - 1) - (idx // cols)
        else:
            r = idx // cols
        return r.long(), c.long()

    def forward(self, K: torch.Tensor, past_length: int, L_gen: int, rows: int, cols: int, ntp_order: str, device: torch.device):
        if K.dim() == 2:
            K = K.unsqueeze(0)
        if K.dim() != 3 or K.size(-1) != 3 or K.size(-2) != 3:
            raise ValueError(f"K must be (B,3,3) or (3,3), got {tuple(K.shape)}")

        B = K.size(0)
        idx = torch.arange(past_length, past_length + L_gen, device=device)
        r, c = self._seq_index_to_rc(idx, rows=rows, cols=cols, ntp_order=ntp_order)

        # token center pixel coords (u,v) in pixel space
        # token grid: (rows, cols) corresponds to image pixels (rows*16, cols*16)
        u = (c.float() * 16.0 + 8.0)  # (L,)
        v = (r.float() * 16.0 + 8.0)  # (L,)
        ones = torch.ones_like(u)
        pix = torch.stack([u, v, ones], dim=-1)  # (L,3)
        pix = pix.unsqueeze(0).expand(B, -1, -1)  # (B,L,3)

        Kinv = torch.inverse(K.to(device))  # (B,3,3)
        d = torch.bmm(pix, Kinv.transpose(1, 2))  # (B,L,3)
        d = F.normalize(d, dim=-1, eps=1e-6)  # unit ray direction in camera frame

        ray_pe = self.mlp(d)  # (B,L,d_model)
        return ray_pe


def build_2d_sincos_pe(h: int, w: int, d_model: int) -> torch.Tensor:
    """Fixed 2D sine-cosine positional encoding, relative to grid center.

    Coordinates: each cell (i,j) maps to:
      y_rel = (i - (h-1)/2) / ((h-1)/2)  in [-1, 1]  (row, top-to-bottom)
      x_rel = (j - (w-1)/2) / ((w-1)/2)  in [-1, 1]  (col, left-to-right)

    Encoding: d_model/2 dims for y, d_model/2 dims for x.
    Each half uses interleaved sin/cos at exponentially spaced frequencies.

    Returns: (1, d_model, h, w) — fixed, non-trainable
    """
    half_d = d_model // 2
    y_coords = (torch.arange(h, dtype=torch.float32) - (h - 1) / 2) / ((h - 1) / 2)  # [-1, 1]
    x_coords = (torch.arange(w, dtype=torch.float32) - (w - 1) / 2) / ((w - 1) / 2)  # [-1, 1]
    gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (h, w)

    dim_t = torch.arange(0, half_d, 2, dtype=torch.float32)  # 0,2,4,...
    freq = 1.0 / (10000.0 ** (dim_t / half_d))  # exponential frequencies

    # y encoding: (h, w, half_d)
    pe_y_sin = torch.sin(gy.unsqueeze(-1) * freq)  # (h, w, half_d/2)
    pe_y_cos = torch.cos(gy.unsqueeze(-1) * freq)
    pe_y = torch.stack([pe_y_sin, pe_y_cos], dim=-1).flatten(-2)  # (h, w, half_d)

    # x encoding: (h, w, half_d)
    pe_x_sin = torch.sin(gx.unsqueeze(-1) * freq)
    pe_x_cos = torch.cos(gx.unsqueeze(-1) * freq)
    pe_x = torch.stack([pe_x_sin, pe_x_cos], dim=-1).flatten(-2)  # (h, w, half_d)

    pe = torch.cat([pe_y, pe_x], dim=-1)  # (h, w, d_model)
    return pe.permute(2, 0, 1).unsqueeze(0)  # (1, d_model, h, w)


class PoseRouteCrossAttn(nn.Module):
    """Pose-routed cross-attention over BEV features with geometric queries.

    Key design:
    - Q anchors: 64 FOV-fan-shaped spatial anchors, rotated/translated by camera pose,
      encoded with the SAME sincos PE as K. This gives Q-K dot product natural spatial
      selectivity that tracks camera FOV.
    - K/V split: PE only on K (routing), V remains pure semantic content.
    - learned_slots kept as residual for flexibility beyond strict geometry.

    Q = W_q(sincos_PE(pose_transformed_anchors) + learned_slots + pose_emb)
    K = W_k(bev_feat + sincos_PE)
    V = W_v(bev_feat)
    attn_bias = bias_conv(visibility_mask)
    """

    def __init__(self, d_model: int = 512, n_queries: int = 64, nhead: int = 8,
                 pose_dim: int = 13, bev_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.bev_size = bev_size

        # Pose projection: 13D → D (global conditioning)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=False),
            nn.Linear(d_model, d_model),
        )

        # Learned query slots (kept as residual for flexibility)
        self.query_slots = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)

        # FOV fan-shaped anchor points (fixed buffer)
        base_anchors = self._build_fan_anchors(n_queries)       # (N_q, 2)
        self.register_buffer('base_anchors', base_anchors)

        # Learnable anchor scale (controls spatial spread)
        self.anchor_scale = nn.Parameter(torch.tensor(1.0))

        # Q/K/V projections (separate, because K≠V)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # BEV grid coordinates in [-1, 1] for position bias (Gaussian kernel)
        cell_idx = torch.arange(bev_size, dtype=torch.float32)
        y_coords = (cell_idx - (bev_size - 1) / 2) / ((bev_size - 1) / 2)  # [-1, 1]
        x_coords = (cell_idx - (bev_size - 1) / 2) / ((bev_size - 1) / 2)
        gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_xy = torch.stack([gx, gy], dim=-1).reshape(1, bev_size * bev_size, 2)  # (1, 4096, 2)
        self.register_buffer('grid_xy', grid_xy)

        # Geometric attention bias from visibility mask
        self.bias_conv = nn.Sequential(
            nn.Conv2d(1, nhead, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(nhead, nhead, kernel_size=1),
        )

        # Per-head Gaussian width: small σ for fine-grained, large σ for broad context
        # sigma clamped to [0.05, 0.6]: too small → hard mask, too large → no spatial selectivity
        self.log_sigma = nn.Parameter(
            torch.linspace(math.log(0.08), math.log(0.5), nhead)
        )  # (nhead,)

        # Learnable strength of position bias relative to content scores
        # Starts small (0.3) so content matching dominates early, geometry grows as needed
        self.pos_alpha = nn.Parameter(torch.tensor(0.3))

        # Post-attention: LayerNorm + FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self._init_weights()

    @staticmethod
    def _build_fan_anchors(n_queries: int = 64) -> torch.Tensor:
        """Build FOV fan-shaped anchor points in polar coordinates.

        Layout: 8 angular bins x 8 radial bins = 64 anchors.
        Angular range: [-60deg, +60deg] (slightly wider than typical 80deg HFOV).
        Radial range: [0.08, 0.7] in BEV-normalized units (~4m to ~36m).
        Reference forward direction: +y (rotated to actual camera direction at runtime).

        Returns:
            anchors: (N_q, 2) in Cartesian, centered at origin
        """
        n_angles = int(math.sqrt(n_queries))
        n_radii = n_queries // n_angles
        # Ensure we get exactly n_queries points
        while n_angles * n_radii != n_queries:
            n_angles -= 1
            n_radii = n_queries // n_angles

        angles = torch.linspace(-math.pi / 3, math.pi / 3, n_angles)  # [-60deg, +60deg]
        radii = torch.linspace(0.12, 0.74, n_radii)                     # near to far
        grid_a, grid_r = torch.meshgrid(angles, radii, indexing='ij')   # (n_angles, n_radii)
        x = grid_r * torch.sin(grid_a)                                  # lateral
        y = grid_r * torch.cos(grid_a)                                  # forward
        return torch.stack([x, y], dim=-1).reshape(n_queries, 2)

    def _pose_to_bev_anchors(self, pose_vec: torch.Tensor) -> torch.Tensor:
        """Transform base fan anchors using camera pose to BEV coordinates.

        Extracts camera forward direction and position from pose_vec,
        rotates and translates the anchors to align with the actual FOV.

        BEV coordinate convention (matching build_2d_sincos_pe):
          x_rel: [-1, 1], left-to-right (east)
          y_rel: [-1, 1], top-to-bottom (south)

        Args:
            pose_vec: (B, 13) = [rot6d(6), t_rel(3), intr(4)]

        Returns:
            anchors: (B, N_q, 2) in [-1, 1] BEV-normalized coordinates
        """
        B = pose_vec.size(0)
        device = pose_vec.device

        # 1. Recover R_cam_to_world from rot6d
        rot6d = pose_vec[:, :6]                                          # (B, 6)
        col0 = F.normalize(rot6d[:, :3], dim=-1, eps=1e-6)              # (B, 3)
        col1_raw = rot6d[:, 3:6]
        # Gram-Schmidt orthogonalization
        col1 = col1_raw - (col1_raw * col0).sum(-1, keepdim=True) * col0
        col1 = F.normalize(col1, dim=-1, eps=1e-6)                      # (B, 3)
        col2 = torch.cross(col0, col1, dim=-1)                          # (B, 3) = forward

        # 2. Camera forward direction projected to BEV plane
        # World: X=east, Y=north, Z=up
        # BEV:   x_rel=east (right), y_rel=south (down) = -north
        fwd_bev_x = col2[:, 0]                                          # east component
        fwd_bev_y = -col2[:, 1]                                         # -north = south
        fwd_norm = torch.sqrt(fwd_bev_x ** 2 + fwd_bev_y ** 2).clamp(min=1e-6)
        fwd_bev_x = fwd_bev_x / fwd_norm
        fwd_bev_y = fwd_bev_y / fwd_norm

        # 3. Build 2D rotation matrix (rotates reference +y to camera forward)
        # We want: R maps anchor's +y axis to (fwd_bev_x, fwd_bev_y)
        # R = [[fwd_y, fwd_x], [-fwd_x, fwd_y]] so that R @ (0,1)^T = (fwd_x, fwd_y)
        cos_theta = fwd_bev_y                                           # (B,)
        sin_theta = fwd_bev_x                                           # (B,)
        R_2d = torch.stack([
            torch.stack([cos_theta, sin_theta], dim=-1),
            torch.stack([-sin_theta, cos_theta], dim=-1),
        ], dim=-2)                                                       # (B, 2, 2)

        # 4. Camera offset in BEV-normalized coordinates
        # t_rel = t_cam - t_imu in world frame
        # PE grid: x_rel = (j - 31.5)/31.5, physical = (j*8+4 - 256)*0.2
        # So x_rel = physical / ((bev_size-1)/2 * 8 * 0.2) = physical / 50.4
        t_rel = pose_vec[:, 6:9]                                         # (B, 3)
        bev_half_extent = (self.bev_size - 1) / 2.0 * 8.0 * 0.2        # 50.4m
        cam_bev_x = t_rel[:, 0] / bev_half_extent                       # east
        cam_bev_y = -t_rel[:, 1] / bev_half_extent                      # -north = south
        cam_offset = torch.stack([cam_bev_x, cam_bev_y], dim=-1)        # (B, 2)

        # 5. Transform anchors: scale → rotate → translate
        scale = self.anchor_scale.clamp(min=0.1)
        anchors = self.base_anchors.unsqueeze(0).expand(B, -1, -1)      # (B, N_q, 2)
        anchors = anchors * scale
        anchors = torch.bmm(anchors, R_2d.transpose(1, 2))              # (B, N_q, 2)
        anchors = anchors + cam_offset.unsqueeze(1)                      # (B, N_q, 2)

        # 6. Soft clamp to [-1, 1] via tanh (differentiable)
        anchors = torch.tanh(anchors)

        return anchors

    def _init_weights(self):
        for m in self.pose_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for lin in [self.W_q, self.W_k, self.W_v, self.out_proj]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for m in self.bias_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pose_vec: torch.Tensor, bev_feat: torch.Tensor,
                bev_vis_mask: torch.Tensor) -> torch.Tensor:
        """
        Content-Position decoupled attention with Gaussian position bias:
          content_scores = W_q(slots + pose) @ W_k(bev_feat)^T / √d_k
          pos_bias       = -||anchor_i - grid_j||^2 / (2σ²)   (Gaussian kernel)
          attn_scores    = content_scores + pos_bias + b_geom

        Args:
            pose_vec: (B, pose_dim)
            bev_feat: (B, D, H_bev, W_bev) — from bev_proj, NO PE added
            bev_vis_mask: (B, 1, H_bev, W_bev)

        Returns:
            C_bev: (B, N_q, D)
        """
        B = pose_vec.size(0)
        N_q = self.query_slots.size(1)  # 64
        N_kv = bev_feat.shape[2] * bev_feat.shape[3]  # 4096

        # --- Geometric anchors ---
        transformed_anchors = self._pose_to_bev_anchors(pose_vec)         # (B, 64, 2)
        self._last_anchors = transformed_anchors.detach()                  # for visualization

        # --- Content Q: learned_slots + pose (NO PE) ---
        pose_emb = self.pose_proj(pose_vec)                                # (B, D)
        q_input = self.query_slots.expand(B, -1, -1) + pose_emb.unsqueeze(1)  # (B, 64, D)
        Q = self.W_q(q_input)                                             # (B, 64, D)

        # --- Content K: bev_feat only (NO grid PE) ---
        kv_input = bev_feat.flatten(2).transpose(1, 2)                    # (B, 4096, D)
        K = self.W_k(kv_input)                                            # (B, 4096, D)

        # --- V: bev_feat only (unchanged) ---
        V = self.W_v(kv_input)                                            # (B, 4096, D)

        # --- Position bias: Gaussian kernel on L2 distance ---
        # anchor: (B, 64, 2), grid: (1, 4096, 2) -> dist_sq: (B, 64, 4096)
        diff = transformed_anchors.unsqueeze(2) - self.grid_xy.unsqueeze(1)  # (B, 64, 4096, 2)
        dist_sq = (diff ** 2).sum(dim=-1)                                     # (B, 64, 4096)
        sigma = torch.exp(self.log_sigma).clamp(0.05, 0.6)                      # (nhead,)
        sigma = sigma.view(1, self.nhead, 1, 1)                                    # (1, nhead, 1, 1)
        dist_sq_4d = dist_sq.unsqueeze(1)                                          # (B, 1, 64, 4096)
        pos_bias = -dist_sq_4d / (2.0 * sigma ** 2)                               # (B, nhead, 64, 4096)

        # --- Geometric attention bias from visibility mask ---
        b_geom = self.bias_conv(bev_vis_mask)                              # (B, nhead, 64, 64)
        b_geom = b_geom.flatten(2)                                         # (B, nhead, 4096)
        b_geom = b_geom.unsqueeze(2).expand(-1, -1, N_q, -1)              # (B, nhead, 64, 4096)

        # --- Multi-head content attention + position bias ---
        Q = Q.view(B, N_q, self.nhead, self.d_k).transpose(1, 2)          # (B, nhead, 64, d_k)
        K = K.view(B, N_kv, self.nhead, self.d_k).transpose(1, 2)         # (B, nhead, 4096, d_k)
        V = V.view(B, N_kv, self.nhead, self.d_k).transpose(1, 2)         # (B, nhead, 4096, d_k)

        content_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)  # (B, nhead, 64, 4096)
        alpha = F.softplus(self.pos_alpha)                                  # positive, smooth
        attn_scores = content_scores + alpha * pos_bias + b_geom
        attn_weights = F.softmax(attn_scores, dim=-1)                      # (B, nhead, 64, 4096)
        self._last_attn_weights = attn_weights.detach()                    # for visualization

        attn_out = attn_weights @ V                                        # (B, nhead, 64, d_k)
        attn_out = attn_out.transpose(1, 2).reshape(B, N_q, self.d_model)  # (B, 64, D)
        attn_out = self.out_proj(attn_out)

        # --- Post-attention (residual uses q_input, not anchor_pe) ---
        C_bev = self.norm1(q_input + attn_out)                             # residual + LN
        C_bev = self.norm2(C_bev + self.ffn(C_bev))                        # FFN + LN
        return C_bev, transformed_anchors  # (B, 64, D), (B, 64, 2)


class AnchorBasedSpatialFiLM(nn.Module):
    """Anchor-based Spatial FiLM: spatially-adaptive modulation using anchor positions.

    Instead of global pooling, this module:
    1. Uses anchor positions from PoseRouteCrossAttn to spatially interpolate C_bev features
    2. Generates per-pixel gamma and beta via convolutions
    3. Applies pixel-wise modulation: output = gamma * semantic + beta

    This preserves spatial structure and eliminates the "smearing" effect.
    """

    def __init__(self, d_model: int = 512, target_rows: int = 16, target_cols: int = 16,
                 rbf_sigma: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.rbf_sigma = rbf_sigma

        # Convolutional networks to generate spatially-adaptive gamma and beta
        self.gamma_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(16, d_model),
            nn.ReLU(inplace=False),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )
        self.beta_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(16, d_model),
            nn.ReLU(inplace=False),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )

        # Identity initialization: gamma≈1, beta≈0 at training start
        nn.init.zeros_(self.gamma_conv[-1].weight)
        nn.init.ones_(self.gamma_conv[-1].bias)
        nn.init.zeros_(self.beta_conv[-1].weight)
        nn.init.zeros_(self.beta_conv[-1].bias)

        # Pre-compute semantic grid coordinates in normalized space [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, target_rows),
            torch.linspace(-1, 1, target_cols),
            indexing='ij'
        )
        self.register_buffer('grid_xy', torch.stack([grid_x, grid_y], dim=-1))  # (R, C, 2)

    def forward(self, C_bev: torch.Tensor, semantic_feat: torch.Tensor,
                anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C_bev: (B, N_q, D) - BEV features from PoseRouteCrossAttn
            semantic_feat: (B, D, R, C) - semantic features from IPM
            anchors: (B, N_q, 2) - anchor positions in normalized coords [-1, 1]

        Returns:
            modulated: (B, D, R, C) - spatially-modulated semantic features
        """
        B, N_q, D = C_bev.shape
        R, C = self.target_rows, self.target_cols

        # Reshape grid for batch processing: (R, C, 2) -> (1, R*C, 2) -> (B, R*C, 2)
        grid = self.grid_xy.reshape(1, R * C, 2).expand(B, -1, -1)  # (B, R*C, 2)

        # Compute pairwise distances between each pixel and each anchor
        # grid: (B, R*C, 2), anchors: (B, N_q, 2)
        # dist: (B, R*C, N_q)
        dist = torch.cdist(grid, anchors)  # L2 distance

        # RBF kernel weighting: closer anchors have higher influence
        weights = torch.exp(-dist ** 2 / (2 * self.rbf_sigma ** 2))  # (B, R*C, N_q)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # normalize

        # Spatially interpolate C_bev features using RBF weights
        # weights: (B, R*C, N_q), C_bev: (B, N_q, D)
        C_bev_spatial = torch.bmm(weights, C_bev)  # (B, R*C, D)

        # Reshape to 2D spatial layout
        C_bev_spatial = C_bev_spatial.transpose(1, 2).reshape(B, D, R, C)  # (B, D, R, C)

        # Generate spatially-adaptive gamma and beta
        gamma = self.gamma_conv(C_bev_spatial)  # (B, D, R, C)
        beta = self.beta_conv(C_bev_spatial)    # (B, D, R, C)

        # Pixel-wise modulation (each pixel has its own gamma and beta!)
        modulated = gamma * semantic_feat + beta
        return modulated  # (B, D, R, C)


class SimplifiedTokenPredictor(nn.Module):
    """Simplified single-scale autoregressive token predictor.

    Key changes from multi-scale version:
    1. Single-scale condition (fine resolution only)
    2. All decoder layers use same condition
    3. Enhanced geometric encoding (Fourier features)
    4. Viewpoint encoding from camera pose
    5. Simpler, more efficient architecture
    """

    # ----- Sequence/Grid helpers (top-left row-major default) -----
    @staticmethod
    def grid_to_seq(tokens_grid: torch.Tensor) -> torch.Tensor:
        """
        Default top-left row-major: (R,C)->(R*C), (B,R,C)->(B,R*C)
        """
        if tokens_grid.dim() == 2:
            return tokens_grid.reshape(-1)
        if tokens_grid.dim() == 3:
            return tokens_grid.reshape(tokens_grid.shape[0], -1)
        raise ValueError(f"grid_to_seq expects (R,C) or (B,R,C), got {tuple(tokens_grid.shape)}")

    def seq_to_grid(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        """
        Default inverse for top-left row-major.
        Accepts (L,) or (B,L); returns (R,C) or (B,R,C) in NORMAL top-down grid.
        """
        L = self.target_rows * self.target_cols
        if tokens_seq.dim() == 1:
            if tokens_seq.numel() != L:
                raise ValueError(f"seq length {tokens_seq.numel()} != rows*cols {L}")
            return tokens_seq.view(self.target_rows, self.target_cols)
        if tokens_seq.dim() == 2:
            if tokens_seq.shape[1] != L:
                raise ValueError(f"seq length {tokens_seq.shape[1]} != rows*cols {L}")
            return tokens_seq.view(tokens_seq.shape[0], self.target_rows, self.target_cols)
        raise ValueError(f"seq_to_grid expects (L,) or (B,L), got {tuple(tokens_seq.shape)}")

    def make_teacher_forcing(self, target_token_grid: torch.Tensor, bos_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create (input_seq, target_seq) for AR training in TOP-LEFT row-major order.
        target_token_grid: (B,R,C) or (R,C) in NORMAL top-down grid
        Returns: (input_seq, target_seq) each (B,L)
        """
        if target_token_grid.dim() == 2:
            # Can be (R, C) or (1, L) from VQ
            if target_token_grid.shape[0] == 1 and target_token_grid.shape[1] == self.target_rows * self.target_cols:
                target_token_grid = target_token_grid.reshape(1, self.target_rows, self.target_cols)
            else:
                target_token_grid = target_token_grid.unsqueeze(0)

        B, R, C = target_token_grid.shape
        if R != self.target_rows or C != self.target_cols:
            # Try to reshape if it's a flat sequence with batch dim
            if R == 1 and C == self.target_rows * self.target_cols:
                target_token_grid = target_token_grid.reshape(B, self.target_rows, self.target_cols)
                B, R, C = target_token_grid.shape
            else:
                raise ValueError(f"target grid is ({R},{C}) but model expects ({self.target_rows},{self.target_cols})")
        target_seq = self.grid_to_seq(target_token_grid)  # (B,L)
        bos = torch.full((B, 1), int(bos_token), dtype=torch.long, device=target_seq.device)
        input_seq = torch.cat([bos, target_seq[:, :-1]], dim=1)
        return input_seq, target_seq

    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 1024,
        num_layers: int = 8,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1080,
        target_rows: int = 18,
        target_cols: int = 60,
        semantic_dim: int = 4,  # RGB(3) + mask(1)

        fourier_freqs: int = 10,
        train_bev_encoder: bool = False,  # If True, the BEV encoder's weights will be trained
        no_bev_pretrain: bool = False,    # If True, BEV encoder will be initialized from scratch
        pose_dim: int = 13,
        use_pose_token: bool = True,
        n_pose_queries: int = 64,
        mode: str = "direct",  # "vanilla", "direct", or "hybrid"
        use_ipm_semantic: bool = True,  # Whether to use IPM semantic features
        hybrid_memory_source: str = "enhanced",  # "enhanced" | "anchor" | "anchor_tokens"
        use_explicit_token_pos: bool = False,  # add explicit token pos embedding (used together with RoPE)
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.mode = mode.lower()
        self.use_ipm_semantic = bool(use_ipm_semantic)
        self.hybrid_memory_source = str(hybrid_memory_source).lower()
        self.use_explicit_token_pos = bool(use_explicit_token_pos)
        if self.mode not in {"vanilla", "direct", "hybrid"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'vanilla', 'direct', or 'hybrid'.")
        if self.hybrid_memory_source not in {"enhanced", "anchor", "anchor_tokens"}:
            raise ValueError(
                f"Unsupported hybrid_memory_source '{hybrid_memory_source}'. Expected 'enhanced', 'anchor', or 'anchor_tokens'."
            )

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.token_pos_embed = nn.Embedding(max_seq_len, d_model) if self.use_explicit_token_pos else None
        self.ntp_order = "topleft"  # default; subclass may override

        # Position encoding: RayRoPE (direct/hybrid) or Vanilla 2D PE (vanilla)
        if self.mode == "vanilla":
            from .vanilla_components import VanillaPositionEncoder
            self.vanilla_pos_encoder = VanillaPositionEncoder(
                d_model=d_model,
                max_h=self.target_rows,
                max_w=self.target_cols
            )
            self.vanilla_pose_projector = None  # Will be initialized later
        else:
            # direct and hybrid modes use RayRoPE (world coordinate ray encoding)
            self.ray_encoder = RayRoPEEncoder(d_model)

        self.input_norm = nn.LayerNorm(d_model)

        # Pose routing config
        self.use_pose_token = bool(use_pose_token)
        self.pose_dim = int(pose_dim)
        self.n_pose_queries = int(n_pose_queries)

        self.pose_route = None
        self.film = None
        self.pose_aware_anchor_query = None

        if self.mode == "vanilla":
            from .vanilla_components import VanillaPoseProjector
            self.vanilla_pose_projector = VanillaPoseProjector(
                pose_dim=self.pose_dim,
                d_model=d_model
            )
            self.pose_mlp = None
        elif self.mode == "direct":
            # Still need a simple pose projector for consistency
            from .vanilla_components import VanillaPoseProjector
            self.vanilla_pose_projector = VanillaPoseProjector(
                pose_dim=self.pose_dim,
                d_model=d_model
            )
        elif self.mode == "hybrid":
            # Hybrid mode: Direct sampling + Pose-aware Anchor Query
            from .pose_aware_anchor_query import PoseAwareAnchorQuery
            self.pose_aware_anchor_query = PoseAwareAnchorQuery(
                d_model=d_model,
                n_queries=self.n_pose_queries,
                nhead=nhead,
                pose_dim=self.pose_dim,
                use_ipm_semantic=self.use_ipm_semantic
            )
            from .vanilla_components import VanillaPoseProjector
            self.vanilla_pose_projector = VanillaPoseProjector(
                pose_dim=self.pose_dim,
                d_model=d_model
            )

        # Single-scale condition encoders (for direct and hybrid modes)
        self.semantic_encoder = SemanticEncoder(semantic_dim, d_model, hidden_channels=128, num_blocks=3) if (self.mode in ["direct", "hybrid"] and self.use_ipm_semantic) else None
        self.coord_encoder = FourierCoordEncoder(d_model, num_freqs=fourier_freqs) if self.mode in ["direct", "hybrid"] else None

        self.bev_encoder = MultiScaleViTBEVEncoder(
            output_channels=256,
            output_size=64,
            input_size=512,
            freeze_backbone=not train_bev_encoder,
            pretrained_path='ckpts/fmow_pretrain.pth' if not no_bev_pretrain else None
        )
        self.bev_feature_dim = 256
        self.bev_proj = nn.Conv2d(self.bev_feature_dim, d_model, kernel_size=1)

        # Runtime monitors for training-time diagnostics
        self.last_fusion_gates: Optional[torch.Tensor] = None  # (B,3)
        self.last_bev_feat_mean: Optional[torch.Tensor] = None
        self.last_bev_feat_std: Optional[torch.Tensor] = None

        # Intermediate features for consistency loss (direct mode only)
        self.last_sampled_bev_feat: Optional[torch.Tensor] = None  # (B, D, R, C)
        self.last_semantic_feat: Optional[torch.Tensor] = None   # (B, D, R, C)
        self.last_coords_map: Optional[torch.Tensor] = None      # (B, 2, R, C)

        # Anchor view conditioning support
        # 用于将 warp 后的 anchor 特征投影到 d_model 维度
        self.anchor_condition_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        nn.init.kaiming_normal_(self.anchor_condition_proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.anchor_condition_proj.bias)
        self.anchor_condition_gate = nn.Conv2d(d_model, d_model, kernel_size=1)
        nn.init.zeros_(self.anchor_condition_gate.weight)
        nn.init.zeros_(self.anchor_condition_gate.bias)

        # Transformer decoder layers
        decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        # Wrap attention modules with caching support and RoPE
        apply_rope = self.mode in ["direct", "hybrid"]
        for layer in decoder_layers:
            layer.self_attn = CacheableAttention(layer.self_attn, apply_rope=apply_rope)
            layer.multihead_attn = CacheableAttention(layer.multihead_attn, apply_rope=False)

        self.decoder_layers = decoder_layers

        # Output
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _apply_anchor_view_condition(
        self,
        fused_direct: torch.Tensor,
        anchor_memory: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if anchor_memory is None:
            return fused_direct

        anchor_proj = self.anchor_condition_proj(anchor_memory)
        anchor_gate = torch.sigmoid(self.anchor_condition_gate(anchor_memory))
        return fused_direct + anchor_gate * anchor_proj

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.token_pos_embed is not None:
            nn.init.normal_(self.token_pos_embed.weight, std=0.02)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Init semantic encoder (only for our mode)
        if self.semantic_encoder is not None:
            for module in self.semantic_encoder.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        # Init BEV projection
        nn.init.kaiming_normal_(self.bev_proj.weight, mode='fan_out', nonlinearity='linear')
        if self.bev_proj.bias is not None:
            nn.init.zeros_(self.bev_proj.bias)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _build_explicit_token_pos(self, batch_size: int, start_idx: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.token_pos_embed is None:
            raise RuntimeError("token_pos_embed is None while explicit token position is requested")
        end_idx = start_idx + seq_len
        if end_idx > self.max_seq_len:
            raise ValueError(f"Position end {end_idx} exceeds max_seq_len={self.max_seq_len}")
        pos_idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
        token_pos = self.token_pos_embed(pos_idx).unsqueeze(0).expand(batch_size, -1, -1)
        return token_pos

    def seq_index_to_rc(self, idx: torch.Tensor):
        """
        Map sequence indices (absolute/global) to spatial (row, col) according to self.ntp_order.
        This encodes absolute spatial position, not "generation-order index" semantics.
        Args:
            idx: (L,) absolute positions in the current forward [past_length, past_length+L)
        Returns:
            r, c: each (L,) long tensors
        """
        cols = int(self.target_cols)
        rows = int(self.target_rows)
        c = idx % cols
        if getattr(self, 'ntp_order', 'topleft') == 'bottomup':
            # bottom-up expects global indices: last row first
            r = (rows - 1) - (idx // cols)
        else:
            r = idx // cols
        return r.long(), c.long()


    def forward(
        self,
        generated_tokens: torch.Tensor,
        condition_tokens: dict,
        aligned_bev_feature_map: Optional[torch.Tensor] = None,
        bev_vis_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        anchor_memory: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the token predictor.
        """
        _, L_gen = generated_tokens.shape
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0][0].size(-2)

        if L_gen + past_length > self.max_seq_len:
            raise ValueError(f"Sequence length {L_gen + past_length} exceeds max_seq_len={self.max_seq_len}")

        tgt = self.token_embed(generated_tokens)  # (B, L_gen, D)

        if self.use_explicit_token_pos and self.mode in ["direct", "hybrid"]:
            token_pos = self._build_explicit_token_pos(
                batch_size=tgt.size(0),
                start_idx=past_length,
                seq_len=L_gen,
                device=tgt.device,
            )
            tgt = tgt + token_pos

        if self.mode == "vanilla":
            # Vanilla mode: 2D positional encoding
            # Calculate normalized coordinates for each token
            device = tgt.device
            B, L_gen_tgt = tgt.shape[0], tgt.shape[1]

            # Generate token positions
            y_coords = []
            x_coords = []
            for b in range(B):
                for l in range(L_gen):
                    idx = past_length + l
                    row = idx // self.target_cols
                    col = idx % self.target_cols

                    y = (row + 0.5) / self.target_rows
                    x = (col + 0.5) / self.target_cols
                    y_coords.append(y)
                    x_coords.append(x)

            y_coords = torch.tensor(y_coords, device=device).view(B, L_gen)
            x_coords = torch.tensor(x_coords, device=device).view(B, L_gen)

            token_pos = self.vanilla_pos_encoder(y_coords, x_coords)
            tgt = self.input_norm(tgt + token_pos)
        else:
            # Our mode: RayRoPE (world coordinate ray encoding)
            K = condition_tokens.get("K", None)
            T_cam_to_world = condition_tokens.get("T_cam_to_world", None)
            if K is None:
                raise KeyError("condition_tokens must contain 'K' for ray RoPE")
            if T_cam_to_world is None:
                raise KeyError("condition_tokens must contain 'T_cam_to_world' for ray RoPE")
            # Get RoPE rotation terms
            cos, sin = self.ray_encoder(
                K=K,
                T_cam_to_world=T_cam_to_world,
                past_length=past_length,
                L_gen=L_gen,
                rows=self.target_rows,
                cols=self.target_cols,
                ntp_order=getattr(self, "ntp_order", "topleft"),
                device=tgt.device,
            )
            # RayRoPE is applied as rotation on query/key, no need to add to embedding
            tgt = self.input_norm(tgt)

        semantic_input = condition_tokens.get('semantic')
        coord_input = condition_tokens.get('coords')
        pose_input = condition_tokens.get('pose', None)
        need_coords = self.mode in ["direct", "hybrid"]
        need_semantic = self.use_ipm_semantic and self.mode in ["direct", "hybrid"]

        if need_coords and coord_input is None:
            raise KeyError("condition_tokens must contain 'coords' for direct/hybrid mode")
        if need_semantic and semantic_input is None:
            raise KeyError("condition_tokens must contain 'semantic' for this mode/config")

        if self.use_pose_token:
            if pose_input is None:
                raise KeyError("condition_tokens must contain 'pose' when use_pose_token=True")
            if pose_input.dim() != 2 or pose_input.size(-1) != self.pose_dim:
                raise ValueError(f"pose must be (B,{self.pose_dim}), got {tuple(pose_input.shape)}")

        if self.mode == "vanilla":
            # Vanilla mode: ONLY use BEV features (ablation: no IPM, no semantic features)
            memory_ipm = None

            bev_memory = None
            if aligned_bev_feature_map is not None:
                bev_feat = self.bev_proj(aligned_bev_feature_map)  # (B, D, H_bev, W_bev)
                bev_memory = bev_feat.flatten(2).transpose(1, 2)  # (B, L_bev, D)
                self.last_bev_feat_mean = bev_feat.mean().detach()
                self.last_bev_feat_std = bev_feat.std(unbiased=False).detach()
            else:
                self.last_bev_feat_mean = None
                self.last_bev_feat_std = None
        elif self.mode == "direct":
            # Direct mode: Use coords_map to sample BEV features directly (+ optional semantic features)
            if self.use_ipm_semantic:
                semantic_feat = self.semantic_encoder(semantic_input)  # (B, D, R, C)
            else:
                semantic_feat = None

            # Project BEV features first
            if aligned_bev_feature_map is not None:
                bev_feat_256 = aligned_bev_feature_map  # (B, 256, 64, 64)
            else:
                B = generated_tokens.size(0)
                bev_feat_256 = torch.zeros(B, 256, 64, 64, device=tgt.device, dtype=tgt.dtype)

            # Direct sampling using coord_input (coords_map)
            # coord_input: (B, 2, R, C) — 每个 token 的 BEV 坐标
            sampled_bev = F.grid_sample(
                bev_feat_256,
                coord_input.permute(0, 2, 3, 1),  # (B, R, C, 2) for grid_sample
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )  # (B, 256, R, C)

            # Project to d_model
            sampled_bev = self.bev_proj(sampled_bev)  # (B, D, R, C)

            # 编码坐标信息
            coord_feat = self.coord_encoder(coord_input)  # (B, D, R, C)

            # 保存中间特征用于一致性损失计算（保持梯度链路，便于一致性项反向传播）
            self.last_sampled_bev_feat = sampled_bev  # (B, D, R, C)
            self.last_semantic_feat = semantic_feat
            self.last_coords_map = coord_input        # (B, 2, R, C)

            # Assemble memory: sampled_bev + coord_feat (+ optional semantic_feat)
            bev_memory = sampled_bev.flatten(2).transpose(1, 2)  # (B, R*C, D)
            coord_memory = coord_feat.flatten(2).transpose(1, 2)      # (B, R*C, D)

            # Memory assembly
            memory_parts = [bev_memory, coord_memory]
            if semantic_feat is not None:
                sem_memory = semantic_feat.flatten(2).transpose(1, 2)  # (B, R*C, D)
                memory_parts.insert(1, sem_memory)
            
            # Add anchor memory if provided
            if anchor_memory is not None:
                # Project anchor memory to d_model and add to memory
                anchor_proj = self.anchor_condition_proj(anchor_memory)  # (B, D, R, C)
                anchor_mem_flat = anchor_proj.flatten(2).transpose(1, 2)  # (B, R*C, D)
                memory_parts.append(anchor_mem_flat)
            
            memory = torch.cat(memory_parts, dim=1)  # (B, ((2|3)+N)*R*C, D)

            self.last_bev_feat_mean = sampled_bev.mean().detach()
            self.last_bev_feat_std = sampled_bev.std(unbiased=False).detach()
        elif self.mode == "hybrid":
            # Hybrid mode: Direct sampling + Pose-aware Anchor Query
            if aligned_bev_feature_map is not None:
                bev_feat_256 = aligned_bev_feature_map
            else:
                B = generated_tokens.size(0)
                bev_feat_256 = torch.zeros(B, 256, 64, 64, device=tgt.device, dtype=tgt.dtype)

            # Direct sampling
            coord_input = condition_tokens.get('coords')
            if coord_input is None:
                raise KeyError("condition_tokens must contain 'coords'")

            sampled_bev = F.grid_sample(
                bev_feat_256,
                coord_input.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            sampled_bev = self.bev_proj(sampled_bev)

            # 保存中间特征用于一致性损失计算（和direct模式保持一致）
            self.last_sampled_bev_feat = sampled_bev  # (B, D, R, C)
            self.last_semantic_feat = None
            self.last_coords_map = coord_input        # (B, 2, R, C)

            # Bind content and explicit coordinates at the feature level before anchors read memory.
            coord_feat = self.coord_encoder(coord_input)
            fused_direct = sampled_bev + coord_feat

            if self.use_ipm_semantic:
                semantic_input = condition_tokens.get('semantic')
                if semantic_input is None:
                    raise KeyError("condition_tokens must contain 'semantic' when use_ipm_semantic=True")
                semantic_feat = self.semantic_encoder(semantic_input)
                self.last_semantic_feat = semantic_feat   # (B, D, R, C)
                fused_direct = fused_direct + semantic_feat

            fused_direct = self._apply_anchor_view_condition(
                fused_direct=fused_direct,
                anchor_memory=anchor_memory,
            )

            direct_memory = fused_direct.flatten(2).transpose(1, 2)
            # 准备direct token坐标 (B, 2, R, C) → (B, R*C, 2)
            direct_coords = coord_input.flatten(2).transpose(1, 2)

            # Pose-aware Anchor Query + Global Context
            use_spatial_broadcast = self.hybrid_memory_source in {"enhanced", "anchor"}
            enhanced_memory, anchor_context, anchor_tokens, anchor_pos = self.pose_aware_anchor_query(
                pose_input=pose_input,
                direct_memory=direct_memory,
                direct_coords=direct_coords,
                use_spatial_broadcast=use_spatial_broadcast,
            )
            if self.hybrid_memory_source == "anchor_tokens":
                memory = anchor_tokens
            elif self.hybrid_memory_source == "anchor":
                if anchor_context is None:
                    raise RuntimeError("anchor_context is None while hybrid_memory_source='anchor'")
                memory = anchor_context
            else:
                memory = enhanced_memory

            self.last_bev_feat_mean = sampled_bev.mean().detach()
            self.last_bev_feat_std = sampled_bev.std(unbiased=False).detach()
        else:
            raise RuntimeError(f"Unsupported mode '{self.mode}' during forward pass.")

        memory_parts = []
        if self.use_pose_token:
            if self.mode == "vanilla":
                # Vanilla mode: Project pose to token
                pose_token = self.vanilla_pose_projector(pose_input)
                memory_parts.append(pose_token)
            elif self.mode == "direct" or self.mode == "hybrid":
                # Direct/hybrid mode: Project pose to token
                pose_token = self.vanilla_pose_projector(pose_input)
                memory_parts.append(pose_token)
            else:
                raise RuntimeError(f"Unsupported mode '{self.mode}' during pose token construction.")

        if self.mode == "vanilla":
            # Vanilla mode: ONLY BEV memory (no IPM memory)
            if bev_memory is not None:
                memory_parts.append(bev_memory)
            memory = torch.cat(memory_parts, dim=1) if memory_parts else None
        elif self.mode == "direct":
            # Direct mode: Prepend pose token to [bev_memory, sem_memory]
            if memory_parts:
                memory = torch.cat([memory_parts[0], memory], dim=1)
        elif self.mode == "hybrid":
            # Hybrid mode: Prepend pose token to enhanced memory
            if memory_parts:
                memory = torch.cat([memory_parts[0], memory], dim=1)
        else:
            raise RuntimeError(f"Unsupported mode '{self.mode}' during memory assembly.")

        full_seq_len = past_length + L_gen
        causal_mask = self._generate_causal_mask(full_seq_len, tgt.device)
        if use_cache and past_length > 0:
            causal_mask = causal_mask[-L_gen:, :]

        present_key_values = [] if use_cache else None
        hidden_states = tgt

        for i, layer in enumerate(self.decoder_layers):
            past_sa_kv = past_key_values[i][0] if past_key_values is not None else None
            past_ca_kv = past_key_values[i][1] if past_key_values is not None else None

            # Self attention with optional RayRoPE
            sa_kwargs = {}
            if self.mode in ["direct", "hybrid"] and cos is not None and sin is not None:
                sa_kwargs["rope_cos"] = cos
                sa_kwargs["rope_sin"] = sin

            sa_out, sa_present_kv = layer.self_attn(
                hidden_states,
                hidden_states,
                hidden_states,
                past_kv=past_sa_kv,
                use_cache=use_cache,
                attn_mask=causal_mask,
                need_weights=False,
                **sa_kwargs
            )
            hidden_states = layer.norm1(hidden_states + layer.dropout1(sa_out))

            ca_out, ca_present_kv = layer.multihead_attn(
                hidden_states,
                memory,
                memory,
                past_kv=past_ca_kv,
                use_cache=use_cache,
                key_padding_mask=None,
            )
            hidden_states = layer.norm2(hidden_states + layer.dropout2(ca_out))

            ffn_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden_states))))
            hidden_states = layer.norm3(hidden_states + layer.dropout3(ffn_output))

            if use_cache:
                present_key_values.append((sa_present_kv, ca_present_kv))

        hidden_states = self.output_norm(hidden_states)
        logits = self.head(hidden_states)

        if use_cache:
            return logits, present_key_values
        return logits


class BottomUpSimplifiedTokenPredictor(SimplifiedTokenPredictor):
    """
    Same core model as SimplifiedTokenPredictor.
    The "new NTP method" here is the autoregressive factorization order:
      - predict tokens in bottom-up raster order (last row -> first row),
        and within each row left -> right.

    IMPORTANT:
      The Transformer itself is unchanged. Only the token <-> sequence ordering changes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ntp_order = "bottomup"

    @staticmethod
    def grid_to_seq(tokens_grid: torch.Tensor) -> torch.Tensor:
        return grid_to_seq_bottomup(tokens_grid)

    def seq_to_grid(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        return seq_to_grid_bottomup(tokens_seq, self.target_rows, self.target_cols)

    def make_teacher_forcing(
        self,
        target_token_grid: torch.Tensor,
        bos_token: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create (input_seq, target_seq) for AR training in bottom-up order.

        Args:
            target_token_grid: (B,R,C) or (R,C) token indices (NORMAL top-down grid)
            bos_token: int

        Returns:
            input_seq:  (B, L)  where input_seq[:,0]=BOS, input_seq[:,1:]=target_seq[:,:-1]
            target_seq: (B, L)  bottom-up order
        """
        if target_token_grid.dim() == 2:
            target_token_grid = target_token_grid.unsqueeze(0)

        B, R, C = target_token_grid.shape
        if R != self.target_rows or C != self.target_cols:
            raise ValueError(f"target grid is ({R},{C}) but model expects ({self.target_rows},{self.target_cols})")

        target_seq = grid_to_seq_bottomup(target_token_grid)  # (B,L)
        bos = torch.full((B, 1), int(bos_token), dtype=torch.long, device=target_seq.device)
        input_seq = torch.cat([bos, target_seq[:, :-1]], dim=1)
        return input_seq, target_seq

    def logits_seq_to_grid(self, logits_seq: torch.Tensor) -> torch.Tensor:
        """
        Convert logits from (B,L,V) bottom-up order -> (B,R,C,V) NORMAL top-down grid.
        """
        if logits_seq.dim() != 3:
            raise ValueError(f"logits_seq must be (B,L,V), got {tuple(logits_seq.shape)}")
        B, L, V = logits_seq.shape
        expected = self.target_rows * self.target_cols
        if L != expected:
            raise ValueError(f"logits L={L} != rows*cols={expected}")
        return logits_seq.view(B, self.target_rows, self.target_cols, V).flip(1)

    def forward(
        self,
        generated_tokens: torch.Tensor,
        condition_tokens: dict,
        aligned_bev_feature_map: Optional[torch.Tensor] = None,
        bev_vis_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        anchor_memory: Optional[torch.Tensor] = None,
    ):
        """
        Accept either:
          - generated_tokens: (B,L) sequence (bottom-up order), OR
          - generated_tokens: (B,R,C) grid (top-down), will be converted internally.
        """
        if generated_tokens.dim() == 3:
            generated_tokens = grid_to_seq_bottomup(generated_tokens)
        return super().forward(
            generated_tokens=generated_tokens,
            condition_tokens=condition_tokens,
            aligned_bev_feature_map=aligned_bev_feature_map,
            bev_vis_mask=bev_vis_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            anchor_memory=anchor_memory,
        )


class MaskGITTokenPredictor(SimplifiedTokenPredictor):
    """
    MaskGIT-style parallel token predictor.

    Differences from AR SimplifiedTokenPredictor:
      - Bidirectional self-attention (no causal mask)
      - Learnable [MASK] token embedding for masked positions
      - No KV cache (full forward pass each iteration)
      - generate() method for iterative parallel decoding
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Learnable [MASK] token embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        condition_tokens: dict,
        aligned_bev_feature_map: Optional[torch.Tensor] = None,
        bev_vis_mask: Optional[torch.Tensor] = None,
        anchor_memory: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            tokens: (B, L) token indices. Values at masked positions are ignored.
            mask: (B, L) bool tensor. True = masked (to be predicted).
            condition_tokens: dict with 'semantic', 'pose', 'K'.
            aligned_bev_feature_map: (B, 256, 64, 64) or None.
            bev_vis_mask: (B, 1, 64, 64) or None.
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = tokens.shape

        # --- Token embedding: real tokens + mask token for masked positions ---
        tgt = self.token_embed(tokens)  # (B, L, D)
        mask_emb = self.mask_token.expand(B, L, -1)  # (B, L, D)
        # Replace masked positions with mask_token
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        tgt = tgt * (1 - mask_expanded) + mask_emb * mask_expanded

        if self.use_explicit_token_pos and self.mode in ["direct", "hybrid"]:
            token_pos = self._build_explicit_token_pos(
                batch_size=B,
                start_idx=0,
                seq_len=L,
                device=tgt.device,
            )
            tgt = tgt + token_pos

        # --- Ray RoPE (world coordinate ray encoding) ---
        if self.mode in ["direct", "hybrid"]:
            K_intr = condition_tokens.get('K', None)
            T_cam_to_world = condition_tokens.get("T_cam_to_world", None)
            if K_intr is None:
                raise KeyError("condition_tokens must contain 'K' for ray RoPE")
            if T_cam_to_world is None:
                raise KeyError("condition_tokens must contain 'T_cam_to_world' for ray RoPE")

            # Get RoPE rotation terms
            cos, sin = self.ray_encoder(
                K=K_intr,
                T_cam_to_world=T_cam_to_world,
                past_length=0,
                L_gen=L,
                rows=self.target_rows,
                cols=self.target_cols,
                ntp_order=getattr(self, 'ntp_order', 'topleft'),
                device=tgt.device,
            )
        else:
            cos, sin = None, None

        # No addition to embedding, RoPE is applied as rotation during attention
        tgt = self.input_norm(tgt)

        # --- Condition encoding ---
        semantic_input = condition_tokens.get('semantic')
        pose_input = condition_tokens.get('pose', None)
        if self.use_ipm_semantic and semantic_input is None:
            raise KeyError("condition_tokens must contain 'semantic' when use_ipm_semantic=True")
        if pose_input is None:
            raise KeyError("condition_tokens must contain 'pose'")

        if self.use_ipm_semantic:
            semantic_feat = self.semantic_encoder(semantic_input)
        else:
            semantic_feat = None

        if self.mode == "direct":
            semantic_input = condition_tokens.get('semantic')
            pose_input = condition_tokens.get('pose', None)
            if self.use_ipm_semantic and semantic_input is None:
                raise KeyError("condition_tokens must contain 'semantic' when use_ipm_semantic=True")
            if pose_input is None:
                raise KeyError("condition_tokens must contain 'pose'")

            if self.use_ipm_semantic:
                semantic_feat = self.semantic_encoder(semantic_input)
            else:
                semantic_feat = None

            if aligned_bev_feature_map is not None:
                bev_feat = aligned_bev_feature_map  # 保持原始的 256 通道 BEV 特征
            else:
                bev_feat = torch.zeros(B, 256, 64, 64, device=tgt.device, dtype=tgt.dtype)

            # Coordinate feature encoding
            coord_input = condition_tokens.get('coords')
            if coord_input is None:
                raise KeyError("condition_tokens must contain 'coords' (Direct mode)")
            coord_feat = self.coord_encoder(coord_input)

            # Direct mode: BEV sampling using coords_map (exact as SimplifiedTokenPredictor)
            sampled_bev = F.grid_sample(
                bev_feat,
                coord_input.permute(0, 2, 3, 1),  # (B, R, C, 2) for grid_sample
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )  # (B, 256, R, C)

            # Project to d_model
            sampled_bev = self.bev_proj(sampled_bev)  # (B, D, R, C)

            # 保存中间特征用于一致性损失计算（保持梯度链路）
            self.last_sampled_bev_feat = sampled_bev  # (B, D, R, C)
            self.last_semantic_feat = semantic_feat
            self.last_coords_map = coord_input        # (B, 2, R, C)

            # Assemble memory: sampled_bev + coord_feat (+ optional semantic_feat)
            bev_memory = sampled_bev.flatten(2).transpose(1, 2)  # (B, R*C, D)
            coord_memory = coord_feat.flatten(2).transpose(1, 2)      # (B, R*C, D)

            # Memory assembly
            memory_parts = [bev_memory, coord_memory]
            if semantic_feat is not None:
                sem_memory = semantic_feat.flatten(2).transpose(1, 2)  # (B, R*C, D)
                memory_parts.insert(1, sem_memory)
            memory = torch.cat(memory_parts, dim=1)

            # Prepend pose token if using pose token
            if self.use_pose_token:
                pose_token = self.vanilla_pose_projector(pose_input)
                memory = torch.cat([pose_token, memory], dim=1)
        elif self.mode == "hybrid":
            # Hybrid mode: Direct sampling + Pose-aware Anchor Query
            if aligned_bev_feature_map is not None:
                bev_feat_256 = aligned_bev_feature_map
            else:
                B = tokens.size(0)
                bev_feat_256 = torch.zeros(B, 256, 64, 64, device=tgt.device, dtype=tgt.dtype)

            # Direct sampling
            coord_input = condition_tokens.get('coords')
            if coord_input is None:
                raise KeyError("condition_tokens must contain 'coords'")

            sampled_bev = F.grid_sample(
                bev_feat_256,
                coord_input.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            sampled_bev = self.bev_proj(sampled_bev)

            # 保存中间特征用于一致性损失计算（和direct模式保持一致）
            self.last_sampled_bev_feat = sampled_bev  # (B, D, R, C)
            self.last_semantic_feat = None
            self.last_coords_map = coord_input        # (B, 2, R, C)

            # Bind content and explicit coordinates at the feature level before anchors read memory.
            coord_feat = self.coord_encoder(coord_input)
            fused_direct = sampled_bev + coord_feat

            if self.use_ipm_semantic:
                semantic_input = condition_tokens.get('semantic')
                if semantic_input is None:
                    raise KeyError("condition_tokens must contain 'semantic' when use_ipm_semantic=True")
                semantic_feat = self.semantic_encoder(semantic_input)
                self.last_semantic_feat = semantic_feat   # (B, D, R, C)
                fused_direct = fused_direct + semantic_feat

            fused_direct = self._apply_anchor_view_condition(
                fused_direct=fused_direct,
                anchor_memory=anchor_memory,
            )

            direct_memory = fused_direct.flatten(2).transpose(1, 2)
            # 准备direct token坐标 (B, 2, R, C) → (B, R*C, 2)
            direct_coords = coord_input.flatten(2).transpose(1, 2)

            # Pose-aware Anchor Query + Global Context
            use_spatial_broadcast = self.hybrid_memory_source in {"enhanced", "anchor"}
            enhanced_memory, anchor_context, anchor_tokens, anchor_pos = self.pose_aware_anchor_query(
                pose_input=pose_input,
                direct_memory=direct_memory,
                direct_coords=direct_coords,
                use_spatial_broadcast=use_spatial_broadcast,
            )
            if self.hybrid_memory_source == "anchor_tokens":
                memory = anchor_tokens
            elif self.hybrid_memory_source == "anchor":
                if anchor_context is None:
                    raise RuntimeError("anchor_context is None while hybrid_memory_source='anchor'")
                memory = anchor_context
            else:
                memory = enhanced_memory

            # Prepend pose token if using pose token
            if self.use_pose_token:
                pose_token = self.vanilla_pose_projector(pose_input)
                memory = torch.cat([pose_token, memory], dim=1)

        else:
            # Our mode: use pose_route and film
            if aligned_bev_feature_map is not None and bev_vis_mask is not None:
                bev_enriched = torch.cat([aligned_bev_feature_map, bev_vis_mask], dim=1)
                bev_feat = self.bev_proj(bev_enriched)
            else:
                bev_feat = torch.zeros(B, self.d_model, 64, 64, device=tgt.device, dtype=tgt.dtype)
                if bev_vis_mask is None:
                    bev_vis_mask = torch.zeros(B, 1, 64, 64, device=tgt.device, dtype=tgt.dtype)

            C_bev, anchors = self.pose_route(pose_input, bev_feat, bev_vis_mask)
            modulated_sem = self.film(C_bev, semantic_feat, anchors)
            sem_memory = modulated_sem.flatten(2).transpose(1, 2)
            memory = torch.cat([C_bev, sem_memory], dim=1)

        # --- Transformer decoder: BIDIRECTIONAL self-attention (no causal mask) ---
        hidden_states = tgt
        for layer in self.decoder_layers:
            # Self attention with optional RayRoPE
            sa_kwargs = {}
            if self.mode in ["direct", "hybrid"] and cos is not None and sin is not None:
                sa_kwargs["rope_cos"] = cos
                sa_kwargs["rope_sin"] = sin

            sa_out, _ = layer.self_attn(
                hidden_states, hidden_states, hidden_states,
                past_kv=None, use_cache=False, attn_mask=None,
                need_weights=False,
                **sa_kwargs
            )
            hidden_states = layer.norm1(hidden_states + layer.dropout1(sa_out))

            ca_out, _ = layer.multihead_attn(
                hidden_states, memory, memory,
                past_kv=None, use_cache=False, key_padding_mask=None,
            )
            hidden_states = layer.norm2(hidden_states + layer.dropout2(ca_out))

            ffn_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden_states))))
            hidden_states = layer.norm3(hidden_states + layer.dropout3(ffn_output))

        hidden_states = self.output_norm(hidden_states)
        logits = self.head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(
        self,
        condition_tokens: dict,
        aligned_bev_feature_map: Optional[torch.Tensor] = None,
        bev_vis_mask: Optional[torch.Tensor] = None,
        num_steps: int = 12,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Iterative parallel decoding (MaskGIT-style).

        Returns:
            token_grid: (B, target_rows, target_cols)
        """
        B = condition_tokens["pose"].size(0)
        L = self.max_seq_len
        device = condition_tokens["pose"].device

        tokens = torch.zeros(B, L, dtype=torch.long, device=device)
        mask = torch.ones(B, L, dtype=torch.bool, device=device)  # all masked

        for step in range(num_steps):
            logits = self.forward(
                tokens, mask, condition_tokens,
                aligned_bev_feature_map, bev_vis_mask,
            )
            # Clamp logits to real vocab (exclude BOS if present)
            logits_sample = logits[:, :, :self.vocab_size]

            # Top-k sampling with temperature
            if temperature != 1.0:
                logits_sample = logits_sample / temperature
            k = min(top_k, logits_sample.size(-1))
            top_vals, top_idx = torch.topk(logits_sample, k, dim=-1)
            probs = F.softmax(top_vals, dim=-1)
            # Sample for each position
            flat_probs = probs.view(-1, k)
            flat_sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
            sampled = torch.gather(top_idx.view(-1, k), -1, flat_sampled.unsqueeze(-1))
            sampled = sampled.squeeze(-1).view(B, L)

            # Confidence: probability of the sampled token
            full_probs = F.softmax(logits[:, :, :self.vocab_size], dim=-1)
            confidence = torch.gather(full_probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
            # Already-determined tokens get infinite confidence
            confidence[~mask] = float('inf')

            # Update tokens at currently masked positions
            tokens = torch.where(mask, sampled, tokens)

            # Cosine schedule: how many tokens remain masked after this step
            ratio = math.cos((step + 1) / num_steps * math.pi / 2)
            num_remain = max(0, int(ratio * L))

            if num_remain > 0 and step < num_steps - 1:
                # Keep the least confident tokens as masked
                _, low_conf_idx = confidence.topk(num_remain, dim=-1, largest=False)
                new_mask = torch.zeros_like(mask)
                new_mask.scatter_(1, low_conf_idx, True)
                mask = new_mask
            else:
                mask = torch.zeros_like(mask)

        return tokens.view(B, self.target_rows, self.target_cols)

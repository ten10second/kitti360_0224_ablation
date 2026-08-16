"""
Pose-aware Anchor Query module for Hybrid mode (Direct + Global Context)

This module implements:
- Learnable anchor positions across entire BEV
- Current pose information utilization
- Anchor-to-direct memory cross-attention
- Anchor self-attention for global understanding
- Global context broadcast to direct tokens

"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions, for RoPE."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


class PoseAwareAnchorQuery(nn.Module):
    """
    Elegant Hybrid mode core module:
    - Learnable anchor positions (full BEV range)
    - Utilize current pose information
    - Anchor aggregates information from direct memory
    - Anchors self-attention for global understanding
    - Broadcast global context to each direct token
    """

    def __init__(self, d_model: int = 512, n_queries: int = 64, nhead: int = 8, pose_dim: int = 13, use_ipm_semantic: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.nhead = nhead
        self.use_ipm_semantic = use_ipm_semantic
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got d_model={self.d_model}, nhead={self.nhead}")
        if self.d_model % 4 != 0:
            raise ValueError(f"PoseAwareAnchorQuery geometry RoPE requires d_model divisible by 4, got {self.d_model}")
        self.head_dim = self.d_model // self.nhead
        self.cross_geom_dim = self.d_model // 2
        geom_freqs = 10000.0 ** (
            -2.0 * torch.arange(self.cross_geom_dim // 2, dtype=torch.float32) / float(self.cross_geom_dim)
        )
        self.register_buffer("cross_geom_freqs", geom_freqs, persistent=False)

        # 1. Learnable anchor positions (full BEV range)
        self.learnable_anchors = nn.Parameter(
            self._build_initial_anchors(n_queries) + torch.randn(n_queries, 2) * 0.1
        )

        # 2. Pose projection (for global context)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=False),
            nn.Linear(d_model, d_model)
        )

        # 3. Anchor position encoding
        from .vanilla_components import VanillaPositionEncoder
        self.anchor_pos_encoder = VanillaPositionEncoder(
            d_model=d_model,
            max_h=10,  # Large enough range
            max_w=10
        )

        # 4. Cross-Attention: Anchor Query → Direct Memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)

        # 5. Self-Attention: Anchor interaction
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(0.1)

        # 6. Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(0.1)

        # 7. Global context broadcast network
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=False),
            nn.Linear(d_model, d_model)
        )

        # Runtime caches for visualization
        self._last_cross_attn_weights: Optional[torch.Tensor] = None
        self._last_self_attn_weights: Optional[torch.Tensor] = None
        self._last_anchor_positions: Optional[torch.Tensor] = None
        self._last_anchors: Optional[torch.Tensor] = None  # alias for compatibility

    @staticmethod
    def _build_initial_anchors(n_queries: int = 64) -> torch.Tensor:
        """Initialize anchors covering larger range, allow free learning"""
        n_angles = int(math.sqrt(n_queries))
        n_radii = n_queries // n_angles

        angles = torch.linspace(-math.pi / 2, math.pi / 2, n_angles)  # [-90°, 90°]
        radii = torch.linspace(0.05, 0.9, n_radii)                     # Wider range

        grid_a, grid_r = torch.meshgrid(angles, radii, indexing='ij')
        x = grid_r * torch.sin(grid_a)
        y = grid_r * torch.cos(grid_a)

        return torch.stack([x, y], dim=-1).reshape(n_queries, 2)

    def _build_xy_rope(self, positions: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build 2D RoPE terms from BEV-plane coordinates."""
        if positions.dim() != 3 or positions.size(-1) != 2:
            raise ValueError(f"positions must be (B, L, 2), got {tuple(positions.shape)}")
        pos = positions.to(torch.float32).clamp(min=-1.5, max=1.5) * math.pi
        x = pos[..., 0]
        y = pos[..., 1]
        freqs = self.cross_geom_freqs.to(device=positions.device, dtype=torch.float32).clone()

        x_theta = x.unsqueeze(-1) * freqs.view(1, 1, -1)
        y_theta = y.unsqueeze(-1) * freqs.view(1, 1, -1)

        x_cos = torch.cos(x_theta).repeat_interleave(2, dim=-1)
        x_sin = torch.sin(x_theta).repeat_interleave(2, dim=-1)
        y_cos = torch.cos(y_theta).repeat_interleave(2, dim=-1)
        y_sin = torch.sin(y_theta).repeat_interleave(2, dim=-1)

        cos = torch.cat([x_cos, y_cos], dim=-1).to(dtype=dtype)
        sin = torch.cat([x_sin, y_sin], dim=-1).to(dtype=dtype)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE terms to projected multi-head Q/K tensors."""
        B, _, L, _ = x.shape
        cos_exp = cos.reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)
        sin_exp = sin.reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)
        return x * cos_exp + rotate_half(x) * sin_exp

    def _cross_attention_with_geometry(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention where Q/K logits are explicitly geometry-aware."""
        attn = self.cross_attn
        B, L_q, D = query.shape
        L_k = key.shape[1]

        if attn.in_proj_weight is not None:
            w_q, w_k, w_v = attn.in_proj_weight.chunk(3)
            if attn.in_proj_bias is not None:
                b_q, b_k, b_v = attn.in_proj_bias.chunk(3)
            else:
                b_q = b_k = b_v = None
        else:
            w_q = attn.q_proj_weight
            w_k = attn.k_proj_weight
            w_v = attn.v_proj_weight
            if attn.in_proj_bias is not None:
                b_q, b_k, b_v = attn.in_proj_bias.chunk(3)
            else:
                b_q = b_k = b_v = None

        q = F.linear(query, w_q, b_q).view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)
        k = F.linear(key, w_k, b_k).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        v = F.linear(value, w_v, b_v).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)

        q_cos, q_sin = self._build_xy_rope(query_pos, dtype=query.dtype)
        k_cos, k_sin = self._build_xy_rope(key_pos, dtype=key.dtype)
        q = self._apply_rope(q, q_cos, q_sin)
        k = self._apply_rope(k, k_cos, k_sin)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_probs = F.dropout(attn_weights, p=float(attn.dropout), training=self.training)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        attn_output = attn.out_proj(attn_output)
        return attn_output, attn_weights.mean(dim=1)

    def forward(
        self,
        pose_input: torch.Tensor,
        direct_memory: torch.Tensor,
        direct_coords: torch.Tensor,
        use_spatial_broadcast: bool = True,
    ):
        """
        Args:
            pose_input: (B, 13) — Current pose information
            direct_memory: (B, L, D) — Direct sampling features of each token
            direct_coords: (B, L, 2) — BEV coordinates of each direct token

        Returns:
            enhanced_memory: (B, L, D) — direct + anchor residual fused memory
            anchor_context: (B, L, D) or None — pure anchor context feature (broadcasted)
            anchor_tokens: (B, N_q, D) — anchor global tokens after cross/self/ffn
            anchor_positions: (B, N_q, 2) — learned anchor positions
        """
        B = pose_input.size(0)
        L = direct_memory.size(1)
        device = pose_input.device

        # Step 1: Pose encoding
        pose_emb = self.pose_proj(pose_input)  # (B, D)

        # Step 2: Anchor position encoding
        anchor_pos = self.learnable_anchors.unsqueeze(0).expand(B, -1, 2)  # (B, N_q, 2)
        anchor_cache = anchor_pos.detach()
        self._last_anchor_positions = anchor_cache
        self._last_anchors = anchor_cache

        # Simple position encoding (can be optimized)
        anchor_enc = self.anchor_pos_encoder(
            y_coords=anchor_pos[..., 1],
            x_coords=anchor_pos[..., 0]
        )  # (B, N_q, D)

        # Step 3: Pose-aware anchor query
        anchor_query = anchor_enc + pose_emb.unsqueeze(1)  # (B, N_q, D)

        # Step 4: Cross-Attention — Anchor aggregates info from direct memory
        cross_attn_output, cross_attn_weights = self._cross_attention_with_geometry(
            query=anchor_query,
            key=direct_memory,
            value=direct_memory,
            query_pos=anchor_pos,
            key_pos=direct_coords,
        )  # (B, N_q, D)
        self._last_cross_attn_weights = cross_attn_weights.detach()
        # 残差连接 + 层归一化
        attn_output = self.norm1(anchor_query + self.dropout1(cross_attn_output))

        # Step 5: Self-Attention — Anchor interaction
        self_attn_output, self_attn_weights = self.self_attn(
            query=attn_output,
            key=attn_output,
            value=attn_output
        )  # (B, N_q, D)
        self._last_self_attn_weights = self_attn_weights.detach()
        # 残差连接 + 层归一化
        attn_output = self.norm2(attn_output + self.dropout2(self_attn_output))

        # Step 6: Feed Forward Network (FFN)
        ffn_output = self.ffn(attn_output)
        # 残差连接 + 层归一化
        attn_output = self.norm3(attn_output + self.dropout3(ffn_output))

        if use_spatial_broadcast:
            # Step 7: 空间广播 (Spatial Broadcast) - 距离加权融合
            # 计算每个direct token到所有anchor的欧氏距离 (B, L, N_q)
            dist_matrix = torch.cdist(direct_coords, anchor_pos, p=2)
            # 距离转权重：距离越近权重越大，温度系数控制平滑度
            temperature = 0.1  # 可以做成可学习参数或配置项
            weights = F.softmax(-dist_matrix / temperature, dim=-1)  # (B, L, N_q)
            # 加权融合多个相邻anchor的特征 (B, L, D)
            anchor_context = torch.bmm(weights, attn_output)
            # 投影增强
            anchor_context = self.context_proj(anchor_context)

            # Step 8: 残差连接，每个token加自己空间位置对应的anchor上下文
            # 如果 use_ipm_semantic=True，direct_memory是bev + semantic拼接的 (B, 640+640, D)，需要复制anchor_context
            if self.use_ipm_semantic:
                anchor_context = torch.cat([anchor_context, anchor_context], dim=1)
            enhanced_memory = direct_memory + anchor_context
        else:
            anchor_context = None
            enhanced_memory = direct_memory

        return enhanced_memory, anchor_context, attn_output, anchor_pos

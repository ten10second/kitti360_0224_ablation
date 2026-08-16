"""Legacy vanilla AR token predictor (ICASSP27 refactor, stripped).

What survived from the MM26 version (checklist 保留项):
  - GPT-style decoder skeleton: causal self-attn + cross-attn to memory
  - CacheableAttention (KV cache for AR inference)
  - teacher-forcing helpers (topleft / bottomup raster orders)
  - e_pose 13-dim single pose token (VanillaPoseProjector)

What was removed here (checklist 删除项):
  - RayRoPEEncoder / RayDirectionEncoder (ray rotary)  -> vanilla 2D PE retained
  - PoseRouteCrossAttn / AnchorBasedSpatialFiLM / PoseAwareAnchorQuery wiring
    (anchor routing; module file deleted)
  - BEV ground-unproject + F_bev grid_sample memory path + SemanticEncoder
    + FourierCoordEncoder (BEV-coord variant; the metric-PE successor now
    lives in world3d/models/icassp27_predictor.py as MetricPE)
  - MultiScaleViTBEVEncoder (SatMAE) from the main path; the encoder file is
    kept on disk solely as the future --sat_encoder satmae ablation branch.
  - direct / hybrid modes (they existed only on top of the BEV/anchor paths)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vanilla_components import VanillaPositionEncoder, VanillaPoseProjector


# --------------------------------------------------------------------- utils
def seq_to_grid_topdown(seq: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    L = rows * cols
    if seq.dim() == 1:
        if seq.numel() != L:
            raise ValueError(f"seq length {seq.numel()} != rows*cols {L}")
        return seq.view(rows, cols)
    if seq.dim() == 2:
        if seq.shape[1] != L:
            raise ValueError(f"seq length {seq.shape[1]} != rows*cols {L}")
        return seq.view(seq.shape[0], rows, cols)
    raise ValueError(f"expects (L,) or (B,L), got {tuple(seq.shape)}")


def grid_to_seq_bottomup(tokens_grid: torch.Tensor) -> torch.Tensor:
    """(B,R,C) or (R,C) top-down grid -> (B,L)/(L,) bottom-up raster sequence."""
    if tokens_grid.dim() == 2:
        return tokens_grid.flip(0).reshape(-1)
    if tokens_grid.dim() == 3:
        return tokens_grid.flip(1).reshape(tokens_grid.shape[0], -1)
    raise ValueError(f"expects (R,C) or (B,R,C), got {tuple(tokens_grid.shape)}")


def seq_to_grid_bottomup(seq: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
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
    """nn.MultiheadAttention wrapper with KV caching for fast AR inference.

    (RoPE branches removed with RayRoPE; signature kept rope-free.)
    """

    def __init__(self, attn_module: nn.MultiheadAttention):
        super().__init__()
        self.attn = attn_module
        self.head_dim = attn_module.embed_dim // attn_module.num_heads
        self.num_heads = attn_module.num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        is_cross_attention = key.shape[1] != query.shape[1]

        if self.attn.in_proj_weight is not None:
            w_q, w_k, w_v = self.attn.in_proj_weight.chunk(3)
            b_q = b_k = b_v = None
            if self.attn.in_proj_bias is not None:
                b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3)
        else:
            w_q, w_k, w_v = self.attn.q_proj_weight, self.attn.k_proj_weight, self.attn.v_proj_weight
            b_q = b_k = b_v = None
            if self.attn.in_proj_bias is not None:
                b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3)

        B, L_q, _ = query.shape
        q = F.linear(query, w_q, b_q)
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)

        if is_cross_attention and use_cache and past_kv is not None:
            k, v = past_kv
            present_kv = past_kv
        else:
            k_current = F.linear(key, w_k, b_k).view(B, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            v_current = F.linear(value, w_v, b_v).view(B, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            if use_cache and past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k_current], dim=2)
                v = torch.cat([past_v, v_current], dim=2)
            else:
                k, v = k_current, v_current
            present_kv = (k, v) if use_cache else None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L_q, -1)
        out = self.attn.out_proj(out)
        return out, present_kv


# ------------------------------------------------------------------ model
class SimplifiedTokenPredictor(nn.Module):
    """Vanilla AR predictor: causal decoder over VQ tokens, memory = e_pose token."""

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
        pose_dim: int = 13,
        use_pose_token: bool = True,
        mode: str = "vanilla",  # only "vanilla" survives the refactor
    ):
        super().__init__()
        if str(mode).lower() != "vanilla":
            raise ValueError(
                f"mode '{mode}' was removed in the ICASSP27 refactor "
                "(direct/hybrid depended on the deleted BEV/anchor paths); only 'vanilla' remains."
            )
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.mode = "vanilla"
        self.ntp_order = "topleft"

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.vanilla_pos_encoder = VanillaPositionEncoder(
            d_model=d_model, max_h=self.target_rows, max_w=self.target_cols
        )
        self.use_pose_token = bool(use_pose_token)
        self.pose_dim = int(pose_dim)
        self.vanilla_pose_projector = VanillaPoseProjector(pose_dim=self.pose_dim, d_model=d_model)

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
        for layer in decoder_layers:
            layer.self_attn = CacheableAttention(layer.self_attn)
            layer.multihead_attn = CacheableAttention(layer.multihead_attn)
        self.decoder_layers = decoder_layers

        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    # --------------------------------------------------------------- helpers
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def _token_2d_pos(self, batch_size: int, past_length: int, seq_len: int, device) -> torch.Tensor:
        rows = (past_length + torch.arange(seq_len, device=device)) // self.target_cols
        cols = (past_length + torch.arange(seq_len, device=device)) % self.target_cols
        y = ((rows + 0.5) / self.target_rows).view(1, -1).expand(batch_size, -1)
        x = ((cols + 0.5) / self.target_cols).view(1, -1).expand(batch_size, -1)
        return self.vanilla_pos_encoder(y, x)

    def seq_to_grid(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        return seq_to_grid_topdown(tokens_seq, self.target_rows, self.target_cols)

    def make_teacher_forcing(
        self, target_token_grid: torch.Tensor, bos_token: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B,R,C) top-down grid -> (input=[BOS, t_{<L-1}], label=t_{1..L}) topleft raster."""
        if target_token_grid.dim() == 2:
            target_token_grid = target_token_grid.unsqueeze(0)
        B, R, C = target_token_grid.shape
        target_seq = target_token_grid.reshape(B, R * C)
        bos = torch.full((B, 1), bos_token, dtype=target_seq.dtype, device=target_seq.device)
        input_seq = torch.cat([bos, target_seq[:, :-1]], dim=1)
        return input_seq, target_seq

    # --------------------------------------------------------------- forward
    def forward(
        self,
        generated_tokens: torch.Tensor,
        condition_tokens: dict,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        _, L_gen = generated_tokens.shape
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0][0].size(-2)
        if L_gen + past_length > self.max_seq_len:
            raise ValueError(f"Sequence length {L_gen + past_length} exceeds max_seq_len={self.max_seq_len}")

        tgt = self.token_embed(generated_tokens)
        tgt = self.input_norm(tgt + self._token_2d_pos(tgt.size(0), past_length, L_gen, tgt.device))

        pose_input = condition_tokens.get("pose", None)
        memory_parts = []
        if self.use_pose_token:
            if pose_input is None:
                raise KeyError("condition_tokens must contain 'pose' when use_pose_token=True")
            if pose_input.dim() != 2 or pose_input.size(-1) != self.pose_dim:
                raise ValueError(f"pose must be (B,{self.pose_dim}), got {tuple(pose_input.shape)}")
            memory_parts.append(self.vanilla_pose_projector(pose_input))
        memory = torch.cat(memory_parts, dim=1) if memory_parts else None

        full_seq_len = past_length + L_gen
        causal_mask = self._generate_causal_mask(full_seq_len, tgt.device)
        if use_cache and past_length > 0:
            causal_mask = causal_mask[-L_gen:, :]

        present_key_values = [] if use_cache else None
        hidden_states = tgt
        for i, layer in enumerate(self.decoder_layers):
            past_sa_kv = past_key_values[i][0] if past_key_values is not None else None
            past_ca_kv = past_key_values[i][1] if past_key_values is not None else None

            sa_out, sa_present_kv = layer.self_attn(
                hidden_states, hidden_states, hidden_states,
                past_kv=past_sa_kv, use_cache=use_cache, attn_mask=causal_mask,
            )
            hidden_states = layer.norm1(hidden_states + layer.dropout1(sa_out))

            if memory is not None:
                ca_out, ca_present_kv = layer.multihead_attn(
                    hidden_states, memory, memory,
                    past_kv=past_ca_kv, use_cache=use_cache,
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
    """Same model; AR factorization order is bottom-up raster (last row -> first)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ntp_order = "bottomup"

    @staticmethod
    def grid_to_seq(tokens_grid: torch.Tensor) -> torch.Tensor:
        return grid_to_seq_bottomup(tokens_grid)

    def seq_to_grid(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        return seq_to_grid_bottomup(tokens_seq, self.target_rows, self.target_cols)

    def make_teacher_forcing(
        self, target_token_grid: torch.Tensor, bos_token: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_token_grid.dim() == 2:
            target_token_grid = target_token_grid.unsqueeze(0)
        target_seq = grid_to_seq_bottomup(target_token_grid)
        B, L = target_seq.shape
        bos = torch.full((B, 1), bos_token, dtype=target_seq.dtype, device=target_seq.device)
        input_seq = torch.cat([bos, target_seq[:, :-1]], dim=1)
        return input_seq, target_seq

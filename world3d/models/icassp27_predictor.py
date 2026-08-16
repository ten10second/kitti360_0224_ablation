"""ICASSP27 predictor: satellite-anchored AR street-view token decoder.

Doc mapping (ICASSP27_method_framework.md):
- §3.1 tokenizer stays external (models/stage1/maskgit/tokenizer.py), frozen.
- §3.2 satellite branch: DINOv2(sat) patch tokens + metric PE (Fourier on
  per-patch world (x,y) meters, added per token). No BEV sampling.
- §3.3 street branch: shared DINOv2 backbone; per-source rel-pose
  (dt(3)+rot6d(6) -> MLP) added to ALL tokens of that source; variable K
  concatenated in order.
- §3.4 e_pose: reused 13-dim pose vector + VanillaPoseProjector (legacy).
- §3.5 decoder: causal self-attn over target tokens with STANDARD 1D learned
  positional embedding (RayRoPE removed -> must re-add 1D PE, doc pitfall #1)
  + cross-attn to memory M = e_pose || f_sat || h_1..h_K. Single L_ce loss.

Ablation switches (doc §6/§7):
  use_sat / use_src            -> B0 (sat only) / B1 (src only) / B2 (both)
  sat_encoder {dino, satmae}   -> SatMAE legacy encoder kept as ablation branch
  geo {pose_add, rayrope, ipm} -> pose_add implemented here; rayrope/ipm are
                                  legacy-reuse ablation rows (not wired in the
                                  pilot; raise NotImplementedError for now).

Note: DINOv2 official checkpoints are patch-14 (no /16 variant exists). Inputs
are resized to multiples of 14 (518 for sat 512px, 518x252 for 640x256 street).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stage2.vanilla_components import VanillaPoseProjector


def _load_dinov2(arch: str):
    """Load DINOv2 from the local hub cache when available (avoids hanging on
    flaky GitHub access); falls back to a normal hub.load otherwise."""
    import torch.hub as hub
    from pathlib import Path as _P

    name = f"dinov2_{arch}"
    repo_dir = _P(hub.get_dir()) / "facebookresearch_dinov2_main"
    ckpt = _P(hub.get_dir()) / "checkpoints" / f"{name}_pretrain.pth"
    if repo_dir.exists() and ckpt.exists():
        return hub.load(str(repo_dir), name, pretrained=True, source="local")
    return hub.load("facebookresearch/dinov2", name, pretrained=True)


class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 patch-token extractor (shared by sat & street branches)."""

    def __init__(self, arch: str = "vitb14"):
        super().__init__()
        self.backbone = _load_dinov2(arch)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.dim = self.backbone.embed_dim
        self.patch = self.backbone.patch_size
        # ImageNet normalization (DINOv2 training stats)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1] -> patch tokens (B, N, D)."""
        x = (x - self.mean) / self.std
        feats = self.backbone.forward_features(x)
        return feats["x_norm_patchtokens"]


class MetricPE(nn.Module):
    """Fourier encoding of metric world (x,y) + MLP, added per token (doc §3.2).

    Reuses the design of the legacy FourierCoordEncoder (kept item) but as a
    standalone module decoupled from the BEV/IPM data path.
    """

    def __init__(self, d_model: int, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(2 * 2 * num_freqs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (B, N, 2) meters -> (B, N, d_model)."""
        freqs = 2.0 ** torch.arange(self.num_freqs, device=xy.device, dtype=xy.dtype)
        ang = xy[..., None] * freqs  # (B,N,2,F)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B,N,2F)
        pe = pe.reshape(*xy.shape[:-1], 2 * 2 * self.num_freqs)
        return self.mlp(pe)


class RelPoseProjector(nn.Module):
    """Per-source relative pose (dt 3 + rot6d 6) -> D, broadcast over tokens."""

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, dp: torch.Tensor) -> torch.Tensor:
        return self.mlp(dp)  # (B,K,D)


class TargetRayPE(nn.Module):
    """Per-token target-view ray embedding (geo=raymap, mainstream conditioning).

    Each target token (40x16 grid) gets the 6-dim (o, d) of its pixel's camera
    ray: origin o = camera center in the WINDOW-LOCAL frame (this is what makes
    the absolute position within the shared satellite window observable), and
    unit direction d = R @ K^-1 (u,v,1) / |.|. Encoded by a small MLP and added
    to the token embedding — same injection pattern as MetricPE / RelPose.
    """

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, o: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """o: (B,N,3) meters, d: (B,N,3) unit -> (B,N,d_model)."""
        return self.mlp(torch.cat([o, d], dim=-1))


class ICASSP27Predictor(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1080,
        pose_dim: int = 13,
        dino_arch: str = "vitb14",
        sat_encoder: str = "dino",       # "dino" | "satmae"
        geo: str = "raymap",             # "raymap" (per-token target rays) | "pose_add" (e_pose only) | "proj" (PVSM-style, ablation placeholder)
        use_sat: bool = True,
        use_src: bool = True,
        fourier_freqs: int = 10,
        sat_px: int = 512,               # on-disk satellite size
        sat_m_per_px: float = 0.196,
        src_size: tuple = (640, 256),    # (W,H) of street images
        target_rows: int = 16,
        target_cols: int = 40,
    ):
        super().__init__()
        assert sat_encoder in ("dino", "satmae")
        assert geo in ("raymap", "pose_add", "proj")
        if sat_encoder == "satmae":
            raise NotImplementedError("SatMAE ablation branch: wire models/multiscale_vit_encoder.py here")
        if geo == "proj":
            raise NotImplementedError("proj ablation row (PVSM-style source->target projection): not wired yet")
        assert use_sat or use_src, "B0/B1/B2 all need at least one condition branch"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_sat = use_sat
        self.use_src = use_src
        self.sat_px = sat_px
        self.sat_m_per_px = sat_m_per_px

        # encoders (shared DINOv2 for sat & street, doc §3.3)
        self.dino = DinoV2Encoder(dino_arch)
        self.sat_proj = nn.Linear(self.dino.dim, d_model)
        self.src_proj = nn.Linear(self.dino.dim, d_model)
        self.metric_pe = MetricPE(d_model, fourier_freqs)
        self.rel_pose = RelPoseProjector(d_model)
        self.ray_pe = TargetRayPE(d_model)
        self.pose_proj = VanillaPoseProjector(pose_dim, d_model)  # e_pose reuse
        self.geo = geo
        self.target_rows = int(target_rows)
        self.target_cols = int(target_cols)

        # DINOv2 input sizes (patch 14)
        self.dino_sat_size = (518, 518)
        self.dino_src_size = (518, 252)  # 640x256 -> (37, 18) patches
        self.img_w, self.img_h = int(src_size[0]), int(src_size[1])
        self.sat_grid = (self.dino_sat_size[0] // self.dino.patch, self.dino_sat_size[1] // self.dino.patch)
        self.src_tokens_per_view = (self.dino_src_size[0] // self.dino.patch) * (self.dino_src_size[1] // self.dino.patch)

        # AR decoder over target tokens
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)  # +1 = BOS slot
        self.bos_idx = vocab_size
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))  # 1D learned PE
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------- memory
    # ------------------------------------------------------------ target rays
    def _target_rays(self, tgt_K: torch.Tensor, tgt_T_cam: torch.Tensor, window_origin_xyz: torch.Tensor):
        """(o, d) per target token. o = camera center in the window frame;
        d = unit ray direction from intrinsics + camera rotation.
        Returns two (B, rows*cols, 3) tensors in token raster order."""
        B = tgt_K.shape[0]
        dev = tgt_K.device
        rows, cols = self.target_rows, self.target_cols
        ph = self.img_h / rows
        pw = self.img_w / cols
        v = (torch.arange(rows, device=dev, dtype=torch.float32) + 0.5) * ph
        u = (torch.arange(cols, device=dev, dtype=torch.float32) + 0.5) * pw
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # row-major: rows outer, cols inner
        pix = torch.stack(
            [uu.reshape(-1), vv.reshape(-1), torch.ones(rows * cols, device=dev)],
            dim=-1,
        )  # (N,3), N == rows*cols == target token count
        p_cam = torch.einsum("bij,nj->bni", torch.inverse(tgt_K), pix)  # (B,N,3)
        d_cam = p_cam / p_cam.norm(dim=-1, keepdim=True)
        d = torch.einsum("bij,bnj->bni", tgt_T_cam[:, :3, :3], d_cam)
        o = (tgt_T_cam[:, :3, 3] - window_origin_xyz)[:, None, :].expand(-1, rows * cols, -1)
        return o, d

    def _token_ray_pe(self, tgt_K, tgt_T_cam, window_origin_xyz, L: torch.Tensor):
        """Ray PE aligned to input positions: position 0 is BOS (zero), position
        i>=1 carries the ray of image token i-1."""
        o, d = self._target_rays(tgt_K, tgt_T_cam, window_origin_xyz)
        pe = self.ray_pe(o, d)  # (B,N,D)
        zero = pe.new_zeros(pe.shape[0], 1, pe.shape[2])
        return torch.cat([zero, pe[:, : L - 1]], dim=1)  # (B,L,D)

    def _sat_patch_world_xy(self, window_origin_xy: torch.Tensor) -> torch.Tensor:
        """World (x,y) of satellite patch centers. Sat crop is north-up and
        centered at the window-center vehicle position (512px @ mpp).
        Returns (B, Ns, 2) meters."""
        B = window_origin_xy.shape[0]
        gh, gw = self.sat_grid  # rows (v, south+), cols (u, east+)
        dev = window_origin_xy.device
        v = (torch.arange(gh, device=dev, dtype=torch.float32) + 0.5) * self.dino_sat_size[0] / gh
        u = (torch.arange(gw, device=dev, dtype=torch.float32) + 0.5) * self.dino_sat_size[1] / gw
        # resized-input coords -> on-disk 512px coords
        u = u * self.sat_px / self.dino_sat_size[1]
        v = v * self.sat_px / self.dino_sat_size[0]
        # DINOv2 patch tokens are row-major over image rows (v) then cols (u)
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # each (gh, gw)
        x = window_origin_xy[:, 0:1] + (uu.reshape(1, -1) - self.sat_px / 2.0) * self.sat_m_per_px
        y = window_origin_xy[:, 1:2] - (vv.reshape(1, -1) - self.sat_px / 2.0) * self.sat_m_per_px
        return torch.stack([x, y], dim=-1)  # (B, Ns, 2)

    def build_memory(
        self,
        pose_vec: torch.Tensor,          # (B, 13)
        sat: Optional[torch.Tensor],     # (B,3,512,512) [0,1]
        window_origin_xyz: Optional[torch.Tensor],  # (B,3) world meters (window frame origin)
        src_rgbs: Optional[torch.Tensor],  # (B,K,3,H,W) [0,1]
        rel_poses: Optional[torch.Tensor],  # (B,K,9)
        src_mask: Optional[torch.Tensor] = None,  # (B,K) bool
    ):
        parts = [self.pose_proj(pose_vec)]  # e_pose first token
        key_pad = [torch.zeros(pose_vec.shape[0], 1, dtype=torch.bool, device=pose_vec.device)]
        if self.use_sat:
            assert sat is not None and window_origin_xyz is not None
            B, K = src_rgbs.shape[:2] if src_rgbs is not None else (sat.shape[0], 0)
            f = self.dino(F.interpolate(sat, size=self.dino_sat_size, mode="bilinear", align_corners=False))
            f = self.sat_proj(f) + self.metric_pe(self._sat_patch_world_xy(window_origin_xyz[:, :2]))
            parts.append(f)
            key_pad.append(torch.zeros(B, f.shape[1], dtype=torch.bool, device=f.device))
        if self.use_src:
            assert src_rgbs is not None and rel_poses is not None
            B, K, C, H, W = src_rgbs.shape
            flat = src_rgbs.reshape(B * K, C, H, W)
            flat = F.interpolate(flat, size=self.dino_src_size, mode="bilinear", align_corners=False)
            f = self.dino(flat).reshape(B, K, self.src_tokens_per_view, -1)
            h = self.src_proj(f) + self.rel_pose(rel_poses)[:, :, None, :]  # rel-pose add per source
            parts.append(h.reshape(B, K * self.src_tokens_per_view, -1))
            pad = (~src_mask) if src_mask is not None else torch.zeros(B, K, dtype=torch.bool, device=h.device)
            key_pad.append(pad[:, :, None].expand(B, K, self.src_tokens_per_view).reshape(B, -1))
        memory = torch.cat(parts, dim=1)
        key_padding = torch.cat(key_pad, dim=1)  # True = ignore
        return memory, key_padding

    # ------------------------------------------------------------- forward
    def forward(
        self,
        input_tokens: torch.Tensor,   # (B, L) teacher-forcing input (BOS-shifted)
        pose_vec: torch.Tensor,
        sat: Optional[torch.Tensor] = None,
        window_origin_xyz: Optional[torch.Tensor] = None,   # (B,3) required when geo=raymap+use_sat
        src_rgbs: Optional[torch.Tensor] = None,
        rel_poses: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_K: Optional[torch.Tensor] = None,               # (B,3,3) required when geo=raymap
        tgt_T_cam: Optional[torch.Tensor] = None,           # (B,4,4) required when geo=raymap
    ) -> torch.Tensor:
        memory, key_padding = self.build_memory(pose_vec, sat, window_origin_xyz, src_rgbs, rel_poses, src_mask)
        B, L = input_tokens.shape
        x = self.token_embed(input_tokens) + self.pos_embed[:, :L]
        if self.geo == "raymap":
            assert tgt_K is not None and tgt_T_cam is not None, "geo=raymap needs tgt_K and tgt_T_cam"
            x = x + self._token_ray_pe(tgt_K, tgt_T_cam, window_origin_xyz if window_origin_xyz is not None else tgt_T_cam.new_zeros(B, 3), L)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, memory, tgt_mask=causal, memory_key_padding_mask=key_padding)
        return self.head(self.norm(x))  # (B, L, vocab)

    @torch.no_grad()
    def generate(self, pose_vec, *, max_len: int, temperature: float = 1.0, top_p: float = 0.95,
                 sat=None, window_origin_xyz=None, src_rgbs=None, rel_poses=None, src_mask=None,
                 tgt_K=None, tgt_T_cam=None):
        """AR sampling (temperature + top-p fixed policy, doc §3.5)."""
        B = pose_vec.shape[0]
        tokens = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=pose_vec.device)
        memory, key_padding = self.build_memory(pose_vec, sat, window_origin_xyz, src_rgbs, rel_poses, src_mask)
        for _ in range(max_len):
            L = tokens.shape[1]
            x = self.token_embed(tokens) + self.pos_embed[:, :L]
            if self.geo == "raymap":
                assert tgt_K is not None and tgt_T_cam is not None, "geo=raymap needs tgt_K and tgt_T_cam"
                x = x + self._token_ray_pe(tgt_K, tgt_T_cam, window_origin_xyz if window_origin_xyz is not None else tgt_T_cam.new_zeros(B, 3), L)
            causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
            for blk in self.blocks:
                x = blk(x, memory, tgt_mask=causal, memory_key_padding_mask=key_padding)
            logits = self.head(self.norm(x[:, -1]))
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
            cum = sorted_p.cumsum(-1)
            keep = cum - sorted_p < top_p
            sorted_p = sorted_p * keep
            sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
            idx = torch.multinomial(sorted_p, 1)
            nxt = torch.gather(sorted_i, 1, idx)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens[:, 1:]

    def make_teacher_forcing(self, target_tokens: torch.Tensor) -> tuple:
        """(input=[BOS, t_{<L-1}], label=t_{1..L})"""
        B, L = target_tokens.shape
        bos = torch.full((B, 1), self.bos_idx, dtype=target_tokens.dtype, device=target_tokens.device)
        inp = torch.cat([bos, target_tokens[:, :-1]], dim=1)
        return inp, target_tokens

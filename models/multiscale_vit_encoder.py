"""
Multi-scale Vision Transformer BEV Feature Extractor based on SatMAE.
Extracts features from intermediate layers of a ViT-Large model.
"""

import numpy as np
import torch
import torch.nn as nn
from functools import partial
import timm.models.vision_transformer

# --------------------------------------------------------
# Position Embedding Utilities (from SatMAE)
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = 1. / 10000**(np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.))
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

# --------------------------------------------------------
# Multi-Scale Vision Transformer
# --------------------------------------------------------
class VisionTransformerForMultiScale(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, output_layers=[6, 12, 18, 24], **kwargs):
        super(VisionTransformerForMultiScale, self).__init__(**kwargs)
        self.output_layers = output_layers
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size[0], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 in self.output_layers:
                # Return patch tokens (without cls token)
                features.append(self.norm(x[:, 1:, :]))
        return features

def vit_large_patch16_ms(**kwargs):
    model = VisionTransformerForMultiScale(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# --------------------------------------------------------
# Multi-Scale BEV Encoder Wrapper
# --------------------------------------------------------
class MultiScaleViTBEVEncoder(nn.Module):
    def __init__(self, output_channels=256, output_size=64, input_size=512, freeze_backbone=True, pretrained_path=None):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.output_size = output_size
        self.patch_dim = input_size // 16
        self.backbone = vit_large_patch16_ms(img_size=input_size, output_layers=[6, 12, 18, 24])
        
        # from satMAE
        self.register_buffer("fmow_mean", torch.tensor([0.418, 0.421, 0.399]).view(1, 3, 1, 1))
        self.register_buffer("fmow_std",  torch.tensor([0.288, 0.275, 0.276]).view(1, 3, 1, 1))


        if pretrained_path:
            print(f"[MultiScaleViT] Loading weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint['model']
            # The pretrained checkpoint does not have an 'encoder.' prefix.
            # We pass the whole state_dict and let strict=False handle non-matching keys.
            interpolate_pos_embed(self.backbone, state_dict)
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[MultiScaleViT] Pretrained weights loaded with status: {msg}")

        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("[MultiScaleViT] Backbone frozen.")

        # Projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Conv2d(1024, 256, kernel_size=1) for _ in range(4)
        ])
        # Final fusion layer
        self.fusion = nn.Conv2d(256 * 4, output_channels, kernel_size=1)

    def forward(self, x):

        # ---- fMoW RGB normalization (expect x in [0,1]) ----
        # x: [B, 3, H, W]
        if not torch.is_floating_point(x):
            x = x.float()
        # Ensure fmow_mean and fmow_std are on the same device as x
        fmow_mean = self.fmow_mean.to(device=x.device)
        fmow_std = self.fmow_std.to(device=x.device)
        x = (x - fmow_mean) / (fmow_std + 1e-6)
        # ---------------------------------------------------

        if self.freeze_backbone:
            with torch.no_grad():
                # 确保在评估模式下运行
                self.backbone.eval()
                multi_scale_features = self.backbone.forward_features(x)
                # 确保输出不会参与梯度计算
                for i, features in enumerate(multi_scale_features):
                    multi_scale_features[i] = features.detach()
        else:
            multi_scale_features = self.backbone.forward_features(x)

        processed_features = []
        for i, features in enumerate(multi_scale_features):
            B, N, C = features.shape
            features_2d = features.view(B, self.patch_dim, self.patch_dim, C).permute(0, 3, 1, 2)
            projected = self.projections[i](features_2d)
            upsampled = nn.functional.interpolate(projected, size=(self.patch_dim, self.patch_dim), mode='bilinear', align_corners=False)
            processed_features.append(upsampled)

        fused = torch.cat(processed_features, dim=1)
        bev_features = self.fusion(fused)

        return nn.functional.interpolate(
            bev_features, size=(self.output_size, self.output_size),
            mode='bilinear', align_corners=False
        )

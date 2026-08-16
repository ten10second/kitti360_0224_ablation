#!/usr/bin/env python3
"""
几何一致性损失 (Geometric Consistency Loss)

实现可微分的逆投影损失，确保生成的前视图在投影到卫星图时与真实卫星图一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

from ..geometry.differentiable_projection import differentiable_camera_to_sat_warp


class GeometricLoss(nn.Module):
    """
    几何一致性损失
    
    核心思想：
    1. 将预测的 logits 通过 soft embedding 转换为可微分的 latent
    2. 使用 VQGAN decoder 解码为 RGB 图像  
    3. 投影到卫星图并与真实卫星图比较
    4. 只在有效 mask 区域计算损失
    """
    
    def __init__(
        self,
        vqgan_decoder,
        token_embedding: nn.Embedding,
        vqgan_z_channels: int = 256,  # VQGAN latent 的通道数
        loss_type: str = 'l1',  # 'l1', 'l2'
        soft_token_method: str = 'adaptive',  # 'softmax', 'gumbel', 'adaptive'
        initial_temperature: float = 2.0,
        final_temperature: float = 0.1,
        sat_size: int = 512,
        resolution: float = 0.2,
        lambda_geometric: float = 0.1,
        warmup_epochs: int = 5,
        transition_epochs: int = 15,
    ):
        super().__init__()

        self.vqgan_decoder = vqgan_decoder
        self.token_embedding = token_embedding
        self.vqgan_z_channels = vqgan_z_channels
        self.loss_type = loss_type
        self.soft_token_method = soft_token_method
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.sat_size = sat_size
        self.resolution = resolution
        self.lambda_geometric = lambda_geometric
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs

        # 投影层：将 token embedding (d_model) 投影到 VQGAN latent space (z_channels)
        d_model = token_embedding.embedding_dim
        if d_model != vqgan_z_channels:
            self.projection = nn.Linear(d_model, vqgan_z_channels)
            print(f"[GeometricLoss] 添加投影层: {d_model} → {vqgan_z_channels}")
        else:
            self.projection = None

        # 冻结 VQGAN decoder（可选）
        for param in self.vqgan_decoder.parameters():
            param.requires_grad = False

        print(f"[GeometricLoss] 初始化完成:")
        print(f"  Token embedding dim: {d_model}")
        print(f"  VQGAN z_channels: {vqgan_z_channels}")
        print(f"  Loss type: {loss_type}")
        print(f"  Soft token method: {soft_token_method}")
        print(f"  Temperature: {initial_temperature} → {final_temperature}")
        print(f"  Lambda geometric: {lambda_geometric}")
        print(f"  VQGAN decoder frozen: True")
    
    def get_temperature(self, epoch: int) -> float:
        """
        获取当前 epoch 的温度
        """
        if epoch < self.warmup_epochs:
            return self.initial_temperature
        elif epoch < self.warmup_epochs + self.transition_epochs:
            # 指数衰减
            progress = (epoch - self.warmup_epochs) / self.transition_epochs
            return self.initial_temperature * (self.final_temperature / self.initial_temperature) ** progress
        else:
            return self.final_temperature
    
    def get_lambda_weight(self, epoch: int, ce_loss: float, geometric_loss: float) -> float:
        """
        自适应调整几何损失权重
        """
        # 预热期不使用几何损失
        if epoch < self.warmup_epochs:
            return 0.0
        
        # 基础权重
        base_weight = self.lambda_geometric
        
        # 根据损失比例调整
        if geometric_loss > 0 and ce_loss > 0:
            ratio = geometric_loss / ce_loss
            if ratio > 10:  # 几何损失太大
                base_weight *= 0.1
            elif ratio > 5:
                base_weight *= 0.5
            elif ratio < 0.1:  # 几何损失太小
                base_weight *= 2.0
        
        return min(base_weight, 0.2)  # 上限
    
    def soft_token_to_latent(
        self, 
        logits: torch.Tensor,  # (B, L, vocab_size)
        latent_shape: Tuple[int, int, int, int],  # (B, C, H, W)
        epoch: int,
    ) -> Tuple[torch.Tensor, str]:
        """
        将 logits 转换为可微分的 latent representation
        """
        B, L, vocab_size = logits.shape
        B_target, C_target, H_target, W_target = latent_shape
        
        if self.soft_token_method == 'softmax':
            # 标准 Softmax
            soft_tokens = F.softmax(logits, dim=-1)
            method_info = "Softmax"
            
        elif self.soft_token_method == 'gumbel':
            # Gumbel Softmax
            temperature = self.get_temperature(epoch)
            soft_tokens = F.gumbel_softmax(
                logits, 
                tau=temperature, 
                hard=False, 
                dim=-1
            )
            method_info = f"Gumbel(τ={temperature:.2f})"
            
        elif self.soft_token_method == 'adaptive':
            # 自适应方法
            temperature = self.get_temperature(epoch)
            
            if epoch < self.warmup_epochs:
                # 预热期：不计算几何损失
                soft_tokens = F.softmax(logits, dim=-1)
                method_info = "Warmup-Softmax"
            elif epoch < self.warmup_epochs + self.transition_epochs:
                # 过渡期：Gumbel Softmax
                soft_tokens = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
                method_info = f"Gumbel(τ={temperature:.2f})"
            else:
                # 稳定期：Softmax
                soft_tokens = F.softmax(logits, dim=-1)
                method_info = "Stable-Softmax"
        else:
            raise ValueError(f"Unknown soft_token_method: {self.soft_token_method}")
        
        # 通过概率加权获得 soft embedding
        # 注意：MaskTransformer 使用 vocab_size+1 的 token_emb（最后一个是mask token），
        # 而 logits 的最后一维是 vocab_size。因此这里对齐维度，截取前 V 行。
        V = logits.shape[-1]
        emb_weight = self.token_embedding.weight
        if emb_weight.shape[0] != V:
            emb_weight = emb_weight[:V, :]
        z_latent_flat = torch.einsum(
            'blv,vd->bld',
            soft_tokens,
            emb_weight
        )  # (B, L, d_model)

        # 投影到 VQGAN latent space
        if self.projection is not None:
            z_latent_flat = self.projection(z_latent_flat)  # (B, L, z_channels)

        # 重新整形为 spatial layout
        z_channels = z_latent_flat.shape[-1]
        z_latent = z_latent_flat.reshape(B, H_target, W_target, z_channels).permute(0, 3, 1, 2)
        # z_latent: (B, z_channels, H, W)

        return z_latent, method_info
    
    def forward(
        self,
        logits: torch.Tensor,  # (B, L, vocab_size) - 预测的 token logits
        K: torch.Tensor,       # (3, 3) or (B, 3, 3) - 相机内参
        T_cam_to_world: torch.Tensor,  # (4, 4) or (B, 4, 4) - 相机到世界变换
        sat_image_gt: torch.Tensor,    # (B, 3, sat_size, sat_size) - 真实卫星图
        latent_shape: Tuple[int, int, int, int],  # VQGAN latent 的形状 (B, C, H, W)
        epoch: int,
        ce_loss: float = 0.0,  # 用于自适应权重调整
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算几何一致性损失
        
        Returns:
            geometric_loss: 几何一致性损失
            info: 调试信息字典
        """
        device = logits.device
        
        # 步骤 1: 将 logits 转换为可微分的 latent
        z_latent, method_info = self.soft_token_to_latent(logits, latent_shape, epoch)
        
        # 步骤 2: 使用 VQGAN decoder 解码为 RGB
        with torch.no_grad() if epoch < self.warmup_epochs else torch.enable_grad():
            if epoch < self.warmup_epochs:
                # 预热期：返回零损失
                geometric_loss = torch.tensor(0.0, device=device)
                info = {
                    'geometric_loss': 0.0,
                    'valid_ratio': 0.0,
                    'valid_pixel_count': 0,
                    'warped_range': (0.0, 0.0),
                    'method': method_info,
                    'temperature': self.get_temperature(epoch),
                    'lambda_weight': 0.0,
                }
                return geometric_loss, info
        
        x_rgb_pred = self.vqgan_decoder(z_latent)  # (B, 3, H_img, W_img)
        
        # 步骤 3: 可微分投影到卫星图
        sat_image_warped, valid_mask = differentiable_camera_to_sat_warp(
            x_rgb_pred,
            K,
            T_cam_to_world,
            sat_size=self.sat_size,
            resolution=self.resolution,
        )  # (B, 3, sat_size, sat_size), (B, 1, sat_size, sat_size)
        
        # 步骤 4: 计算带掩码的损失
        valid_mask_3ch = valid_mask.expand_as(sat_image_warped)  # (B, 3, sat_size, sat_size)
        
        # 只在有效区域计算损失
        warped_masked = sat_image_warped * valid_mask_3ch
        target_masked = sat_image_gt * valid_mask_3ch
        
        if self.loss_type == 'l1':
            loss_per_pixel = F.l1_loss(warped_masked, target_masked, reduction='none')
        elif self.loss_type == 'l2':
            loss_per_pixel = F.mse_loss(warped_masked, target_masked, reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # 只对有效像素求平均
        valid_pixel_count = valid_mask.sum().clamp(min=1e-6)
        geometric_loss = loss_per_pixel.sum() / valid_pixel_count
        
        # 调试信息
        info = {
            'geometric_loss': geometric_loss.item(),
            'valid_ratio': valid_mask.mean().item(),
            'valid_pixel_count': valid_pixel_count.item(),
            'warped_range': (sat_image_warped.min().item(), sat_image_warped.max().item()),
            'method': method_info,
            'temperature': self.get_temperature(epoch),
            'lambda_weight': self.get_lambda_weight(epoch, ce_loss, geometric_loss.item()),
        }
        
        return geometric_loss, info

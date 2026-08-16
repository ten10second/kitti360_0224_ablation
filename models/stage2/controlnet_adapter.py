"""
ControlNet Adapter for Camera Pose-conditioned View Synthesis.

This module provides a unified interface for working with different ControlNet
architectures in the diffusion model pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from diffusers import ControlNetModel


class ControlNetAdapter(nn.Module):
    """
    Adapter for integrating ControlNet into the diffusion pipeline.

    Provides a unified interface for working with different ControlNet architectures
    while maintaining compatibility with Stable Diffusion.
    """

    def __init__(
        self,
        controlnet_type: str = "custom",
        pretrained_model_name_or_path: Optional[str] = None,
        in_channels: int = 3,
        out_channels: int = 320,
        num_timesteps: int = 1000,
    ):
        super().__init__()

        self.controlnet_type = controlnet_type

        if controlnet_type == "custom":
            # Custom ControlNet for our pipeline
            self.controlnet = ControlNetModel.from_pretrained(
                pretrained_model_name_or_path or "runwayml/stable-diffusion-v1-5",
                subfolder="unet",
                in_channels=in_channels,
            )
        elif controlnet_type == "ip2p":
            # Image Prompt to Prompt ControlNet
            self.controlnet = ControlNetModel.from_pretrained(
                pretrained_model_name_or_path or "lllyasviel/sd-controlnet-ip2p",
            )
        elif controlnet_type == "canny":
            # Canny edge ControlNet
            self.controlnet = ControlNetModel.from_pretrained(
                pretrained_model_name_or_path or "lllyasviel/sd-controlnet-canny",
            )
        elif controlnet_type == "depth":
            # Depth map ControlNet
            self.controlnet = ControlNetModel.from_pretrained(
                pretrained_model_name_or_path or "lllyasviel/sd-controlnet-depth",
            )
        elif controlnet_type == "openpose":
            # OpenPose ControlNet (for human pose)
            self.controlnet = ControlNetModel.from_pretrained(
                pretrained_model_name_or_path or "lllyasviel/sd-controlnet-openpose",
            )
        else:
            raise ValueError(f"Unknown controlnet_type: {controlnet_type}")

        # Additional configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps

        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1000, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # ControlNet conditioning adapter
        self.control_adapter = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for new layers."""
        for m in self.timestep_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.control_adapter.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_control_condition(self, control_image: torch.Tensor) -> torch.Tensor:
        """
        Encode control image into appropriate format for ControlNet.

        Args:
            control_image: (B, C, H, W) control image

        Returns:
            encoded_control: (B, C', H, W) encoded control condition
        """
        return self.control_adapter(control_image)

    def encode_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Encode timesteps for ControlNet.

        Args:
            timesteps: (B,) timesteps

        Returns:
            embeddings: (B, 512) timestep embeddings
        """
        # Convert to one-hot
        one_hot = F.one_hot(timesteps.clamp(0, self.num_timesteps - 1), self.num_timesteps)
        return self.timestep_embedding(one_hot.float())

    def forward(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ControlNet.

        Args:
            x: (B, 4, H, W) latent representation
            hint: (B, C, H, W) control image
            timesteps: (B,) timesteps
            encoder_hidden_states: (B, seq_len, d_model) prompt embeddings

        Returns:
            down_block_res_samples: list of downsampling block outputs
            mid_block_res_sample: mid block output
        """
        # Encode control condition
        encoded_control = self.encode_control_condition(hint)

        # Forward through ControlNet
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=encoded_control,
            return_dict=False,
        )

        return down_block_res_samples, mid_block_res_sample

    def freeze_controlnet(self):
        """Freeze ControlNet weights."""
        for param in self.controlnet.parameters():
            param.requires_grad = False
        self.controlnet.eval()

    def unfreeze_controlnet(self):
        """Unfreeze ControlNet weights."""
        for param in self.controlnet.parameters():
            param.requires_grad = True
        self.controlnet.train()

    def get_trainable_parameters(self, include_controlnet: bool = True):
        """
        Get trainable parameters.

        Args:
            include_controlnet: whether to include ControlNet parameters

        Returns:
            list of trainable parameters
        """
        params = []

        if include_controlnet:
            params.extend(list(self.controlnet.parameters()))

        # Always include adapter parameters
        params.extend(list(self.timestep_embedding.parameters()))
        params.extend(list(self.control_adapter.parameters()))

        return params

    def save_pretrained(self, save_directory: str):
        """Save ControlNet to disk."""
        self.controlnet.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load from pretrained model."""
        adapter = cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
        return adapter


class ControlNetWrapper(nn.Module):
    """
    Wrapper to simplify ControlNet integration into the pipeline.

    Provides a clean interface for using ControlNet with Stable Diffusion.
    """

    def __init__(
        self,
        controlnet_type: str = "custom",
        pretrained_model_name_or_path: Optional[str] = None,
        use_condition_encoding: bool = True,
    ):
        super().__init__()

        self.use_condition_encoding = use_condition_encoding

        # Load ControlNet adapter
        self.adapter = ControlNetAdapter(
            controlnet_type=controlnet_type,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
        )

        # Condition processor
        self.condition_processor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ControlNet wrapper.

        Args:
            x: (B, 4, H, W) latent representation
            condition: (B, 3, H, W) control image
            timesteps: (B,) timesteps
            encoder_hidden_states: (B, seq_len, d_model) prompt embeddings

        Returns:
            down_block_res_samples: list of downsampling block outputs
            mid_block_res_sample: mid block output
        """
        if self.use_condition_encoding:
            condition = self.condition_processor(condition)

        return self.adapter(x, condition, timesteps, encoder_hidden_states)

    def freeze_controlnet(self):
        """Freeze ControlNet weights."""
        self.adapter.freeze_controlnet()

    def unfreeze_controlnet(self):
        """Unfreeze ControlNet weights."""
        self.adapter.unfreeze_controlnet()

    def get_trainable_parameters(self, include_controlnet: bool = True):
        """Get trainable parameters."""
        params = self.adapter.get_trainable_parameters(include_controlnet)
        params.extend(list(self.condition_processor.parameters()))
        return params

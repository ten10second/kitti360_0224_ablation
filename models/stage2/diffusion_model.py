"""
Diffusion Model for Camera Pose-conditioned View Synthesis.

This module implements a diffusion-based view synthesis model using
Stable Diffusion v1.5 with ControlNet, conditioned on satellite images
and camera poses.

Architecture:
  - Base: Stable Diffusion v1.5 (frozen)
  - Control: Custom ControlNet (trainable)
  - Image Condition: MultiScaleViTBEVEncoder (frozen) extracts satellite features
  - Pose Condition: MLP maps pose to 768-dim embedding
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer

# Import existing components
from models.multiscale_vit_encoder import MultiScaleViTBEVEncoder
from world3d.train.pose_ar import build_pose_vec


class PoseEmbeddingProjector(nn.Module):
    """
    Project camera pose to embedding matching CLIP text embedding dimension (768).

    Takes a 13-dimensional pose vector (6D rotation + 3D translation + 4D intrinsics)
    and maps it to a 768-dimensional embedding that can replace or augment
    Stable Diffusion's text prompt embedding.
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: List[int] = [256, 512, 768],
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:  # No activation on last layer
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pose_vec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pose_vec: (B, 13) pose vector from build_pose_vec()

        Returns:
            pose_emb: (B, 768) embedding matching CLIP text embedding dimension
        """
        return self.mlp(pose_vec)


class BEVFeatureAdapter(nn.Module):
    """
    Adapt BEV features to ControlNet's expected input format.

    Takes 256-dimensional BEV features from MultiScaleViTBEVEncoder
    and adapts them to 4 channels for ControlNet input.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,  # ControlNet expects RGB
        target_size: int = 512,
    ):
        super().__init__()

        self.target_size = target_size

        # Project BEV features to 3 channels (RGB-like)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            bev_features: (B, 256, H, W) BEV features from MultiScaleViTBEVEncoder

        Returns:
            control_image: (B, 3, 512, 512) control image for ControlNet
        """
        # Project channels
        x = self.proj(bev_features)

        # Upsample to target size
        x = F.interpolate(
            x,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        return x


class DiffusionPoseModel(nn.Module):
    """
    Main diffusion model for camera pose-conditioned view synthesis.

    Combines:
    - Stable Diffusion v1.5 as base model (frozen)
    - Custom ControlNet for conditioning (trainable)
    - MultiScaleViTBEVEncoder for satellite image features (frozen)
    - PoseEmbeddingProjector for camera pose conditioning
    """

    def __init__(
        self,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        bev_encoder_ckpt: str = "ckpts/fmow_pretrain.pth",
        freeze_sd: bool = True,
        freeze_bev_encoder: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Load Stable Diffusion components
        print(f"[DiffusionModel] Loading Stable Diffusion from {sd_model_id}")
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_id, subfolder="vae"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_id, subfolder="unet"
        ).to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_model_id, subfolder="text_encoder"
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            sd_model_id, subfolder="tokenizer"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            sd_model_id, subfolder="scheduler"
        )

        # Create custom ControlNet from UNet
        print("[DiffusionModel] Creating custom ControlNet")
        self.controlnet = ControlNetModel.from_unet(self.unet).to(self.device)

        # Load BEV encoder
        print(f"[DiffusionModel] Loading BEV encoder from {bev_encoder_ckpt}")
        self.bev_encoder = MultiScaleViTBEVEncoder(
            output_channels=256,
            output_size=64,
            input_size=512,
            freeze_backbone=freeze_bev_encoder,
            pretrained_path=bev_encoder_ckpt,
        ).to(self.device)

        # Pose embedding projector
        self.pose_embedder = PoseEmbeddingProjector(
            input_dim=13,
            hidden_dims=[256, 512, 768],
            dropout=0.0,
        ).to(self.device)

        # BEV feature adapter for ControlNet
        self.bev_adapter = BEVFeatureAdapter(
            in_channels=256,
            out_channels=3,
            target_size=512,
        ).to(self.device)

        # Freeze components as requested
        if freeze_sd:
            self.freeze_stable_diffusion()

        if freeze_bev_encoder:
            self.freeze_bev_encoder()

        # Cache empty text embedding for unconditional generation
        self._empty_text_emb = None
        self._empty_text_attention_mask = None

    def freeze_stable_diffusion(self):
        """Freeze Stable Diffusion components."""
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()

        print("[DiffusionModel] Stable Diffusion frozen")

    def freeze_bev_encoder(self):
        """Freeze BEV encoder (already handled by MultiScaleViTBEVEncoder)."""
        for param in self.bev_encoder.parameters():
            param.requires_grad = False
        self.bev_encoder.eval()

    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        return list(self.controlnet.parameters()) + \
               list(self.pose_embedder.parameters()) + \
               list(self.bev_adapter.parameters())

    def get_prompt_embedding(
        self,
        pose_vec: torch.Tensor,
        num_images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get text embedding from pose vector.

        Args:
            pose_vec: (B, 13) pose vector
            num_images_per_prompt: number of images per prompt

        Returns:
            prompt_embeds: (B * num_images, seq_len, 768) prompt embeddings
            attention_mask: (B * num_images, seq_len) attention mask
        """
        batch_size = pose_vec.shape[0]

        # Project pose to embedding
        pose_emb = self.pose_embedder(pose_vec)  # (B, 768)

        # Get CLIP's empty text embedding for structure
        if self._empty_text_emb is None:
            empty_tokens = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                self._empty_text_emb = self.text_encoder(
                    empty_tokens.input_ids
                ).last_hidden_state
                self._empty_text_attention_mask = empty_tokens.attention_mask

        # Start with empty embedding structure
        prompt_embeds = self._empty_text_emb.repeat(batch_size, 1, 1).clone()
        attention_mask = self._empty_text_attention_mask.repeat(batch_size, 1).clone()

        # Replace the first token (SOS) with our pose embedding
        # Or replace middle token - SD often uses token 1 for condition
        # Using position 1 as it's often used for the main condition
        prompt_embeds[:, 1, :] = pose_emb

        # Handle num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, attention_mask

    def encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to VAE latents."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to images."""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return images

    def forward(
        self,
        sat_images: torch.Tensor,
        pose_vec: torch.Tensor,
        target_images: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            sat_images: (B, 3, H, W) satellite images in [0, 1]
            pose_vec: (B, 13) pose vectors
            target_images: (B, 3, H, W) target images in [-1, 1]
            generator: optional random generator

        Returns:
            dict with loss and other outputs
        """
        batch_size = sat_images.shape[0]

        # Extract BEV features from satellite image
        with torch.no_grad():
            bev_features = self.bev_encoder(sat_images)  # (B, 256, 64, 64)

        # Adapt BEV features for ControlNet
        control_image = self.bev_adapter(bev_features)  # (B, 3, 512, 512)

        # Get pose-based prompt embedding
        prompt_embeds, _ = self.get_prompt_embedding(pose_vec)

        # Encode target images to latents
        if target_images is not None:
            # Resize target images to SD's expected size
            target_resized = F.interpolate(
                target_images,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            latents = self.encode_latents(target_resized)
        else:
            # If no target, generate random latents
            latents_shape = (batch_size, 4, 64, 64)
            if generator is not None:
                latents = torch.randn(
                    latents_shape,
                    device=self.device,
                    generator=generator,
                )
            else:
                latents = torch.randn(
                    latents_shape,
                    device=self.device,
                )

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get ControlNet output
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_image,
            return_dict=False,
        )

        # Get UNet prediction
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # Compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            loss = F.mse_loss(model_pred, noise)
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            # For v-prediction
            loss = F.mse_loss(model_pred, self.noise_scheduler.get_velocity(latents, noise, timesteps))
        else:
            loss = F.mse_loss(model_pred, noise)

        return {
            "loss": loss,
            "latents": latents,
            "control_image": control_image,
        }

    @torch.no_grad()
    def generate(
        self,
        sat_images: torch.Tensor,
        pose_vec: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate images from satellite images and poses.

        Args:
            sat_images: (B, 3, H, W) satellite images in [0, 1]
            pose_vec: (B, 13) pose vectors
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale
            generator: optional random generator

        Returns:
            generated_images: (B, 3, 512, 512) generated images in [-1, 1]
        """
        batch_size = sat_images.shape[0]

        # Extract BEV features
        with torch.no_grad():
            bev_features = self.bev_encoder(sat_images)

        # Adapt BEV features for ControlNet
        control_image = self.bev_adapter(bev_features)

        # Get conditional and unconditional embeddings for classifier-free guidance
        cond_embeds, _ = self.get_prompt_embedding(pose_vec)

        # Create unconditional embedding (empty text)
        uncond_tokens = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        uncond_embeds = self.text_encoder(uncond_tokens.input_ids).last_hidden_state
        uncond_embeds = uncond_embeds.repeat(batch_size, 1, 1)

        # Concatenate for classifier-free guidance
        prompt_embeds = torch.cat([uncond_embeds, cond_embeds])

        # Double control image for CFG
        control_image = torch.cat([control_image, control_image])

        # Initialize latents
        latents_shape = (batch_size, 4, 64, 64)
        latents = randn_tensor(
            latents_shape,
            device=self.device,
            generator=generator,
        )

        # Set up scheduler
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            # Double latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            # ControlNet forward
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_image,
                return_dict=False,
            )

            # UNet forward
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to images
        images = self.decode_latents(latents)

        return images

    def save_pretrained(self, save_dir: Union[str, Path]):
        """Save trainable components to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save ControlNet
        self.controlnet.save_pretrained(save_dir / "controlnet")

        # Save pose embedder and BEV adapter
        torch.save(
            {
                "pose_embedder": self.pose_embedder.state_dict(),
                "bev_adapter": self.bev_adapter.state_dict(),
            },
            save_dir / "adaptors.pt",
        )

        print(f"[DiffusionModel] Saved to {save_dir}")

    def load_pretrained(self, load_dir: Union[str, Path]):
        """Load trainable components from disk."""
        load_dir = Path(load_dir)

        # Load ControlNet
        if (load_dir / "controlnet").exists():
            self.controlnet = ControlNetModel.from_pretrained(
                load_dir / "controlnet"
            ).to(self.device)

        # Load pose embedder and BEV adapter
        adaptors_path = load_dir / "adaptors.pt"
        if adaptors_path.exists():
            state_dict = torch.load(adaptors_path, map_location=self.device)
            self.pose_embedder.load_state_dict(state_dict["pose_embedder"])
            self.bev_adapter.load_state_dict(state_dict["bev_adapter"])

        print(f"[DiffusionModel] Loaded from {load_dir}")

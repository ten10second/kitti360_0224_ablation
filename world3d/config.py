from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Configuration for diffusion model."""
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_type: str = "custom"
    pose_encoding_type: str = "6d"
    pose_embed_dim: int = 768
    use_existing_bev_encoder: bool = True
    bev_encoder_ckpt: str = "ckpts/fmow_pretrain.pth"
    freeze_sd: bool = True
    freeze_bev_encoder: bool = True


@dataclass
class ArTrainConfig:
    # Repro / sampling
    data_seed: int = 0
    deterministic_data: bool = False
    shuffle: bool = False
    seed: int = 42

    # Data
    data_root: str = ""
    drive: str = "2013_05_28_drive_0003_sync"  # Single-drive mode (backward compat)
    drives: Optional[list] = None  # Multi-drive mode: list of {name, frames_file, weight}
    p_front: float = 0.0
    yaw_min_abs: float = 0.0
    yaw_max_abs: float = 40.0
    use_fixed_five_views: bool = True
    fixed_view_turn_deg: float = 30.0
    subset: Optional[int] = None
    exclude_frames: Optional[list] = None  # Test frames to exclude from training

    # Performance optimization parameters
    perf: dict = field(default_factory=lambda: {
        "enable_tf32": True,
        "enable_sdpa": True,
        "channels_last": False,
        "torch_compile": False,
    })

    # DataLoader optimization
    loader: dict = field(default_factory=lambda: {
        "num_workers": 8,
        "prefetch_factor": 2,
        "persistent_workers": True,
    })

    # DDP optimization
    dist: dict = field(default_factory=lambda: {
        "find_unused_parameters": False,
        "static_graph": False,
    })

    # DDP / DataLoader (backward compatibility)
    num_workers: int = 0
    ddp_strict_view: bool = True

    # Reproducibility
    full_determinism: bool = False

    virtual_hfov: float = 80.0
    virtual_w: int = 640
    virtual_h: int = 256
    sat_dir: Optional[str] = None

    # Runtime
    device: str = "cuda"
    out_dir: str = "runs/ar_simplified"
    vq_ckpt: str = "ckpts/maskgit-vqgan-imagenet-f16-256.bin"
    resume_ckpt: Optional[str] = None
    warm_start_ckpt: Optional[str] = None

    # Train
    steps: int = 120000
    print_every: int = 100
    save_every: int = 4000
    vis_every: int = 100
    coords_vis_every: int = 0
    plot_every: int = 100
    eval_every: int = 100
    eval_samples: int = 4
    compute_fid: bool = False
    lpips_net: str = "alex"
    vis_top_k: int = 50
    vis_temperature: float = 1.0

    # IPM correction angles (degrees)
    roll_deg: float = 0.0  # Roll correction for virtual-view rectification (dataloader)
    pitch_deg: float = 0.0  # Pitch correction for virtual-view rectification (dataloader)

    # Model
    # (mode/fourier_freqs/train_bev_encoder/no_bev_pretrain/n_pose_queries/
    #  hybrid_memory_source/use_ipm_semantic/use_explicit_token_pos/semantic_dim
    #  removed with the direct/hybrid/anchor/RayRoPE/BEV paths in the ICASSP27 refactor)
    ntp_order: str = "topleft"  # topleft|bottomup
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1080
    pose_dim: int = 13
    use_pose_token: bool = True
    vocab_size: int = 1024
    grid_cols: int = 40
    grid_rows: int = 16
    bos_token: int = 1024

    # Optim
    lr: float = 1e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    accum_steps: int = 4
    batch_size: int = 4
    label_smoothing: float = 0.0
    ce_weight: float = 1.0

    # (pair-consistency / multi-view consistency / anchor-view stage-2 /
    #  MaskGIT / OneSlot blocks removed with their trainers in the ICASSP27 refactor)

    # Scheduler
    use_warmup_cosine: bool = False
    warmup_updates: int = 4000
    min_lr: float = 1e-6


@dataclass
class DiffusionTrainConfig:
    """Configuration for diffusion model training."""
    # Base configuration (shared with AR)
    device: str = "cuda"
    data_root: str = ""
    drives: Optional[list] = None
    seed: int = 42
    data_seed: int = 42
    deterministic_data: bool = False
    shuffle: bool = True
    subset: Optional[int] = None

    # View configuration
    p_front: float = 0.6
    yaw_min_abs: float = 0.0
    yaw_max_abs: float = 40.0
    use_fixed_five_views: bool = True
    fixed_view_turn_deg: float = 30.0
    fourier_freqs: int = 10

    # Performance optimization parameters
    perf: dict = field(default_factory=lambda: {
        "enable_tf32": True,
        "enable_sdpa": True,
        "channels_last": False,
        "torch_compile": False,
    })

    # DataLoader optimization
    loader: dict = field(default_factory=lambda: {
        "num_workers": 8,
        "prefetch_factor": 2,
        "persistent_workers": True,
    })

    # DDP optimization
    dist: dict = field(default_factory=lambda: {
        "find_unused_parameters": True,
        "static_graph": False,
    })

    # Pair consistency optimization
    pair_sat_size: int = 256
    pair_max_pairs_per_frame: int = 2
    min_overlap_pixels: int = 200

    # Backward compatibility
    num_workers: int = 8
    ddp_strict_view: bool = True
    full_determinism: bool = False
    virtual_hfov: float = 80.0
    virtual_w: int = 640
    virtual_h: int = 256

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training
    steps: int = 80000
    batch_size: int = 8
    accum_steps: int = 2
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_warmup_cosine: bool = True
    warmup_updates: int = 4000
    min_lr: float = 1.0e-6

    # Checkpointing & Logging
    out_dir: str = "runs/diffusion_1d_cond"
    print_every: int = 100
    save_every: int = 5000
    vis_every: int = 1000
    plot_every: int = 100

    # Evaluation
    eval_every: int = 5000
    eval_batch_size: int = 4
    eval_num_samples: int = 1000

    # Sampling & Visualization
    vis_top_k: int = 50
    vis_temperature: float = 0.9


def load_cfg(path: str, overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Load configuration from file."""
    cfg_dict = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            cfg_dict[k] = v

    # Determine which config class to use
    if "model" in cfg_dict and isinstance(cfg_dict["model"], dict):
        # Diffusion model config
        if "model" in cfg_dict:
            model_dict = cfg_dict.pop("model")
            cfg_dict["model"] = ModelConfig(**model_dict)
        return DiffusionTrainConfig(**cfg_dict)
    else:
        # AR model config
        return ArTrainConfig(**cfg_dict)

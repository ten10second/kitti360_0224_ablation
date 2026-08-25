"""Minimal two-stage georeferenced BEV latent pipeline."""

from .geometry import (
    bilinear_splat,
    geometry_supervision_support,
    observation_partition,
    project_points_to_image,
    relative_height_map,
    render_volume,
    se3_inverse,
    target_pixels_supported_by_bev,
)
from .models import (
    ColumnFieldDecoder,
    CompletionOutput,
    GroundBEVEncoder,
    LatentCompletion,
    SatelliteBEVEncoder,
)
from .readouts import BEVHeightDecoder, freeze_module

__all__ = [
    "bilinear_splat",
    "geometry_supervision_support",
    "observation_partition",
    "project_points_to_image",
    "relative_height_map",
    "render_volume",
    "se3_inverse",
    "target_pixels_supported_by_bev",
    "BEVHeightDecoder",
    "ColumnFieldDecoder",
    "CompletionOutput",
    "freeze_module",
    "GroundBEVEncoder",
    "LatentCompletion",
    "SatelliteBEVEncoder",
]

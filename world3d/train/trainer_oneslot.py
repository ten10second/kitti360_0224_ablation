"""OneSlot trainer: one-shot token prediction with configurable mask curriculum."""

from __future__ import annotations

import math

import torch

from world3d.train.trainer_maskgit import MaskGITTrainer


class OneSlotTrainer(MaskGITTrainer):
    """Independent one-shot trainer.

    Core difference from MaskGIT:
    - Uses a configurable mask curriculum, with optional ramp to full masking.
    """

    def _trainer_name(self) -> str:
        return "OneSlot"

    def _checkpoint_model_type(self) -> str:
        return "oneslot"

    def _sample_mask(self, B: int, L: int, device: torch.device, step: int | None = None) -> torch.Tensor:
        start_ratio = float(getattr(self.cfg, "oneslot_mask_start_ratio", 1.0))
        end_ratio = float(getattr(self.cfg, "oneslot_mask_end_ratio", 1.0))
        ramp_steps = max(0, int(getattr(self.cfg, "oneslot_mask_ramp_steps", 0)))
        schedule = str(getattr(self.cfg, "oneslot_mask_schedule", "cosine")).lower()

        start_ratio = min(max(start_ratio, 0.0), 1.0)
        end_ratio = min(max(end_ratio, 0.0), 1.0)

        if step is None or ramp_steps <= 0 or abs(end_ratio - start_ratio) < 1e-8:
            ratio = end_ratio
        else:
            progress = min(max((int(step) - int(self.start_step)) / float(ramp_steps), 0.0), 1.0)
            if schedule == "linear":
                mix = progress
            else:
                mix = 0.5 * (1.0 - math.cos(math.pi * progress))
            ratio = start_ratio + (end_ratio - start_ratio) * mix

        ratio = min(max(ratio, 1.0 / max(1, L)), 1.0)
        num_masked = max(1, int(round(ratio * L)))
        if num_masked >= L:
            return torch.ones(B, L, dtype=torch.bool, device=device)

        noise = torch.rand(B, L, device=device)
        sorted_indices = noise.argsort(dim=1)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            mask[i, sorted_indices[i, :num_masked]] = True
        return mask

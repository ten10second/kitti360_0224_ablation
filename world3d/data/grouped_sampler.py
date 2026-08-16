"""GroupedFrameSampler for FixedFiveViewDataset.

Ensures each batch contains complete groups of 5 views from the same frames,
enabling pairwise geometric consistency loss during training.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch.utils.data


class GroupedFrameSampler(torch.utils.data.Sampler):
    """Sampler that groups indices by frame for FixedFiveViewDataset.

    FixedFiveViewDataset expands each frame into 5 views, so len(dataset) = N_frames * 5.
    This sampler ensures each batch contains complete groups of 5 views from the same frames.

    Args:
        dataset: FixedFiveViewDataset instance
        batch_size: Total batch size (must be divisible by 5)
        shuffle: Whether to shuffle frame groups
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size % 5 != 0:
            raise ValueError(f"batch_size must be divisible by 5, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Total indices = N_frames * 5
        total_indices = len(dataset)
        if total_indices % 5 != 0:
            raise ValueError(f"Dataset length must be divisible by 5, got {total_indices}")

        self.n_frames = total_indices // 5
        self.groups_per_batch = batch_size // 5

        # Group indices: [[0,1,2,3,4], [5,6,7,8,9], ...]
        self.frame_groups = [
            list(range(i * 5, (i + 1) * 5))
            for i in range(self.n_frames)
        ]

    def __iter__(self) -> Iterator[int]:
        # Shuffle frame groups if requested
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            frame_order = list(range(self.n_frames))
            rng.shuffle(frame_order)
            groups = [self.frame_groups[i] for i in frame_order]
        else:
            groups = self.frame_groups

        # Yield batches of complete frame groups
        for batch_start in range(0, self.n_frames, self.groups_per_batch):
            batch_groups = groups[batch_start : batch_start + self.groups_per_batch]
            # Flatten groups into indices
            for group in batch_groups:
                yield from group

    def __len__(self) -> int:
        # Total number of indices yielded
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.epoch = epoch

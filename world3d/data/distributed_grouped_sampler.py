"""
Distributed Grouped Frame Sampler

This sampler ensures that:
1. Each batch contains complete groups of 5 views from the same frames (for pair consistency)
2. Each rank gets exclusive frames (no duplicates across ranks)
3. Batch size must be a multiple of 5

Modified from: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
"""

import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.data.sampler import Sampler
import math
import random


class DistributedGroupedFrameSampler(Sampler):
    """
    Sampler that groups frames into 5-view sets and distributes them across ranks.

    Args:
        dataset: Dataset to sample from. Should have fixed 5 views per frame.
        batch_size: Number of samples per batch (must be multiple of 5)
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: Whether to shuffle the data (default: True)
        seed: Random seed (default: 0)
        drop_last: Whether to drop the last incomplete batch (default: True)
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if batch_size % 5 != 0:
            raise ValueError(f"batch_size must be a multiple of 5, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Number of views per frame (must be 5 for fixed five views)
        self.views_per_frame = 5

        # Calculate number of frames (groups) in dataset
        if len(dataset) % self.views_per_frame != 0:
            raise ValueError(
                f"Dataset size must be multiple of {self.views_per_frame}, got {len(dataset)}"
            )
        self.num_frames = len(dataset) // self.views_per_frame
        self.groups_per_batch = self.batch_size // self.views_per_frame

        # Calculate number of groups per rank
        if self.drop_last:
            self.num_groups_per_rank = math.ceil(
                (self.num_frames - self.num_replicas) / self.num_replicas
            )
            self.total_groups = self.num_groups_per_rank * self.num_replicas
        else:
            self.num_groups_per_rank = math.ceil(self.num_frames / self.num_replicas)
            self.total_groups = self.num_groups_per_rank * self.num_replicas

        self.num_samples = self.num_groups_per_rank * self.views_per_frame

    def __iter__(self):
        # Seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Generate list of frame indices (groups)
        frame_indices = list(range(self.num_frames))

        # Shuffle if needed
        if self.shuffle:
            # Deterministically shuffle based on seed
            frame_indices = [
                frame_indices[i]
                for i in torch.randperm(len(frame_indices), generator=g).tolist()
            ]

        # Drop frames to make it evenly divisible across ranks
        if self.drop_last:
            # Keep only full batches
            drop = self.num_frames % self.num_replicas
            if drop > 0:
                frame_indices = frame_indices[:-drop]

        # Distribute frames across ranks
        rank_frame_indices = frame_indices[self.rank :: self.num_replicas]

        # Ensure each rank gets exactly self.num_groups_per_rank groups
        if len(rank_frame_indices) < self.num_groups_per_rank:
            # This should only happen when drop_last=False and there are remaining frames
            # We'll pad with dummy indices (which will be dropped later)
            pad = self.num_groups_per_rank - len(rank_frame_indices)
            rank_frame_indices += rank_frame_indices[:pad]

        # Expand frame indices to view indices (each frame has 5 views)
        indices = []
        for frame_idx in rank_frame_indices:
            start = frame_idx * self.views_per_frame
            indices.extend(range(start, start + self.views_per_frame))

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. This ensures all replicas
        use a different random ordering for each epoch.

        Args:
            epoch (int): Epoch number
        """
        self.epoch = epoch
        self.seed = epoch

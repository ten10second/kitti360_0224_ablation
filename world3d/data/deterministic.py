from __future__ import annotations

import random


def make_rng(seed: int, epoch: int, idx: int, *, salt: int = 0) -> random.Random:
    # Centralized deterministic RNG for per-item decisions.
    # salt can be used to derive independent streams (e.g., view vs yaw).
    return random.Random(int(seed) + int(epoch) * 1000003 + int(idx) * 10007 + int(salt))


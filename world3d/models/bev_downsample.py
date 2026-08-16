 #!/usr/bin/env python3
"""
Downsample BEV 64x64 -> 32x32 over the full 100m field.
Prefer anti-aliased average pooling; optionally max.
"""
import numpy as np
from typing import Literal

Array = np.ndarray


def downsample_2x(bev: Array, mode: Literal['avg','max']='avg') -> Array:
    """bev: (H,W) float32, expected H=W=64 -> returns (32,32)
    Covers the full field, not a central crop.
    """
    H, W = bev.shape
    assert H % 2 == 0 and W % 2 == 0, 'H,W must be even'
    if mode == 'avg':
        a = bev[0::2, 0::2]
        b = bev[0::2, 1::2]
        c = bev[1::2, 0::2]
        d = bev[1::2, 1::2]
        out = (a + b + c + d) * 0.25
    elif mode == 'max':
        out = np.maximum.reduce([bev[0::2,0::2], bev[0::2,1::2], bev[1::2,0::2], bev[1::2,1::2]])
    else:
        raise ValueError(f'Unknown mode={mode}')
    return out.astype(np.float32)


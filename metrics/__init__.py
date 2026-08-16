from .psnr import PSNR
from .ssim import SSIM
from .lpips import LPIPS
from .fid import FID
from .psqueeze import P_Squeeze
from .dino_similarity import DINOSimilarity
# from .segany_consistency import SegAnyConsistency  # requires segment_anything library
from .depth_consistency import DepthConsistency
from .lrce import LRCE
from .multiview_consistency import MultiViewConsistency

__all__ = [
    'PSNR',
    'SSIM',
    'LPIPS',
    'FID',
    'P_Squeeze',
    'DINOSimilarity',
    # 'SegAnyConsistency',
    'DepthConsistency',
    'LRCE',
    'MultiViewConsistency',
]
#!/usr/bin/env python3
"""
Batch evaluation script for all image generation metrics.
Reads image pairs from txt file and outputs all metrics results.

Input txt format (each line):
img1_path img2_path [depth1_path depth2_path K1_path K2_path T_src_to_tgt_path]
- Minimum: 2 columns (image paths)
- For multi-view metrics: need 8 columns including camera parameters and depth
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import List, Dict, Optional

# Import all metrics
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import (
    PSNR, SSIM, LPIPS, FID,
    P_Squeeze, DINOSimilarity, SegAnyConsistency,
    DepthConsistency, LRCE, MultiViewConsistency
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate image generation metrics")
    parser.add_argument("--txt_path", type=str, required=True, help="Path to txt file with image pairs")
    parser.add_argument("--output_path", type=str, default="metrics_results.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (only for metrics that support batching)")

    # Metric selection flags
    parser.add_argument("--psnr", action="store_true", help="Calculate PSNR")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")
    parser.add_argument("--lpips", action="store_true", help="Calculate LPIPS")
    parser.add_argument("--fid", action="store_true", help="Calculate FID (requires full dataset)")
    parser.add_argument("--p_squeeze", action="store_true", help="Calculate P_Squeeze")
    parser.add_argument("--dino", action="store_true", help="Calculate DINOSimilarity")
    parser.add_argument("--segany", action="store_true", help="Calculate SegAnyConsistency")
    parser.add_argument("--depth", action="store_true", help="Calculate DepthConsistency")
    parser.add_argument("--lrce", action="store_true", help="Calculate LRCE")
    parser.add_argument("--mv_depth", action="store_true", help="Calculate MultiView depth consistency")
    parser.add_argument("--mv_lpips", action="store_true", help="Calculate MultiView warp LPIPS")

    # Model parameters
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="Path to SAM checkpoint")
    parser.add_argument("--dino_model", type=str, default="facebook/dinov2-small", help="DINOv2 model name")
    parser.add_argument("--midas_model", type=str, default="MiDaS_small", help="MiDaS model type")

    args = parser.parse_args()

    # If no metrics specified, enable all except FID and multi-view
    if not any([args.psnr, args.ssim, args.lpips, args.fid, args.p_squeeze,
                args.dino, args.segany, args.depth, args.lrce, args.mv_depth, args.mv_lpips]):
        print("No metrics specified, enabling all standard metrics (psnr, ssim, lpips, p_squeeze, dino, depth, lrce)")
        args.psnr = args.ssim = args.lpips = args.p_squeeze = args.dino = args.depth = args.lrce = True

    return args


def load_image(path: str, device: str) -> torch.Tensor:
    """ Load image to tensor in [0, 1] range, shape (1, 3, H, W) """
    img = Image.open(path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def load_matrix(path: str, device: str) -> torch.Tensor:
    """ Load numpy matrix from file (npy or txt) """
    if path.endswith('.npy'):
        mat = np.load(path)
    else:
        mat = np.loadtxt(path)
    return torch.tensor(mat, dtype=torch.float32).unsqueeze(0).to(device)


class ImagePairDataset:
    def __init__(self, txt_path: str, device: str, load_extra: bool = False):
        self.device = device
        self.load_extra = load_extra

        with open(txt_path, 'r') as f:
            self.lines = [line.strip().split() for line in f.readlines() if line.strip()]

        print(f"Loaded {len(self.lines)} image pairs from {txt_path}")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict:
        parts = self.lines[idx]
        img1_path = parts[0]
        img2_path = parts[1]

        # Load images
        img1 = load_image(img1_path, self.device)
        img2 = load_image(img2_path, self.device)

        result = {
            'img1': img1,
            'img2': img2,
            'img1_path': img1_path,
            'img2_path': img2_path,
        }

        # Load extra data if needed
        if self.load_extra and len(parts) >= 8:
            depth1_path = parts[2]
            depth2_path = parts[3]
            K1_path = parts[4]
            K2_path = parts[5]
            T_path = parts[6]

            result['depth1'] = load_matrix(depth1_path, self.device) if depth1_path != 'None' else None
            result['depth2'] = load_matrix(depth2_path, self.device) if depth2_path != 'None' else None
            result['K1'] = load_matrix(K1_path, self.device)
            result['K2'] = load_matrix(K2_path, self.device)
            result['T_src_to_tgt'] = load_matrix(T_path, self.device)

        return result


def init_metrics(args) -> Dict:
    """ Initialize all selected metrics """
    metrics = {}

    if args.psnr:
        metrics['psnr'] = PSNR(reduction='none').to(args.device)
    if args.ssim:
        metrics['ssim'] = SSIM(reduction='none').to(args.device)
    if args.lpips:
        metrics['lpips'] = LPIPS(reduction='none').to(args.device)
    if args.fid:
        metrics['fid'] = FID(reduction='none').to(args.device)
    if args.p_squeeze:
        metrics['p_squeeze'] = P_Squeeze(reduction='none', device=args.device)
    if args.dino:
        metrics['dino'] = DINOSimilarity(
            reduction='none',
            model_name=args.dino_model,
            device=args.device
        )
    if args.segany:
        metrics['segany'] = SegAnyConsistency(
            reduction='none',
            sam_checkpoint=args.sam_checkpoint,
            device=args.device
        )
    if args.depth:
        metrics['depth'] = DepthConsistency(
            reduction='none',
            model_type=args.midas_model,
            device=args.device
        )
    if args.lrce:
        metrics['lrce'] = LRCE(reduction='none')
    if args.mv_depth:
        metrics['mv_depth'] = MultiViewConsistency(
            reduction='none',
            metric_type='depth_consistency',
            model_type=args.midas_model,
            device=args.device
        )
    if args.mv_lpips:
        metrics['mv_lpips'] = MultiViewConsistency(
            reduction='none',
            metric_type='warp_lpips',
            model_type=args.midas_model,
            device=args.device
        )

    print(f"Initialized {len(metrics)} metrics: {list(metrics.keys())}")
    return metrics


def main():
    args = parse_args()

    # Check if multi-view metrics are enabled
    need_extra = args.mv_depth or args.mv_lpips

    # Load dataset
    dataset = ImagePairDataset(args.txt_path, args.device, load_extra=need_extra)

    # Initialize metrics
    metrics = init_metrics(args)

    # Results storage
    results = []

    # Process all image pairs
    for i in tqdm(range(len(dataset)), desc="Evaluating metrics"):
        item = dataset[i]
        img1 = item['img1']
        img2 = item['img2']

        # Calculate metrics
        res = {
            'idx': i,
            'img1_path': item['img1_path'],
            'img2_path': item['img2_path'],
        }

        try:
            # Standard metrics
            for name, metric in metrics.items():
                if name in ['mv_depth', 'mv_lpips']:
                    continue  # handle multi-view separately
                value = metric(img1, img2).item()
                res[name] = value

            # Multi-view metrics
            if need_extra:
                if args.mv_depth:
                    value = metrics['mv_depth'](
                        img1, img2,
                        K_src=item['K1'],
                        K_tgt=item['K2'],
                        T_src_to_tgt=item['T_src_to_tgt'],
                        depth_src=item.get('depth1'),
                        depth_tgt=item.get('depth2'),
                    ).item()
                    res['mv_depth'] = value

                if args.mv_lpips:
                    value = metrics['mv_lpips'](
                        img1, img2,
                        K_src=item['K1'],
                        K_tgt=item['K2'],
                        T_src_to_tgt=item['T_src_to_tgt'],
                        depth_src=item.get('depth1'),
                        depth_tgt=item.get('depth2'),
                    ).item()
                    res['mv_lpips'] = value

        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            # Fill with NaN for failed cases
            for name in metrics.keys():
                if name not in res:
                    res[name] = float('nan')

        results.append(res)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Add mean row
    mean_row = {'idx': 'mean', 'img1_path': '', 'img2_path': ''}
    for col in df.columns:
        if col not in ['idx', 'img1_path', 'img2_path']:
            mean_row[col] = df[col].mean()

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save to CSV
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

    # Print summary
    print("\n=== Metrics Summary (Mean) ===")
    for name in metrics.keys():
        if name in mean_row:
            print(f"{name:15s}: {mean_row[name]:.6f}")


if __name__ == "__main__":
    main()

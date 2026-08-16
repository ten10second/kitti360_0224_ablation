#!/usr/bin/env python3
"""
Compute generation metrics for results saved in direct/hybrid inference layouts.
Supports per-view metrics, adjacent-view multi-view consistency metrics, and dataset-level FID.

Supported directory structures:

Legacy:
results/
├─ direct/
│  ├─ {sync_name}/
│  │  ├─ left/
│  │  │  ├─ {frame_id}/
│  │  │  │  ├─ gt.png
│  │  │  │  ├─ generated.png
│  │  │  │  ├─ K.npy
│  │  │  │  └─ T_cam_to_world.npy
│  │  ├─ left_to_front_30/
│  │  ├─ front/
│  │  ├─ right_to_front_30/
│  │  └─ right/

Grouped:
results/
├─ direct/
│  ├─ {sync_name}/
│  │  ├─ fixed5/
│  │  │  ├─ left/
│  │  │  ├─ left_to_front_30/
│  │  │  ├─ front/
│  │  │  ├─ right_to_front_30/
│  │  │  └─ right/
│  │  └─ zero_shot/
│  │     ├─ left_back_30/
│  │     └─ right_back_30/
├─ hybrid/
│  └─ ... same structure
"""

import os
import shutil
import tempfile
import argparse
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
from pytorch_fid.fid_score import calculate_fid_given_paths

# Import all metrics
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import (
    PSNR, SSIM, LPIPS,
    P_Squeeze, DINOSimilarity,
    DepthConsistency, LRCE, MultiViewConsistency,
)

# Optional SegAnyConsistency (requires segment_anything library)
try:
    from metrics.segany_consistency import SegAnyConsistency
    HAS_SEGANY = True
except ImportError:
    SegAnyConsistency = None
    HAS_SEGANY = False


FIXED5_VIEW_ORDER = ["left", "left_to_front_30", "front", "right_to_front_30", "right"]
ZERO_SHOT_VIEW_ORDER = ["left_back_30", "right_back_30"]
ALL_VIEWS = FIXED5_VIEW_ORDER + ZERO_SHOT_VIEW_ORDER
SUBSET_VIEW_ORDERS = {
    "fixed5": FIXED5_VIEW_ORDER,
    "zero_shot": ZERO_SHOT_VIEW_ORDER,
}
ZERO_SHOT_REFERENCE_PAIRS = [
    ("left", "left_back_30"),
    ("right", "right_back_30"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute all metrics for generated results")
    parser.add_argument("--results-root", type=str, default="results/", help="Root directory of results, or a direct/hybrid mode directory itself")
    parser.add_argument("--mode", type=str, choices=["auto", "direct", "hybrid", "both"], default="auto",
                        help="Which mode to evaluate. auto: infer from results-root; both: expect direct/ and hybrid/ under results-root")
    parser.add_argument("--syncs", type=str, nargs="+", default=None, help="Specific sync directories to evaluate (default: all)")
    parser.add_argument("--subsets", type=str, nargs="+", choices=["fixed5", "zero_shot"], default=None,
                        help="Specific result subsets to evaluate. Default: auto-discover all available subsets")
    parser.add_argument("--views", type=str, nargs="+", choices=ALL_VIEWS, default=ALL_VIEWS,
                        help="Specific views to evaluate (default: all supported views)")
    parser.add_argument("--output-path", type=str, default="metrics_results.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Metric selection flags
    parser.add_argument("--psnr", action="store_true", help="Calculate PSNR")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")
    parser.add_argument("--lpips", action="store_true", help="Calculate LPIPS")
    parser.add_argument("--p_squeeze", action="store_true", help="Calculate P_Squeeze")
    parser.add_argument("--dino", action="store_true", help="Calculate DINOSimilarity")
    parser.add_argument("--depth", action="store_true", help="Calculate DepthConsistency")
    parser.add_argument("--lrce", action="store_true", help="Calculate LRCE")
    parser.add_argument("--fid", action="store_true", help="Calculate dataset-level FID for each evaluated mode")
    parser.add_argument("--mv-depth", action="store_true", help="Calculate multi-view depth consistency (adjacent views)")
    parser.add_argument("--mv-mask-dino", dest="mv_mask_dino", action="store_true", help="Calculate multi-view masked DINO similarity (adjacent views)")
    parser.add_argument("--mv-seg-miou", action="store_true", help="Calculate multi-view warp segmentation mIoU (adjacent views)")

    # Model parameters
    parser.add_argument("--sam-checkpoint", type=str, default=None, help="Path to SAM checkpoint")
    if HAS_SEGANY:
        parser.add_argument("--segany", action="store_true", help="Calculate SegAnyConsistency")
    parser.add_argument("--dino-model", type=str, default="facebook/dinov2-small", help="DINOv2 model name")
    parser.add_argument("--midas-model", type=str, default="MiDaS_small", help="MiDaS model type")

    args = parser.parse_args()

    # If no metrics specified, enable all standard metrics
    metric_flags = [
        args.psnr, args.ssim, args.lpips, args.p_squeeze, args.dino,
        args.depth, args.lrce, args.fid, args.mv_depth,
        args.mv_mask_dino, args.mv_seg_miou,
    ]
    if HAS_SEGANY:
        metric_flags.append(getattr(args, 'segany', False))

    if not any(metric_flags):
        print("No metrics specified, enabling all standard metrics: psnr, ssim, lpips, p_squeeze, dino, depth, lrce, fid")
        args.psnr = args.ssim = args.lpips = args.p_squeeze = args.dino = args.depth = args.lrce = args.fid = True

    return args


def load_image(path: str, device: str) -> torch.Tensor:
    """Load image to tensor in [0, 1] range, shape (1, 3, H, W)."""
    img = Image.open(path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def load_matrix(path: str, device: str) -> torch.Tensor:
    """Load numpy matrix from npy file."""
    mat = np.load(path)
    return torch.tensor(mat, dtype=torch.float32).unsqueeze(0).to(device)


def _link_or_copy(src: str, dst: Path):
    src_path = Path(src)
    try:
        os.link(src_path, dst)
    except OSError:
        try:
            os.symlink(src_path, dst)
        except OSError:
            shutil.copy2(src_path, dst)


def compute_dataset_fid(samples: List[Dict], device: str, batch_size: int = 50) -> float:
    """Compute one FID score over all selected samples of a mode."""
    if not samples:
        return float('nan')

    with tempfile.TemporaryDirectory(prefix='fid_gt_') as gt_dir, tempfile.TemporaryDirectory(prefix='fid_gen_') as gen_dir:
        gt_dir_path = Path(gt_dir)
        gen_dir_path = Path(gen_dir)
        for idx, sample in enumerate(samples):
            _link_or_copy(sample['gt_path'], gt_dir_path / f"{idx:06d}.png")
            _link_or_copy(sample['gen_path'], gen_dir_path / f"{idx:06d}.png")

        return float(calculate_fid_given_paths(
            [str(gt_dir_path), str(gen_dir_path)],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=2,
        ))


def _looks_like_eval_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    return any(child.is_dir() and "sync" in child.name for child in root.iterdir())



def resolve_mode_roots(results_root: str, mode: str) -> List[tuple]:
    """Resolve evaluation roots.

    Default behavior is simple: evaluate the directory the user passed.
    Only use direct/ or hybrid/ subdirectories when explicitly requested.
    """
    root = Path(results_root)
    root_name = root.name.lower()

    if mode == "direct":
        target = root if root_name == "direct" or _looks_like_eval_root(root) else root / "direct"
        return [("direct", target)]

    if mode == "hybrid":
        target = root if root_name == "hybrid" or _looks_like_eval_root(root) else root / "hybrid"
        return [("hybrid", target)]

    if mode == "both":
        return [("direct", root / "direct"), ("hybrid", root / "hybrid")]

    # auto mode: always prefer evaluating the exact folder the user passed.
    if _looks_like_eval_root(root):
        inferred_mode = root_name if root_name in {"direct", "hybrid"} else root_name
        return [(inferred_mode, root)]

    if root_name in {"direct", "hybrid"}:
        return [(root_name, root)]

    candidates = []
    if (root / "direct").exists():
        candidates.append(("direct", root / "direct"))
    if (root / "hybrid").exists():
        candidates.append(("hybrid", root / "hybrid"))
    return candidates


def find_all_samples(
    results_root: str,
    mode: str,
    syncs: Optional[List[str]] = None,
    views: Optional[List[str]] = None,
    subsets: Optional[List[str]] = None,
) -> List[Dict]:
    """Find all valid samples in one resolved mode directory."""
    samples = []
    mode_dir = Path(results_root)

    if not mode_dir.exists():
        print(f"Warning: {mode_dir} does not exist, skipping {mode} mode")
        return []

    if syncs is None:
        sync_dirs = [d for d in mode_dir.iterdir() if d.is_dir() and "sync" in d.name]
    else:
        sync_dirs = [mode_dir / s for s in syncs if (mode_dir / s).exists()]

    selected_views = views or ALL_VIEWS
    requested_subsets = set(subsets) if subsets else None

    def collect_samples_from_view_root(view_root: Path, sync_name: str, subset_name: str):
        for view in selected_views:
            view_dir = view_root / view
            if not view_dir.exists():
                continue
            for frame_dir in sorted(view_dir.iterdir()):
                if not frame_dir.is_dir() or not frame_dir.name.isdigit():
                    continue
                frame_id = frame_dir.name
                gt_path = frame_dir / "gt.png"
                gen_path = frame_dir / "generated.png"
                k_path = frame_dir / "K.npy"
                t_path = frame_dir / "T_cam_to_world.npy"

                if gt_path.exists() and gen_path.exists() and k_path.exists() and t_path.exists():
                    samples.append({
                        "mode": mode,
                        "subset": subset_name,
                        "sync": sync_name,
                        "view": view,
                        "frame_id": frame_id,
                        "gt_path": str(gt_path),
                        "gen_path": str(gen_path),
                        "K_path": str(k_path),
                        "T_path": str(t_path),
                    })

    for sync_dir in sorted(sync_dirs):
        sync_name = sync_dir.name
        has_legacy_layout = any((sync_dir / view).is_dir() for view in FIXED5_VIEW_ORDER)
        if has_legacy_layout and (requested_subsets is None or "fixed5" in requested_subsets):
            collect_samples_from_view_root(sync_dir, sync_name, "fixed5")

        for subset_name in ["fixed5", "zero_shot"]:
            if requested_subsets is not None and subset_name not in requested_subsets:
                continue
            subset_dir = sync_dir / subset_name
            if subset_dir.is_dir():
                collect_samples_from_view_root(subset_dir, sync_name, subset_name)

    print(f"Found {len(samples)} valid samples for {mode} mode")
    return samples


def init_metrics(args) -> Dict:
    """Initialize all selected non-dataset metrics."""
    metrics = {}

    if args.psnr:
        print("[Init] PSNR", flush=True)
        metrics['psnr'] = PSNR(reduction='none')
    if args.ssim:
        print("[Init] SSIM", flush=True)
        metrics['ssim'] = SSIM(reduction='none')
    if args.lpips:
        print("[Init] LPIPS", flush=True)
        metrics['lpips'] = LPIPS(reduction='none').to(args.device)
    if args.p_squeeze:
        print("[Init] P_Squeeze", flush=True)
        metrics['p_squeeze'] = P_Squeeze(reduction='none', device=args.device)
    if args.dino:
        print(f"[Init] DINO ({args.dino_model})", flush=True)
        metrics['dino'] = DINOSimilarity(
            reduction='none',
            model_name=args.dino_model,
            device=args.device,
        )
    if args.depth:
        print(f"[Init] DepthConsistency / MiDaS ({args.midas_model})", flush=True)
        metrics['depth'] = DepthConsistency(
            reduction='none',
            model_type=args.midas_model,
            device=args.device,
        )
    if args.lrce:
        print("[Init] LRCE", flush=True)
        metrics['lrce'] = LRCE(reduction='none')
    if HAS_SEGANY and getattr(args, 'segany', False):
        print("[Init] SegAnyConsistency", flush=True)
        metrics['segany'] = SegAnyConsistency(
            reduction='none',
            sam_checkpoint=args.sam_checkpoint,
            device=args.device,
        )
    if args.mv_depth:
        print(f"[Init] MultiViewConsistency depth ({args.midas_model})", flush=True)
        metrics['mv_depth'] = MultiViewConsistency(
            reduction='none',
            metric_type='depth_consistency',
            model_type=args.midas_model,
            device=args.device,
        )
    if args.mv_mask_dino:
        print(f"[Init] MultiViewConsistency masked DINO ({args.dino_model})", flush=True)
        metrics['mv_mask_dino'] = MultiViewConsistency(
            reduction='none',
            metric_type='warp_dino',
            model_type=args.midas_model,
            dino_model_name=args.dino_model,
            device=args.device,
        )
    if args.mv_seg_miou:
        if not HAS_SEGANY:
            raise ImportError("mv-seg-miou requires segment_anything to be installed.")
        print("[Init] MultiViewConsistency seg mIoU (SAM)", flush=True)
        metrics['mv_seg_miou'] = MultiViewConsistency(
            reduction='none',
            metric_type='warp_seg_miou',
            model_type=args.midas_model,
            sam_checkpoint=args.sam_checkpoint,
            device=args.device,
        )

    print(f"[Init] Initialized {len(metrics)} metrics: {list(metrics.keys())}", flush=True)
    return metrics


def compute_per_view_metrics(sample: Dict, metrics: Dict, device: str) -> Dict:
    """Compute per-view standard metrics for a single sample."""
    img_gt = load_image(sample['gt_path'], device)
    img_gen = load_image(sample['gen_path'], device)

    result = sample.copy()
    for name, metric in metrics.items():
        if name.startswith('mv_'):
            continue
        try:
            value = metric(img_gen, img_gt).item()
            result[name] = value
        except Exception as e:
            print(f"Error computing {name} for {sample['sync']}/{sample['view']}/{sample['frame_id']}: {e}")
            result[name] = float('nan')

    return result


def compute_multi_view_metrics(frame_group: Dict, metrics: Dict, device: str) -> List[Dict]:
    """Compute multi-view consistency metrics for adjacent view pairs of the same frame."""
    results = []
    sync = frame_group['sync']
    frame_id = frame_group['frame_id']
    views = frame_group['views']

    requested_subsets = frame_group.get('requested_subsets')
    if requested_subsets is None:
        requested_subsets = {"fixed5", "zero_shot"}
    else:
        requested_subsets = set(requested_subsets)

    pair_specs = []
    if "fixed5" in requested_subsets:
        for i in range(len(FIXED5_VIEW_ORDER) - 1):
            pair_specs.append(("fixed5", FIXED5_VIEW_ORDER[i], FIXED5_VIEW_ORDER[i + 1]))
    if "zero_shot" in requested_subsets:
        for ref_view, zero_view in ZERO_SHOT_REFERENCE_PAIRS:
            pair_specs.append(("zero_shot", ref_view, zero_view))

    for subset_name, view1_name, view2_name in pair_specs:
        if view1_name not in views or view2_name not in views:
            continue

        view1 = views[view1_name]
        view2 = views[view2_name]

        img1 = load_image(view1['gen_path'], device)
        img2 = load_image(view2['gen_path'], device)
        K1 = load_matrix(view1['K_path'], device)
        K2 = load_matrix(view2['K_path'], device)
        T1 = load_matrix(view1['T_path'], device)
        T2 = load_matrix(view2['T_path'], device)

        T_src_to_tgt = torch.bmm(torch.inverse(T2), T1)

        mv_result = {
            "mode": frame_group['mode'],
            "subset": subset_name,
            "sync": sync,
            "frame_id": frame_id,
            "view_pair": f"{view1_name}→{view2_name}",
        }

        for name, metric in metrics.items():
            if not name.startswith('mv_'):
                continue
            try:
                value = metric(
                    img1, img2,
                    K_src=K1,
                    K_tgt=K2,
                    T_src_to_tgt=T_src_to_tgt,
                ).item()
                mv_result[name] = value
            except Exception as e:
                print(f"Error computing {name} for {sync}/{frame_id}/{view1_name}→{view2_name}: {e}")
                mv_result[name] = float('nan')

        results.append(mv_result)

    return results


def main():
    args = parse_args()
    device = args.device
    print(f"[Start] results_root={args.results_root} mode={args.mode} device={device}", flush=True)

    resolved_mode_roots = resolve_mode_roots(args.results_root, args.mode)
    all_samples = []
    if not resolved_mode_roots:
        raise ValueError(
            f"Could not find any evaluable results under results-root={args.results_root}. "
            "Pass a directory that directly contains sync folders, or use --mode both for a parent directory."
        )

    for mode_name, mode_root in resolved_mode_roots:
        print(f"[Scan] mode={mode_name} root={mode_root}", flush=True)
        all_samples.extend(find_all_samples(str(mode_root), mode_name, args.syncs, args.views, args.subsets))

    if not all_samples:
        print("No valid samples found!")
        return

    print(f"[Scan] total valid samples={len(all_samples)}", flush=True)
    print("[Init] Building metric models...", flush=True)
    metrics = init_metrics(args)
    selected_per_view_metrics = [name for name in metrics.keys() if not name.startswith('mv_')]
    selected_multi_view_metrics = [name for name in metrics.keys() if name.startswith('mv_')]
    print(f"[Init] per-view metrics={selected_per_view_metrics}", flush=True)
    print(f"[Init] multi-view metrics={selected_multi_view_metrics}", flush=True)

    from collections import defaultdict
    frame_groups = defaultdict(lambda: defaultdict(dict))
    mv_samples = all_samples
    if selected_multi_view_metrics and args.subsets is not None and "zero_shot" in args.subsets:
        mv_samples = []
        for mode_name, mode_root in resolved_mode_roots:
            mv_samples.extend(find_all_samples(
                str(mode_root),
                mode_name,
                args.syncs,
                ALL_VIEWS,
                ["fixed5", "zero_shot"],
            ))

    for sample in mv_samples:
        key = (sample['mode'], sample['sync'], sample['frame_id'])
        frame_groups[key]['mode'] = sample['mode']
        frame_groups[key]['sync'] = sample['sync']
        frame_groups[key]['frame_id'] = sample['frame_id']
        frame_groups[key]['requested_subsets'] = set(args.subsets) if args.subsets else {"fixed5", "zero_shot"}
        frame_groups[key]['views'][sample['view']] = sample

    per_view_results = []
    if selected_per_view_metrics:
        print("\n[Run] Computing per-view metrics...", flush=True)
        for sample in tqdm(all_samples, desc="Per-view metrics", file=sys.stdout, dynamic_ncols=True):
            res = compute_per_view_metrics(sample, metrics, device)
            per_view_results.append(res)

    multi_view_results = []
    if selected_multi_view_metrics:
        print("\n[Run] Computing multi-view consistency metrics...", flush=True)
        for group in tqdm(frame_groups.values(), desc="Multi-view metrics", file=sys.stdout, dynamic_ncols=True):
            mv_res = compute_multi_view_metrics(group, metrics, device)
            multi_view_results.extend(mv_res)

    dataset_results = []
    if args.fid:
        print("\n[Run] Computing dataset-level FID...", flush=True)
        mode_subset_pairs = sorted({(sample['mode'], sample['subset']) for sample in all_samples})
        for mode_name, subset_name in mode_subset_pairs:
            print(f"[FID] mode={mode_name} subset={subset_name}", flush=True)
            mode_samples = [
                sample for sample in all_samples
                if sample['mode'] == mode_name and sample['subset'] == subset_name
            ]
            try:
                fid_value = compute_dataset_fid(mode_samples, device=device)
            except Exception as e:
                print(f"Error computing fid for {mode_name}/{subset_name}: {e}")
                fid_value = float('nan')
            dataset_results.append({
                'type': 'dataset',
                'mode': mode_name,
                'subset': subset_name,
                'sync': 'ALL',
                'frame_id': '',
                'view': 'all_selected',
                'view_pair': '',
                'fid': fid_value,
            })

    all_results = []
    for res in per_view_results:
        row = {
            'type': 'per_view',
            'mode': res['mode'],
            'subset': res.get('subset', ''),
            'sync': res['sync'],
            'frame_id': res['frame_id'],
            'view': res['view'],
            'view_pair': '',
        }
        for name in selected_per_view_metrics:
            row[name] = res.get(name, float('nan'))
        all_results.append(row)

    for res in multi_view_results:
        row = {
            'type': 'multi_view',
            'mode': res['mode'],
            'subset': res.get('subset', ''),
            'sync': res['sync'],
            'frame_id': res['frame_id'],
            'view': '',
            'view_pair': res['view_pair'],
        }
        for name in selected_multi_view_metrics:
            row[name] = res.get(name, float('nan'))
        all_results.append(row)

    all_results.extend(dataset_results)
    df = pd.DataFrame(all_results)

    print("\n=== Metrics Summary ===", flush=True)
    summary_pairs = [
        (mode_name, subset_name)
        for mode_name, subset_name in sorted({
            (row['mode'], row.get('subset', '')) for row in all_results if row.get('mode')
        })
    ]
    for mode_name, subset_name in summary_pairs:
        group_df = df[(df['mode'] == mode_name) & (df['subset'] == subset_name)]
        print(f"\n{mode_name.upper()} MODE / {subset_name.upper()}:", flush=True)

        per_view_df = group_df[group_df['type'] == 'per_view']
        if not per_view_df.empty:
            print("\nPer-view metrics (mean):", flush=True)
            for col in selected_per_view_metrics:
                mean_val = per_view_df[col].mean()
                std_val = per_view_df[col].std()
                print(f"  {col:15s}: {mean_val:.6f} ± {std_val:.6f}", flush=True)

        multi_view_df = group_df[group_df['type'] == 'multi_view']
        if not multi_view_df.empty:
            print("\nMulti-view metrics (mean):", flush=True)
            for col in selected_multi_view_metrics:
                mean_val = multi_view_df[col].mean()
                std_val = multi_view_df[col].std()
                print(f"  {col:15s}: {mean_val:.6f} ± {std_val:.6f}", flush=True)

        dataset_df = group_df[group_df['type'] == 'dataset']
        if args.fid and not dataset_df.empty and 'fid' in dataset_df.columns:
            fid_value = dataset_df['fid'].iloc[0]
            print("\nDataset metrics:", flush=True)
            print(f"  {'fid':15s}: {fid_value:.6f}", flush=True)

    df.to_csv(args.output_path, index=False)
    print(f"\n[Done] Full results saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

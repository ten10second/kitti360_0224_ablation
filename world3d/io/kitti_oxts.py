#!/usr/bin/env python3
"""
Small helpers to read KITTI OXTS yaw per frame.
Looks for yaw column in dataformat.txt; falls back to common index 5 (lat,lon,alt,roll,pitch,yaw,...)
"""
import os
from typing import Dict, List, Optional


def parse_oxts_dataformat(dataformat_path: str) -> Optional[Dict[str, int]]:
    if not os.path.isfile(dataformat_path):
        return None
    try:
        with open(dataformat_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines()]
        # Find the last line that looks like a whitespace-separated list of column names
        candidates = []
        for ln in lines:
            if not ln or ln.startswith('#'):
                continue
            toks = [t.strip().lower() for t in ln.replace(',', ' ').split() if t.strip()]
            # heuristic: must contain 'yaw' and at least 6 tokens
            if 'yaw' in toks and len(toks) >= 6:
                candidates.append(toks)
        cols = candidates[-1] if candidates else None
        if not cols:
            return None
        return {name: idx for idx, name in enumerate(cols)}
    except Exception:
        return None


def load_yaw_series(oxts_dir: str) -> List[float]:
    data_dir = os.path.join(oxts_dir, 'data') if not oxts_dir.endswith('data') else oxts_dir
    fmt_path = os.path.join(os.path.dirname(data_dir), 'dataformat.txt')
    name_to_idx = parse_oxts_dataformat(fmt_path)
    yaw_idx = name_to_idx.get('yaw', 5) if name_to_idx else 5

    files = sorted([fn for fn in os.listdir(data_dir) if fn.endswith('.txt')])
    yaws: List[float] = []
    for fn in files:
        fp = os.path.join(data_dir, fn)
        try:
            with open(fp, 'r') as f:
                ln = f.read().strip()
            vals = [float(x) for x in ln.split()]
            if yaw_idx >= len(vals):
                # fallback: try last value
                yaw = float(vals[-1])
            else:
                yaw = float(vals[yaw_idx])
            yaws.append(yaw)
        except Exception:
            yaws.append(0.0)
    return yaws


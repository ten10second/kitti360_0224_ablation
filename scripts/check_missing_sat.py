#!/usr/bin/env python3
"""
Satellite Folder Checker

Two modes:
1) Presence vs list (if --list_txt is provided):
   - --list_txt: text file lines: <image_path> <bin_path>
   - Compares image base names against files in --sat_dir (png/jpg/jpeg), reports missing.

2) Folder health check (no --list_txt):
   - Only scans --sat_dir, reports corrupt/unreadable files, non-RGB images, too-small images, and duplicate base names.

Examples:
  # Compare against training list (report missing names)
  python scripts/check_missing_sat.py --list_txt data/kitti_raw/train_list.txt --sat_dir /path/to/satellite

  # Just check the satellite folder quality (no list)
  python scripts/check_missing_sat.py --sat_dir /path/to/satellite --min_size 32

Notes:
  - Recursive scan under --sat_dir is enabled by default; disable with --no_recursive.
  - Use --write_missing to save missing base names (mode 1) or problematic file paths (mode 2).
"""
import argparse
import os
import sys
from typing import Dict, List, Set, Tuple


ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def load_list_file(list_txt: str) -> List[str]:
    """Read list file and return image base names (without extension)."""
    if not os.path.isfile(list_txt):
        raise FileNotFoundError(f"list_txt not found: {list_txt}")
    out: List[str] = []
    with open(list_txt, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 1:
                continue
            img_path = parts[0]
            base = os.path.splitext(os.path.basename(img_path))[0]
            out.append(base)
    return out


def scan_sat_dir(sat_dir: str, recursive: bool) -> Tuple[Set[str], int]:
    """Return set of available base names and total files matched."""
    names: Set[str] = set()
    total = 0
    if recursive:
        for root, _dirs, files in os.walk(sat_dir):
            for fn in files:
                if fn.endswith(ALLOWED_EXT):
                    total += 1
                    names.add(os.path.splitext(fn)[0])
    else:
        for fn in os.listdir(sat_dir):
            if fn.endswith(ALLOWED_EXT):
                total += 1
                names.add(os.path.splitext(fn)[0])
    return names, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list_txt", type=str, default=None, help="Optional: list file (<image_path> <bin_path>) for presence check")
    ap.add_argument("--sat_dir", required=True, help="Directory containing satellite images")
    ap.add_argument("--no_recursive", action="store_true", help="Disable recursive scan under sat_dir")
    ap.add_argument("--min_size", type=int, default=16, help="Flag images with min(H, W) < this size in health check mode")
    ap.add_argument("--write_missing", type=str, default=None, help="Write missing base names (mode 1) or bad file paths (mode 2)")
    args = ap.parse_args()

    if not os.path.isdir(args.sat_dir):
        print(f"[ERROR] sat_dir not found: {args.sat_dir}", file=sys.stderr)
        return 2

    # Mode selection
    if args.list_txt:
        # Mode 1: presence vs list
        expected = load_list_file(args.list_txt)
        expected_set = set(expected)
        present_set, present_files = scan_sat_dir(args.sat_dir, recursive=not args.no_recursive)

        missing = sorted(expected_set - present_set)
        found = len(expected_set) - len(missing)

        print("========================================")
        print("Satellite Presence Check (vs list)")
        print("========================================")
        print(f"List file        : {args.list_txt}")
        print(f"Satellite dir    : {args.sat_dir}")
        print(f"Recursive scan   : {not args.no_recursive}")
        print("")
        print(f"Expected (unique): {len(expected_set)} base names")
        print(f"Present files    : {present_files} (across allowed extensions)")
        print(f"Present (unique) : {len(present_set)} base names")
        print(f"Found            : {found}/{len(expected_set)}")
        print(f"Missing          : {len(missing)}")

        if missing:
            print("")
            print("Missing examples (first 20):")
            for name in missing[:20]:
                print(f"  - {name}")

        if args.write_missing:
            out_path = os.path.abspath(args.write_missing)
            os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
            with open(out_path, "w") as f:
                for name in missing:
                    f.write(name + "\n")
            print("")
            print(f"Missing list written to: {out_path}")

        return 0 if not missing else 1

    # Mode 2: health check of the folder
    print("========================================")
    print("Satellite Folder Health Check")
    print("========================================")
    print(f"Satellite dir  : {args.sat_dir}")
    print(f"Recursive scan : {not args.no_recursive}")
    print(f"Min size (px)  : {args.min_size}")

    bad_unreadable: List[str] = []
    bad_not_rgb: List[str] = []
    bad_too_small: List[str] = []
    duplicates: Dict[str, List[str]] = {}

    # gather files
    paths: List[str] = []
    if not args.no_recursive:
        for root, _dirs, files in os.walk(args.sat_dir):
            for fn in files:
                if fn.endswith(ALLOWED_EXT):
                    paths.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(args.sat_dir):
            if fn.endswith(ALLOWED_EXT):
                paths.append(os.path.join(args.sat_dir, fn))

    # check
    import cv2  # lazy import to avoid unnecessary dependency if only comparing names
    base_to_paths: Dict[str, List[str]] = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        base_to_paths.setdefault(base, []).append(p)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            bad_unreadable.append(p)
            continue
        h, w = img.shape[:2]
        if img.ndim != 3 or img.shape[2] != 3:
            bad_not_rgb.append(p)
        if min(h, w) < args.min_size:
            bad_too_small.append(p)

    for base, ps in base_to_paths.items():
        if len(ps) > 1:
            duplicates[base] = ps

    # report
    print("")
    print(f"Total images      : {len(paths)}")
    print(f"Unreadable        : {len(bad_unreadable)}")
    print(f"Non-RGB           : {len(bad_not_rgb)}")
    print(f"Too small (<{args.min_size}) : {len(bad_too_small)}")
    print(f"Duplicate basenames: {len(duplicates)}")

    def print_list(title: str, items: List[str], limit: int = 10):
        if not items:
            return
        print("")
        print(f"{title} (first {min(limit, len(items))}):")
        for x in items[:limit]:
            print(f"  - {x}")

    print_list("Unreadable files", bad_unreadable)
    print_list("Non-RGB files", bad_not_rgb)
    print_list("Too small files", bad_too_small)
    if duplicates:
        print("")
        print("Duplicate basenames examples (first 5 bases):")
        i = 0
        for base, ps in duplicates.items():
            print(f"  - {base}:")
            for p in ps[:3]:
                print(f"      * {p}")
            i += 1
            if i >= 5:
                break

    if args.write_missing:
        out_path = os.path.abspath(args.write_missing)
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        with open(out_path, "w") as f:
            for x in bad_unreadable:
                f.write(x + "\n")
            for x in bad_not_rgb:
                f.write(x + "\n")
            for x in bad_too_small:
                f.write(x + "\n")
        print("")
        print(f"Problem file list written to: {out_path}")

    # Return non-zero if any problems found
    problems = len(bad_unreadable) + len(bad_not_rgb) + len(bad_too_small)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

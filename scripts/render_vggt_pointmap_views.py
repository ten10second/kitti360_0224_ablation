#!/usr/bin/env python3
"""Render a metric VGGT RGB point map from its recovered camera viewpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


VIEW_NAMES = (
    "front_left_crop",
    "front_right_crop",
    "left_fisheye_yaw_-45",
    "left_fisheye_yaw_0",
    "left_fisheye_yaw_+45",
    "right_fisheye_yaw_-45",
    "right_fisheye_yaw_0",
    "right_fisheye_yaw_+45",
)


def render_points(
    points_world: np.ndarray,
    colors: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
    output_size: tuple[int, int],
    input_size: tuple[int, int],
    *,
    near_m: float,
    far_m: float,
    radius_px: int,
    background: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    """Project world points with a nearest-point z-buffer and circular splats."""
    width, height = output_size
    input_width, input_height = input_size
    R_world_cam = T_world_cam[:3, :3]
    camera_center = T_world_cam[:3, 3]

    # T_world_cam stores camera-to-world rotation.  For row vectors the
    # inverse rotation is therefore (p_world - c_world) @ R_world_cam.
    local = points_world - camera_center
    distance = np.linalg.norm(local, axis=1)
    nearby = (distance >= near_m) & (distance <= far_m)
    point_indices = np.flatnonzero(nearby)
    camera_points = local[nearby] @ R_world_cam
    depth = camera_points[:, 2]

    K_scaled = K.copy().astype(np.float64)
    K_scaled[0] *= width / float(input_width)
    K_scaled[1] *= height / float(input_height)
    positive = depth > near_m
    camera_points = camera_points[positive]
    depth = depth[positive]
    point_indices = point_indices[positive]
    projected = camera_points @ K_scaled.T
    u = np.rint(projected[:, 0] / depth).astype(np.int32)
    v = np.rint(projected[:, 1] / depth).astype(np.int32)
    margin = radius_px
    visible = (
        (u >= -margin) & (u < width + margin)
        & (v >= -margin) & (v < height + margin)
    )
    u, v, depth = u[visible], v[visible], depth[visible]
    point_indices = point_indices[visible]

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = background
    zbuffer = np.full(height * width, np.inf, dtype=np.float32)
    flat_image = image.reshape(-1, 3)
    offsets = [
        (dx, dy)
        for dy in range(-radius_px, radius_px + 1)
        for dx in range(-radius_px, radius_px + 1)
        if dx * dx + dy * dy <= radius_px * radius_px
    ]
    if not offsets:
        offsets = [(0, 0)]

    for dx, dy in offsets:
        uu, vv = u + dx, v + dy
        in_bounds = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        if not np.any(in_bounds):
            continue
        pixel = vv[in_bounds].astype(np.int64) * width + uu[in_bounds]
        z = depth[in_bounds]
        source = point_indices[in_bounds]

        # Sort by pixel first and depth second, then retain the closest point
        # for each pixel.  Comparing against the shared z-buffer also resolves
        # overlap between neighbouring splat offsets.
        order = np.lexsort((z, pixel))
        sorted_pixel = pixel[order]
        first = np.empty(len(order), dtype=bool)
        first[0] = True
        first[1:] = sorted_pixel[1:] != sorted_pixel[:-1]
        chosen = order[first]
        chosen_pixel = pixel[chosen]
        closer = z[chosen] < zbuffer[chosen_pixel]
        chosen, chosen_pixel = chosen[closer], chosen_pixel[closer]
        zbuffer[chosen_pixel] = z[chosen]
        flat_image[chosen_pixel] = colors[source[chosen]]

    return image, int(np.isfinite(zbuffer).sum())


def save_contact_sheet(paths: list[Path], output: Path, columns: int = 4) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    label_height = 34
    cell_width = images[0].width
    cell_height = images[0].height + label_height
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (244, 239, 247))
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(zip(paths, images)):
        x = index % columns * cell_width
        y = index // columns * cell_height
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 10, y + 9), path.stem, fill=(55, 43, 62))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame_index", type=int, default=8)
    parser.add_argument("--input_width", type=int, default=518)
    parser.add_argument("--input_height", type=int, default=350)
    parser.add_argument("--width", type=int, default=1036)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--near_m", type=float, default=0.4)
    parser.add_argument("--far_m", type=float, default=80.0)
    parser.add_argument("--radius_px", type=int, default=1)
    args = parser.parse_args()
    if args.frame_index < 0 or args.radius_px < 0:
        raise ValueError("frame_index and radius_px must be non-negative")

    data = np.load(args.pointmap)
    required = {"points", "colors", "camera_T_world", "camera_K", "view_camera_ids"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(f"point-map archive is missing {sorted(missing)}; re-export it")
    points = data["points"].astype(np.float32, copy=False)
    colors = data["colors"].astype(np.uint8, copy=False)
    cameras = data["camera_T_world"].astype(np.float32, copy=False)
    intrinsics = data["camera_K"].astype(np.float32, copy=False)
    views_per_frame = len(VIEW_NAMES)
    first = args.frame_index * views_per_frame
    if first + views_per_frame > len(cameras):
        raise IndexError(
            f"frame_index {args.frame_index} exceeds {len(cameras) // views_per_frame} frames"
        )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for local_index, name in enumerate(VIEW_NAMES):
        camera_index = first + local_index
        image, occupied = render_points(
            points,
            colors,
            cameras[camera_index],
            intrinsics[camera_index],
            (args.width, args.height),
            (args.input_width, args.input_height),
            near_m=args.near_m,
            far_m=args.far_m,
            radius_px=args.radius_px,
            background=(244, 239, 247),
        )
        path = output / f"{local_index:02d}_{name}.png"
        Image.fromarray(image).save(path)
        paths.append(path)
        coverage = occupied / float(args.width * args.height)
        print(f"{path}: occupied={occupied}, coverage={coverage:.3f}")

    save_contact_sheet(paths, output / "camera_view_contact_sheet.png")
    print(output / "camera_view_contact_sheet.png")


if __name__ == "__main__":
    main()

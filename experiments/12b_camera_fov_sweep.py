"""Camera/FOV sweep for the v1.2 full 3D Schwarzschild renderer."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.backgrounds import make_background
from src.camera import generate_camera_rays
from src.disk_models import disk_colorize
from src.geodesic_3d import (
    STATUS_CAPTURED,
    STATUS_DISK_HIT,
    STATUS_ESCAPED,
    STATUS_INCOMPLETE,
    integrate_ray_bundle_3d,
    sample_background_from_directions,
)


SWEEP_OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "sweeps" / "camera_fov"
SUMMARY_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep camera framing settings for the full 3D geodesic renderer."
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--fovs", nargs="+", type=float, default=[14.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    parser.add_argument("--camera-heights", nargs="+", type=float, default=[20.0, 40.0, 60.0, 80.0])
    parser.add_argument("--camera-distances", nargs="+", type=float, default=[100.0])
    parser.add_argument("--background", choices=["stars", "checkerboard", "radial", "galaxy"], default="stars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disk-inner-radius", type=float, default=6.0)
    parser.add_argument("--disk-outer-radius", type=float, default=30.0)
    parser.add_argument("--emissivity-power", type=float, default=1.75)
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--r-escape", type=float, default=150.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-scale", type=float, default=4.0)
    parser.add_argument("--shadow-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def save_tensor_image(tensor: torch.Tensor, path: pathlib.Path) -> None:
    """Save a (H, W, 3) float tensor in [0, 1] as PNG."""
    try:
        import torchvision.utils as tvu

        path.parent.mkdir(parents=True, exist_ok=True)
        tvu.save_image(tensor.permute(2, 0, 1).cpu(), str(path))
    except ImportError:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), tensor.cpu().numpy())


def make_status_rgb(status: np.ndarray) -> torch.Tensor:
    """Create a simple debug color map for per-pixel ray status."""
    height, width = status.shape
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[status == STATUS_CAPTURED] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    image[status == STATUS_DISK_HIT] = np.array([1.0, 0.55, 0.1], dtype=np.float32)
    image[status == STATUS_ESCAPED] = np.array([0.2, 0.45, 0.95], dtype=np.float32)
    image[status == STATUS_INCOMPLETE] = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    return torch.from_numpy(image)


def format_scalar(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def build_background(args: argparse.Namespace, width: int, height: int, device: torch.device) -> torch.Tensor:
    background_width = max(width, int(width * args.background_scale))
    background_height = max(height, int(background_width / 2))
    return make_background(
        args.background,
        width=background_width,
        height=background_height,
        seed=args.seed,
        device=device,
    )


def render_configuration(
    *,
    width: int,
    height: int,
    fov: float,
    camera_distance: float,
    camera_height: float,
    args: argparse.Namespace,
    background: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    camera_position = torch.tensor([0.0, -camera_distance, camera_height], device=device)
    target = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_hint = torch.tensor([0.0, 0.0, 1.0], device=device)

    directions, _, _, _, camera_position = generate_camera_rays(
        width=width,
        height=height,
        fov=fov,
        camera_position=camera_position,
        target=target,
        up_hint=up_hint,
        device=device,
    )

    result = integrate_ray_bundle_3d(
        camera_position=camera_position.cpu().numpy(),
        ray_directions=directions.cpu().numpy(),
        disk_inner_radius=args.disk_inner_radius,
        disk_outer_radius=args.disk_outer_radius,
        horizon_radius=2.05,
        r_escape=args.r_escape,
        max_steps=args.max_steps,
        step_size=args.step_size,
    )

    image = torch.zeros((height, width, 3), dtype=torch.float32)
    escaped_mask = result.status_map == STATUS_ESCAPED
    disk_mask = result.status_map == STATUS_DISK_HIT
    incomplete_mask = result.status_map == STATUS_INCOMPLETE

    if np.any(escaped_mask):
        safe_escape_direction = np.where(
            np.isfinite(result.escape_direction),
            result.escape_direction,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        background_image = sample_background_from_directions(background, safe_escape_direction)
        image[escaped_mask] = background_image[escaped_mask]

    if np.any(disk_mask) and not args.shadow_only:
        hit_radius = torch.from_numpy(result.hit_radius.astype(np.float32))
        raw = torch.pow((hit_radius / args.disk_inner_radius).clamp(min=1.0), -args.emissivity_power)
        raw_min = float((args.disk_outer_radius / args.disk_inner_radius) ** (-args.emissivity_power))
        intensity = ((raw - raw_min) / max(1.0 - raw_min, 1e-6)).clamp(0.0, 1.0)
        disk_rgb = disk_colorize(intensity)
        image[disk_mask] = disk_rgb[disk_mask]

    if np.any(incomplete_mask):
        image[incomplete_mask] = torch.tensor([0.18, 0.18, 0.18], dtype=torch.float32)

    capture_mask_rgb = torch.from_numpy(result.capture_mask.astype(np.float32)).unsqueeze(-1).repeat(1, 1, 3)
    disk_mask_rgb = torch.from_numpy(result.disk_hit_mask.astype(np.float32)).unsqueeze(-1).repeat(1, 1, 3)
    escaped_mask_rgb = torch.from_numpy(result.escaped_mask.astype(np.float32)).unsqueeze(-1).repeat(1, 1, 3)
    status_rgb = make_status_rgb(result.status_map)

    n_pixels = width * height
    metrics = {
        "captured_fraction": float(result.capture_mask.sum()) / n_pixels,
        "disk_hit_fraction": float(result.disk_hit_mask.sum()) / n_pixels,
        "escaped_fraction": float(result.escaped_mask.sum()) / n_pixels,
        "incomplete_fraction": float(result.incomplete_mask.sum()) / n_pixels,
    }
    return image.clamp(0.0, 1.0), status_rgb, capture_mask_rgb, disk_mask_rgb, escaped_mask_rgb, metrics


def score_configuration(row: dict[str, float], shadow_only: bool) -> float:
    captured_target = 0.45
    escaped_target = 0.30
    disk_target = 0.0 if shadow_only else 0.25
    score = 0.0
    score -= abs(row["captured_fraction"] - captured_target) * 2.2
    score -= abs(row["escaped_fraction"] - escaped_target) * 1.8
    score -= abs(row["disk_hit_fraction"] - disk_target) * 1.5
    score -= row["incomplete_fraction"] * 10.0
    if not shadow_only:
        if row["captured_fraction"] > 0.65:
            score -= (row["captured_fraction"] - 0.65) * 3.0
        if row["disk_hit_fraction"] > 0.55:
            score -= (row["disk_hit_fraction"] - 0.55) * 3.0
        if row["escaped_fraction"] < 0.12:
            score -= (0.12 - row["escaped_fraction"]) * 4.0
    return score


def write_summary_csv(summary_path: pathlib.Path, rows: list[dict[str, float]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fov",
        "camera_distance",
        "camera_height",
        "captured_fraction",
        "disk_hit_fraction",
        "escaped_fraction",
        "incomplete_fraction",
        "score",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))

    mode_name = "shadow_only" if args.shadow_only else "full"
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else SWEEP_OUTPUT_DIR / mode_name
    summary_path = pathlib.Path(args.summary_csv) if args.summary_csv else SUMMARY_DIR / f"camera_fov_sweep_{mode_name}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    background = build_background(args, width, height, device)
    rows: list[dict[str, float]] = []
    configurations = [
        (fov, camera_distance, camera_height)
        for camera_distance in args.camera_distances
        for camera_height in args.camera_heights
        for fov in args.fovs
    ]
    if args.limit is not None:
        configurations = configurations[: args.limit]

    for index, (fov, camera_distance, camera_height) in enumerate(configurations, start=1):
        tag = (
            f"fov_{format_scalar(fov)}"
            f"__dist_{format_scalar(camera_distance)}"
            f"__height_{format_scalar(camera_height)}"
        )
        config_dir = output_dir / tag
        config_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(configurations)}] Rendering "
            f"fov={fov}, camera_distance={camera_distance}, camera_height={camera_height}"
        )

        image, status_rgb, capture_mask_rgb, disk_mask_rgb, escaped_mask_rgb, metrics = render_configuration(
            width=width,
            height=height,
            fov=fov,
            camera_distance=camera_distance,
            camera_height=camera_height,
            args=args,
            background=background,
            device=device,
        )

        save_tensor_image(image, config_dir / "render.png")
        save_tensor_image(status_rgb, config_dir / "status_map.png")
        save_tensor_image(capture_mask_rgb, config_dir / "capture_mask.png")
        save_tensor_image(disk_mask_rgb, config_dir / "disk_hit_mask.png")
        save_tensor_image(escaped_mask_rgb, config_dir / "escaped_mask.png")

        row = {
            "fov": fov,
            "camera_distance": camera_distance,
            "camera_height": camera_height,
            **metrics,
        }
        row["score"] = score_configuration(row, shadow_only=args.shadow_only)
        rows.append(row)
        print(
            "  Fractions: "
            f"captured={row['captured_fraction']:.4f}, "
            f"disk={row['disk_hit_fraction']:.4f}, "
            f"escaped={row['escaped_fraction']:.4f}, "
            f"incomplete={row['incomplete_fraction']:.4f}, "
            f"score={row['score']:.4f}"
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    write_summary_csv(summary_path, rows)
    print(f"Saved summary CSV to {summary_path}")

    if rows:
        best = rows[0]
        print(
            "Recommended default: "
            f"fov={best['fov']}, "
            f"camera_distance={best['camera_distance']}, "
            f"camera_height={best['camera_height']} "
            f"(captured={best['captured_fraction']:.4f}, "
            f"disk={best['disk_hit_fraction']:.4f}, "
            f"escaped={best['escaped_fraction']:.4f}, "
            f"incomplete={best['incomplete_fraction']:.4f})"
        )


if __name__ == "__main__":
    main()
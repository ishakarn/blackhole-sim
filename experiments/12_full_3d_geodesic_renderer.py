"""v1.2 full 3D Schwarzschild curved-ray renderer."""

from __future__ import annotations

import argparse
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


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "12_full_3d_geodesic_renderer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full 3D Schwarzschild ray marching renderer with disk-plane crossings."
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=14.0)
    parser.add_argument("--background", choices=["stars", "checkerboard", "radial", "galaxy"], default="stars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-distance", type=float, default=40.0)
    parser.add_argument("--camera-height", type=float, default=8.0)
    parser.add_argument("--disk-inner-radius", type=float, default=6.0)
    parser.add_argument("--disk-outer-radius", type=float, default=20.0)
    parser.add_argument("--emissivity-power", type=float, default=1.75)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--r-escape", type=float, default=90.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-scale", type=float, default=4.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--show", action="store_true")
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
    print(f"Saved {path}")


def make_status_rgb(status: np.ndarray) -> torch.Tensor:
    """Create a simple debug color map for per-pixel ray status."""
    height, width = status.shape
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[status == STATUS_CAPTURED] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    image[status == STATUS_DISK_HIT] = np.array([1.0, 0.55, 0.1], dtype=np.float32)
    image[status == STATUS_ESCAPED] = np.array([0.2, 0.45, 0.95], dtype=np.float32)
    image[status == STATUS_INCOMPLETE] = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    return torch.from_numpy(image)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))

    background_width = max(width, int(width * args.background_scale))
    background_height = max(height, int(background_width / 2))
    background = make_background(
        args.background,
        width=background_width,
        height=background_height,
        seed=args.seed,
        device=device,
    )

    camera_position = torch.tensor([0.0, -args.camera_distance, args.camera_height], device=device)
    target = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_hint = torch.tensor([0.0, 0.0, 1.0], device=device)

    directions, _, _, _, camera_position = generate_camera_rays(
        width=width,
        height=height,
        fov=args.fov,
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
    background_image = torch.zeros_like(image)

    if np.any(escaped_mask):
        safe_escape_direction = np.where(
            np.isfinite(result.escape_direction),
            result.escape_direction,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        background_image = sample_background_from_directions(background, safe_escape_direction)
        image[escaped_mask] = background_image[escaped_mask]

    if np.any(disk_mask):
        hit_radius = torch.from_numpy(result.hit_radius.astype(np.float32))
        raw = torch.pow((hit_radius / args.disk_inner_radius).clamp(min=1.0), -args.emissivity_power)
        raw_min = float((args.disk_outer_radius / args.disk_inner_radius) ** (-args.emissivity_power))
        intensity = ((raw - raw_min) / max(1.0 - raw_min, 1e-6)).clamp(0.0, 1.0)
        disk_rgb = disk_colorize(intensity)
        image[disk_mask] = disk_rgb[disk_mask]

    if np.any(incomplete_mask):
        image[incomplete_mask] = torch.tensor([0.18, 0.18, 0.18], dtype=torch.float32)

    image = image.clamp(0.0, 1.0)

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "full3d_geodesic_disk.png"
    save_tensor_image(image, output_path)

    capture_mask_rgb = torch.from_numpy(result.capture_mask.astype(np.float32)).unsqueeze(-1).repeat(1, 1, 3)
    disk_mask_rgb = torch.from_numpy(result.disk_hit_mask.astype(np.float32)).unsqueeze(-1).repeat(1, 1, 3)
    escaped_mask_rgb = torch.from_numpy(result.escaped_mask.astype(np.float32)).unsqueeze(-1).repeat(1, 1, 3)
    incomplete_mask_rgb = torch.from_numpy(result.incomplete_mask.astype(np.float32)).unsqueeze(-1).repeat(1, 1, 3)
    status_rgb = make_status_rgb(result.status_map)
    save_tensor_image(capture_mask_rgb, OUTPUT_DIR / "full3d_capture_mask.png")
    save_tensor_image(disk_mask_rgb, OUTPUT_DIR / "full3d_disk_hit_mask.png")
    save_tensor_image(escaped_mask_rgb, OUTPUT_DIR / "full3d_escaped_mask.png")
    save_tensor_image(incomplete_mask_rgb, OUTPUT_DIR / "full3d_incomplete_mask.png")
    save_tensor_image(status_rgb, OUTPUT_DIR / "full3d_status_map.png")

    n_pixels = width * height
    captured_count = int(result.capture_mask.sum())
    disk_count = int(result.disk_hit_mask.sum())
    escaped_count = int(result.escaped_mask.sum())
    incomplete_count = int(result.incomplete_mask.sum())
    print(
        f"Ray statuses: captured={captured_count}, "
        f"disk={disk_count}, escaped={escaped_count}, incomplete={incomplete_count} / {n_pixels}"
    )
    print(
        "Fractions: "
        f"captured={captured_count / n_pixels:.4f}, "
        f"disk={disk_count / n_pixels:.4f}, "
        f"escaped={escaped_count / n_pixels:.4f}, "
        f"incomplete={incomplete_count / n_pixels:.4f}"
    )

    if args.show:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(capture_mask_rgb.cpu().numpy())
        axes[0].set_title("Capture Mask")
        axes[0].axis("off")
        axes[1].imshow(disk_mask_rgb.cpu().numpy())
        axes[1].set_title("Disk Hit Mask")
        axes[1].axis("off")
        axes[2].imshow(status_rgb.cpu().numpy())
        axes[2].set_title("Status Map")
        axes[2].axis("off")
        axes[3].imshow(image.cpu().numpy())
        axes[3].set_title("Full 3D Geodesic Render")
        axes[3].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
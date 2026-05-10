"""v1.8b photon momentum diagnostics for Schwarzschild transfer rendering."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.backgrounds import make_background
from src.camera import generate_camera_rays
from src.geodesic_3d import (
    STATUS_CAPTURED,
    STATUS_DISK_HIT,
    STATUS_ESCAPED,
    STATUS_INCOMPLETE,
    integrate_ray_bundle_3d,
    sample_background_from_directions,
)
from src.photon_transfer import (
    momentum_transfer_g_factor,
    photon_momentum_from_direction,
    schwarzschild_null_residual,
    tangent_transfer_g_factor,
    tetrad_null_residual,
    transfer_difference_stats,
)
from src.relativistic_disk import scalar_to_rgb


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "photon_diagnostics"
PRESETS: dict[str, tuple[int, int]] = {
    "preview": (256, 1),
    "quality": (512, 2),
    "high_quality": (1024, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Photon momentum diagnostics for the Schwarzschild transfer renderer."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quality")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--supersample", type=int, default=None)
    parser.add_argument("--fov", type=float, default=14.0)
    parser.add_argument("--background", choices=["stars", "checkerboard", "radial", "galaxy"], default="stars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-distance", type=float, default=100.0)
    parser.add_argument("--camera-height", type=float, default=80.0)
    parser.add_argument("--disk-inner-radius", type=float, default=6.0)
    parser.add_argument("--disk-outer-radius", type=float, default=25.0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset_resolution, preset_supersample = PRESETS[args.preset]
    if args.resolution is None:
        args.resolution = preset_resolution
    if args.supersample is None:
        args.supersample = preset_supersample
    args.supersample = int(np.clip(args.supersample, 1, 4))
    return args


def save_tensor_image(tensor: torch.Tensor, path: pathlib.Path) -> None:
    try:
        import torchvision.utils as tvu

        path.parent.mkdir(parents=True, exist_ok=True)
        tvu.save_image(tensor.permute(2, 0, 1).cpu(), str(path))
    except ImportError:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), tensor.cpu().numpy())
    print(f"Saved {path}")


def make_status_rgb(status: np.ndarray) -> torch.Tensor:
    height, width = status.shape
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[status == STATUS_CAPTURED] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    image[status == STATUS_DISK_HIT] = np.array([1.0, 0.55, 0.1], dtype=np.float32)
    image[status == STATUS_ESCAPED] = np.array([0.2, 0.45, 0.95], dtype=np.float32)
    image[status == STATUS_INCOMPLETE] = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    return torch.from_numpy(image)


def gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d / kernel_2d.sum()


def downsample_image(image: torch.Tensor, supersample: int, width: int, height: int) -> torch.Tensor:
    if supersample <= 1:
        return image
    image_nchw = image.permute(2, 0, 1).unsqueeze(0)
    downsampled = F.avg_pool2d(image_nchw, kernel_size=supersample, stride=supersample)
    return downsampled.squeeze(0).permute(1, 2, 0)[:height, :width]


def summarize_scalar(name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
    masked = values[mask > 0.0]
    if masked.numel() == 0:
        print(f"{name}: no disk-hit samples")
        return
    print(
        f"{name}: min={float(masked.min()):.6e}, max={float(masked.max()):.6e}, "
        f"mean={float(masked.mean()):.6e}, mean_abs={float(masked.abs().mean()):.6e}"
    )


def main() -> None:
    args = apply_preset(parse_args())
    device = torch.device("cpu")
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))
    supersample = args.supersample
    internal_width = width * supersample
    internal_height = height * supersample

    background_width = max(internal_width, int(internal_width * 4.0))
    background_height = max(internal_height, int(background_width / 2))
    background = make_background(
        args.background,
        width=background_width,
        height=background_height,
        seed=args.seed,
        device=device,
    )

    camera_position = torch.tensor([0.0, -args.camera_distance, args.camera_height], device=device, dtype=torch.float32)
    target = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_hint = torch.tensor([0.0, 0.0, 1.0], device=device)

    directions, _, _, _, camera_position = generate_camera_rays(
        width=internal_width,
        height=internal_height,
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
        r_escape=150.0,
        max_steps=7000,
        step_size=0.005,
    )

    disk_mask = torch.from_numpy(result.status_map == STATUS_DISK_HIT).to(device=device, dtype=torch.float32)
    hit_radius = torch.from_numpy(result.hit_radius.astype(np.float32)).to(device=device)
    hit_position = torch.from_numpy(result.hit_position.astype(np.float32)).to(device=device)
    hit_direction = torch.from_numpy(result.hit_direction.astype(np.float32)).to(device=device)

    momentum = photon_momentum_from_direction(hit_position, hit_direction, disk_mask)
    g_tangent = tangent_transfer_g_factor(
        hit_position,
        hit_direction,
        hit_radius.clamp(min=args.disk_inner_radius),
        disk_mask,
    )
    g_momentum, _ = momentum_transfer_g_factor(
        hit_position,
        hit_direction,
        hit_radius.clamp(min=args.disk_inner_radius),
        disk_mask,
    )
    g_stats = transfer_difference_stats(g_tangent, g_momentum, disk_mask)

    null_residual = schwarzschild_null_residual(momentum, disk_mask)
    tetrad_residual, tetrad_momentum = tetrad_null_residual(momentum, disk_mask)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_tensor_image(downsample_image(scalar_to_rgb(momentum["k_r"], disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_k_r.png")
    save_tensor_image(downsample_image(scalar_to_rgb(momentum["k_theta"], disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_k_theta.png")
    save_tensor_image(downsample_image(scalar_to_rgb(momentum["k_phi"], disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_k_phi.png")
    save_tensor_image(downsample_image(scalar_to_rgb(momentum["k_t"], disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_k_t.png")
    save_tensor_image(downsample_image(scalar_to_rgb(momentum["theta"], disk_mask, min_value=0.0, max_value=float(np.pi)), supersample, width, height), OUTPUT_DIR / "full3d_hit_theta.png")
    save_tensor_image(downsample_image(scalar_to_rgb(momentum["phi"], disk_mask, min_value=-float(np.pi), max_value=float(np.pi)), supersample, width, height), OUTPUT_DIR / "full3d_hit_phi.png")
    save_tensor_image(downsample_image(scalar_to_rgb(null_residual.abs(), disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_null_residual.png")
    save_tensor_image(downsample_image(scalar_to_rgb(tetrad_residual.abs(), disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_tetrad_null_residual.png")
    save_tensor_image(downsample_image(scalar_to_rgb(g_tangent, disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_transfer_g_factor_tangent.png")
    save_tensor_image(downsample_image(scalar_to_rgb(g_momentum, disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_transfer_g_factor_momentum.png")
    save_tensor_image(downsample_image(scalar_to_rgb((g_momentum - g_tangent).abs(), disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_transfer_g_factor_difference.png")
    save_tensor_image(downsample_image(scalar_to_rgb((g_momentum / g_tangent.clamp(min=1e-6)), disk_mask), supersample, width, height), OUTPUT_DIR / "full3d_transfer_g_factor_ratio.png")
    save_tensor_image(downsample_image(make_status_rgb(result.status_map).to(device=device), supersample, width, height), OUTPUT_DIR / "full3d_transfer_status_map.png")

    summarize_scalar("null residual", null_residual, disk_mask)
    summarize_scalar("tetrad null residual", tetrad_residual, disk_mask)
    print(
        f"g tangent: min={g_stats['tangent_min']:.6f}, max={g_stats['tangent_max']:.6f}, mean={g_stats['tangent_mean']:.6f}"
    )
    print(
        f"g momentum: min={g_stats['momentum_min']:.6f}, max={g_stats['momentum_max']:.6f}, mean={g_stats['momentum_mean']:.6f}"
    )
    print(
        f"g difference: mean_abs={g_stats['mean_abs_diff']:.6f}, max_abs={g_stats['max_abs_diff']:.6f}"
    )

    if args.show:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        panels = [
            (downsample_image(scalar_to_rgb(momentum["k_r"], disk_mask), supersample, width, height), "k_r"),
            (downsample_image(scalar_to_rgb(momentum["k_phi"], disk_mask), supersample, width, height), "k_phi"),
            (downsample_image(scalar_to_rgb(g_momentum, disk_mask), supersample, width, height), "g Momentum"),
            (downsample_image(scalar_to_rgb(null_residual.abs(), disk_mask), supersample, width, height), "Null Residual"),
            (downsample_image(scalar_to_rgb(tetrad_residual.abs(), disk_mask), supersample, width, height), "Tetrad Null Residual"),
            (downsample_image(scalar_to_rgb((g_momentum - g_tangent).abs(), disk_mask), supersample, width, height), "|g diff|"),
        ]
        for axis, (panel, title) in zip(axes.flat, panels):
            axis.imshow(panel.cpu().numpy())
            axis.set_title(title)
            axis.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
"""v1.3 refined full 3D Schwarzschild renderer with smoother disk emission."""

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


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "13_refined_3d_geodesic_renderer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refined full 3D Schwarzschild ray marching renderer with smooth disk emission."
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=14.0)
    parser.add_argument("--background", choices=["stars", "checkerboard", "radial", "galaxy"], default="stars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-distance", type=float, default=100.0)
    parser.add_argument("--camera-height", type=float, default=80.0)
    parser.add_argument("--disk-inner-radius", type=float, default=6.0)
    parser.add_argument("--disk-outer-radius", type=float, default=30.0)
    parser.add_argument("--emissivity-power", type=float, default=2.5)
    parser.add_argument("--disk-alpha", type=float, default=0.9)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--r-escape", type=float, default=150.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-scale", type=float, default=4.0)
    parser.add_argument("--bloom", action="store_true")
    parser.add_argument("--bloom-strength", type=float, default=0.35)
    parser.add_argument("--bloom-radius", type=int, default=5)
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


def make_disk_radius_rgb(hit_radius: torch.Tensor, disk_mask: torch.Tensor, r_in: float, r_out: float) -> torch.Tensor:
    """Visualize disk-hit radius as a warm scalar field for debugging."""
    radius_clamped = hit_radius.clamp(min=r_in, max=r_out)
    radius_t = ((radius_clamped - r_in) / max(r_out - r_in, 1e-6)).clamp(0.0, 1.0)
    image = torch.zeros((*hit_radius.shape, 3), dtype=torch.float32, device=hit_radius.device)
    image[..., 0] = 1.0 - 0.45 * radius_t
    image[..., 1] = 0.95 - 0.85 * radius_t
    image[..., 2] = 0.2 + 0.55 * radius_t
    return image * disk_mask.unsqueeze(-1)


def warm_disk_colorize(intensity: torch.Tensor) -> torch.Tensor:
    """Map normalized disk intensity to a warm white-yellow-orange-red ramp."""
    t = intensity.clamp(0.0, 1.0)
    dark = torch.tensor([0.08, 0.005, 0.005], device=t.device, dtype=torch.float32)
    deep_red = torch.tensor([0.33, 0.03, 0.01], device=t.device, dtype=torch.float32)
    orange = torch.tensor([0.95, 0.33, 0.05], device=t.device, dtype=torch.float32)
    yellow_white = torch.tensor([1.0, 0.96, 0.82], device=t.device, dtype=torch.float32)

    low_mix = torch.clamp(t / 0.45, 0.0, 1.0).unsqueeze(-1)
    mid_mix = torch.clamp((t - 0.25) / 0.45, 0.0, 1.0).unsqueeze(-1)
    high_mix = torch.clamp((t - 0.7) / 0.3, 0.0, 1.0).unsqueeze(-1)

    color = dark + low_mix * (deep_red - dark)
    color = color + mid_mix * (orange - color)
    color = color + high_mix * (yellow_white - color)
    return color.clamp(0.0, 1.0)


def make_disk_emission(
    hit_radius: torch.Tensor,
    disk_mask: torch.Tensor,
    disk_inner_radius: float,
    disk_outer_radius: float,
    emissivity_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build smooth disk emission and normalized intensity from disk-hit radii."""
    radius_clamped = hit_radius.clamp(min=disk_inner_radius, max=disk_outer_radius)
    raw = torch.pow(radius_clamped / disk_inner_radius, -emissivity_power)
    raw_outer = float((disk_outer_radius / disk_inner_radius) ** (-emissivity_power))
    intensity = ((raw - raw_outer) / max(1.0 - raw_outer, 1e-6)).clamp(0.0, 1.0)
    intensity = torch.pow(intensity, 0.8)
    disk_rgb = warm_disk_colorize(intensity)
    disk_rgb = disk_rgb * intensity.unsqueeze(-1)
    disk_rgb = disk_rgb * disk_mask.unsqueeze(-1)
    return disk_rgb.clamp(0.0, 1.0), intensity * disk_mask


def gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def apply_bloom(image: torch.Tensor, emission: torch.Tensor, strength: float, radius: int) -> torch.Tensor:
    """Add a soft glow around bright disk emission."""
    if strength <= 0.0 or radius <= 0:
        return image

    kernel_size = radius * 2 + 1
    sigma = max(radius / 2.0, 1.0)
    kernel = gaussian_kernel(kernel_size, sigma, image.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(3, 1, 1, 1)

    emission_nchw = emission.permute(2, 0, 1).unsqueeze(0)
    blurred = F.conv2d(emission_nchw, kernel, padding=radius, groups=3)
    blurred = blurred.squeeze(0).permute(1, 2, 0)
    return (image + strength * blurred).clamp(0.0, 1.0)


def downsample_image(image: torch.Tensor, supersample: int, width: int, height: int) -> torch.Tensor:
    """Average downsample a supersampled image back to output size."""
    if supersample <= 1:
        return image
    image_nchw = image.permute(2, 0, 1).unsqueeze(0)
    downsampled = F.avg_pool2d(image_nchw, kernel_size=supersample, stride=supersample)
    downsampled = downsampled.squeeze(0).permute(1, 2, 0)
    return downsampled[:height, :width]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    supersample = max(1, args.supersample)
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))
    internal_width = width * supersample
    internal_height = height * supersample

    background_width = max(internal_width, int(internal_width * args.background_scale))
    background_height = max(internal_height, int(background_width / 2))
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
        r_escape=args.r_escape,
        max_steps=args.max_steps,
        step_size=args.step_size,
    )

    image = torch.zeros((internal_height, internal_width, 3), dtype=torch.float32, device=device)
    escaped_mask = torch.from_numpy(result.status_map == STATUS_ESCAPED).to(device=device)
    disk_mask = torch.from_numpy(result.status_map == STATUS_DISK_HIT).to(device=device)
    incomplete_mask = torch.from_numpy(result.status_map == STATUS_INCOMPLETE).to(device=device)
    background_image = torch.zeros_like(image)
    direct_background = sample_background_from_directions(background, directions.cpu().numpy()).to(device=device)

    if np.any(result.status_map == STATUS_ESCAPED):
        safe_escape_direction = np.where(
            np.isfinite(result.escape_direction),
            result.escape_direction,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        background_image = sample_background_from_directions(background, safe_escape_direction).to(device=device)
        image[escaped_mask] = background_image[escaped_mask]

    hit_radius = torch.from_numpy(result.hit_radius.astype(np.float32)).to(device=device)
    disk_emission, disk_intensity = make_disk_emission(
        hit_radius=hit_radius,
        disk_mask=disk_mask.float(),
        disk_inner_radius=args.disk_inner_radius,
        disk_outer_radius=args.disk_outer_radius,
        emissivity_power=args.emissivity_power,
    )
    disk_alpha = (args.disk_alpha * torch.pow(disk_intensity, 0.6)).clamp(0.0, 1.0).unsqueeze(-1)
    image = torch.where(disk_mask.unsqueeze(-1), disk_emission * disk_alpha + direct_background * (1.0 - disk_alpha), image)

    if np.any(result.status_map == STATUS_INCOMPLETE):
        image[incomplete_mask] = torch.tensor([0.18, 0.18, 0.18], dtype=torch.float32, device=device)

    raw_image = image.clamp(0.0, 1.0)
    bloomed_image = apply_bloom(raw_image, disk_emission, args.bloom_strength, args.bloom_radius) if args.bloom else raw_image

    final_image = downsample_image(bloomed_image, supersample, width, height).clamp(0.0, 1.0)
    disk_emission_only = downsample_image(disk_emission.clamp(0.0, 1.0), supersample, width, height)
    hit_radius_rgb = make_disk_radius_rgb(hit_radius, disk_mask.float(), args.disk_inner_radius, args.disk_outer_radius)
    hit_radius_rgb = downsample_image(hit_radius_rgb.clamp(0.0, 1.0), supersample, width, height)
    status_rgb = downsample_image(make_status_rgb(result.status_map).to(device=device), supersample, width, height)

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "full3d_geodesic_disk_refined.png"
    save_tensor_image(final_image, output_path)
    save_tensor_image(disk_emission_only, OUTPUT_DIR / "full3d_disk_emission_only.png")
    save_tensor_image(hit_radius_rgb, OUTPUT_DIR / "full3d_disk_hit_radius.png")
    save_tensor_image(status_rgb, OUTPUT_DIR / "full3d_status_map.png")
    if args.bloom:
        raw_downsampled = downsample_image(raw_image, supersample, width, height).clamp(0.0, 1.0)
        save_tensor_image(raw_downsampled, OUTPUT_DIR / "full3d_geodesic_disk_refined_raw.png")

    n_pixels = internal_width * internal_height
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
        axes[0].imshow(status_rgb.cpu().numpy())
        axes[0].set_title("Status Map")
        axes[0].axis("off")
        axes[1].imshow(hit_radius_rgb.cpu().numpy())
        axes[1].set_title("Disk Hit Radius")
        axes[1].axis("off")
        axes[2].imshow(disk_emission_only.cpu().numpy())
        axes[2].set_title("Disk Emission")
        axes[2].axis("off")
        axes[3].imshow(final_image.cpu().numpy())
        axes[3].set_title("Refined Full 3D Render")
        axes[3].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
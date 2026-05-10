"""v1.6 full 3D Schwarzschild renderer with transfer-function disk shading."""

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
from src.relativistic_disk import (
    disk_emissivity,
    heuristic_g_proxy,
    normalize_masked,
    scalar_to_rgb,
    transfer_g_factor,
    warm_disk_colorize,
)


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "16_relativistic_transfer_renderer"
PRESETS: dict[str, tuple[int, int]] = {
    "preview": (256, 1),
    "quality": (512, 2),
    "high_quality": (1024, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full 3D Schwarzschild render with relativistic transfer-function disk shading."
    )
    parser.add_argument("--physics-mode", choices=["transfer", "heuristic"], default="transfer")
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
    parser.add_argument("--emissivity-mode", choices=["thin_disk", "power_law"], default="thin_disk")
    parser.add_argument("--emissivity-power", type=float, default=3.0)
    parser.add_argument("--rotation-direction", choices=["prograde", "retrograde"], default="prograde")
    parser.add_argument("--intensity-power", type=int, choices=[3, 4], default=3)
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--r-escape", type=float, default=150.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-scale", type=float, default=4.0)
    parser.add_argument("--bloom", action="store_true")
    parser.add_argument("--bloom-strength", type=float, default=0.15)
    parser.add_argument("--bloom-radius", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
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


def blur_image(image: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0.0:
        return image
    kernel_radius = max(1, int(np.ceil(radius * 2.0)))
    kernel_size = kernel_radius * 2 + 1
    sigma = max(radius, 0.5)
    kernel = gaussian_kernel(kernel_size, sigma, image.device).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(3, 1, 1, 1)
    image_nchw = image.permute(2, 0, 1).unsqueeze(0)
    blurred = F.conv2d(image_nchw, kernel, padding=kernel_radius, groups=3)
    return blurred.squeeze(0).permute(1, 2, 0)


def apply_bloom(image: torch.Tensor, emission: torch.Tensor, strength: float, radius: int) -> torch.Tensor:
    if strength <= 0.0 or radius <= 0:
        return image
    kernel_size = radius * 2 + 1
    sigma = max(radius / 2.0, 1.0)
    kernel = gaussian_kernel(kernel_size, sigma, image.device).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(3, 1, 1, 1)
    emission_nchw = emission.permute(2, 0, 1).unsqueeze(0)
    blurred = F.conv2d(emission_nchw, kernel, padding=radius, groups=3)
    blurred = blurred.squeeze(0).permute(1, 2, 0)
    return (image + strength * blurred).clamp(0.0, 1.0)


def downsample_image(image: torch.Tensor, supersample: int, width: int, height: int) -> torch.Tensor:
    if supersample <= 1:
        return image
    image_nchw = image.permute(2, 0, 1).unsqueeze(0)
    downsampled = F.avg_pool2d(image_nchw, kernel_size=supersample, stride=supersample)
    return downsampled.squeeze(0).permute(1, 2, 0)[:height, :width]


def main() -> None:
    args = apply_preset(parse_args())
    device = torch.device(args.device)
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))
    supersample = args.supersample
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
        r_escape=args.r_escape,
        max_steps=args.max_steps,
        step_size=args.step_size,
    )

    captured_mask = torch.from_numpy(result.status_map == STATUS_CAPTURED).to(device=device, dtype=torch.bool)
    escaped_mask = torch.from_numpy(result.status_map == STATUS_ESCAPED).to(device=device, dtype=torch.bool)
    disk_mask = torch.from_numpy(result.status_map == STATUS_DISK_HIT).to(device=device, dtype=torch.float32)
    incomplete_mask = torch.from_numpy(result.status_map == STATUS_INCOMPLETE).to(device=device, dtype=torch.bool)

    direct_background = sample_background_from_directions(background, directions.cpu().numpy()).to(device=device)
    blurred_background = blur_image(direct_background, max(1.0, 0.75 * supersample))
    escaped_background = direct_background.clone()
    if np.any(result.status_map == STATUS_ESCAPED):
        safe_escape_direction = np.where(
            np.isfinite(result.escape_direction),
            result.escape_direction,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        escaped_background = sample_background_from_directions(background, safe_escape_direction).to(device=device)

    image = torch.zeros((internal_height, internal_width, 3), dtype=torch.float32, device=device)
    image[escaped_mask] = escaped_background[escaped_mask]
    image[captured_mask] = 0.0

    hit_radius = torch.from_numpy(result.hit_radius.astype(np.float32)).to(device=device)
    hit_position = torch.from_numpy(result.hit_position.astype(np.float32)).to(device=device)
    hit_direction = torch.from_numpy(result.hit_direction.astype(np.float32)).to(device=device)

    emissivity = disk_emissivity(
        hit_radius,
        disk_mask,
        disk_inner_radius=args.disk_inner_radius,
        emissivity_mode=args.emissivity_mode,
        emissivity_power=args.emissivity_power,
    )
    emissivity_map = normalize_masked(emissivity, disk_mask)

    transfer_g = transfer_g_factor(
        hit_position,
        hit_direction,
        hit_radius.clamp(min=args.disk_inner_radius),
        disk_mask,
        rotation_direction=args.rotation_direction,
    )
    heuristic_g = heuristic_g_proxy(
        hit_position,
        hit_direction,
        hit_radius.clamp(min=args.disk_inner_radius),
        disk_mask,
        rotation_direction=args.rotation_direction,
    )
    g_factor = transfer_g if args.physics_mode == "transfer" else heuristic_g

    observed_intensity_raw = emissivity * torch.pow(g_factor, float(args.intensity_power))
    observed_intensity = normalize_masked(observed_intensity_raw, disk_mask)
    disk_color = warm_disk_colorize(observed_intensity)
    disk_emission = disk_color * observed_intensity.unsqueeze(-1) * disk_mask.unsqueeze(-1)

    disk_composite = disk_emission + 0.02 * blurred_background * disk_mask.unsqueeze(-1)
    image = torch.where(disk_mask.unsqueeze(-1) > 0.0, disk_composite.clamp(0.0, 1.0), image)
    if torch.any(incomplete_mask):
        image[incomplete_mask] = torch.tensor([0.18, 0.18, 0.18], dtype=torch.float32, device=device)

    raw_internal = image.clamp(0.0, 1.0)
    final_internal = apply_bloom(raw_internal, disk_emission, args.bloom_strength, args.bloom_radius) if args.bloom else raw_internal

    final_image = downsample_image(final_internal, supersample, width, height).clamp(0.0, 1.0)
    raw_image = downsample_image(raw_internal, supersample, width, height).clamp(0.0, 1.0)
    emissivity_out = downsample_image(scalar_to_rgb(emissivity_map, disk_mask), supersample, width, height).clamp(0.0, 1.0)
    g_out = downsample_image(scalar_to_rgb(g_factor, disk_mask), supersample, width, height).clamp(0.0, 1.0)
    intensity_out = downsample_image(scalar_to_rgb(observed_intensity, disk_mask), supersample, width, height).clamp(0.0, 1.0)
    status_out = downsample_image(make_status_rgb(result.status_map).to(device=device), supersample, width, height).clamp(0.0, 1.0)

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "full3d_transfer_render.png"
    save_tensor_image(final_image, output_path)
    save_tensor_image(raw_image, OUTPUT_DIR / "full3d_transfer_raw.png")
    save_tensor_image(emissivity_out, OUTPUT_DIR / "full3d_transfer_emissivity.png")
    save_tensor_image(g_out, OUTPUT_DIR / "full3d_transfer_g_factor.png")
    save_tensor_image(intensity_out, OUTPUT_DIR / "full3d_transfer_intensity.png")
    save_tensor_image(status_out, OUTPUT_DIR / "full3d_transfer_status_map.png")

    n_pixels = internal_width * internal_height
    captured_count = int(result.capture_mask.sum())
    disk_count = int(result.disk_hit_mask.sum())
    escaped_count = int(result.escaped_mask.sum())
    incomplete_count = int(result.incomplete_mask.sum())
    print(f"Preset: {args.preset}, resolution={width}, supersample={supersample}, physics_mode={args.physics_mode}")
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
    if torch.any(disk_mask > 0.0):
        g_masked = g_factor[disk_mask > 0.0]
        print(
            f"g range: min={float(g_masked.min()):.4f}, "
            f"max={float(g_masked.max()):.4f}, mean={float(g_masked.mean()):.4f}"
        )

    if args.show:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        panels = [
            (emissivity_out, "Emissivity"),
            (g_out, "g Factor"),
            (intensity_out, "Observed Intensity"),
            (status_out, "Status Map"),
            (raw_image, "Raw Transfer"),
            (final_image, "Transfer Render"),
        ]
        for axis, (panel, title) in zip(axes.flat, panels):
            axis.imshow(panel.cpu().numpy())
            axis.set_title(title)
            axis.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
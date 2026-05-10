"""v1.5 polished full 3D Schwarzschild renderer with improved image quality."""

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
from src.constants import M
from src.geodesic_3d import (
    STATUS_CAPTURED,
    STATUS_DISK_HIT,
    STATUS_ESCAPED,
    STATUS_INCOMPLETE,
    integrate_ray_bundle_3d,
    sample_background_from_directions,
)


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "15_polished_render"
PRESETS: dict[str, tuple[int, int]] = {
    "preview": (256, 1),
    "quality": (512, 2),
    "high_quality": (1024, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polished full 3D Schwarzschild render with improved anti-aliasing and compositing."
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
    parser.add_argument("--emissivity-power", type=float, default=2.5)
    parser.add_argument("--disk-alpha", type=float, default=1.0)
    parser.add_argument("--disk-background-mix", type=float, default=0.02)
    parser.add_argument("--inner-edge-width", type=float, default=1.5)
    parser.add_argument("--outer-edge-width", type=float, default=3.0)
    parser.add_argument("--beaming-strength", type=float, default=0.8)
    parser.add_argument("--beaming-power", type=float, default=3.0)
    parser.add_argument("--rotation-direction", choices=["ccw", "cw"], default="ccw")
    parser.add_argument("--redshift-strength", type=float, default=0.5)
    parser.add_argument("--redshift-power", type=float, default=1.0)
    parser.add_argument("--shadow-edge-smoothing", type=float, default=0.75)
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--r-escape", type=float, default=150.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-scale", type=float, default=4.0)
    parser.add_argument("--bloom", action="store_true")
    parser.add_argument("--bloom-strength", type=float, default=0.2)
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


def smoothstep(edge0: float, edge1: float, values: torch.Tensor) -> torch.Tensor:
    denom = max(edge1 - edge0, 1e-6)
    t = ((values - edge0) / denom).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def warm_disk_colorize(intensity: torch.Tensor) -> torch.Tensor:
    t = intensity.clamp(0.0, 1.0)
    dark = torch.tensor([0.06, 0.004, 0.004], device=t.device, dtype=torch.float32)
    deep_red = torch.tensor([0.34, 0.025, 0.01], device=t.device, dtype=torch.float32)
    orange = torch.tensor([0.96, 0.34, 0.05], device=t.device, dtype=torch.float32)
    yellow = torch.tensor([1.0, 0.86, 0.38], device=t.device, dtype=torch.float32)
    white_hot = torch.tensor([1.0, 0.98, 0.9], device=t.device, dtype=torch.float32)

    low_mix = torch.clamp(t / 0.35, 0.0, 1.0).unsqueeze(-1)
    mid_mix = torch.clamp((t - 0.2) / 0.45, 0.0, 1.0).unsqueeze(-1)
    high_mix = torch.clamp((t - 0.6) / 0.25, 0.0, 1.0).unsqueeze(-1)
    white_mix = torch.clamp((t - 0.85) / 0.15, 0.0, 1.0).unsqueeze(-1)

    color = dark + low_mix * (deep_red - dark)
    color = color + mid_mix * (orange - color)
    color = color + high_mix * (yellow - color)
    color = color + white_mix * (white_hot - color)
    return color.clamp(0.0, 1.0)


def gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d / kernel_2d.sum()


def blur_scalar(values: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0.0:
        return values
    kernel_radius = max(1, int(np.ceil(radius * 2.0)))
    kernel_size = kernel_radius * 2 + 1
    sigma = max(radius, 0.5)
    kernel = gaussian_kernel(kernel_size, sigma, values.device).view(1, 1, kernel_size, kernel_size)
    value_nchw = values.unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(value_nchw, kernel, padding=kernel_radius)
    return blurred.squeeze(0).squeeze(0)


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


def build_polished_disk_maps(
    *,
    hit_radius: torch.Tensor,
    hit_position: torch.Tensor,
    disk_mask: torch.Tensor,
    camera_position: torch.Tensor,
    disk_inner_radius: float,
    disk_outer_radius: float,
    emissivity_power: float,
    disk_alpha: float,
    inner_edge_width: float,
    outer_edge_width: float,
    beaming_strength: float,
    beaming_power: float,
    rotation_direction: str,
    redshift_strength: float,
    redshift_power: float,
) -> dict[str, torch.Tensor]:
    safe_hit_radius = torch.where(
        disk_mask > 0.0,
        torch.nan_to_num(hit_radius, nan=disk_outer_radius),
        torch.full_like(hit_radius, disk_outer_radius),
    )
    safe_hit_position = torch.where(
        disk_mask.unsqueeze(-1) > 0.0,
        torch.nan_to_num(hit_position, nan=0.0),
        torch.zeros_like(hit_position),
    )

    radius = safe_hit_radius.clamp(min=disk_inner_radius, max=disk_outer_radius)
    emissivity_raw = torch.pow(radius / disk_inner_radius, -emissivity_power)
    emissivity_outer = float((disk_outer_radius / disk_inner_radius) ** (-emissivity_power))
    emissivity = ((emissivity_raw - emissivity_outer) / max(1.0 - emissivity_outer, 1e-6)).clamp(0.0, 1.0)

    inner_highlight = 1.0
    if inner_edge_width > 0.0:
        inner_highlight = 1.0 + 0.35 * (1.0 - smoothstep(disk_inner_radius, disk_inner_radius + inner_edge_width, radius))
    emissivity = (emissivity * inner_highlight).clamp(0.0, 1.5) * disk_mask

    x = safe_hit_position[..., 0]
    y = safe_hit_position[..., 1]
    tangential = torch.stack([-y, x, torch.zeros_like(x)], dim=-1)
    if rotation_direction == "cw":
        tangential = -tangential
    tangential = F.normalize(tangential, dim=-1, eps=1e-6)

    beta = torch.sqrt(torch.clamp(torch.tensor(M, device=radius.device, dtype=radius.dtype) / radius, min=0.0, max=0.36))
    beta = beta.clamp(0.0, 0.6)
    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - beta * beta, min=1e-6))
    view_hat = F.normalize(camera_position.view(1, 1, 3) - safe_hit_position, dim=-1, eps=1e-6)
    los = (tangential * view_hat).sum(dim=-1).clamp(-1.0, 1.0)
    doppler = 1.0 / (gamma * (1.0 - beta * los))
    beaming = torch.pow(doppler.clamp(min=1e-6), beaming_power).clamp(0.2, 5.0)
    beaming_factor = ((1.0 - beaming_strength) + beaming_strength * beaming).clamp(0.2, 5.0) * disk_mask

    g_grav = torch.sqrt(torch.clamp(1.0 - (2.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / radius), min=1e-6))
    redshift = torch.pow(g_grav, redshift_power).clamp(0.0, 1.0)
    redshift_factor = ((1.0 - redshift_strength) + redshift_strength * redshift).clamp(0.0, 1.0) * disk_mask

    combined_intensity = emissivity * torch.nan_to_num(beaming_factor, nan=0.0) * torch.nan_to_num(redshift_factor, nan=0.0)
    combined_intensity = (combined_intensity / combined_intensity.max().clamp_min(1e-6)).clamp(0.0, 1.0)
    color = warm_disk_colorize(combined_intensity)
    disk_emission = (color * combined_intensity.unsqueeze(-1) * disk_mask.unsqueeze(-1)).clamp(0.0, 1.0)

    inner_opacity = torch.ones_like(radius)
    outer_opacity = torch.ones_like(radius)
    if inner_edge_width > 0.0:
        inner_opacity = smoothstep(disk_inner_radius, disk_inner_radius + inner_edge_width, radius)
    if outer_edge_width > 0.0:
        outer_opacity = 1.0 - smoothstep(disk_outer_radius - outer_edge_width, disk_outer_radius, radius)
    alpha_mask = (disk_alpha * inner_opacity * outer_opacity).clamp(0.0, 1.0) * disk_mask

    return {
        "disk_emission": disk_emission,
        "alpha_mask": alpha_mask,
        "beaming_factor": (beaming_factor / 5.0).clamp(0.0, 1.0),
        "redshift_factor": redshift_factor.clamp(0.0, 1.0),
    }


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

    captured_mask = torch.from_numpy(result.status_map == STATUS_CAPTURED).to(device=device, dtype=torch.float32)
    escaped_mask = torch.from_numpy(result.status_map == STATUS_ESCAPED).to(device=device, dtype=torch.bool)
    disk_mask = torch.from_numpy(result.status_map == STATUS_DISK_HIT).to(device=device, dtype=torch.float32)
    incomplete_mask = torch.from_numpy(result.status_map == STATUS_INCOMPLETE).to(device=device, dtype=torch.bool)

    direct_background = sample_background_from_directions(background, directions.cpu().numpy()).to(device=device)
    blurred_disk_background = blur_image(direct_background, max(1.0, 0.75 * supersample))
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

    hit_radius = torch.from_numpy(result.hit_radius.astype(np.float32)).to(device=device)
    hit_position = torch.from_numpy(result.hit_position.astype(np.float32)).to(device=device)
    disk_maps = build_polished_disk_maps(
        hit_radius=hit_radius,
        hit_position=hit_position,
        disk_mask=disk_mask,
        camera_position=camera_position,
        disk_inner_radius=args.disk_inner_radius,
        disk_outer_radius=args.disk_outer_radius,
        emissivity_power=args.emissivity_power,
        disk_alpha=args.disk_alpha,
        inner_edge_width=args.inner_edge_width,
        outer_edge_width=args.outer_edge_width,
        beaming_strength=args.beaming_strength,
        beaming_power=args.beaming_power,
        rotation_direction=args.rotation_direction,
        redshift_strength=args.redshift_strength,
        redshift_power=args.redshift_power,
    )

    alpha_mask = blur_scalar(disk_maps["alpha_mask"], max((supersample - 1) * 0.5, 0.0))
    alpha_mask = alpha_mask.clamp(0.0, 1.0)
    background_mix = torch.clamp(torch.tensor(args.disk_background_mix, device=device), 0.0, 1.0)
    disk_background_weight = background_mix + (1.0 - background_mix) * (1.0 - alpha_mask)
    disk_background_weight = disk_background_weight.unsqueeze(-1)
    disk_emission = disk_maps["disk_emission"]
    disk_composite = disk_emission * (1.0 - disk_background_weight) + blurred_disk_background * disk_background_weight
    image = torch.where(disk_mask.unsqueeze(-1) > 0.0, disk_composite, image)

    if np.any(result.status_map == STATUS_INCOMPLETE):
        image[incomplete_mask] = torch.tensor([0.18, 0.18, 0.18], dtype=torch.float32, device=device)

    shadow_soft_mask = captured_mask
    if args.shadow_edge_smoothing > 0.0:
        shadow_soft_mask = blur_scalar(captured_mask, args.shadow_edge_smoothing * supersample)
        shadow_soft_mask = shadow_soft_mask.clamp(0.0, 1.0)
    image = image * (1.0 - shadow_soft_mask.unsqueeze(-1))

    raw_internal = image.clamp(0.0, 1.0)
    final_internal = apply_bloom(raw_internal, disk_emission, args.bloom_strength, args.bloom_radius) if args.bloom else raw_internal

    polished_render = downsample_image(final_internal, supersample, width, height).clamp(0.0, 1.0)
    raw_render = downsample_image(raw_internal, supersample, width, height).clamp(0.0, 1.0)
    disk_emission_out = downsample_image(disk_emission, supersample, width, height).clamp(0.0, 1.0)
    alpha_mask_out = downsample_image(alpha_mask.unsqueeze(-1).repeat(1, 1, 3), supersample, width, height).clamp(0.0, 1.0)
    status_out = downsample_image(make_status_rgb(result.status_map).to(device=device), supersample, width, height).clamp(0.0, 1.0)

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "full3d_polished_render.png"
    save_tensor_image(polished_render, output_path)
    save_tensor_image(raw_render, OUTPUT_DIR / "full3d_polished_raw.png")
    save_tensor_image(disk_emission_out, OUTPUT_DIR / "full3d_polished_disk_emission.png")
    save_tensor_image(status_out, OUTPUT_DIR / "full3d_polished_status_map.png")
    save_tensor_image(alpha_mask_out, OUTPUT_DIR / "full3d_polished_alpha_mask.png")
    if args.bloom:
        save_tensor_image(polished_render, OUTPUT_DIR / "full3d_polished_bloom.png")

    n_pixels = internal_width * internal_height
    captured_count = int(result.capture_mask.sum())
    disk_count = int(result.disk_hit_mask.sum())
    escaped_count = int(result.escaped_mask.sum())
    incomplete_count = int(result.incomplete_mask.sum())
    print(
        f"Preset: {args.preset}, resolution={width}, supersample={supersample}"
    )
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
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        panels = [
            (status_out, "Status Map"),
            (disk_emission_out, "Disk Emission"),
            (alpha_mask_out, "Alpha Mask"),
            (raw_render, "Raw Render"),
            (polished_render, "Polished Render"),
        ]
        for axis, (panel, title) in zip(axes.flat, panels):
            axis.imshow(panel.cpu().numpy())
            axis.set_title(title)
            axis.axis("off")
        axes.flat[-1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
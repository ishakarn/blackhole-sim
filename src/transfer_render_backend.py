"""Reusable backend for Schwarzschild transfer-function rendering."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F

from .backgrounds import make_background
from .camera import generate_camera_rays
from .geodesic_3d import (
    STATUS_CAPTURED,
    STATUS_DISK_HIT,
    STATUS_ESCAPED,
    STATUS_INCOMPLETE,
    integrate_ray_bundle_3d,
    sample_background_from_directions,
)
from .photon_transfer import momentum_transfer_g_factor, tangent_transfer_g_factor
from .relativistic_disk import disk_emissivity, normalize_masked, warm_disk_colorize


@dataclass(slots=True)
class CameraParameters:
    azimuth_deg: float = 0.0
    distance: float = 100.0
    height: float = 80.0
    fov: float = 14.0


@dataclass(slots=True)
class RenderParameters:
    resolution: int = 512
    width: int | None = None
    height: int | None = None
    aspect: float = 1.0
    supersample: int = 2
    background: str = "stars"
    seed: int = 42
    disk_inner_radius: float = 6.0
    disk_outer_radius: float = 25.0
    emissivity_mode: str = "thin_disk"
    emissivity_power: float = 3.0
    rotation_direction: str = "prograde"
    intensity_power: int = 3
    transfer_mode: str = "momentum"
    max_steps: int = 7000
    step_size: float = 0.005
    r_escape: float = 150.0
    device: str = "cpu"
    background_scale: float = 4.0
    background_blur_radius: float | None = None
    bloom: bool = False
    bloom_strength: float = 0.15
    bloom_radius: int = 4
    diagnostics: bool = False


@dataclass(slots=True)
class RenderResult:
    image: torch.Tensor
    raw_image: torch.Tensor | None
    render_time_seconds: float
    width: int
    height: int
    internal_width: int
    internal_height: int
    status_map: np.ndarray | None
    counts: dict[str, int]


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


def camera_position_from_orbit(camera: CameraParameters) -> torch.Tensor:
    azimuth = np.deg2rad(camera.azimuth_deg)
    x = camera.distance * np.sin(azimuth)
    y = -camera.distance * np.cos(azimuth)
    z = camera.height
    return torch.tensor([x, y, z], dtype=torch.float32)


def render_black_hole(camera: CameraParameters, params: RenderParameters) -> RenderResult:
    start_time = time.perf_counter()
    device = torch.device(params.device)
    width = params.width if params.width is not None else params.resolution
    height = params.height if params.height is not None else max(1, int(width * params.aspect))
    supersample = max(1, params.supersample)
    internal_width = width * supersample
    internal_height = height * supersample

    with torch.inference_mode():
        background_width = max(internal_width, int(internal_width * params.background_scale))
        background_height = max(internal_height, int(background_width / 2))
        background = make_background(
            params.background,
            width=background_width,
            height=background_height,
            seed=params.seed,
            device=device,
        )

        camera_position = camera_position_from_orbit(camera).to(device=device)
        target = torch.tensor([0.0, 0.0, 0.0], device=device)
        up_hint = torch.tensor([0.0, 0.0, 1.0], device=device)

        directions, _, _, _, camera_position = generate_camera_rays(
            width=internal_width,
            height=internal_height,
            fov=camera.fov,
            camera_position=camera_position,
            target=target,
            up_hint=up_hint,
            device=device,
        )

        ray_directions = directions.cpu().numpy()
        result = integrate_ray_bundle_3d(
            camera_position=camera_position.cpu().numpy(),
            ray_directions=ray_directions,
            disk_inner_radius=params.disk_inner_radius,
            disk_outer_radius=params.disk_outer_radius,
            horizon_radius=2.05,
            r_escape=params.r_escape,
            max_steps=params.max_steps,
            step_size=params.step_size,
        )

        status_map = result.status_map if params.diagnostics else None
        captured_mask = torch.from_numpy(result.status_map == STATUS_CAPTURED).to(device=device, dtype=torch.bool)
        escaped_mask = torch.from_numpy(result.status_map == STATUS_ESCAPED).to(device=device, dtype=torch.bool)
        disk_mask = torch.from_numpy(result.status_map == STATUS_DISK_HIT).to(device=device, dtype=torch.float32)
        incomplete_mask = torch.from_numpy(result.status_map == STATUS_INCOMPLETE).to(device=device, dtype=torch.bool)

        direct_background = sample_background_from_directions(background, ray_directions).to(device=device)
        if params.background_blur_radius is None:
            background_blur_radius = max(1.0, 0.75 * supersample)
        else:
            background_blur_radius = max(0.0, params.background_blur_radius)
        blurred_background = (
            blur_image(direct_background, background_blur_radius)
            if background_blur_radius > 0.0
            else direct_background
        )
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
            disk_inner_radius=params.disk_inner_radius,
            emissivity_mode=params.emissivity_mode,
            emissivity_power=params.emissivity_power,
        )

        radius_for_transfer = hit_radius.clamp(min=params.disk_inner_radius)
        if params.transfer_mode == "momentum":
            g_factor, _ = momentum_transfer_g_factor(
                hit_position,
                hit_direction,
                radius_for_transfer,
                disk_mask,
                rotation_direction=params.rotation_direction,
            )
        else:
            g_factor = tangent_transfer_g_factor(
                hit_position,
                hit_direction,
                radius_for_transfer,
                disk_mask,
                rotation_direction=params.rotation_direction,
            )

        observed_intensity = normalize_masked(emissivity * torch.pow(g_factor, float(params.intensity_power)), disk_mask)
        disk_color = warm_disk_colorize(observed_intensity)
        disk_emission = disk_color * observed_intensity.unsqueeze(-1) * disk_mask.unsqueeze(-1)
        disk_composite = disk_emission + 0.02 * blurred_background * disk_mask.unsqueeze(-1)
        image = torch.where(disk_mask.unsqueeze(-1) > 0.0, disk_composite.clamp(0.0, 1.0), image)

        if torch.any(incomplete_mask):
            image[incomplete_mask] = torch.tensor([0.18, 0.18, 0.18], dtype=torch.float32, device=device)

        raw_internal = image.clamp(0.0, 1.0)
        final_internal = apply_bloom(raw_internal, disk_emission, params.bloom_strength, params.bloom_radius) if params.bloom else raw_internal

        final_image = downsample_image(final_internal, supersample, width, height).clamp(0.0, 1.0)
        raw_image = None
        if params.diagnostics or params.bloom:
            raw_image = downsample_image(raw_internal, supersample, width, height).clamp(0.0, 1.0)

    elapsed = time.perf_counter() - start_time
    counts = {
        "captured": int(result.capture_mask.sum()),
        "disk": int(result.disk_hit_mask.sum()),
        "escaped": int(result.escaped_mask.sum()),
        "incomplete": int(result.incomplete_mask.sum()),
    }
    return RenderResult(
        image=final_image,
        raw_image=raw_image,
        render_time_seconds=elapsed,
        width=width,
        height=height,
        internal_width=internal_width,
        internal_height=internal_height,
        status_map=status_map,
        counts=counts,
    )


__all__ = [
    "CameraParameters",
    "RenderParameters",
    "RenderResult",
    "camera_position_from_orbit",
    "render_black_hole",
]
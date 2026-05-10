"""Approximate accretion disk models for static black hole rendering."""

import math

import torch


def make_disk_coordinates(
    x: torch.Tensor,
    y: torch.Tensor,
    tilt_deg: float = 70.0,
    rotation_deg: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return disk-plane coordinates and elliptical disk radius."""
    rotation = math.radians(rotation_deg)
    tilt = math.radians(tilt_deg)

    cos_rot = math.cos(rotation)
    sin_rot = math.sin(rotation)
    cos_tilt = max(math.cos(tilt), eps)

    x_rot = cos_rot * x + sin_rot * y
    y_rot = -sin_rot * x + cos_rot * y

    x_disk = x_rot
    y_disk = y_rot / cos_tilt
    r_disk = torch.sqrt(x_disk**2 + y_disk**2).clamp(min=eps)
    return x_disk, y_disk, r_disk


def make_disk_mask(
    r_disk: torch.Tensor,
    inner_radius: float,
    outer_radius: float,
    edge_softness: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a hard mask and a soft alpha mask for the disk annulus."""
    hard_mask = (r_disk >= inner_radius) & (r_disk <= outer_radius)
    if edge_softness <= 0.0:
        alpha = hard_mask.float()
    else:
        inner = torch.sigmoid((r_disk - inner_radius) / edge_softness)
        outer = torch.sigmoid((outer_radius - r_disk) / edge_softness)
        alpha = inner * outer
    return hard_mask, alpha.clamp(0.0, 1.0)


def disk_brightness_profile(
    r_disk: torch.Tensor,
    inner_radius: float,
    outer_radius: float,
    power: float = 0.75,
    contrast: float = 1.6,
) -> torch.Tensor:
    """Return normalized disk brightness using a power-law profile."""
    clamped_radius = r_disk.clamp(min=inner_radius, max=outer_radius)
    raw = torch.pow(clamped_radius / inner_radius, -power)
    raw_min = torch.pow(torch.tensor(outer_radius / inner_radius, device=r_disk.device), -power)
    normalized = (raw - raw_min) / max(1.0 - raw_min.item(), 1e-6)
    return torch.pow(normalized.clamp(0.0, 1.0), 1.0 / contrast)


def apply_doppler_beaming(
    intensity: torch.Tensor,
    x_disk: torch.Tensor,
    r_disk: torch.Tensor,
    beaming_strength: float = 0.35,
) -> torch.Tensor:
    """Approximate one-sided brightening for the approaching disk side."""
    approach = x_disk / r_disk
    beaming = 1.0 + beaming_strength * approach
    return (intensity * beaming.clamp(min=0.1)).clamp(0.0, 1.5)


def disk_colorize(intensity: torch.Tensor) -> torch.Tensor:
    """Map normalized intensity to a warm accretion-disk color palette."""
    t = intensity.clamp(0.0, 1.0)
    low = torch.tensor([0.18, 0.02, 0.01], device=t.device, dtype=torch.float32)
    mid = torch.tensor([0.95, 0.25, 0.04], device=t.device, dtype=torch.float32)
    high = torch.tensor([1.0, 0.95, 0.78], device=t.device, dtype=torch.float32)

    mid_mix = torch.clamp(t / 0.6, 0.0, 1.0).unsqueeze(-1)
    high_mix = torch.clamp((t - 0.55) / 0.45, 0.0, 1.0).unsqueeze(-1)

    color = low + mid_mix * (mid - low)
    color = color + high_mix * (high - color)
    return color.clamp(0.0, 1.0)


def render_disk_image(
    x: torch.Tensor,
    y: torch.Tensor,
    inner_radius: float = 6.0,
    outer_radius: float = 12.0,
    tilt_deg: float = 70.0,
    rotation_deg: float = 0.0,
    beaming_strength: float = 0.35,
    edge_softness: float = 0.15,
    brightness_power: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return disk RGB image, alpha mask, and hard mask in image space."""
    x_disk, _, r_disk = make_disk_coordinates(
        x,
        y,
        tilt_deg=tilt_deg,
        rotation_deg=rotation_deg,
    )
    hard_mask, alpha = make_disk_mask(
        r_disk,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        edge_softness=edge_softness,
    )
    brightness = disk_brightness_profile(
        r_disk,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        power=brightness_power,
    )
    brightness = apply_doppler_beaming(
        brightness,
        x_disk=x_disk,
        r_disk=r_disk,
        beaming_strength=beaming_strength,
    )
    disk_rgb = disk_colorize(brightness) * alpha.unsqueeze(-1)
    return disk_rgb.clamp(0.0, 1.0), alpha, hard_mask.float()
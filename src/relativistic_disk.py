"""Relativistic thin-disk shading helpers for Schwarzschild disk-hit renders."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import M


def warm_disk_colorize(intensity: torch.Tensor) -> torch.Tensor:
    """Map normalized intensity to a warm accretion-disk palette."""
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


def normalize_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    max_value = masked.max().clamp_min(1e-6)
    return (masked / max_value).clamp(0.0, 1.0)


def disk_emissivity(
    radius: torch.Tensor,
    disk_mask: torch.Tensor,
    disk_inner_radius: float,
    emissivity_mode: str = "thin_disk",
    emissivity_power: float = 3.0,
) -> torch.Tensor:
    """Return emitted disk intensity before relativistic transfer effects."""
    safe_radius = torch.where(
        disk_mask > 0.0,
        torch.nan_to_num(radius, nan=disk_inner_radius),
        torch.full_like(radius, disk_inner_radius),
    ).clamp(min=disk_inner_radius + 1e-6)

    if emissivity_mode == "thin_disk":
        emissivity = torch.pow(safe_radius, -3.0) * (1.0 - torch.sqrt(disk_inner_radius / safe_radius))
        emissivity = emissivity.clamp(min=0.0)
    else:
        emissivity = torch.pow(safe_radius / disk_inner_radius, -emissivity_power)

    return emissivity * disk_mask


def emitter_four_velocity(radius: torch.Tensor, rotation_direction: str = "prograde") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Schwarzschild circular-orbit u^t, u^phi, and Omega for the disk emitter."""
    safe_radius = radius.clamp(min=3.0 + 1e-6)
    omega = torch.sqrt(torch.tensor(M, device=radius.device, dtype=radius.dtype) / torch.pow(safe_radius, 3.0))
    u_t = 1.0 / torch.sqrt(torch.clamp(1.0 - (3.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / safe_radius), min=1e-6))
    sign = 1.0 if rotation_direction == "prograde" else -1.0
    u_phi = sign * omega * u_t
    return u_t, u_phi, omega


def transfer_g_factor(
    hit_position: torch.Tensor,
    hit_direction: torch.Tensor,
    radius: torch.Tensor,
    disk_mask: torch.Tensor,
    rotation_direction: str = "prograde",
    clamp_min: float = 0.1,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """Compute a Schwarzschild emitter-to-observer frequency shift factor g.

    The ray marcher traces from the observer toward the disk, so the physical
    photon direction at the emitter is the negative of the stored hit tangent.
    We use a consistent proportional 4-momentum form at the disk hit; only the
    ratio matters for g.
    """
    safe_position = torch.where(
        disk_mask.unsqueeze(-1) > 0.0,
        torch.nan_to_num(hit_position, nan=0.0),
        torch.zeros_like(hit_position),
    )
    emission_direction = -F.normalize(torch.nan_to_num(hit_direction, nan=0.0), dim=-1, eps=1e-6)

    x = safe_position[..., 0]
    y = safe_position[..., 1]
    e_phi = torch.stack([-y, x, torch.zeros_like(x)], dim=-1)
    e_phi = F.normalize(e_phi, dim=-1, eps=1e-6)
    n_phi = (emission_direction * e_phi).sum(dim=-1).clamp(-1.0, 1.0)

    safe_radius = radius.clamp(min=3.0 + 1e-6)
    alpha = torch.sqrt(torch.clamp(1.0 - (2.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / safe_radius), min=1e-6))
    u_t, u_phi, _ = emitter_four_velocity(safe_radius, rotation_direction=rotation_direction)

    k_t = -alpha
    k_phi = safe_radius * n_phi

    numerator = -k_t
    denominator = -(k_t * u_t + k_phi * u_phi)
    g_factor = numerator / denominator.clamp(min=1e-6)
    g_factor = torch.nan_to_num(g_factor, nan=clamp_min, posinf=clamp_max, neginf=clamp_min)
    return g_factor.clamp(clamp_min, clamp_max) * disk_mask


def heuristic_g_proxy(
    hit_position: torch.Tensor,
    hit_direction: torch.Tensor,
    radius: torch.Tensor,
    disk_mask: torch.Tensor,
    rotation_direction: str = "prograde",
    clamp_min: float = 0.1,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """Return the older heuristic gravitational/Doppler proxy as a comparison path."""
    safe_position = torch.where(
        disk_mask.unsqueeze(-1) > 0.0,
        torch.nan_to_num(hit_position, nan=0.0),
        torch.zeros_like(hit_position),
    )
    emission_direction = -F.normalize(torch.nan_to_num(hit_direction, nan=0.0), dim=-1, eps=1e-6)

    x = safe_position[..., 0]
    y = safe_position[..., 1]
    e_phi = torch.stack([-y, x, torch.zeros_like(x)], dim=-1)
    if rotation_direction == "retrograde":
        e_phi = -e_phi
    e_phi = F.normalize(e_phi, dim=-1, eps=1e-6)
    los = (emission_direction * e_phi).sum(dim=-1).clamp(-1.0, 1.0)

    safe_radius = radius.clamp(min=3.0 + 1e-6)
    beta = torch.sqrt(torch.clamp(torch.tensor(M, device=radius.device, dtype=radius.dtype) / safe_radius, min=0.0, max=0.36)).clamp(0.0, 0.6)
    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - beta * beta, min=1e-6))
    doppler = 1.0 / (gamma * (1.0 - beta * los))
    g_grav = torch.sqrt(torch.clamp(1.0 - (2.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / safe_radius), min=1e-6))
    g_factor = doppler * g_grav
    g_factor = torch.nan_to_num(g_factor, nan=clamp_min, posinf=clamp_max, neginf=clamp_min)
    return g_factor.clamp(clamp_min, clamp_max) * disk_mask


def scalar_to_rgb(values: torch.Tensor, mask: torch.Tensor, min_value: float | None = None, max_value: float | None = None) -> torch.Tensor:
    if min_value is None:
        min_value = float(values[mask > 0.0].min().item()) if torch.any(mask > 0.0) else 0.0
    if max_value is None:
        max_value = float(values[mask > 0.0].max().item()) if torch.any(mask > 0.0) else 1.0
    denom = max(max_value - min_value, 1e-6)
    normalized = ((values - min_value) / denom).clamp(0.0, 1.0)
    return normalized.unsqueeze(-1).repeat(1, 1, 3) * mask.unsqueeze(-1)


__all__ = [
    "disk_emissivity",
    "emitter_four_velocity",
    "heuristic_g_proxy",
    "normalize_masked",
    "scalar_to_rgb",
    "transfer_g_factor",
    "warm_disk_colorize",
]
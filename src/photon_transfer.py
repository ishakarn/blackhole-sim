"""Photon momentum reconstruction helpers for Schwarzschild transfer rendering."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .constants import M
from .relativistic_disk import emitter_four_velocity


def _safe_hit_tensors(
    hit_position: torch.Tensor,
    hit_direction: torch.Tensor,
    disk_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_position = torch.where(
        disk_mask.unsqueeze(-1) > 0.0,
        torch.nan_to_num(hit_position, nan=0.0),
        torch.zeros_like(hit_position),
    )
    safe_direction = torch.where(
        disk_mask.unsqueeze(-1) > 0.0,
        torch.nan_to_num(hit_direction, nan=0.0),
        torch.zeros_like(hit_direction),
    )
    emission_direction = -F.normalize(safe_direction, dim=-1, eps=1e-6)
    return safe_position, emission_direction


def spherical_coordinates_from_cartesian(position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = position[..., 0]
    y = position[..., 1]
    z = position[..., 2]
    radius = torch.sqrt(torch.clamp(x * x + y * y + z * z, min=1e-12))
    theta = torch.arccos((z / radius).clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)
    return radius, theta, phi


def spherical_basis(theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    e_r = torch.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dim=-1)
    e_theta = torch.stack([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], dim=-1)
    e_phi = torch.stack([-sin_phi, cos_phi, torch.zeros_like(phi)], dim=-1)
    return e_r, e_theta, e_phi


def photon_momentum_from_direction(
    hit_position: torch.Tensor,
    hit_direction: torch.Tensor,
    disk_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Convert local Cartesian hit directions into Schwarzschild-coordinate photon momentum."""
    safe_position, emission_direction = _safe_hit_tensors(hit_position, hit_direction, disk_mask)
    radius, theta, phi = spherical_coordinates_from_cartesian(safe_position)
    radius = radius.clamp(min=2.0 + 1e-6)

    e_r, e_theta, e_phi = spherical_basis(theta, phi)
    sin_theta = torch.sin(theta).clamp(min=1e-6)

    k_r = (emission_direction * e_r).sum(dim=-1)
    k_theta = (emission_direction * e_theta).sum(dim=-1) / radius
    k_phi = (emission_direction * e_phi).sum(dim=-1) / (radius * sin_theta)

    lapse = torch.clamp(1.0 - (2.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / radius), min=1e-6)
    spatial_term = (
        (k_r * k_r) / lapse
        + radius * radius * k_theta * k_theta
        + radius * radius * sin_theta * sin_theta * k_phi * k_phi
    )
    k_t_contravariant = torch.sqrt(spatial_term / lapse)

    k_t_covariant = -lapse * k_t_contravariant
    k_phi_covariant = radius * radius * sin_theta * sin_theta * k_phi

    return {
        "radius": radius,
        "theta": theta,
        "phi": phi,
        "k_t": k_t_contravariant * disk_mask,
        "k_r": k_r * disk_mask,
        "k_theta": k_theta * disk_mask,
        "k_phi": k_phi * disk_mask,
        "k_t_covariant": k_t_covariant * disk_mask,
        "k_phi_covariant": k_phi_covariant * disk_mask,
    }


def tangent_transfer_g_factor(
    hit_position: torch.Tensor,
    hit_direction: torch.Tensor,
    radius: torch.Tensor,
    disk_mask: torch.Tensor,
    rotation_direction: str = "prograde",
    clamp_min: float = 0.1,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """Old v1.6 tangent-based transfer factor for comparison."""
    safe_position, emission_direction = _safe_hit_tensors(hit_position, hit_direction, disk_mask)

    x = safe_position[..., 0]
    y = safe_position[..., 1]
    e_phi = torch.stack([-y, x, torch.zeros_like(x)], dim=-1)
    e_phi = F.normalize(e_phi, dim=-1, eps=1e-6)
    n_phi = (emission_direction * e_phi).sum(dim=-1).clamp(-1.0, 1.0)

    safe_radius = radius.clamp(min=3.0 + 1e-6)
    lapse = torch.sqrt(torch.clamp(1.0 - (2.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / safe_radius), min=1e-6))
    u_t, u_phi, _ = emitter_four_velocity(safe_radius, rotation_direction=rotation_direction)

    k_t_covariant = -lapse
    k_phi_covariant = safe_radius * n_phi

    numerator = -k_t_covariant
    denominator = -(k_t_covariant * u_t + k_phi_covariant * u_phi)
    g_factor = numerator / denominator.clamp(min=1e-6)
    g_factor = torch.nan_to_num(g_factor, nan=clamp_min, posinf=clamp_max, neginf=clamp_min)
    return g_factor.clamp(clamp_min, clamp_max) * disk_mask


def momentum_transfer_g_factor(
    hit_position: torch.Tensor,
    hit_direction: torch.Tensor,
    radius: torch.Tensor,
    disk_mask: torch.Tensor,
    rotation_direction: str = "prograde",
    clamp_min: float = 0.1,
    clamp_max: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute g using reconstructed Schwarzschild-coordinate photon momentum."""
    momentum = photon_momentum_from_direction(hit_position, hit_direction, disk_mask)
    safe_radius = radius.clamp(min=3.0 + 1e-6)
    u_t, u_phi, _ = emitter_four_velocity(safe_radius, rotation_direction=rotation_direction)

    numerator = -momentum["k_t_covariant"]
    denominator = -(momentum["k_t_covariant"] * u_t + momentum["k_phi_covariant"] * u_phi)
    g_factor = numerator / denominator.clamp(min=1e-6)
    g_factor = torch.nan_to_num(g_factor, nan=clamp_min, posinf=clamp_max, neginf=clamp_min)
    g_factor = g_factor.clamp(clamp_min, clamp_max) * disk_mask
    return g_factor, momentum


def transfer_difference_stats(g_tangent: torch.Tensor, g_momentum: torch.Tensor, disk_mask: torch.Tensor) -> dict[str, float]:
    masked = disk_mask > 0.0
    if not torch.any(masked):
        return {
            "tangent_min": 0.0,
            "tangent_max": 0.0,
            "tangent_mean": 0.0,
            "momentum_min": 0.0,
            "momentum_max": 0.0,
            "momentum_mean": 0.0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
        }

    tangent = g_tangent[masked]
    momentum = g_momentum[masked]
    diff = (momentum - tangent).abs()
    return {
        "tangent_min": float(tangent.min()),
        "tangent_max": float(tangent.max()),
        "tangent_mean": float(tangent.mean()),
        "momentum_min": float(momentum.min()),
        "momentum_max": float(momentum.max()),
        "momentum_mean": float(momentum.mean()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
    }


__all__ = [
    "momentum_transfer_g_factor",
    "photon_momentum_from_direction",
    "schwarzschild_null_residual",
    "spherical_basis",
    "spherical_coordinates_from_cartesian",
    "static_tetrad_momentum",
    "tangent_transfer_g_factor",
    "tetrad_null_residual",
    "transfer_difference_stats",
]


def schwarzschild_null_residual(momentum: dict[str, torch.Tensor], disk_mask: torch.Tensor) -> torch.Tensor:
    """Return g_{mu nu} k^mu k^nu for reconstructed coordinate momentum."""
    radius = momentum["radius"].clamp(min=2.0 + 1e-6)
    theta = momentum["theta"]
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    lapse = torch.clamp(1.0 - (2.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / radius), min=1e-6)

    residual = (
        -lapse * momentum["k_t"] * momentum["k_t"]
        + (momentum["k_r"] * momentum["k_r"]) / lapse
        + radius * radius * momentum["k_theta"] * momentum["k_theta"]
        + radius * radius * sin_theta * sin_theta * momentum["k_phi"] * momentum["k_phi"]
    )
    return torch.nan_to_num(residual, nan=0.0) * disk_mask


def static_tetrad_momentum(momentum: dict[str, torch.Tensor], disk_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert coordinate momentum to a local static orthonormal frame."""
    radius = momentum["radius"].clamp(min=2.0 + 1e-6)
    theta = momentum["theta"]
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    lapse = torch.clamp(1.0 - (2.0 * torch.tensor(M, device=radius.device, dtype=radius.dtype) / radius), min=1e-6)
    sqrt_lapse = torch.sqrt(lapse)

    k_hat_t = sqrt_lapse * momentum["k_t"]
    k_hat_r = momentum["k_r"] / sqrt_lapse
    k_hat_theta = radius * momentum["k_theta"]
    k_hat_phi = radius * sin_theta * momentum["k_phi"]

    return {
        "k_hat_t": torch.nan_to_num(k_hat_t, nan=0.0) * disk_mask,
        "k_hat_r": torch.nan_to_num(k_hat_r, nan=0.0) * disk_mask,
        "k_hat_theta": torch.nan_to_num(k_hat_theta, nan=0.0) * disk_mask,
        "k_hat_phi": torch.nan_to_num(k_hat_phi, nan=0.0) * disk_mask,
    }


def tetrad_null_residual(momentum: dict[str, torch.Tensor], disk_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return local-frame null residual and the local tetrad momentum components."""
    tetrad = static_tetrad_momentum(momentum, disk_mask)
    residual = (
        -tetrad["k_hat_t"] * tetrad["k_hat_t"]
        + tetrad["k_hat_r"] * tetrad["k_hat_r"]
        + tetrad["k_hat_theta"] * tetrad["k_hat_theta"]
        + tetrad["k_hat_phi"] * tetrad["k_hat_phi"]
    )
    return torch.nan_to_num(residual, nan=0.0) * disk_mask, tetrad
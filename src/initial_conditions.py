"""Initial particle distributions for the 2D simulator."""

from __future__ import annotations

import math

import torch

from .constants import G, M


def disk_particles(
    num_particles: int,
    radius_min: float = 4.0,
    radius_max: float = 18.0,
    velocity_multiplier_mean: float = 0.98,
    velocity_multiplier_std: float = 0.10,
    radial_velocity_std: float = 0.015,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create particles in a disk with approximately circular velocities."""

    if radius_min <= 0 or radius_max <= radius_min:
        raise ValueError("radius_min must be positive and smaller than radius_max.")

    device = torch.device(device)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    radii = torch.empty(num_particles, dtype=dtype, device=device).uniform_(
        radius_min, radius_max, generator=generator
    )
    theta = torch.empty(num_particles, dtype=dtype, device=device).uniform_(
        0.0, 2.0 * math.pi, generator=generator
    )

    positions = torch.stack((radii * torch.cos(theta), radii * torch.sin(theta)), dim=1)

    radial_hat = positions / radii.unsqueeze(1)
    tangential_hat = torch.stack((-radial_hat[:, 1], radial_hat[:, 0]), dim=1)

    circular_speed = torch.sqrt(torch.as_tensor(G * M, dtype=dtype, device=device) / radii)
    multiplier = torch.normal(
        mean=velocity_multiplier_mean,
        std=velocity_multiplier_std,
        size=(num_particles,),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    radial_velocity = torch.normal(
        mean=0.0,
        std=radial_velocity_std,
        size=(num_particles,),
        generator=generator,
        dtype=dtype,
        device=device,
    )

    velocities = (
        tangential_hat * (circular_speed * multiplier).unsqueeze(1)
        + radial_hat * radial_velocity.unsqueeze(1)
    )

    return positions, velocities


def accretion_disk_particles(
    num_particles: int,
    radius_min: float = 6.0,
    radius_max: float = 40.0,
    velocity_multiplier: float = 0.98,
    velocity_noise: float = 0.06,
    radial_noise: float = 0.01,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create an accretion-disk-like particle distribution mostly outside the ISCO."""

    return disk_particles(
        num_particles=num_particles,
        radius_min=radius_min,
        radius_max=radius_max,
        velocity_multiplier_mean=velocity_multiplier,
        velocity_multiplier_std=velocity_noise,
        radial_velocity_std=radial_noise,
        device=device,
        dtype=dtype,
        seed=seed,
    )

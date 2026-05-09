"""Time integration and force calculations."""

from __future__ import annotations

import torch

from .constants import EVENT_HORIZON_RADIUS, G, M


def newtonian_acceleration(
    positions: torch.Tensor,
    active: torch.Tensor | None = None,
    softening: float = 0.03,
) -> torch.Tensor:
    """Compute central Newtonian acceleration toward the origin."""

    radius_squared = (positions * positions).sum(dim=1).clamp_min(softening * softening)
    inv_radius_cubed = torch.rsqrt(radius_squared) / radius_squared
    accelerations = -(G * M) * positions * inv_radius_cubed.unsqueeze(1)

    if active is not None:
        accelerations = torch.where(active.unsqueeze(1), accelerations, torch.zeros_like(accelerations))
    return accelerations


def velocity_verlet_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    active: torch.Tensor,
    dt: float,
    horizon_radius: float = EVENT_HORIZON_RADIUS,
    softening: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance active particles by one Velocity Verlet step."""

    acceleration = newtonian_acceleration(positions, active=active, softening=softening)
    new_positions = positions + velocities * dt + 0.5 * acceleration * (dt * dt)

    radii = torch.linalg.norm(new_positions, dim=1)
    still_active = active & (radii > horizon_radius)

    new_acceleration = newtonian_acceleration(new_positions, active=still_active, softening=softening)
    new_velocities = velocities + 0.5 * (acceleration + new_acceleration) * dt

    new_positions = torch.where(active.unsqueeze(1), new_positions, positions)
    new_velocities = torch.where(still_active.unsqueeze(1), new_velocities, torch.zeros_like(new_velocities))

    return new_positions, new_velocities, still_active

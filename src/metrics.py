"""Metrics and outcome classification for particle experiments."""

from __future__ import annotations

import torch


def measure_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    active: torch.Tensor,
    num_particles: int,
) -> dict[str, torch.Tensor]:
    """Measure one simulation snapshot on the current device."""

    radii = torch.linalg.norm(positions, dim=1)
    speeds = torch.linalg.norm(velocities, dim=1)
    active_count = active.sum()

    if active_count.item() > 0:
        mean_radius = radii[active].mean()
        mean_speed = speeds[active].mean()
    else:
        mean_radius = torch.full((), float("nan"), device=positions.device)
        mean_speed = torch.full((), float("nan"), device=positions.device)

    active_count_float = active_count.to(torch.float32)
    return {
        "active_count": active_count_float,
        "swallowed_fraction": 1.0 - active_count_float / float(num_particles),
        "mean_radius": mean_radius,
        "mean_speed": mean_speed,
    }


def classify_outcomes(
    positions: torch.Tensor,
    active: torch.Tensor,
    escape_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return swallowed, escaped, and orbiting masks on the input device."""

    radii = torch.linalg.norm(positions, dim=1)
    swallowed = ~active
    escaped = active & (radii > escape_radius)
    orbiting = active & ~escaped
    return swallowed, escaped, orbiting


def summarize_outcomes(
    swallowed: torch.Tensor,
    escaped: torch.Tensor,
    orbiting: torch.Tensor,
) -> tuple[dict[str, int], dict[str, float]]:
    """Create CPU scalar outcome counts and fractions from masks."""

    total = swallowed.numel()
    counts = {
        "swallowed": int(swallowed.sum().item()),
        "escaped": int(escaped.sum().item()),
        "orbiting": int(orbiting.sum().item()),
    }
    fractions = {name: count / total for name, count in counts.items()}
    return counts, fractions

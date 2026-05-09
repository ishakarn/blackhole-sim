"""Metrics and outcome classification for particle experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from .constants import EVENT_HORIZON_RADIUS, ISCO_RADIUS, PHOTON_SPHERE_RADIUS


REGION_METRIC_NAMES = (
    "swallowed_fraction",
    "plunge_fraction",
    "unstable_fraction",
    "stable_fraction",
    "escaped_fraction",
)


def classify_regions(
    positions: torch.Tensor,
    active: torch.Tensor,
    escape_radius: float,
    enable_escape: bool = True,
) -> dict[str, torch.Tensor]:
    """Classify particles into pseudo-relativistic radial regions."""

    radii = torch.linalg.norm(positions, dim=1)
    swallowed = (~active) | (radii < EVENT_HORIZON_RADIUS)
    escaped = active & (radii > escape_radius) if enable_escape else torch.zeros_like(active)
    bound_active = active & ~escaped

    plunge = bound_active & (radii >= EVENT_HORIZON_RADIUS) & (radii < PHOTON_SPHERE_RADIUS)
    unstable = bound_active & (radii >= PHOTON_SPHERE_RADIUS) & (radii < ISCO_RADIUS)
    stable = bound_active & (radii >= ISCO_RADIUS)

    return {
        "swallowed": swallowed,
        "plunge_region": plunge,
        "unstable_orbit_region": unstable,
        "stable_orbit_region": stable,
        "escaped": escaped,
    }


def measure_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    active: torch.Tensor,
    num_particles: int,
    escape_radius: float,
    enable_escape: bool = True,
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
    regions = classify_regions(
        positions,
        active,
        escape_radius=escape_radius,
        enable_escape=enable_escape,
    )
    total = float(num_particles)

    return {
        "active_count": active_count_float,
        "swallowed_fraction": regions["swallowed"].sum().to(torch.float32) / total,
        "plunge_fraction": regions["plunge_region"].sum().to(torch.float32) / total,
        "unstable_fraction": regions["unstable_orbit_region"].sum().to(torch.float32) / total,
        "stable_fraction": regions["stable_orbit_region"].sum().to(torch.float32) / total,
        "escaped_fraction": regions["escaped"].sum().to(torch.float32) / total,
        "mean_radius": mean_radius,
        "mean_speed": mean_speed,
        "mean_radius_active": mean_radius,
        "mean_speed_active": mean_speed,
    }


def classify_outcomes(
    positions: torch.Tensor,
    active: torch.Tensor,
    escape_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return swallowed, escaped, and orbiting masks on the input device."""

    regions = classify_regions(positions, active, escape_radius=escape_radius)
    swallowed = regions["swallowed"]
    escaped = regions["escaped"]
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


def save_metrics_csv(
    times: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    output_path: str | Path,
) -> Path:
    """Export metric time series to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = [name for name, values in metrics.items() if values.numel() > 0]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["time", *metric_names])
        writer.writeheader()
        for index, time_value in enumerate(times.tolist()):
            row = {"time": time_value}
            for name in metric_names:
                row[name] = metrics[name][index].item()
            writer.writerow(row)

    return output_path

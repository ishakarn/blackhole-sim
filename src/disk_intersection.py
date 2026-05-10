"""Geodesic-based accretion disk intersection helpers."""

from __future__ import annotations

import numpy as np
import torch

from .constants import B_CRIT
from .disk_models import disk_colorize
from .geodesic_renderer import make_deflection_lookup_b_grid
from .geodesics import integrate_many_null_geodesics


def find_first_disk_intersection_radius(
    radius_values: np.ndarray,
    inner_radius: float,
    outer_radius: float,
) -> float | None:
    """Return the first annulus crossing on the outgoing branch after periapsis."""
    finite = np.isfinite(radius_values)
    if not np.any(finite):
        return None

    valid_r = radius_values[finite]
    periapsis_index = int(np.argmin(valid_r))
    outgoing_r = valid_r[periapsis_index:]
    in_annulus = (outgoing_r >= inner_radius) & (outgoing_r <= outer_radius)
    if not np.any(in_annulus):
        return None
    return float(outgoing_r[np.argmax(in_annulus)])


def build_disk_intersection_lookup_table(
    b_max: float,
    inner_radius: float,
    outer_radius: float,
    phi_max: float = 18.0,
    num_points: int = 12000,
    n_near: int = 320,
    n_far: int = 320,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lookup tables for deflection-valid b and first disk-hit radii."""
    b_grid = make_deflection_lookup_b_grid(
        b_max=b_max,
        n_near=n_near,
        n_far=n_far,
    )
    trajectories = integrate_many_null_geodesics(
        b_grid.tolist(),
        phi_max=phi_max,
        num_points=num_points,
    )

    hit_radius_grid = np.full_like(b_grid, np.nan, dtype=float)
    escaped_mask = np.zeros_like(b_grid, dtype=bool)
    for index, trajectory in enumerate(trajectories):
        if trajectory.status != "escaped":
            continue
        escaped_mask[index] = True
        hit_radius = find_first_disk_intersection_radius(
            trajectory.r,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )
        if hit_radius is not None:
            hit_radius_grid[index] = hit_radius

    return b_grid, hit_radius_grid, escaped_mask


def interpolate_disk_hit_radii(
    b_values: np.ndarray,
    b_grid: np.ndarray,
    hit_radius_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate disk-hit radii over b, preserving no-hit regions as NaN."""
    valid = np.isfinite(hit_radius_grid)
    if not np.any(valid):
        return np.full_like(b_values, np.nan, dtype=float)

    valid_b = b_grid[valid]
    valid_r = hit_radius_grid[valid]
    interpolated = np.interp(b_values, valid_b, valid_r, left=np.nan, right=np.nan)
    below = b_values < valid_b.min()
    above = b_values > valid_b.max()
    interpolated[below | above] = np.nan
    return interpolated


def render_disk_emission_from_radii(
    hit_radii: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    power: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert disk-hit radii into RGB emission and a visibility mask."""
    hit_tensor = torch.from_numpy(hit_radii.astype(np.float32))
    visible = torch.isfinite(hit_tensor)

    clamped = hit_tensor.clone()
    clamped[~visible] = outer_radius
    raw = torch.pow((clamped / inner_radius).clamp(min=1.0), -power)
    raw_min = float((outer_radius / inner_radius) ** (-power))
    normalized = (raw - raw_min) / max(1.0 - raw_min, 1e-6)
    normalized = normalized.clamp(0.0, 1.0)
    colors = disk_colorize(normalized)
    colors[~visible] = 0.0
    return colors.clamp(0.0, 1.0), visible.float()


__all__ = [
    "build_disk_intersection_lookup_table",
    "find_first_disk_intersection_radius",
    "interpolate_disk_hit_radii",
    "render_disk_emission_from_radii",
]
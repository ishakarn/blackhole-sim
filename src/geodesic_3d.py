"""Vectorized 3D Schwarzschild null-ray marching using planar geodesic lifting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .camera import normalize
from .constants import EVENT_HORIZON_RADIUS


STATUS_CAPTURED = 0
STATUS_DISK_HIT = 1
STATUS_ESCAPED = 2
STATUS_INCOMPLETE = 3


@dataclass(slots=True)
class RayBundleResult:
    """Per-pixel results for the full 3D curved-ray renderer."""

    status: np.ndarray
    hit_radius: np.ndarray
    hit_position: np.ndarray
    hit_direction: np.ndarray
    escape_direction: np.ndarray
    capture_mask: np.ndarray
    disk_hit_mask: np.ndarray
    escaped_mask: np.ndarray
    incomplete_mask: np.ndarray
    status_map: np.ndarray


def _normalize_np(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return vectors / norms


def _rk4_step(u: np.ndarray, up: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance the null-orbit equation by one RK4 step for many rays."""
    k1_u = up
    k1_up = 3.0 * u * u - u

    u2 = u + 0.5 * step * k1_u
    up2 = up + 0.5 * step * k1_up
    k2_u = up2
    k2_up = 3.0 * u2 * u2 - u2

    u3 = u + 0.5 * step * k2_u
    up3 = up + 0.5 * step * k2_up
    k3_u = up3
    k3_up = 3.0 * u3 * u3 - u3

    u4 = u + step * k3_u
    up4 = up + step * k3_up
    k4_u = up4
    k4_up = 3.0 * u4 * u4 - u4

    u_next = u + (step / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    up_next = up + (step / 6.0) * (k1_up + 2.0 * k2_up + 2.0 * k3_up + k4_up)
    return u_next, up_next


def sample_background_from_directions(background: torch.Tensor, directions: np.ndarray) -> torch.Tensor:
    """Sample a procedural background as an equirectangular environment map."""
    direction_tensor = torch.from_numpy(directions.astype(np.float32))
    dx = direction_tensor[..., 0]
    dy = direction_tensor[..., 1]
    dz = direction_tensor[..., 2]

    lon = torch.atan2(dx, dy)
    lat = torch.asin(dz.clamp(-1.0, 1.0))
    u = lon / math.pi
    v = -(2.0 * lat / math.pi)

    grid = torch.stack([u, v], dim=-1).unsqueeze(0)
    bg_t = background.permute(2, 0, 1).unsqueeze(0).float()
    sampled = torch.nn.functional.grid_sample(
        bg_t,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0).permute(1, 2, 0)


def integrate_ray_bundle_3d(
    camera_position: np.ndarray,
    ray_directions: np.ndarray,
    disk_inner_radius: float = 6.0,
    disk_outer_radius: float = 20.0,
    horizon_radius: float = 2.05,
    r_escape: float = 80.0,
    max_steps: int = 3000,
    step_size: float = 0.01,
) -> RayBundleResult:
    """Integrate a bundle of 3D Schwarzschild null rays.

    Each ray is marched in its orbital plane using the validated orbit equation
    and reconstructed back into 3D to test actual disk-plane crossings.
    """
    height, width, _ = ray_directions.shape
    n_rays = height * width
    dirs = ray_directions.reshape(n_rays, 3).astype(np.float64)
    camera = camera_position.astype(np.float64)

    r0 = float(np.linalg.norm(camera))
    er0 = camera / r0
    k_r = dirs @ er0
    tangential = dirs - k_r[:, None] * er0[None, :]
    k_t = np.linalg.norm(tangential, axis=1)

    ephi = np.zeros_like(tangential)
    valid_t = k_t > 1e-8
    ephi[valid_t] = tangential[valid_t] / k_t[valid_t, None]

    # Fallback for near-radial rays; they will usually be captured quickly.
    if np.any(~valid_t):
        fallback = np.cross(np.repeat(er0[None, :], (~valid_t).sum(), axis=0), np.array([[0.0, 0.0, 1.0]]))
        fallback = _normalize_np(fallback)
        ephi[~valid_t] = fallback
        k_t[~valid_t] = 1e-8

    u = np.full(n_rays, 1.0 / r0, dtype=np.float64)
    up = -(k_r) / (r0 * k_t)
    phi = np.zeros(n_rays, dtype=np.float64)

    previous_position = np.repeat(camera[None, :], n_rays, axis=0)
    previous_radius = np.full(n_rays, r0, dtype=np.float64)
    status = np.full(n_rays, STATUS_INCOMPLETE, dtype=np.int8)
    hit_radius = np.full(n_rays, np.nan, dtype=np.float64)
    hit_position = np.full((n_rays, 3), np.nan, dtype=np.float64)
    hit_direction = np.full((n_rays, 3), np.nan, dtype=np.float64)
    escape_direction = np.full((n_rays, 3), np.nan, dtype=np.float64)
    active = np.ones(n_rays, dtype=bool)

    cos_phi = np.ones(n_rays, dtype=np.float64)
    sin_phi = np.zeros(n_rays, dtype=np.float64)

    for _ in range(max_steps):
        active_idx = np.nonzero(active)[0]
        if active_idx.size == 0:
            break

        u_next, up_next = _rk4_step(u[active_idx], up[active_idx], step_size)
        phi_next = phi[active_idx] + step_size

        positive = u_next > 1e-8
        r_next = np.full_like(u_next, r_escape + 1.0)
        r_next[positive] = 1.0 / u_next[positive]

        cos_next = np.cos(phi_next)
        sin_next = np.sin(phi_next)
        position_next = (
            r_next[:, None]
            * (cos_next[:, None] * er0[None, :] + sin_next[:, None] * ephi[active_idx])
        )

        prev_pos = previous_position[active_idx]
        prev_z = prev_pos[:, 2]
        curr_z = position_next[:, 2]
        plane_cross = (prev_z * curr_z <= 0.0) & (np.abs(prev_z - curr_z) > 1e-8)
        if np.any(plane_cross):
            cross_idx_local = np.nonzero(plane_cross)[0]
            t_cross = prev_z[cross_idx_local] / (prev_z[cross_idx_local] - curr_z[cross_idx_local])
            hit_points = prev_pos[cross_idx_local] + t_cross[:, None] * (position_next[cross_idx_local] - prev_pos[cross_idx_local])
            hit_r = np.sqrt(hit_points[:, 0] ** 2 + hit_points[:, 1] ** 2)
            hit_norm = np.linalg.norm(hit_points, axis=1)
            prev_r_cross = previous_radius[active_idx[cross_idx_local]]
            curr_r_cross = r_next[cross_idx_local]

            # Reject trivial plane crossings that happen before the ray has
            # meaningfully moved inward from the camera.
            approached = np.minimum(prev_r_cross, curr_r_cross) < (0.98 * r0)
            valid_hit = (
                (hit_r >= disk_inner_radius)
                & (hit_r <= disk_outer_radius)
                & (hit_norm > horizon_radius)
                & approached
            )
            if np.any(valid_hit):
                hit_global = active_idx[cross_idx_local[valid_hit]]
                status[hit_global] = STATUS_DISK_HIT
                hit_radius[hit_global] = hit_r[valid_hit]
                hit_position[hit_global] = hit_points[valid_hit]
                hit_direction[hit_global] = _normalize_np(
                    position_next[cross_idx_local[valid_hit]] - prev_pos[cross_idx_local[valid_hit]]
                )
                active[hit_global] = False

        still_active_idx = np.nonzero(active[active_idx])[0]
        if still_active_idx.size == 0:
            phi[active_idx] = phi_next
            u[active_idx] = u_next
            up[active_idx] = up_next
            previous_position[active_idx] = position_next
            continue

        global_active = active_idx[still_active_idx]
        r_active = r_next[still_active_idx]
        up_active = up_next[still_active_idx]

        captured = r_active <= horizon_radius
        if np.any(captured):
            captured_global = global_active[captured]
            status[captured_global] = STATUS_CAPTURED
            active[captured_global] = False

        escaped = ((r_active >= r_escape) & (up_active < 0.0)) | (~positive[still_active_idx])
        if np.any(escaped):
            escaped_global = global_active[escaped]
            local_escape = still_active_idx[escaped]
            status[escaped_global] = STATUS_ESCAPED
            step_direction = _normalize_np(position_next[local_escape] - prev_pos[local_escape])
            escape_direction[escaped_global] = step_direction
            active[escaped_global] = False

        phi[active_idx] = phi_next
        u[active_idx] = u_next
        up[active_idx] = up_next
        cos_phi[active_idx] = cos_next
        sin_phi[active_idx] = sin_next
        previous_position[active_idx] = position_next
        previous_radius[active_idx] = r_next

    capture_mask = (status == STATUS_CAPTURED).reshape(height, width)
    disk_hit_mask = (status == STATUS_DISK_HIT).reshape(height, width)
    escaped_mask = (status == STATUS_ESCAPED).reshape(height, width)
    incomplete_mask = (status == STATUS_INCOMPLETE).reshape(height, width)
    status_map = status.reshape(height, width)
    return RayBundleResult(
        status=status_map,
        hit_radius=hit_radius.reshape(height, width),
        hit_position=hit_position.reshape(height, width, 3),
        hit_direction=hit_direction.reshape(height, width, 3),
        escape_direction=escape_direction.reshape(height, width, 3),
        capture_mask=capture_mask,
        disk_hit_mask=disk_hit_mask,
        escaped_mask=escaped_mask,
        incomplete_mask=incomplete_mask,
        status_map=status_map,
    )


__all__ = [
    "RayBundleResult",
    "STATUS_CAPTURED",
    "STATUS_DISK_HIT",
    "STATUS_ESCAPED",
    "STATUS_INCOMPLETE",
    "integrate_ray_bundle_3d",
    "sample_background_from_directions",
]
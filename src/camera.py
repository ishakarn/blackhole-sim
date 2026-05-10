"""Simple 3D pinhole camera helpers for static rendering experiments."""

from __future__ import annotations

import math

import torch


def normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize the last dimension of a tensor."""
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True).clamp_min(eps)
    return vector / norm


def rodrigues_rotate(
    vector: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor | float,
) -> torch.Tensor:
    """Rotate vectors about an axis using Rodrigues' rotation formula."""
    axis = normalize(axis)
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, device=vector.device, dtype=vector.dtype)
    while angle.ndim < vector.ndim:
        angle = angle.unsqueeze(-1)

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    dot = (vector * axis).sum(dim=-1, keepdim=True)
    cross = torch.cross(axis.expand_as(vector), vector, dim=-1)
    return vector * cos_angle + cross * sin_angle + axis * dot * (1.0 - cos_angle)


def build_camera_basis(
    camera_position: torch.Tensor,
    target: torch.Tensor,
    up_hint: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized forward/right/up camera basis vectors."""
    forward = normalize(target - camera_position)
    right = normalize(torch.cross(forward, up_hint, dim=-1))
    up = normalize(torch.cross(right, forward, dim=-1))
    return forward, right, up


def generate_camera_rays(
    width: int,
    height: int,
    fov: float,
    camera_position: torch.Tensor,
    target: torch.Tensor,
    up_hint: torch.Tensor,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate normalized 3D ray directions for each image pixel.

    The `fov` parameter follows the existing project convention: it is the
    full image-plane width in world units at the target depth, not an angular
    field-of-view in degrees.
    """
    device = torch.device(device)
    camera_position = camera_position.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    up_hint = up_hint.to(device=device, dtype=torch.float32)

    forward, right, up = build_camera_basis(camera_position, target, up_hint)
    target_distance = torch.linalg.norm(target - camera_position).item()
    half_width = fov / 2.0
    aspect = height / width
    half_height = half_width * aspect

    xs = torch.linspace(-half_width, half_width, width, device=device)
    ys = torch.linspace(-half_height, half_height, height, device=device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")

    directions = (
        forward.view(1, 1, 3) * target_distance
        + x_grid.unsqueeze(-1) * right.view(1, 1, 3)
        + y_grid.unsqueeze(-1) * up.view(1, 1, 3)
    )
    directions = normalize(directions)
    return directions, forward, right, up, camera_position


def compute_impact_parameters(
    camera_position: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Compute the ray impact parameter relative to the origin."""
    origin = camera_position.view(1, 1, 3).expand_as(directions)
    return torch.linalg.norm(torch.cross(origin, directions, dim=-1), dim=-1)
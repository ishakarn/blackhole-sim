"""Geodesic-based static Schwarzschild lensing renderer."""

from __future__ import annotations

import csv
import math
import pathlib

import numpy as np
import torch

from .constants import B_CRIT
from .geodesics import integrate_many_null_geodesics
from .raytracing import compute_impact_parameter, make_camera_grid


LOOKUP_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "lookup"
DEFAULT_LOOKUP_NPZ = LOOKUP_DIR / "deflection_lookup.npz"
DEFAULT_LOOKUP_CSV = LOOKUP_DIR / "deflection_lookup.csv"


def make_deflection_lookup_b_grid(
    b_max: float,
    n_near: int = 320,
    n_far: int = 320,
    b_margin: float = 1e-3,
    near_extent: float = 1.2,
) -> np.ndarray:
    """Return a 1-D impact-parameter grid denser near the critical value."""
    b_min = B_CRIT + b_margin
    near_stop = min(b_max, B_CRIT + near_extent)

    near_grid = B_CRIT + np.geomspace(b_margin, max(near_stop - B_CRIT, b_margin), n_near)
    if near_stop >= b_max:
        return np.unique(near_grid)

    far_grid = np.linspace(near_stop, b_max, n_far)
    return np.unique(np.concatenate([near_grid, far_grid]))


def build_deflection_lookup_table(
    b_max: float,
    phi_max: float = 18.0,
    num_points: int = 12000,
    n_near: int = 320,
    n_far: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a radial lookup table alpha(b) from the geodesic solver."""
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

    alpha_grid = np.full_like(b_grid, np.nan, dtype=float)
    valid_mask = np.zeros_like(b_grid, dtype=bool)
    for index, trajectory in enumerate(trajectories):
        if trajectory.deflection_angle is not None:
            alpha_grid[index] = trajectory.deflection_angle
            valid_mask[index] = True

    if not np.any(valid_mask):
        raise RuntimeError("Deflection lookup table contains no escaping rays.")

    valid_b = b_grid[valid_mask]
    valid_alpha = alpha_grid[valid_mask]
    alpha_grid = np.interp(b_grid, valid_b, valid_alpha)
    return b_grid, alpha_grid


def build_deflection_lookup_payload(
    b_max: float,
    phi_max: float = 18.0,
    num_points: int = 12000,
    n_near: int = 320,
    n_far: int = 320,
) -> dict[str, np.ndarray | float]:
    """Compute lookup arrays and metadata for persistence/reuse."""
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

    alpha_raw = np.full_like(b_grid, np.nan, dtype=float)
    status_grid = np.empty(len(b_grid), dtype="<U16")
    valid_mask = np.zeros_like(b_grid, dtype=bool)
    for index, trajectory in enumerate(trajectories):
        status_grid[index] = trajectory.status
        if trajectory.deflection_angle is not None:
            alpha_raw[index] = trajectory.deflection_angle
            valid_mask[index] = True

    if not np.any(valid_mask):
        raise RuntimeError("Deflection lookup table contains no escaping rays.")

    valid_b = b_grid[valid_mask]
    valid_alpha = alpha_raw[valid_mask]
    alpha_interp = np.interp(b_grid, valid_b, valid_alpha)
    return {
        "b_grid": b_grid,
        "alpha_grid": alpha_interp,
        "alpha_raw": alpha_raw,
        "status_grid": status_grid,
        "b_max": float(b_grid.max()),
        "phi_max": float(phi_max),
        "num_points": float(num_points),
        "n_near": float(n_near),
        "n_far": float(n_far),
    }


def save_deflection_lookup_payload(
    payload: dict[str, np.ndarray | float],
    npz_path: pathlib.Path = DEFAULT_LOOKUP_NPZ,
    csv_path: pathlib.Path = DEFAULT_LOOKUP_CSV,
) -> None:
    """Persist lookup arrays to npz and csv files."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_path,
        b=np.asarray(payload["b_grid"]),
        alpha=np.asarray(payload["alpha_grid"]),
        alpha_raw=np.asarray(payload["alpha_raw"]),
        status=np.asarray(payload["status_grid"]),
        b_max=float(payload["b_max"]),
        phi_max=float(payload["phi_max"]),
        num_points=float(payload["num_points"]),
        n_near=float(payload["n_near"]),
        n_far=float(payload["n_far"]),
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["b", "alpha", "status"])
        for b_value, alpha_value, status in zip(
            np.asarray(payload["b_grid"]),
            np.asarray(payload["alpha_grid"]),
            np.asarray(payload["status_grid"]),
            strict=True,
        ):
            writer.writerow([f"{float(b_value):.12g}", f"{float(alpha_value):.12g}", str(status)])


def load_deflection_lookup_payload(
    npz_path: pathlib.Path = DEFAULT_LOOKUP_NPZ,
) -> dict[str, np.ndarray | float] | None:
    """Load a persisted lookup payload if it exists."""
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        return {
            "b_grid": data["b"],
            "alpha_grid": data["alpha"],
            "alpha_raw": data["alpha_raw"],
            "status_grid": data["status"],
            "b_max": float(data["b_max"]),
            "phi_max": float(data["phi_max"]),
            "num_points": float(data["num_points"]),
            "n_near": float(data["n_near"]),
            "n_far": float(data["n_far"]),
        }


def get_or_build_deflection_lookup_payload(
    b_max: float,
    phi_max: float = 18.0,
    num_points: int = 12000,
    n_near: int = 320,
    n_far: int = 320,
    npz_path: pathlib.Path = DEFAULT_LOOKUP_NPZ,
    csv_path: pathlib.Path = DEFAULT_LOOKUP_CSV,
    regenerate: bool = False,
) -> dict[str, np.ndarray | float]:
    """Load a persisted lookup table when possible, otherwise build and save it."""
    if not regenerate:
        payload = load_deflection_lookup_payload(npz_path=npz_path)
        if payload is not None and float(payload["b_max"]) >= b_max:
            print(f"Loaded deflection lookup from {npz_path}")
            return payload

    payload = build_deflection_lookup_payload(
        b_max=b_max,
        phi_max=phi_max,
        num_points=num_points,
        n_near=n_near,
        n_far=n_far,
    )
    save_deflection_lookup_payload(payload, npz_path=npz_path, csv_path=csv_path)
    print(f"Built deflection lookup and saved to {npz_path}")
    return payload


def interpolate_deflection_angles(
    b_values: np.ndarray,
    b_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate alpha(b) for arbitrary impact parameters."""
    return np.interp(b_values, b_grid, alpha_grid, left=alpha_grid[0], right=alpha_grid[-1])


def sample_background_from_angles(
    background: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: np.ndarray,
    fov: float,
) -> torch.Tensor:
    """Sample a procedural background as a wrapped angular environment map."""
    height, width = x.shape
    theta = torch.atan2(y, x)
    radius = torch.sqrt(x**2 + y**2)
    half_fov = fov / 2.0
    half_fov_y = half_fov * (height / width)
    max_radius = math.sqrt(half_fov * half_fov + half_fov_y * half_fov_y)

    alpha_t = torch.from_numpy(alpha).to(device=x.device, dtype=torch.float32)
    theta_source = theta + alpha_t

    # Horizontal coordinate is the outgoing direction around the lens.
    theta_wrapped = torch.remainder(theta_source + math.pi, 2.0 * math.pi) - math.pi
    u = theta_wrapped / math.pi

    # Use normalized screen radius as a simple second coordinate for the
    # background lookup. This keeps the warp radially symmetric and avoids
    # the obvious horizontal smearing from using screen y directly.
    v = (2.0 * (radius / max_radius) - 1.0).clamp(-1.0, 1.0)

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


def render_geodesic_lensing_image(
    background: torch.Tensor,
    width: int,
    height: int,
    fov: float,
    phi_max: float = 18.0,
    num_points: int = 12000,
    lookup_points_near: int = 320,
    lookup_points_far: int = 320,
    lookup_npz_path: pathlib.Path = DEFAULT_LOOKUP_NPZ,
    lookup_csv_path: pathlib.Path = DEFAULT_LOOKUP_CSV,
    regenerate_lookup: bool = False,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict[str, np.ndarray | float]]:
    """Render a static Schwarzschild lensing image using geodesic alpha(b)."""
    device = torch.device(device)
    background = background.to(device)

    x, y = make_camera_grid(width, height, fov, device=device)
    b = compute_impact_parameter(x, y).cpu().numpy()

    payload = get_or_build_deflection_lookup_payload(
        b_max=float(np.max(b)),
        phi_max=phi_max,
        num_points=num_points,
        n_near=lookup_points_near,
        n_far=lookup_points_far,
        npz_path=lookup_npz_path,
        csv_path=lookup_csv_path,
        regenerate=regenerate_lookup,
    )
    b_grid = np.asarray(payload["b_grid"])
    alpha_grid = np.asarray(payload["alpha_grid"])

    alpha = interpolate_deflection_angles(b, b_grid, alpha_grid)
    image = sample_background_from_angles(
        background=background,
        x=x,
        y=y,
        alpha=alpha,
        fov=fov,
    )

    captured = b < B_CRIT
    captured_mask = torch.from_numpy(captured).to(device=device)
    image[captured_mask] = 0.0

    metadata: dict[str, np.ndarray | float] = {
        "b_grid": b_grid,
        "alpha_grid": alpha_grid,
        "b_max": float(payload["b_max"]),
    }
    return image.clamp(0.0, 1.0), metadata
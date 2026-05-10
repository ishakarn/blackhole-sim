"""
Approximate Schwarzschild gravitational lensing renderer.

Uses dimensionless units:  G = M = c = 1,  r_s = 2,  b_crit = 3*sqrt(3).

The lensing model is a first-order Born approximation:
    alpha(b) = 4 / b          (deflection angle in radians)

Rays with b < b_crit are captured by the black hole (shadow).
"""

import math
import torch


# ---------------------------------------------------------------------------
# Physical constants (dimensionless)
# ---------------------------------------------------------------------------
B_CRIT: float = 3.0 * math.sqrt(3.0)   # ≈ 5.196 — critical impact parameter
EVENT_HORIZON: float = 2.0
PHOTON_SPHERE: float = 3.0


# ---------------------------------------------------------------------------
# Camera / ray grid
# ---------------------------------------------------------------------------

def make_camera_grid(
    width: int,
    height: int,
    fov: float,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x, y) image-plane coordinate tensors, both shape (H, W).

    The field-of-view *fov* is the full width of the image plane in
    dimensionless units.  The coordinate origin is at the image centre.
    """
    half = fov / 2.0
    aspect = height / width
    xs = torch.linspace(-half, half, width, device=device)
    ys = torch.linspace(-half * aspect, half * aspect, height, device=device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
    return x_grid, y_grid


# ---------------------------------------------------------------------------
# Lensing physics
# ---------------------------------------------------------------------------

def compute_impact_parameter(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Euclidean distance from the optical axis — shape (H, W)."""
    return torch.sqrt(x**2 + y**2).clamp(min=eps)


def compute_deflection_angle(
    b: torch.Tensor,
    G: float = 1.0,
    M: float = 1.0,
    c: float = 1.0,
) -> torch.Tensor:
    """First-order Schwarzschild deflection angle alpha = 4GM / (b c²)."""
    return (4.0 * G * M) / (b * c**2)


def make_shadow_mask(
    b: torch.Tensor,
    b_crit: float = B_CRIT,
    shadow_softness: float = 0.05,
) -> torch.Tensor:
    """Return a soft shadow transmission mask where 0=captured, 1=free."""
    if shadow_softness > 0.0:
        edge = (b - b_crit) / shadow_softness
        return torch.sigmoid(edge)
    return (b >= b_crit).float()


def make_photon_ring_image(
    b: torch.Tensor,
    width: float = 0.12,
    intensity: float = 0.75,
    color: tuple[float, float, float] = (1.0, 0.75, 0.35),
) -> torch.Tensor:
    """Return an additive RGB photon-ring glow image."""
    ring_dist = (b - B_CRIT) / width
    ring_mask = torch.exp(-0.5 * ring_dist**2)
    ring_color = torch.tensor(color, device=b.device, dtype=torch.float32)
    return (intensity * ring_mask).unsqueeze(-1) * ring_color


def make_lensed_source_coordinates(
    x: torch.Tensor,
    y: torch.Tensor,
    b: torch.Tensor,
    lens_strength: float = 1.0,
    max_deflection: float = 2.0 * math.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source-plane coordinates under the approximate thin-lens map."""
    alpha = compute_deflection_angle(b)
    alpha = torch.clamp(alpha * lens_strength, max=max_deflection)
    factor = 1.0 - alpha / b
    return x * factor, y * factor


def sample_lensed_background(
    background: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    b: torch.Tensor,
    fov: float,
    lens_strength: float = 1.0,
    max_deflection: float = 2.0 * math.pi,
) -> torch.Tensor:
    """Sample the procedural background through the approximate lens model."""
    height, width = x.shape
    x_src, y_src = make_lensed_source_coordinates(
        x,
        y,
        b,
        lens_strength=lens_strength,
        max_deflection=max_deflection,
    )

    half_fov = fov / 2.0
    half_fov_y = half_fov * (height / width)

    u = x_src / half_fov
    v = -y_src / half_fov_y

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


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_lensing_image(
    background: torch.Tensor,
    width: int,
    height: int,
    fov: float,
    b_crit: float = B_CRIT,
    lens_strength: float = 1.0,
    max_deflection: float = 2.0 * math.pi,
    shadow_softness: float = 0.05,
    photon_ring: bool = True,
    photon_ring_width: float = 0.25,
    photon_ring_intensity: float = 3.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Render a gravitationally lensed image.

    Parameters
    ----------
    background:
        RGBA or RGB float tensor, shape (H_bg, W_bg, C).  Values in [0, 1].
    width, height:
        Output image dimensions in pixels.
    fov:
        Full width of the image plane in dimensionless units.
    b_crit:
        Critical impact parameter.  Rays with b < b_crit → shadow.
    lens_strength:
        Scale factor applied to the deflection angle.  1.0 = physical.
    max_deflection:
        Upper-clamp on the deflection angle (prevents extreme wrapping).
    shadow_softness:
        Width of the soft shadow edge in image-plane units.
    device:
        PyTorch device string.

    Returns
    -------
    image:
        Float tensor of shape (height, width, C) with values in [0, 1].
    """
    device = torch.device(device)
    background = background.to(device)

    # --- camera grid ---
    x, y = make_camera_grid(width, height, fov, device=device)

    # --- impact parameter ---
    b = compute_impact_parameter(x, y)      # (H, W)

    image = sample_lensed_background(
        background=background,
        x=x,
        y=y,
        b=b,
        fov=fov,
        lens_strength=lens_strength,
        max_deflection=max_deflection,
    )

    # --- shadow mask ---
    mask = make_shadow_mask(
        b,
        b_crit=b_crit,
        shadow_softness=shadow_softness,
    )

    image = image * mask.unsqueeze(-1)

    if photon_ring:
        image = image + make_photon_ring_image(
            b,
            width=photon_ring_width,
            intensity=photon_ring_intensity,
        )

    return image.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def draw_circle_overlay(
    image: torch.Tensor,
    radius_world: float,
    fov: float,
    color: tuple[float, ...] = (1.0, 1.0, 0.0),
    thickness_px: int = 2,
) -> torch.Tensor:
    """Draw a reference circle (in world units) onto *image* (H, W, C)."""
    H, W, C = image.shape
    image = image.clone()
    device = image.device

    half = fov / 2.0
    aspect = H / W

    x, y = make_camera_grid(W, H, fov, device=device)
    b = torch.sqrt(x**2 + y**2)

    # pixel size in world units
    px_world = fov / W
    inner = radius_world - thickness_px * px_world / 2.0
    outer = radius_world + thickness_px * px_world / 2.0

    ring = (b >= inner) & (b <= outer)

    color_t = torch.tensor(color[:C], device=device, dtype=torch.float32)
    image[ring] = color_t

    return image


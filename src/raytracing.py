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
    n_channels = background.shape[-1]

    # --- camera grid ---
    x, y = make_camera_grid(width, height, fov, device=device)

    # --- polar coordinates ---
    b = compute_impact_parameter(x, y)
    theta = torch.atan2(y, x)          # (H, W)

    # --- deflection ---
    alpha = compute_deflection_angle(b)
    alpha = torch.clamp(alpha * lens_strength, max=max_deflection)

    # angular warp: deflect toward the black hole (reduce theta magnitude)
    theta_src = theta + alpha           # wrap theta toward the lens axis

    # convert back to Cartesian for background sampling
    r_src = b                           # radial mapping unchanged
    x_src = r_src * torch.cos(theta_src)
    y_src = r_src * torch.sin(theta_src)

    # --- sample background via normalised grid_sample coordinates ---
    # background may be a different size; map x_src/y_src to [-1, 1]
    H_bg, W_bg = background.shape[:2]
    aspect_bg = H_bg / W_bg

    half_fov = fov / 2.0
    half_fov_y = half_fov * (height / width)

    u = x_src / half_fov                           # [-1, 1] in x
    v = y_src / half_fov_y                         # [-1, 1] in y
    v = -v                                          # flip y (image coords)

    # grid_sample expects (N, C, H, W) input and (N, H, W, 2) grid
    grid = torch.stack([u, v], dim=-1).unsqueeze(0)        # (1, H, W, 2)
    bg_t = background.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H_bg, W_bg)

    sampled = torch.nn.functional.grid_sample(
        bg_t,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, C, H, W)
    image = sampled.squeeze(0).permute(1, 2, 0)             # (H, W, C)

    # --- shadow mask ---
    # soft edge: sigmoid blend from 0 (captured) to 1 (free)
    if shadow_softness > 0.0:
        edge = (b - b_crit) / shadow_softness
        mask = torch.sigmoid(edge)                  # (H, W)
    else:
        mask = (b >= b_crit).float()

    image = image * mask.unsqueeze(-1)

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


"""
Procedural background generators for the gravitational lensing renderer.

Each function returns a float32 tensor of shape (H, W, 3) with values in [0, 1].
The background is generated in image-plane coordinates; the lensing renderer
then warps those coordinates before sampling.

Available backgrounds
---------------------
star_field        — random bright stars on a dark sky
checkerboard      — two-colour checkerboard (useful for debugging distortion)
radial_gradient   — radial colour sweep
galaxy            — multi-arm spiral-ish star field with a bulge
"""

import math
import torch


# ---------------------------------------------------------------------------
# Star field
# ---------------------------------------------------------------------------

def star_field(
    width: int,
    height: int,
    n_stars: int = 4000,
    seed: int = 42,
    bg_brightness: float = 0.01,
    star_radius_px: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Random star field: white/coloured dots on a near-black sky.

    Stars are drawn as soft discs of radius *star_radius_px* pixels using a
    2-D Gaussian splat so they look slightly glow-y.
    """
    device = torch.device(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # dark sky base
    img = torch.full((height, width, 3), bg_brightness, device=device)

    # random star positions (pixel coords)
    sx = (torch.rand(n_stars, generator=rng, device=device) * width).long()
    sy = (torch.rand(n_stars, generator=rng, device=device) * height).long()

    # random brightness
    brightness = torch.rand(n_stars, generator=rng, device=device) * 0.85 + 0.15

    # random colour temperature: blue-white to yellow-white
    r_tint = torch.rand(n_stars, generator=rng, device=device) * 0.3 + 0.7   # 0.7-1.0
    g_tint = torch.rand(n_stars, generator=rng, device=device) * 0.2 + 0.8   # 0.8-1.0
    b_tint = torch.ones(n_stars, device=device)                                # 1.0

    # splat each star with a small Gaussian kernel
    sigma = max(star_radius_px, 1)
    half_k = sigma * 2
    ksize = half_k * 2 + 1
    k1d = torch.arange(-half_k, half_k + 1, device=device).float()
    kernel = torch.exp(-0.5 * (k1d / sigma) ** 2)
    kernel2d = kernel[:, None] * kernel[None, :]           # (ksize, ksize)
    kernel2d = kernel2d / kernel2d.max()

    for i in range(n_stars):
        cx, cy = sx[i].item(), sy[i].item()
        b_val = brightness[i].item()
        col = torch.tensor(
            [r_tint[i].item(), g_tint[i].item(), b_tint[i].item()],
            device=device,
        )

        x0, x1 = cx - half_k, cx + half_k + 1
        y0, y1 = cy - half_k, cy + half_k + 1

        # clamp to image boundaries
        kx0 = max(0, -x0); kx1 = ksize - max(0, x1 - width)
        ky0 = max(0, -y0); ky1 = ksize - max(0, y1 - height)
        ix0, ix1 = max(0, x0), min(width, x1)
        iy0, iy1 = max(0, y0), min(height, y1)

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        patch = kernel2d[ky0:ky1, kx0:kx1] * b_val    # (ph, pw)
        img[iy0:iy1, ix0:ix1] += patch.unsqueeze(-1) * col

    return img.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Checkerboard
# ---------------------------------------------------------------------------

def checkerboard(
    width: int,
    height: int,
    n_squares: int = 20,
    color_a: tuple[float, float, float] = (0.9, 0.9, 0.9),
    color_b: tuple[float, float, float] = (0.15, 0.15, 0.4),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Checkerboard pattern — useful for visualising lensing distortion."""
    device = torch.device(device)

    sq_w = width / n_squares
    sq_h = height / n_squares

    xs = torch.arange(width, device=device).float()
    ys = torch.arange(height, device=device).float()
    yg, xg = torch.meshgrid(ys, xs, indexing="ij")

    cell_x = (xg / sq_w).long()
    cell_y = (yg / sq_h).long()
    parity = (cell_x + cell_y) % 2            # 0 or 1  (H, W)

    ca = torch.tensor(color_a, device=device)
    cb = torch.tensor(color_b, device=device)

    # broadcast: (H, W, 1) * (3,)
    img = torch.where(parity.unsqueeze(-1) == 0, ca, cb)
    return img.float()


# ---------------------------------------------------------------------------
# Radial gradient
# ---------------------------------------------------------------------------

def radial_gradient(
    width: int,
    height: int,
    color_center: tuple[float, float, float] = (0.05, 0.0, 0.15),
    color_edge: tuple[float, float, float] = (0.6, 0.2, 0.05),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Smooth radial colour gradient from centre to edge."""
    device = torch.device(device)

    xs = torch.linspace(-1, 1, width, device=device)
    ys = torch.linspace(-1, 1, height, device=device)
    yg, xg = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(xg**2 + yg**2).clamp(0.0, 1.0)    # (H, W)

    cc = torch.tensor(color_center, device=device)
    ce = torch.tensor(color_edge, device=device)

    img = cc + r.unsqueeze(-1) * (ce - cc)
    return img.float().clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Galaxy (multi-arm spiral + bulge)
# ---------------------------------------------------------------------------

def galaxy(
    width: int,
    height: int,
    n_stars: int = 12000,
    n_arms: int = 3,
    arm_tightness: float = 0.4,
    seed: int = 42,
    bg_brightness: float = 0.005,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rough spiral galaxy background with a bright central bulge."""
    device = torch.device(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    img = torch.full((height, width, 3), bg_brightness, device=device)

    # build star positions in normalised [-1, 1] space
    # --- bulge ---
    n_bulge = n_stars // 4
    br = torch.rand(n_bulge, generator=rng, device=device) ** 0.5 * 0.15
    bt = torch.rand(n_bulge, generator=rng, device=device) * 2 * math.pi
    bx = br * torch.cos(bt)
    by = br * torch.sin(bt)

    # --- arms ---
    n_arm_stars = n_stars - n_bulge
    arm_idx = torch.randint(0, n_arms, (n_arm_stars,), generator=rng, device=device)
    arm_offset = arm_idx.float() * (2 * math.pi / n_arms)

    radius = torch.rand(n_arm_stars, generator=rng, device=device) ** 0.5 * 0.9 + 0.05
    spread = torch.randn(n_arm_stars, generator=rng, device=device) * 0.08
    angle = arm_offset + radius / arm_tightness + spread
    ax = radius * torch.cos(angle)
    ay = radius * torch.sin(angle)

    all_x = torch.cat([bx, ax])
    all_y = torch.cat([by, ay])
    all_r = torch.cat([br, radius])

    # brightness falls off with radius
    brightness = (1.0 - all_r.clamp(0, 1)) ** 2 * 0.8 + 0.05

    # colour: bulge is warm, arms are blue-white
    n_total = len(all_x)
    r_tint = torch.ones(n_total, device=device)
    g_tint = (0.8 + 0.2 * torch.rand(n_total, generator=rng, device=device))
    b_tint = (0.6 + 0.4 * torch.rand(n_total, generator=rng, device=device))
    # bulge: warmer
    r_tint[:n_bulge] = 1.0
    g_tint[:n_bulge] = 0.7 + 0.2 * torch.rand(n_bulge, generator=rng, device=device)
    b_tint[:n_bulge] = 0.4 + 0.2 * torch.rand(n_bulge, generator=rng, device=device)

    # map to pixel coords
    sx = ((all_x + 1) / 2 * width).long().clamp(0, width - 1)
    sy = ((all_y + 1) / 2 * height).long().clamp(0, height - 1)

    for i in range(n_total):
        px, py = sx[i].item(), sy[i].item()
        col = torch.tensor(
            [r_tint[i].item(), g_tint[i].item(), b_tint[i].item()],
            device=device,
        )
        img[py, px] += col * brightness[i].item()

    return img.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKGROUNDS: dict[str, callable] = {
    "stars": star_field,
    "checkerboard": checkerboard,
    "radial": radial_gradient,
    "galaxy": galaxy,
}


def make_background(
    name: str,
    width: int,
    height: int,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convenience factory — returns (H, W, 3) float tensor in [0, 1]."""
    if name not in BACKGROUNDS:
        raise ValueError(
            f"Unknown background '{name}'. "
            f"Choose from: {list(BACKGROUNDS.keys())}"
        )
    fn = BACKGROUNDS[name]
    # pass seed only to functions that accept it
    import inspect
    sig = inspect.signature(fn)
    kwargs: dict = dict(width=width, height=height, device=device)
    if "seed" in sig.parameters:
        kwargs["seed"] = seed
    return fn(**kwargs)

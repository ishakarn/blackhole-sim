"""v1.1 approximate 3D black hole renderer with tilted disk geometry."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.backgrounds import make_background
from src.camera import compute_impact_parameters, generate_camera_rays, normalize, rodrigues_rotate
from src.constants import B_CRIT
from src.disk_models import disk_colorize
from src.geodesic_renderer import (
    DEFAULT_LOOKUP_CSV,
    DEFAULT_LOOKUP_NPZ,
    get_or_build_deflection_lookup_payload,
    interpolate_deflection_angles,
)


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "11_approx_3d_disk_renderer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate 3D black hole renderer using geodesic deflection lookup and a tilted disk."
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=14.0)
    parser.add_argument(
        "--background",
        choices=["stars", "checkerboard", "radial", "galaxy"],
        default="stars",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-distance", type=float, default=30.0)
    parser.add_argument("--camera-height", type=float, default=0.0)
    parser.add_argument("--shadow-scale", type=float, default=0.92)
    parser.add_argument("--disk-inner-radius", type=float, default=6.0)
    parser.add_argument("--disk-outer-radius", type=float, default=15.0)
    parser.add_argument("--disk-tilt", type=float, default=70.0)
    parser.add_argument("--disk-rotation", type=float, default=0.0)
    parser.add_argument("--disk-thickness-proxy", type=float, default=0.18)
    parser.add_argument("--emissivity-power", type=float, default=1.25)
    parser.add_argument("--beaming-strength", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-scale", type=float, default=4.0)
    parser.add_argument("--phi-max", type=float, default=18.0)
    parser.add_argument("--num-points", type=int, default=12000)
    parser.add_argument("--lookup-points-near", type=int, default=320)
    parser.add_argument("--lookup-points-far", type=int, default=320)
    parser.add_argument("--lookup-npz", type=str, default=str(DEFAULT_LOOKUP_NPZ))
    parser.add_argument("--lookup-csv", type=str, default=str(DEFAULT_LOOKUP_CSV))
    parser.add_argument("--regenerate-lookup", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save-background", action="store_true")
    parser.add_argument("--save-disk-mask", action="store_true")
    parser.add_argument("--save-disk-only", action="store_true")
    parser.add_argument("--save-shadow-mask", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def save_tensor_image(tensor: torch.Tensor, path: pathlib.Path) -> None:
    """Save a (H, W, 3) float tensor in [0, 1] as PNG."""
    try:
        import torchvision.utils as tvu

        path.parent.mkdir(parents=True, exist_ok=True)
        tvu.save_image(tensor.permute(2, 0, 1).cpu(), str(path))
    except ImportError:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), tensor.cpu().numpy())
    print(f"Saved {path}")


def make_disk_frame(
    camera_forward: torch.Tensor,
    camera_right: torch.Tensor,
    camera_up: torch.Tensor,
    tilt_deg: float,
    rotation_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a tilted disk frame relative to the camera view."""
    rotation = math.radians(rotation_deg)
    tilt = math.radians(tilt_deg)
    axis = normalize(math.cos(rotation) * camera_right + math.sin(rotation) * camera_up)
    disk_normal = rodrigues_rotate(camera_forward, axis, tilt)
    disk_u = axis
    disk_v = normalize(torch.cross(disk_normal, disk_u, dim=-1))
    return disk_normal, disk_u, disk_v


def bend_rays_toward_center(
    directions: torch.Tensor,
    camera_position: torch.Tensor,
    alpha: torch.Tensor,
    max_bend: float = 1.15,
) -> torch.Tensor:
    """Approximate gravitational bending by rotating rays toward the origin."""
    to_center = normalize((-camera_position).view(1, 1, 3).expand_as(directions))
    axis = torch.cross(directions, to_center, dim=-1)
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    safe_axis = torch.where(axis_norm > 1e-6, axis / axis_norm.clamp_min(1e-6), axis)
    bend_angle = torch.clamp(alpha, min=0.0, max=max_bend)
    rotated = rodrigues_rotate(directions, safe_axis, bend_angle)
    return torch.where(axis_norm > 1e-6, rotated, directions)


def sample_background_from_directions(background: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Sample a procedural background as an equirectangular environment map."""
    dx = directions[..., 0]
    dy = directions[..., 1]
    dz = directions[..., 2]

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


def intersect_rays_with_disk(
    camera_position: torch.Tensor,
    ray_directions: torch.Tensor,
    disk_normal: torch.Tensor,
    disk_u: torch.Tensor,
    disk_v: torch.Tensor,
    inner_radius: float,
    outer_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect bent rays with a tilted thin disk plane through the origin."""
    origin = camera_position.view(1, 1, 3)
    denom = (ray_directions * disk_normal.view(1, 1, 3)).sum(dim=-1)
    numer = -torch.dot(camera_position, disk_normal)
    valid = denom.abs() > 1e-6
    t = torch.zeros_like(denom)
    t[valid] = numer / denom[valid]

    hit_points = origin + t.unsqueeze(-1) * ray_directions
    u_coord = (hit_points * disk_u.view(1, 1, 3)).sum(dim=-1)
    v_coord = (hit_points * disk_v.view(1, 1, 3)).sum(dim=-1)
    radius = torch.sqrt(u_coord * u_coord + v_coord * v_coord)

    hit_mask = valid & (t > 0.0) & (radius >= inner_radius) & (radius <= outer_radius)
    return hit_mask, hit_points, radius, t


def render_disk_emission(
    hit_mask: torch.Tensor,
    hit_points: torch.Tensor,
    ray_directions: torch.Tensor,
    disk_normal: torch.Tensor,
    inner_radius: float,
    outer_radius: float,
    disk_thickness_proxy: float,
    emissivity_power: float,
    beaming_strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render warm disk emission with radial falloff and approximate beaming."""
    radius = torch.linalg.norm(hit_points, dim=-1).clamp_min(inner_radius)
    radial_emissivity = torch.pow((radius / inner_radius).clamp(min=1.0), -emissivity_power)
    ring_width = max(disk_thickness_proxy, 1e-3) * (outer_radius - inner_radius)
    annulus_profile = torch.exp(-0.5 * ((radius - inner_radius) / ring_width) ** 2)
    outer_taper = ((outer_radius - radius) / max(outer_radius - inner_radius, 1e-6)).clamp(0.0, 1.0)
    intensity = (radial_emissivity * annulus_profile * torch.pow(outer_taper, 0.6)).clamp(0.0, 10.0)
    intensity = intensity / intensity.amax().clamp_min(1e-6)

    radial = normalize(hit_points)
    orbital = normalize(torch.cross(disk_normal.view(1, 1, 3).expand_as(hit_points), radial, dim=-1))
    view_dir = -ray_directions
    beaming = (1.0 + beaming_strength * (orbital * view_dir).sum(dim=-1)).clamp(0.2, 1.6)
    intensity = (intensity * beaming).clamp(0.0, 1.0)

    colors = disk_colorize(intensity)
    colors[~hit_mask] = 0.0
    return colors.clamp(0.0, 1.0), hit_mask.float()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))

    background_width = max(width, int(width * args.background_scale))
    background_height = max(height, int(background_width / 2))
    background = make_background(
        args.background,
        width=background_width,
        height=background_height,
        seed=args.seed,
        device=device,
    )

    camera_position = torch.tensor([0.0, -args.camera_distance, args.camera_height], device=device)
    target = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_hint = torch.tensor([0.0, 0.0, 1.0], device=device)

    directions, camera_forward, camera_right, camera_up, camera_position = generate_camera_rays(
        width=width,
        height=height,
        fov=args.fov,
        camera_position=camera_position,
        target=target,
        up_hint=up_hint,
        device=device,
    )
    impact = compute_impact_parameters(camera_position, directions)
    b_np = impact.cpu().numpy()

    lookup_payload = get_or_build_deflection_lookup_payload(
        b_max=float(np.max(b_np)),
        phi_max=args.phi_max,
        num_points=args.num_points,
        n_near=args.lookup_points_near,
        n_far=args.lookup_points_far,
        npz_path=pathlib.Path(args.lookup_npz),
        csv_path=pathlib.Path(args.lookup_csv),
        regenerate=args.regenerate_lookup,
    )
    b_grid = np.asarray(lookup_payload["b_grid"])
    alpha_grid = np.asarray(lookup_payload["alpha_grid"])
    alpha_np = interpolate_deflection_angles(b_np, b_grid, alpha_grid)
    alpha = torch.from_numpy(alpha_np.astype(np.float32)).to(device=device)

    bent_directions = bend_rays_toward_center(directions, camera_position, alpha)
    background_image = sample_background_from_directions(background, bent_directions)

    disk_normal, disk_u, disk_v = make_disk_frame(
        camera_forward,
        camera_right,
        camera_up,
        tilt_deg=args.disk_tilt,
        rotation_deg=args.disk_rotation,
    )
    disk_hit, hit_points, _, _ = intersect_rays_with_disk(
        camera_position,
        directions,
        disk_normal,
        disk_u,
        disk_v,
        inner_radius=args.disk_inner_radius,
        outer_radius=args.disk_outer_radius,
    )
    disk_rgb, disk_mask = render_disk_emission(
        disk_hit,
        hit_points,
        directions,
        disk_normal,
        inner_radius=args.disk_inner_radius,
        outer_radius=args.disk_outer_radius,
        disk_thickness_proxy=args.disk_thickness_proxy,
        emissivity_power=args.emissivity_power,
        beaming_strength=args.beaming_strength,
    )

    captured = impact < (args.shadow_scale * B_CRIT)
    image = background_image.clone()
    visible_disk = disk_mask > 0.0
    image[visible_disk] = disk_rgb[visible_disk]
    image[captured] = 0.0
    image = image.clamp(0.0, 1.0)

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "approx_3d_geodesic_disk.png"
    save_tensor_image(image, output_path)
    if args.save_background:
        save_tensor_image(background_image, OUTPUT_DIR / "approx_3d_geodesic_background.png")
    if args.save_disk_mask:
        mask_rgb = disk_mask.unsqueeze(-1).repeat(1, 1, 3)
        save_tensor_image(mask_rgb, OUTPUT_DIR / "approx_3d_geodesic_diskmask.png")
    if args.save_disk_only:
        save_tensor_image(disk_rgb, OUTPUT_DIR / "approx_3d_geodesic_diskonly.png")
    if args.save_shadow_mask:
        shadow_rgb = captured.float().unsqueeze(-1).repeat(1, 1, 3)
        save_tensor_image(shadow_rgb, OUTPUT_DIR / "approx_3d_geodesic_shadowmask.png")

    print(
        f"Lookup table: {len(b_grid)} samples, b_max={float(np.max(b_np)):.3f}, "
        f"camera=({camera_position[0].item():.1f}, {camera_position[1].item():.1f}, {camera_position[2].item():.1f})"
    )

    if args.show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(background_image.cpu().numpy())
        axes[0].set_title("Lensed Background")
        axes[0].axis("off")
        axes[1].imshow(disk_rgb.cpu().numpy())
        axes[1].set_title("Tilted Disk")
        axes[1].axis("off")
        axes[2].imshow(image.cpu().numpy())
        axes[2].set_title("Approximate 3D Geodesic Disk")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
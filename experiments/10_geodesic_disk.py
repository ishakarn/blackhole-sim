"""v1.0 geodesic-based black hole shadow + disk renderer."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.backgrounds import make_background
from src.constants import B_CRIT
from src.disk_intersection import (
    build_disk_intersection_lookup_table,
    interpolate_disk_hit_radii,
    render_disk_emission_from_radii,
)
from src.geodesic_renderer import render_geodesic_lensing_image
from src.raytracing import compute_impact_parameter, make_camera_grid


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "10_geodesic_disk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a geodesic-based Schwarzschild shadow with accretion disk emission."
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-scale", type=float, default=4.0)
    parser.add_argument("--phi-max", type=float, default=18.0)
    parser.add_argument("--num-points", type=int, default=12000)
    parser.add_argument("--lookup-points-near", type=int, default=320)
    parser.add_argument("--lookup-points-far", type=int, default=320)
    parser.add_argument("--disk-inner-radius", type=float, default=6.0)
    parser.add_argument("--disk-outer-radius", type=float, default=12.0)
    parser.add_argument(
        "--disk-screen-thickness",
        type=float,
        default=0.35,
        help="Approximate visible half-thickness of the thin equatorial disk in screen-plane units.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save-background", action="store_true")
    parser.add_argument("--save-disk-mask", action="store_true")
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))

    background_width = max(width, int(width * args.background_scale))
    background_height = max(height, int(height * max(1.0, args.background_scale / 2.0)))
    background = make_background(
        args.background,
        width=background_width,
        height=background_height,
        seed=args.seed,
        device=device,
    )

    background_image, metadata = render_geodesic_lensing_image(
        background=background,
        width=width,
        height=height,
        fov=args.fov,
        phi_max=args.phi_max,
        num_points=args.num_points,
        lookup_points_near=args.lookup_points_near,
        lookup_points_far=args.lookup_points_far,
        device=device,
    )

    x, y = make_camera_grid(width, height, args.fov, device=device)
    b = compute_impact_parameter(x, y).cpu().numpy()

    b_grid, disk_radius_grid, _ = build_disk_intersection_lookup_table(
        b_max=float(metadata["b_max"]),
        inner_radius=args.disk_inner_radius,
        outer_radius=args.disk_outer_radius,
        phi_max=args.phi_max,
        num_points=args.num_points,
        n_near=args.lookup_points_near,
        n_far=args.lookup_points_far,
    )
    hit_radii = interpolate_disk_hit_radii(b, b_grid, disk_radius_grid)
    disk_rgb, disk_mask = render_disk_emission_from_radii(
        hit_radii,
        inner_radius=args.disk_inner_radius,
        outer_radius=args.disk_outer_radius,
    )
    disk_rgb = disk_rgb.to(device=device)
    disk_mask = disk_mask.to(device=device)

    # The current solver is an equatorial 2-D slice. Use a narrow screen-plane
    # window as the minimal proxy for a thin visible disk instead of letting
    # the annulus flood the whole frame.
    plane_alpha = torch.exp(-0.5 * (y / args.disk_screen_thickness) ** 2)
    disk_rgb = disk_rgb * plane_alpha.unsqueeze(-1)
    disk_mask = disk_mask * plane_alpha

    captured_mask = torch.from_numpy(b < B_CRIT).to(device=device)
    image = background_image.clone()
    visible_disk = (disk_mask > 0.05) & (~captured_mask)
    image[visible_disk] = disk_rgb[visible_disk]
    image[captured_mask] = 0.0
    image = image.clamp(0.0, 1.0)

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "geodesic_black_hole_disk.png"
    save_tensor_image(image, output_path)

    if args.save_background:
        save_tensor_image(background_image, OUTPUT_DIR / "geodesic_black_hole_background.png")
    if args.save_disk_mask:
        mask_rgb = disk_mask.unsqueeze(-1).repeat(1, 1, 3)
        save_tensor_image(mask_rgb, OUTPUT_DIR / "geodesic_black_hole_diskmask.png")

    print(
        f"Lookup tables: deflection={len(metadata['b_grid'])} samples, "
        f"disk={len(b_grid)} samples"
    )

    if args.show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(background_image.cpu().numpy())
        axes[0].set_title("Geodesic Background")
        axes[0].axis("off")
        axes[1].imshow(disk_rgb.cpu().numpy())
        axes[1].set_title("Disk Emission")
        axes[1].axis("off")
        axes[2].imshow(image.cpu().numpy())
        axes[2].set_title("Geodesic Black Hole Disk")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
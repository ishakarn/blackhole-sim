"""v0.7 static black hole renderer with approximate accretion disk."""

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.backgrounds import make_background
from src.disk_models import render_disk_image
from src.raytracing import (
    B_CRIT,
    EVENT_HORIZON,
    PHOTON_SPHERE,
    compute_impact_parameter,
    draw_circle_overlay,
    make_camera_grid,
    make_lensed_source_coordinates,
    make_photon_ring_image,
    make_shadow_mask,
    sample_lensed_background,
)


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "07_black_hole_disk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate static black hole render with accretion disk (v0.7)"
    )
    parser.add_argument("--resolution", type=int, default=1024)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lens-strength", type=float, default=1.0)
    parser.add_argument("--max-deflection", type=float, default=2.0 * math.pi)
    parser.add_argument("--shadow-softness", type=float, default=0.05)

    parser.add_argument("--disk-inner-radius", type=float, default=6.0)
    parser.add_argument("--disk-outer-radius", type=float, default=12.0)
    parser.add_argument("--disk-tilt", type=float, default=70.0)
    parser.add_argument("--disk-rotation", type=float, default=0.0)
    parser.add_argument("--disk-edge-softness", type=float, default=0.15)
    parser.add_argument("--brightness-power", type=float, default=0.75)
    parser.add_argument("--beaming-strength", type=float, default=0.35)

    parser.add_argument("--no-photon-ring", action="store_true")
    parser.add_argument("--photon-ring-width", type=float, default=0.12)
    parser.add_argument("--photon-ring-intensity", type=float, default=0.75)

    parser.add_argument("--overlay-circles", action="store_true")
    parser.add_argument("--no-save-background", action="store_true")
    parser.add_argument("--save-disk-mask", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def save_tensor_image(tensor: torch.Tensor, path: pathlib.Path) -> None:
    """Save a (H, W, 3) float tensor in [0, 1] as PNG."""
    try:
        import torchvision.utils as tvu

        path.parent.mkdir(parents=True, exist_ok=True)
        tvu.save_image(tensor.permute(2, 0, 1).cpu(), str(path))
    except ImportError:
        import matplotlib.pyplot as plt  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), tensor.cpu().numpy())
    print(f"  Saved: {path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))

    print(f"Device: {device}")
    print(f"Image size: {width} x {height}")
    print(
        f"FOV: {args.fov} | b_crit={B_CRIT:.4f} | "
        f"disk=[{args.disk_inner_radius:.1f}, {args.disk_outer_radius:.1f}]"
    )

    background = make_background(
        args.background,
        width=width,
        height=height,
        seed=args.seed,
        device=device,
    )

    if not args.no_save_background:
        background_path = OUTPUT_DIR / f"black_hole_disk_background_{args.background}.png"
        save_tensor_image(background, background_path)

    x, y = make_camera_grid(width, height, args.fov, device=device)
    b = compute_impact_parameter(x, y)

    background_image = sample_lensed_background(
        background=background,
        x=x,
        y=y,
        b=b,
        fov=args.fov,
        lens_strength=args.lens_strength,
        max_deflection=args.max_deflection,
    )

    x_src, y_src = make_lensed_source_coordinates(
        x,
        y,
        b,
        lens_strength=args.lens_strength,
        max_deflection=args.max_deflection,
    )

    disk_rgb, disk_alpha, disk_mask = render_disk_image(
        x_src,
        y_src,
        inner_radius=args.disk_inner_radius,
        outer_radius=args.disk_outer_radius,
        tilt_deg=args.disk_tilt,
        rotation_deg=args.disk_rotation,
        beaming_strength=args.beaming_strength,
        edge_softness=args.disk_edge_softness,
        brightness_power=args.brightness_power,
    )

    image = background_image * (1.0 - disk_alpha.unsqueeze(-1)) + disk_rgb

    shadow_mask = make_shadow_mask(
        b,
        b_crit=B_CRIT,
        shadow_softness=args.shadow_softness,
    )
    image = image * shadow_mask.unsqueeze(-1)

    if not args.no_photon_ring:
        image = image + make_photon_ring_image(
            b,
            width=args.photon_ring_width,
            intensity=args.photon_ring_intensity,
        )

    if args.overlay_circles:
        image = draw_circle_overlay(image, B_CRIT, args.fov, color=(0.0, 1.0, 1.0), thickness_px=2)
        image = draw_circle_overlay(image, PHOTON_SPHERE, args.fov, color=(1.0, 1.0, 0.0), thickness_px=2)
        image = draw_circle_overlay(image, EVENT_HORIZON, args.fov, color=(1.0, 0.2, 0.2), thickness_px=2)

    image = image.clamp(0.0, 1.0)
    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "black_hole_disk.png"
    save_tensor_image(image, output_path)

    if args.save_disk_mask:
        mask_rgb = disk_mask.unsqueeze(-1).repeat(1, 1, 3)
        mask_path = OUTPUT_DIR / "black_hole_disk_diskmask.png"
        save_tensor_image(mask_rgb, mask_path)

    if args.show:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(background.cpu().numpy())
        axes[0].set_title("Background")
        axes[0].axis("off")
        axes[1].imshow(disk_rgb.clamp(0.0, 1.0).cpu().numpy())
        axes[1].set_title("Disk Layer")
        axes[1].axis("off")
        axes[2].imshow(image.cpu().numpy())
        axes[2].set_title("Black Hole Disk Render")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
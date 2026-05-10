"""v0.9 geodesic-based Schwarzschild lensing renderer."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.backgrounds import make_background
from src.geodesic_renderer import render_geodesic_lensing_image


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a static Schwarzschild lensing image using geodesic deflection lookup."
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
    parser.add_argument(
        "--background-scale",
        type=float,
        default=4.0,
        help="Scale factor for the procedural background environment map resolution.",
    )
    parser.add_argument("--phi-max", type=float, default=18.0)
    parser.add_argument("--num-points", type=int, default=12000)
    parser.add_argument("--lookup-points-near", type=int, default=320)
    parser.add_argument("--lookup-points-far", type=int, default=320)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), tensor.cpu().numpy())

    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))

    background_width = max(width, int(width * args.background_scale))
    background_height = max(height, int(height * max(1.0, args.background_scale / 2.0)))

    background = make_background(
        args.background,
        width=background_width,
        height=background_height,
        seed=args.seed,
        device=args.device,
    )
    image, metadata = render_geodesic_lensing_image(
        background=background,
        width=width,
        height=height,
        fov=args.fov,
        phi_max=args.phi_max,
        num_points=args.num_points,
        lookup_points_near=args.lookup_points_near,
        lookup_points_far=args.lookup_points_far,
        device=args.device,
    )

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "geodesic_lensing.png"
    save_tensor_image(image, output_path)
    print(
        f"Lookup table: {len(metadata['b_grid'])} samples, "
        f"b_max={float(metadata['b_max']):.3f}"
    )

    if args.show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(background.cpu().numpy())
        axes[0].set_title(f"Background: {args.background} ({background_width}x{background_height})")
        axes[0].axis("off")
        axes[1].imshow(image.cpu().numpy())
        axes[1].set_title("Geodesic Lensing")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
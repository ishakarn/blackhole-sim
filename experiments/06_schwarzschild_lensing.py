"""
v0.6 — Approximate Schwarzschild gravitational lensing renderer.

Produces a static image showing:
  - central black shadow (b < b_crit)
  - distorted/lensed background (stars / checkerboard / galaxy / radial)
  - optional overlay circles for b_crit, photon sphere, event horizon

Usage examples
--------------
    python -m experiments.06_schwarzschild_lensing --resolution 1024 --fov 14 --background stars
    python -m experiments.06_schwarzschild_lensing --resolution 1024 --fov 14 --background checkerboard --show
    python -m experiments.06_schwarzschild_lensing --resolution 512  --fov 20 --background galaxy --device cuda --show
    python -m experiments.06_schwarzschild_lensing --resolution 2048 --fov 14 --background stars --seed 7 --overlay-circles --device cuda
"""

import argparse
import math
import pathlib
import sys

import torch

# ---------------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.backgrounds import make_background
from src.raytracing import (
    B_CRIT,
    EVENT_HORIZON,
    PHOTON_SPHERE,
    render_lensing_image,
    draw_circle_overlay,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Schwarzschild lensing renderer (v0.6 — approximate)"
    )

    # resolution
    p.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Output image width in pixels (height scaled by --aspect).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override --resolution for width.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override --resolution for height.",
    )
    p.add_argument(
        "--aspect",
        type=float,
        default=1.0,
        help="Height / width ratio (default 1.0 = square).",
    )

    # optics
    p.add_argument(
        "--fov",
        type=float,
        default=14.0,
        help="Full image-plane width in dimensionless units (default 14).",
    )
    p.add_argument(
        "--lens-strength",
        type=float,
        default=1.0,
        help="Multiplier on the deflection angle (default 1.0 = physical).",
    )
    p.add_argument(
        "--max-deflection",
        type=float,
        default=2.0 * math.pi,
        help="Upper clamp on deflection angle in radians (default 2π).",
    )
    p.add_argument(
        "--shadow-softness",
        type=float,
        default=0.05,
        help="Soft-edge width in world units (0 = hard edge).",
    )

    # background
    p.add_argument(
        "--background",
        choices=["stars", "checkerboard", "radial", "galaxy"],
        default="stars",
        help="Procedural background type.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for procedural backgrounds.",
    )

    # overlays
    p.add_argument(
        "--overlay-circles",
        action="store_true",
        help="Draw reference circles for b_crit, photon sphere, event horizon.",
    )
    p.add_argument(
        "--no-save-background",
        action="store_true",
        help="Skip saving the unlensed background reference image.",
    )

    # output
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: outputs/figures/schwarzschild_lensing.png).",
    )
    p.add_argument(
        "--no-photon-ring",
        action="store_true",
        help="Disable the photon ring glow at the shadow edge.",
    )
    p.add_argument(
        "--photon-ring-intensity",
        type=float,
        default=3.0,
        help="Brightness multiplier for the photon ring glow (default 3.0).",
    )
    p.add_argument(
        "--photon-ring-width",
        type=float,
        default=0.25,
        help="Gaussian sigma of the photon ring in world units (default 0.25).",
    )

    p.add_argument(
        "--show",
        action="store_true",
        help="Display the image after saving (requires matplotlib).",
    )

    # device
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: cuda if available).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_tensor_image(tensor: torch.Tensor, path: pathlib.Path) -> None:
    """Save a (H, W, 3) float tensor in [0,1] as a PNG."""
    try:
        import torchvision.utils as tvu

        path.parent.mkdir(parents=True, exist_ok=True)
        # torchvision expects (C, H, W) in [0, 1]
        tvu.save_image(tensor.permute(2, 0, 1).cpu(), str(path))
    except ImportError:
        # fall back to matplotlib
        import matplotlib.pyplot as plt  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), tensor.cpu().numpy())

    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- resolve dimensions ---
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else max(1, int(width * args.aspect))
    print(f"Image size: {width} x {height}")
    print(f"FOV: {args.fov}  |  b_crit = {B_CRIT:.4f}")

    if args.fov <= B_CRIT:
        print(
            f"WARNING: fov ({args.fov}) <= b_crit ({B_CRIT:.3f}). "
            "The shadow will fill the whole image. Increase --fov."
        )

    # --- background ---
    print(f"Generating background: {args.background} (seed={args.seed}) …")
    background = make_background(
        args.background,
        width=width,
        height=height,
        seed=args.seed,
        device=device,
    )

    # optionally save unlensed reference
    if not args.no_save_background:
        bg_path = OUTPUT_DIR / f"schwarzschild_lensing_background_{args.background}.png"
        save_tensor_image(background, bg_path)

    # --- render lensed image ---
    print("Rendering lensed image …")
    lensed = render_lensing_image(
        background=background,
        width=width,
        height=height,
        fov=args.fov,
        b_crit=B_CRIT,
        lens_strength=args.lens_strength,
        max_deflection=args.max_deflection,
        shadow_softness=args.shadow_softness,
        photon_ring=not args.no_photon_ring,
        photon_ring_intensity=args.photon_ring_intensity,
        photon_ring_width=args.photon_ring_width,
        device=device,
    )

    # --- optional overlays ---
    if args.overlay_circles:
        # b_crit — cyan
        lensed = draw_circle_overlay(lensed, B_CRIT, args.fov, color=(0.0, 1.0, 1.0), thickness_px=2)
        # photon sphere — yellow
        lensed = draw_circle_overlay(lensed, PHOTON_SPHERE, args.fov, color=(1.0, 1.0, 0.0), thickness_px=2)
        # event horizon — red
        lensed = draw_circle_overlay(lensed, EVENT_HORIZON, args.fov, color=(1.0, 0.2, 0.2), thickness_px=2)

    # --- save ---
    out_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "schwarzschild_lensing.png"
    save_tensor_image(lensed, out_path)
    print("Done.")

    # --- show ---
    if args.show:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(background.cpu().numpy())
        axes[0].set_title(f"Background: {args.background}")
        axes[0].axis("off")
        axes[1].imshow(lensed.cpu().numpy())
        title = f"Schwarzschild lensing  |  fov={args.fov}  |  b_crit={B_CRIT:.3f}"
        if args.overlay_circles:
            title += "\ncyan=b_crit  yellow=photon sphere  red=event horizon"
        axes[1].set_title(title)
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

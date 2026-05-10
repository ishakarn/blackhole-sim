"""v0.8 Schwarzschild null geodesic trajectory experiment."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.constants import B_CRIT, EVENT_HORIZON_RADIUS, PHOTON_SPHERE_RADIUS
from src.geodesics import integrate_many_null_geodesics


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "08_null_geodesics"


def parse_b_values(raw: str) -> list[float]:
    values: list[float] = []
    for value in raw.split(","):
        token = value.strip().lower()
        if not token:
            continue
        if token in {"bcrit", "b_crit", "critical"}:
            values.append(B_CRIT)
        else:
            values.append(float(token))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate and plot Schwarzschild null geodesics in the equatorial plane."
    )
    parser.add_argument(
        "--b-values",
        type=str,
        default="4.5,5.0,bcrit,5.5,6.0,8.0",
        help="Comma-separated impact parameters.",
    )
    parser.add_argument(
        "--phi-max",
        type=float,
        default=12.0,
        help="Maximum azimuthal range for integration.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=6000,
        help="Number of RK4 integration samples.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the trajectory figure.",
    )
    parser.add_argument(
        "--annotate-b",
        action="store_true",
        help="Annotate each ray near its starting segment with its impact parameter.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot after saving.",
    )
    return parser.parse_args()


def plot_trajectories(trajectories, annotate_b: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))

    horizon = plt.Circle((0.0, 0.0), EVENT_HORIZON_RADIUS, color="black", alpha=0.95)
    photon_sphere = plt.Circle(
        (0.0, 0.0),
        PHOTON_SPHERE_RADIUS,
        color="gold",
        fill=False,
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
    )
    ax.add_patch(horizon)
    ax.add_patch(photon_sphere)

    cmap = plt.get_cmap("plasma")
    for index, trajectory in enumerate(trajectories):
        color = cmap(index / max(len(trajectories) - 1, 1))
        linestyle = "-" if trajectory.status == "escaped" else ":"
        linewidth = 2.0 if abs(trajectory.impact_parameter - B_CRIT) < 0.2 else 1.7
        label = f"b={trajectory.impact_parameter:.3f} ({trajectory.status})"
        ax.plot(trajectory.x, trajectory.y, color=color, linewidth=linewidth, linestyle=linestyle, label=label)

        if annotate_b and len(trajectory.x) > 8:
            ax.text(
                trajectory.x[5],
                trajectory.y[5],
                f"b={trajectory.impact_parameter:.3f}",
                color=color,
                fontsize=8,
            )

    max_extent = max(
        10.0,
        float(np.nanmax([np.nanmax(np.abs(traj.x)) for traj in trajectories])),
        float(np.nanmax([np.nanmax(np.abs(traj.y)) for traj in trajectories])),
    )
    max_extent = min(max_extent, 50.0)

    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Schwarzschild Null Geodesics\n"
        f"event horizon r=2, photon sphere r=3, critical b={B_CRIT:.3f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    b_values = parse_b_values(args.b_values)
    trajectories = integrate_many_null_geodesics(
        b_values,
        phi_max=args.phi_max,
        num_points=args.num_points,
    )

    fig = plot_trajectories(trajectories, annotate_b=args.annotate_b)
    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "null_geodesics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved {output_path}")
    for trajectory in trajectories:
        print(f"b={trajectory.impact_parameter:.3f}: {trajectory.status}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
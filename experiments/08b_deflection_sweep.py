"""Measure null geodesic deflection angle versus impact parameter."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.constants import B_CRIT
from src.geodesics import integrate_many_null_geodesics


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures"
METRICS_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "metrics"


def parse_b_values(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Schwarzschild light deflection as a function of impact parameter."
    )
    parser.add_argument(
        "--b-values",
        type=str,
        default="5.3,5.5,6,8,10,15,20,30,50",
        help="Comma-separated impact parameters to sweep.",
    )
    parser.add_argument("--phi-max", type=float, default=18.0)
    parser.add_argument("--num-points", type=int, default=12000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Output path for the CSV metrics table.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def save_deflection_csv(trajectories, output_path: pathlib.Path) -> None:
    """Save numerical and weak-field deflection data as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "b",
            "alpha_num",
            "alpha_weak",
            "alpha_minus_weak",
            "alpha_ratio",
            "status",
        ])
        for trajectory in trajectories:
            alpha_weak = 4.0 / trajectory.impact_parameter
            if trajectory.deflection_angle is None:
                writer.writerow([
                    f"{trajectory.impact_parameter:.12g}",
                    "",
                    f"{alpha_weak:.12g}",
                    "",
                    "",
                    trajectory.status,
                ])
                continue

            alpha_minus_weak = trajectory.deflection_angle - alpha_weak
            alpha_ratio = trajectory.deflection_angle / alpha_weak
            writer.writerow([
                f"{trajectory.impact_parameter:.12g}",
                f"{trajectory.deflection_angle:.12g}",
                f"{alpha_weak:.12g}",
                f"{alpha_minus_weak:.12g}",
                f"{alpha_ratio:.12g}",
                trajectory.status,
            ])


def main() -> None:
    args = parse_args()
    b_values = parse_b_values(args.b_values)
    trajectories = integrate_many_null_geodesics(
        b_values,
        phi_max=args.phi_max,
        num_points=args.num_points,
    )

    escaped = [trajectory for trajectory in trajectories if trajectory.deflection_angle is not None]
    escaped_b = np.array([trajectory.impact_parameter for trajectory in escaped], dtype=float)
    numerical_alpha = np.array([trajectory.deflection_angle for trajectory in escaped], dtype=float)
    weak_field_alpha = 4.0 / escaped_b

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(escaped_b, numerical_alpha, "o-", linewidth=2.0, label="Numerical geodesic")
    ax.plot(escaped_b, weak_field_alpha, "--", linewidth=1.8, label="Weak-field 4/b")
    ax.axvline(B_CRIT, color="black", linestyle=":", linewidth=1.2, label=f"b_crit={B_CRIT:.3f}")
    ax.set_xlabel("Impact parameter b")
    ax.set_ylabel("Deflection angle alpha [rad]")
    ax.set_title("Schwarzschild Light Deflection vs Impact Parameter")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = pathlib.Path(args.output) if args.output else OUTPUT_DIR / "null_geodesic_deflection.png"
    csv_output_path = pathlib.Path(args.csv_output) if args.csv_output else METRICS_DIR / "null_geodesic_deflection.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    save_deflection_csv(trajectories, csv_output_path)
    print(f"Saved {output_path}")
    print(f"Saved {csv_output_path}")
    print("b, alpha_num, alpha_weak, ratio, status")
    for trajectory in trajectories:
        if trajectory.deflection_angle is None:
            print(f"{trajectory.impact_parameter:.6f}, nan, {4.0 / trajectory.impact_parameter:.6f}, nan, {trajectory.status}")
            continue
        alpha_weak = 4.0 / trajectory.impact_parameter
        ratio = trajectory.deflection_angle / alpha_weak
        print(
            f"{trajectory.impact_parameter:.6f}, {trajectory.deflection_angle:.6f}, "
            f"{alpha_weak:.6f}, {ratio:.6f}, {trajectory.status}"
        )

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
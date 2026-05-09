"""Run v0.2: metrics, outcome analysis, and velocity multiplier sweep."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.simulation import SimulationConfig, run_experiment
from src.sweeps import run_velocity_multiplier_sweep
from src.visualization import (
    save_animation,
    save_metrics_plots,
    save_outcome_sweep_plot,
    save_trajectory_plot,
)


def parse_velocity_values(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=900)
    parser.add_argument("--num-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--radius-min", type=float, default=4.0)
    parser.add_argument("--radius-max", type=float, default=18.0)
    parser.add_argument("--velocity-std", type=float, default=0.13)
    parser.add_argument("--radial-velocity-std", type=float, default=0.02)
    parser.add_argument("--velocity-values", default="0.7,0.9,1.0,1.1,1.3")
    parser.add_argument("--escape-radius", type=float, default=20.0)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--animation", default="outputs/animations/v02_representative_particles.gif")
    parser.add_argument("--no-animation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    velocity_values = parse_velocity_values(args.velocity_values)
    base_config = SimulationConfig(
        num_particles=args.num_particles,
        num_steps=args.num_steps,
        dt=args.dt,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        velocity_multiplier_std=args.velocity_std,
        radial_velocity_std=args.radial_velocity_std,
        escape_radius=args.escape_radius,
        save_every=args.save_every,
        device=args.device,
        seed=args.seed,
    )

    representative_config = replace(base_config, velocity_multiplier_mean=1.0)
    representative = run_experiment(representative_config)
    metric_paths = save_metrics_plots(representative, args.output_dir)
    trajectory_path = save_trajectory_plot(
        representative,
        Path(args.output_dir) / "v02_representative_trajectories.png",
    )
    animation_path = None
    if not args.no_animation:
        animation_path = save_animation(representative, Path(args.animation))

    sweep_results = run_velocity_multiplier_sweep(base_config, velocity_values)
    outcome_path = save_outcome_sweep_plot(
        sweep_results,
        Path(args.output_dir) / "v02_outcome_fractions_by_velocity.png",
    )

    print(f"Device: {representative.device}")
    print(f"Saved active-count plot: {metric_paths['active_count']}")
    print(f"Saved swallowed-fraction plot: {metric_paths['swallowed_fraction']}")
    print(f"Saved outcome sweep plot: {outcome_path}")
    print(f"Saved representative trajectories: {trajectory_path}")
    if animation_path is not None:
        print(f"Saved representative animation: {animation_path}")
    print("Outcome fractions:")
    for item in sweep_results:
        fractions = item.result.outcome_fractions
        print(
            f"  v={item.velocity_multiplier:.2f}: "
            f"swallowed={fractions['swallowed']:.3f}, "
            f"orbiting={fractions['orbiting']:.3f}, "
            f"escaped={fractions['escaped']:.3f}"
        )


if __name__ == "__main__":
    main()

"""Run v0.4: pseudo-relativistic accretion disk visuals and region metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.metrics import save_metrics_csv
from src.simulation import SimulationConfig, run_experiment
from src.visualization import (
    save_accretion_disk_trajectory_plot,
    save_animation,
    save_region_fraction_plot,
)


def log(message: str) -> None:
    print(f"[v0.4] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=12000)
    parser.add_argument("--num-steps", type=int, default=2200)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--radius-min", type=float, default=6.0)
    parser.add_argument("--radius-max", type=float, default=40.0)
    parser.add_argument("--velocity-multiplier", type=float, default=0.985)
    parser.add_argument("--velocity-noise", type=float, default=0.055)
    parser.add_argument("--radial-noise", type=float, default=0.012)
    parser.add_argument("--escape-radius", type=float, default=65.0)
    parser.add_argument("--snapshot-interval", type=int, default=20)
    parser.add_argument("--max-record-particles", type=int, default=2000)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--color-by", choices=["temperature", "radius", "speed"], default="temperature")
    parser.add_argument("--figure", default="outputs/figures/v04_accretion_disk_trajectories.png")
    parser.add_argument("--region-plot", default="outputs/figures/v04_region_fractions.png")
    parser.add_argument("--metrics-csv", default="outputs/metrics/pseudo_relativistic_disk_metrics.csv")
    parser.add_argument("--animation", default="outputs/animations/v04_accretion_disk.gif")
    parser.add_argument("--save-animation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("Building pseudo-relativistic disk config.")
    config = SimulationConfig(
        num_particles=args.num_particles,
        num_steps=args.num_steps,
        dt=args.dt,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        velocity_multiplier_mean=args.velocity_multiplier,
        velocity_multiplier_std=args.velocity_noise,
        radial_velocity_std=args.radial_noise,
        initialization_mode="accretion_disk",
        escape_radius=args.escape_radius,
        snapshot_interval=args.snapshot_interval,
        record_trajectory=True,
        record_metrics=True,
        max_record_particles=args.max_record_particles,
        device=args.device,
        seed=args.seed,
    )

    log(
        f"Running CUDA-aware disk simulation: N={config.num_particles}, "
        f"steps={config.num_steps}, snapshots every {config.snapshot_interval} steps, "
        f"recording up to {config.max_record_particles} particles."
    )
    result = run_experiment(config)
    log(f"Simulation finished on device: {result.device}.")
    log(f"Outcome fractions: {result.outcome_fractions}.")

    log(f"Saving hot-disk trajectory plot to {args.figure}.")
    figure_path = save_accretion_disk_trajectory_plot(
        result,
        Path(args.figure),
        color_by=args.color_by,
    )

    log(f"Saving region-fraction plot to {args.region_plot}.")
    region_path = save_region_fraction_plot(result, Path(args.region_plot))

    log(f"Saving metrics CSV to {args.metrics_csv}.")
    metrics_path = save_metrics_csv(result.times, result.metrics, Path(args.metrics_csv))

    animation_path = None
    if args.save_animation:
        log(f"Saving optional animation to {args.animation}.")
        animation_path = save_animation(
            result,
            Path(args.animation),
            fps=30,
            trail_length=45,
            title="v0.4 pseudo-relativistic accretion disk",
        )

    log(f"Saved trajectory plot: {figure_path}.")
    log(f"Saved region plot: {region_path}.")
    log(f"Saved metrics CSV: {metrics_path}.")
    if animation_path is not None:
        log(f"Saved animation: {animation_path}.")


if __name__ == "__main__":
    main()

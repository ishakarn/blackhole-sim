"""Run v0.1: 2D Newtonian black-hole-like particle simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation import SimulationConfig, run_simulation
from src.visualization import save_animation, save_trajectory_plot


def log(message: str) -> None:
    print(f"[v0.1] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=900)
    parser.add_argument("--num-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--radius-min", type=float, default=4.0)
    parser.add_argument("--radius-max", type=float, default=18.0)
    parser.add_argument("--velocity-mean", type=float, default=0.97)
    parser.add_argument("--velocity-std", type=float, default=0.13)
    parser.add_argument("--radial-velocity-std", type=float, default=0.02)
    parser.add_argument("--escape-radius", type=float, default=20.0)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--snapshot-interval", type=int, default=None)
    parser.add_argument("--max-record-particles", type=int, default=None)
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--figure", default="outputs/figures/v01_trajectories.png")
    parser.add_argument("--animation", default="outputs/animations/v01_particles.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("Building simulation config.")
    config = SimulationConfig(
        num_particles=args.num_particles,
        num_steps=args.num_steps,
        dt=args.dt,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        velocity_multiplier_mean=args.velocity_mean,
        velocity_multiplier_std=args.velocity_std,
        radial_velocity_std=args.radial_velocity_std,
        escape_radius=args.escape_radius,
        save_every=args.save_every,
        snapshot_interval=args.snapshot_interval,
        record_trajectory=not args.no_trajectory,
        max_record_particles=args.max_record_particles,
        device=args.device,
        seed=args.seed,
    )

    log(
        f"Running simulation: N={config.num_particles}, steps={config.num_steps}, "
        f"device={config.device}, snapshot_interval={config.snapshot_interval or config.save_every}."
    )
    result = run_simulation(config)
    log(f"Simulation finished on device: {result.device}.")
    log(f"Active particles: {int(result.final_active.sum())}/{result.final_active.numel()}.")
    log(f"Outcome counts: {result.outcome_counts}.")

    if result.recorded_particle_count > 0:
        log(f"Saving trajectory plot to {args.figure}.")
        figure_path = save_trajectory_plot(result, Path(args.figure))
        log(f"Saved trajectory plot: {figure_path}.")
        if not args.no_animation:
            log(f"Saving animation to {args.animation}.")
            animation_path = save_animation(result, Path(args.animation))
            log(f"Saved animation: {animation_path}.")
    else:
        log("Skipped visualization because trajectory recording is disabled.")


if __name__ == "__main__":
    main()

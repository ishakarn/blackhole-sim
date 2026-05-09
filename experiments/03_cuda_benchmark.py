"""Run v0.3 CPU vs CUDA benchmarks for the particle simulator."""

from __future__ import annotations

import argparse

from src.benchmark import (
    benchmark_particle_counts,
    save_benchmark_csv,
    save_benchmark_plot,
)
from src.simulation import SimulationConfig


def log(message: str) -> None:
    print(f"[v0.3] {message}", flush=True)


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particle-counts",
        default="1000,10000,50000,100000,250000",
        help="Comma-separated particle counts to benchmark.",
    )
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--radius-min", type=float, default=4.0)
    parser.add_argument("--radius-max", type=float, default=18.0)
    parser.add_argument("--velocity-mean", type=float, default=0.97)
    parser.add_argument("--velocity-std", type=float, default=0.13)
    parser.add_argument("--radial-velocity-std", type=float, default=0.02)
    parser.add_argument("--escape-radius", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", default="outputs/benchmarks/cuda_benchmark.csv")
    parser.add_argument("--plot", default="outputs/benchmarks/cuda_benchmark.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    particle_counts = parse_int_list(args.particle_counts)
    log(f"Preparing benchmark sweep for particle counts: {particle_counts}.")
    log(f"Each run uses steps={args.num_steps}, dt={args.dt}.")
    base_config = SimulationConfig(
        num_steps=args.num_steps,
        dt=args.dt,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        velocity_multiplier_mean=args.velocity_mean,
        velocity_multiplier_std=args.velocity_std,
        radial_velocity_std=args.radial_velocity_std,
        escape_radius=args.escape_radius,
        record_trajectory=False,
        record_metrics=False,
        max_record_particles=0,
        seed=args.seed,
    )

    log("Starting CPU/CUDA benchmark runs.")
    results = benchmark_particle_counts(base_config, particle_counts, verbose=True)
    log("Benchmark runs finished.")

    log(f"Saving benchmark CSV to {args.csv}.")
    csv_path = save_benchmark_csv(results, args.csv)
    log(f"Saving benchmark plot to {args.plot}.")
    plot_path = save_benchmark_plot(results, args.plot)

    log(f"Saved benchmark CSV: {csv_path}.")
    log(f"Saved benchmark plot: {plot_path}.")
    print("Results:")
    for result in results:
        print(
            f"  {result.device:>4} | "
            f"N={result.num_particles:<7} "
            f"runtime={result.total_runtime_seconds:>8.3f}s "
            f"steps/s={result.steps_per_second:>8.1f} "
            f"updates/s={result.particle_updates_per_second:>12.1f} "
            f"active={result.final_active_fraction:.3f} "
            f"swallowed={result.final_swallowed_fraction:.3f} "
            f"escaped={result.final_escaped_fraction:.3f}"
        )


if __name__ == "__main__":
    main()

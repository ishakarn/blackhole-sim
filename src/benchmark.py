"""Benchmark utilities for CPU vs CUDA particle simulation runs."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .simulation import SimulationConfig, run_experiment


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    num_particles: int
    num_steps: int
    dt: float
    total_runtime_seconds: float
    steps_per_second: float
    particle_updates_per_second: float
    final_active_fraction: float
    final_swallowed_fraction: float
    final_escaped_fraction: float


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_one(
    config: SimulationConfig,
    device_name: str,
    verbose: bool = False,
) -> BenchmarkResult:
    device = torch.device(device_name)
    benchmark_config = replace(
        config,
        device=device_name,
        record_trajectory=False,
        record_metrics=False,
        max_record_particles=0,
    )

    if verbose:
        print(
            f"[benchmark] Starting {device_name} run: "
            f"N={config.num_particles}, steps={config.num_steps}",
            flush=True,
        )

    _synchronize_if_cuda(device)
    start = time.perf_counter()
    result = run_experiment(benchmark_config)
    _synchronize_if_cuda(device)
    elapsed = time.perf_counter() - start

    steps_per_second = config.num_steps / elapsed
    particle_updates = config.num_particles * config.num_steps
    particle_updates_per_second = particle_updates / elapsed

    if verbose:
        print(
            f"[benchmark] Finished {device_name} run: "
            f"{elapsed:.3f}s, {particle_updates_per_second:.1f} updates/s",
            flush=True,
        )

    return BenchmarkResult(
        device=str(result.device),
        num_particles=config.num_particles,
        num_steps=config.num_steps,
        dt=config.dt,
        total_runtime_seconds=elapsed,
        steps_per_second=steps_per_second,
        particle_updates_per_second=particle_updates_per_second,
        final_active_fraction=1.0 - result.outcome_fractions["swallowed"],
        final_swallowed_fraction=result.outcome_fractions["swallowed"],
        final_escaped_fraction=result.outcome_fractions["escaped"],
    )


def benchmark_devices(
    config: SimulationConfig,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """Run the same simulation on CPU and CUDA when CUDA is available."""

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif verbose:
        print("[benchmark] CUDA is not available; running CPU only.", flush=True)
    return [_benchmark_one(config, device, verbose=verbose) for device in devices]


def benchmark_particle_counts(
    base_config: SimulationConfig,
    num_particles_list: list[int],
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """Run CPU/CUDA benchmarks across particle counts."""

    results: list[BenchmarkResult] = []
    for num_particles in num_particles_list:
        if verbose:
            print(f"[benchmark] Particle-count sweep item: N={num_particles}", flush=True)
        config = replace(base_config, num_particles=num_particles)
        results.extend(benchmark_devices(config, verbose=verbose))
    return results


def save_benchmark_csv(
    results: list[BenchmarkResult],
    output_path: str | Path = "outputs/benchmarks/cuda_benchmark.csv",
) -> Path:
    """Save benchmark rows to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(results[0]).keys()) if results else list(BenchmarkResult.__dataclass_fields__)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return output_path


def save_benchmark_plot(
    results: list[BenchmarkResult],
    output_path: str | Path = "outputs/benchmarks/cuda_benchmark.png",
) -> Path:
    """Plot particle update throughput for CPU and CUDA."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for device in sorted({result.device for result in results}):
        device_results = [result for result in results if result.device == device]
        device_results.sort(key=lambda item: item.num_particles)
        ax.plot(
            [item.num_particles for item in device_results],
            [item.particle_updates_per_second for item in device_results],
            marker="o",
            linewidth=2,
            label=device,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Particle update throughput")
    ax.set_xlabel("number of particles")
    ax.set_ylabel("particle updates / second")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

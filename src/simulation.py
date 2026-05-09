"""Simulation orchestration for 2D black-hole-like test particles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import EVENT_HORIZON_RADIUS
from .initial_conditions import accretion_disk_particles, disk_particles
from .integrators import velocity_verlet_step
from .metrics import classify_outcomes, measure_step, summarize_outcomes


@dataclass(frozen=True)
class SimulationConfig:
    num_particles: int = 900
    num_steps: int = 1600
    dt: float = 0.035
    radius_min: float = 4.0
    radius_max: float = 18.0
    velocity_multiplier_mean: float = 0.97
    velocity_multiplier_std: float = 0.13
    radial_velocity_std: float = 0.02
    initialization_mode: str = "disk"
    horizon_radius: float = EVENT_HORIZON_RADIUS
    escape_radius: float = 20.0
    softening: float = 0.03
    save_every: int = 4
    snapshot_interval: int | None = None
    record_trajectory: bool = True
    record_metrics: bool = True
    max_record_particles: int | None = None
    device: str = "auto"
    seed: int | None = 7


@dataclass
class ExperimentResult:
    positions: torch.Tensor
    velocities: torch.Tensor
    active: torch.Tensor
    metrics: dict[str, torch.Tensor]
    outcome_counts: dict[str, int]
    outcome_fractions: dict[str, float]
    final_positions: torch.Tensor
    final_velocities: torch.Tensor
    final_active: torch.Tensor
    final_swallowed: torch.Tensor
    final_escaped: torch.Tensor
    final_orbiting: torch.Tensor
    dt: float
    snapshot_interval: int
    recorded_particle_count: int
    device: torch.device

    @property
    def times(self) -> torch.Tensor:
        if self.positions.numel() > 0:
            num_snapshots = self.positions.shape[0]
        elif self.metrics:
            num_snapshots = len(next(iter(self.metrics.values())))
        else:
            num_snapshots = 0
        return torch.arange(num_snapshots) * self.dt * self.snapshot_interval

    @property
    def save_every(self) -> int:
        return self.snapshot_interval


SimulationResult = ExperimentResult


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_experiment(config: SimulationConfig = SimulationConfig()) -> ExperimentResult:
    """Run the simulation with trajectories, metrics, and final outcomes."""

    device = resolve_device(config.device)
    snapshot_interval = config.snapshot_interval or config.save_every
    record_count = config.num_particles
    if config.max_record_particles is not None:
        record_count = min(record_count, config.max_record_particles)

    if config.initialization_mode == "accretion_disk":
        positions, velocities = accretion_disk_particles(
            num_particles=config.num_particles,
            radius_min=config.radius_min,
            radius_max=config.radius_max,
            velocity_multiplier=config.velocity_multiplier_mean,
            velocity_noise=config.velocity_multiplier_std,
            radial_noise=config.radial_velocity_std,
            device=device,
            seed=config.seed,
        )
    elif config.initialization_mode == "disk":
        positions, velocities = disk_particles(
            num_particles=config.num_particles,
            radius_min=config.radius_min,
            radius_max=config.radius_max,
            velocity_multiplier_mean=config.velocity_multiplier_mean,
            velocity_multiplier_std=config.velocity_multiplier_std,
            radial_velocity_std=config.radial_velocity_std,
            device=device,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown initialization_mode: {config.initialization_mode!r}")
    active = torch.ones(config.num_particles, dtype=torch.bool, device=device)

    saved_positions = []
    saved_velocities = []
    saved_active = []
    metric_history: dict[str, list[torch.Tensor]] = {
        "active_count": [],
        "swallowed_fraction": [],
        "plunge_fraction": [],
        "unstable_fraction": [],
        "stable_fraction": [],
        "escaped_fraction": [],
        "mean_radius": [],
        "mean_speed": [],
        "mean_radius_active": [],
        "mean_speed_active": [],
    }

    with torch.no_grad():
        for step in range(config.num_steps + 1):
            if step % snapshot_interval == 0:
                if config.record_trajectory and record_count > 0:
                    saved_positions.append(positions[:record_count].detach().cpu())
                    saved_velocities.append(velocities[:record_count].detach().cpu())
                    saved_active.append(active[:record_count].detach().cpu())
                if config.record_metrics:
                    step_metrics = measure_step(
                        positions,
                        velocities,
                        active,
                        config.num_particles,
                        escape_radius=config.escape_radius,
                    )
                    for name, value in step_metrics.items():
                        metric_history[name].append(value.detach().cpu())

            if step == config.num_steps:
                break

            positions, velocities, active = velocity_verlet_step(
                positions,
                velocities,
                active,
                dt=config.dt,
                horizon_radius=config.horizon_radius,
                softening=config.softening,
            )

    final_swallowed_device, final_escaped_device, final_orbiting_device = classify_outcomes(
        positions,
        active,
        escape_radius=config.escape_radius,
    )
    outcome_counts, outcome_fractions = summarize_outcomes(
        final_swallowed_device,
        final_escaped_device,
        final_orbiting_device,
    )
    final_positions = positions.detach().cpu()
    final_velocities = velocities.detach().cpu()
    final_active = active.detach().cpu()
    final_swallowed = final_swallowed_device.detach().cpu()
    final_escaped = final_escaped_device.detach().cpu()
    final_orbiting = final_orbiting_device.detach().cpu()

    return ExperimentResult(
        positions=torch.stack(saved_positions) if saved_positions else torch.empty((0, 0, 2)),
        velocities=torch.stack(saved_velocities) if saved_velocities else torch.empty((0, 0, 2)),
        active=torch.stack(saved_active) if saved_active else torch.empty((0, 0), dtype=torch.bool),
        metrics={
            name: torch.stack(values) if values else torch.empty((0,))
            for name, values in metric_history.items()
        },
        outcome_counts=outcome_counts,
        outcome_fractions=outcome_fractions,
        final_positions=final_positions,
        final_velocities=final_velocities,
        final_active=final_active,
        final_swallowed=final_swallowed,
        final_escaped=final_escaped,
        final_orbiting=final_orbiting,
        dt=config.dt,
        snapshot_interval=snapshot_interval,
        recorded_particle_count=record_count if config.record_trajectory else 0,
        device=device,
    )


def run_simulation(config: SimulationConfig = SimulationConfig()) -> SimulationResult:
    """Backward-compatible name for the v0.2 experiment runner."""

    return run_experiment(config)

"""Simulation orchestration for 2D black-hole-like test particles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import EVENT_HORIZON_RADIUS
from .initial_conditions import disk_particles
from .integrators import velocity_verlet_step


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
    horizon_radius: float = EVENT_HORIZON_RADIUS
    escape_radius: float = 20.0
    softening: float = 0.03
    save_every: int = 4
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
    dt: float
    save_every: int
    device: torch.device

    @property
    def times(self) -> torch.Tensor:
        return torch.arange(self.positions.shape[0]) * self.dt * self.save_every


SimulationResult = ExperimentResult


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _measure_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    active: torch.Tensor,
    num_particles: int,
) -> dict[str, torch.Tensor]:
    radii = torch.linalg.norm(positions, dim=1)
    speeds = torch.linalg.norm(velocities, dim=1)
    active_count = active.sum()

    if active_count.item() > 0:
        mean_radius = radii[active].mean()
        mean_speed = speeds[active].mean()
    else:
        mean_radius = torch.full((), float("nan"), device=positions.device)
        mean_speed = torch.full((), float("nan"), device=positions.device)

    return {
        "active_count": active_count.to(torch.float32),
        "swallowed_fraction": 1.0 - active_count.to(torch.float32) / float(num_particles),
        "mean_radius": mean_radius,
        "mean_speed": mean_speed,
    }


def _classify_outcomes(
    positions: torch.Tensor,
    active: torch.Tensor,
    horizon_radius: float,
    escape_radius: float,
) -> tuple[dict[str, int], dict[str, float]]:
    radii = torch.linalg.norm(positions, dim=1)
    swallowed = ~active
    escaped = active & (radii > escape_radius)
    orbiting = active & ~escaped

    total = positions.shape[0]
    counts = {
        "swallowed": int(swallowed.sum()),
        "escaped": int(escaped.sum()),
        "orbiting": int(orbiting.sum()),
    }
    fractions = {name: count / total for name, count in counts.items()}

    if bool((swallowed & (radii > horizon_radius)).any()):
        counts["swallowed_outside_horizon"] = int((swallowed & (radii > horizon_radius)).sum())
        fractions["swallowed_outside_horizon"] = counts["swallowed_outside_horizon"] / total

    return counts, fractions


def run_experiment(config: SimulationConfig = SimulationConfig()) -> ExperimentResult:
    """Run the simulation with trajectories, metrics, and final outcomes."""

    device = resolve_device(config.device)
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
    active = torch.ones(config.num_particles, dtype=torch.bool, device=device)

    saved_positions = []
    saved_velocities = []
    saved_active = []
    metric_history: dict[str, list[torch.Tensor]] = {
        "active_count": [],
        "swallowed_fraction": [],
        "mean_radius": [],
        "mean_speed": [],
    }

    for step in range(config.num_steps + 1):
        if step % config.save_every == 0:
            saved_positions.append(positions.detach().cpu())
            saved_velocities.append(velocities.detach().cpu())
            saved_active.append(active.detach().cpu())
            step_metrics = _measure_step(positions, velocities, active, config.num_particles)
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

    final_positions = positions.detach().cpu()
    final_velocities = velocities.detach().cpu()
    final_active = active.detach().cpu()
    outcome_counts, outcome_fractions = _classify_outcomes(
        final_positions,
        final_active,
        horizon_radius=config.horizon_radius,
        escape_radius=config.escape_radius,
    )

    return ExperimentResult(
        positions=torch.stack(saved_positions),
        velocities=torch.stack(saved_velocities),
        active=torch.stack(saved_active),
        metrics={name: torch.stack(values) for name, values in metric_history.items()},
        outcome_counts=outcome_counts,
        outcome_fractions=outcome_fractions,
        final_positions=final_positions,
        final_velocities=final_velocities,
        final_active=final_active,
        dt=config.dt,
        save_every=config.save_every,
        device=device,
    )


def run_simulation(config: SimulationConfig = SimulationConfig()) -> SimulationResult:
    """Backward-compatible name for the v0.2 experiment runner."""

    return run_experiment(config)

"""Parameter sweeps for experiment analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .simulation import ExperimentResult, SimulationConfig, run_experiment


@dataclass
class SweepResult:
    velocity_multiplier: float
    result: ExperimentResult


def run_velocity_multiplier_sweep(
    base_config: SimulationConfig,
    velocity_multipliers: list[float],
) -> list[SweepResult]:
    """Run one experiment per mean tangential velocity multiplier."""

    results = []
    for velocity_multiplier in velocity_multipliers:
        config = replace(
            base_config,
            velocity_multiplier_mean=velocity_multiplier,
        )
        results.append(
            SweepResult(
                velocity_multiplier=velocity_multiplier,
                result=run_experiment(config),
            )
        )
    return results


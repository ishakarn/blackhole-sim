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
    verbose: bool = False,
) -> list[SweepResult]:
    """Run one experiment per mean tangential velocity multiplier."""

    results = []
    for velocity_multiplier in velocity_multipliers:
        if verbose:
            print(f"[sweep] Running velocity_multiplier_mean={velocity_multiplier:.3f}", flush=True)
        config = replace(
            base_config,
            velocity_multiplier_mean=velocity_multiplier,
        )
        result = run_experiment(config)
        if verbose:
            fractions = result.outcome_fractions
            print(
                f"[sweep] Finished v={velocity_multiplier:.3f}: "
                f"swallowed={fractions['swallowed']:.3f}, "
                f"orbiting={fractions['orbiting']:.3f}, "
                f"escaped={fractions['escaped']:.3f}",
                flush=True,
            )
        results.append(
            SweepResult(
                velocity_multiplier=velocity_multiplier,
                result=result,
            )
        )
    return results

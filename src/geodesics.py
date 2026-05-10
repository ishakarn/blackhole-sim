"""Schwarzschild null geodesic utilities in the equatorial plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import B_CRIT, EVENT_HORIZON_RADIUS, M, PHOTON_SPHERE_RADIUS


@dataclass(slots=True)
class NullGeodesicTrajectory:
    """Integrated equatorial null geodesic parameterized by azimuth."""

    impact_parameter: float
    phi: np.ndarray
    u: np.ndarray
    r: np.ndarray
    x: np.ndarray
    y: np.ndarray
    status: str


def schwarzschild_null_rhs(phi: float, state: np.ndarray) -> np.ndarray:
    """Return d/dphi [u, du/dphi] for d²u/dφ² + u = 3Mu²."""
    del phi
    u, du_dphi = state
    d2u_dphi2 = 3.0 * M * u * u - u
    return np.array([du_dphi, d2u_dphi2], dtype=float)


def rk4_step(ode, phi: float, state: np.ndarray, step: float) -> np.ndarray:
    """Advance one RK4 step for a first-order ODE system."""
    k1 = ode(phi, state)
    k2 = ode(phi + 0.5 * step, state + 0.5 * step * k1)
    k3 = ode(phi + 0.5 * step, state + 0.5 * step * k2)
    k4 = ode(phi + step, state + step * k3)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def impact_parameter_to_initial_conditions(impact_parameter: float) -> np.ndarray:
    """Initial state at infinity for an incoming ray.

    In flat space, u = sin(phi) / b. Starting at phi = 0 gives
    u(0) = 0 and du/dphi(0) = 1 / b.
    """
    return np.array([0.0, 1.0 / impact_parameter], dtype=float)


def polar_to_cartesian(phi: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar trajectory to Cartesian coordinates."""
    x = -r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def classify_trajectory(u_values: np.ndarray) -> str:
    """Classify a trajectory from its reciprocal radius history."""
    positive = u_values > 0.0
    if not np.any(positive):
        return "escaped"

    finite_r = np.empty_like(u_values)
    finite_r.fill(np.inf)
    finite_r[positive] = 1.0 / u_values[positive]
    if np.any(finite_r <= EVENT_HORIZON_RADIUS):
        return "captured"

    if len(u_values) >= 3 and positive[-1] and u_values[-1] > u_values[-2]:
        return "incomplete"
    return "escaped"


def integrate_null_geodesic(
    impact_parameter: float,
    phi_max: float = 12.0,
    num_points: int = 6000,
    horizon_radius: float = EVENT_HORIZON_RADIUS,
) -> NullGeodesicTrajectory:
    """Integrate an equatorial Schwarzschild null geodesic in azimuth.

    Rays with b < b_crit are expected to be captured, while those with
    b > b_crit escape after a finite deflection.
    """
    phi_values = np.linspace(0.0, phi_max, num_points)
    step = phi_values[1] - phi_values[0]

    state = impact_parameter_to_initial_conditions(impact_parameter)
    states = [state.copy()]
    used_phi = [phi_values[0]]
    status = "escaped"

    for phi in phi_values[:-1]:
        state = rk4_step(schwarzschild_null_rhs, phi, state, step)
        u = state[0]

        if u <= 0.0 and phi > 0.0:
            status = "escaped"
            break

        if u > 0.0 and (1.0 / u) <= horizon_radius:
            status = "captured"
            states.append(state.copy())
            used_phi.append(phi + step)
            break

        states.append(state.copy())
        used_phi.append(phi + step)
    else:
        status = "incomplete"

    used_phi_array = np.asarray(used_phi)
    state_array = np.asarray(states)
    u_values = np.clip(state_array[:, 0], 0.0, None)

    r_values = np.full_like(u_values, np.inf)
    positive = u_values > 0.0
    r_values[positive] = 1.0 / u_values[positive]
    r_values = np.clip(r_values, 0.0, 200.0)

    if status == "incomplete":
        status = classify_trajectory(u_values)

    x_values, y_values = polar_to_cartesian(used_phi_array, r_values)
    return NullGeodesicTrajectory(
        impact_parameter=impact_parameter,
        phi=used_phi_array,
        u=u_values,
        r=r_values,
        x=x_values,
        y=y_values,
        status=status,
    )


def integrate_many_null_geodesics(
    impact_parameters: list[float],
    phi_max: float = 12.0,
    num_points: int = 6000,
) -> list[NullGeodesicTrajectory]:
    """Integrate a batch of impact parameters."""
    return [
        integrate_null_geodesic(b, phi_max=phi_max, num_points=num_points)
        for b in impact_parameters
    ]


__all__ = [
    "B_CRIT",
    "EVENT_HORIZON_RADIUS",
    "PHOTON_SPHERE_RADIUS",
    "NullGeodesicTrajectory",
    "classify_trajectory",
    "impact_parameter_to_initial_conditions",
    "integrate_many_null_geodesics",
    "integrate_null_geodesic",
    "polar_to_cartesian",
    "rk4_step",
    "schwarzschild_null_rhs",
]
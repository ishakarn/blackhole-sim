"""Live simulation state for real-time particle rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import EVENT_HORIZON_RADIUS
from .initial_conditions import accretion_disk_particles
from .integrators import velocity_verlet_step
from .metrics import classify_regions, measure_step
from .simulation import resolve_device


COLOR_MODES = ("radius", "speed", "temperature")


@dataclass
class LiveSimulationConfig:
    num_particles: int = 50_000
    render_particles: int = 10_000
    physics_steps_per_frame: int = 5
    dt: float = 0.005
    radius_min: float = 6.0
    radius_max: float = 40.0
    velocity_multiplier: float = 0.985
    velocity_noise: float = 0.055
    radial_noise: float = 0.012
    escape_radius: float = 65.0
    softening: float = 0.03
    device: str = "auto"
    color_mode: str = "temperature"
    enable_trails: bool = False
    trail_length: int = 40
    seed: int | None = 17
    enable_injection: bool = False


@dataclass
class RenderFrame:
    positions: np.ndarray
    colors: np.ndarray
    active: np.ndarray
    metrics: dict[str, float]


class LiveSimulationState:
    """Owns full particle state on CPU/CUDA and exposes small render frames."""

    def __init__(self, config: LiveSimulationConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.color_mode = config.color_mode
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {COLOR_MODES}.")

        self.render_count = min(config.render_particles, config.num_particles)
        self.render_indices = torch.arange(self.render_count, device=self.device)
        self.frame_index = 0
        self.reset()

    def reset(self) -> None:
        self.positions, self.velocities = accretion_disk_particles(
            num_particles=self.config.num_particles,
            radius_min=self.config.radius_min,
            radius_max=self.config.radius_max,
            velocity_multiplier=self.config.velocity_multiplier,
            velocity_noise=self.config.velocity_noise,
            radial_noise=self.config.radial_noise,
            device=self.device,
            seed=self.config.seed,
        )
        self.active = torch.ones(self.config.num_particles, dtype=torch.bool, device=self.device)
        self.frame_index = 0

    def step(self) -> None:
        with torch.no_grad():
            for _ in range(self.config.physics_steps_per_frame):
                self.positions, self.velocities, self.active = velocity_verlet_step(
                    self.positions,
                    self.velocities,
                    self.active,
                    dt=self.config.dt,
                    horizon_radius=EVENT_HORIZON_RADIUS,
                    softening=self.config.softening,
                )
                if self.config.enable_injection:
                    self._inject_replaceable_particles()
        self.frame_index += 1

    def cycle_color_mode(self) -> str:
        index = COLOR_MODES.index(self.color_mode)
        self.color_mode = COLOR_MODES[(index + 1) % len(COLOR_MODES)]
        return self.color_mode

    def render_frame(self, fps: float = 0.0) -> RenderFrame:
        with torch.no_grad():
            render_positions = self.positions[self.render_indices]
            render_velocities = self.velocities[self.render_indices]
            render_active = self.active[self.render_indices]
            radii = torch.linalg.norm(render_positions, dim=1)
            speeds = torch.linalg.norm(render_velocities, dim=1)

            if self.color_mode == "radius":
                colors = radii
            elif self.color_mode == "speed":
                colors = speeds
            else:
                colors = self._temperature_proxy(radii)

            metrics = self._metrics(fps)

        return RenderFrame(
            positions=render_positions.detach().cpu().numpy(),
            colors=colors.detach().cpu().numpy(),
            active=render_active.detach().cpu().numpy(),
            metrics=metrics,
        )

    def _metrics(self, fps: float) -> dict[str, float]:
        step_metrics = measure_step(
            self.positions,
            self.velocities,
            self.active,
            self.config.num_particles,
            escape_radius=self.config.escape_radius,
        )
        regions = classify_regions(
            self.positions,
            self.active,
            escape_radius=self.config.escape_radius,
        )
        active_count = int(step_metrics["active_count"].item())
        swallowed_count = int(regions["swallowed"].sum().item())
        escaped_count = int(regions["escaped"].sum().item())
        return {
            "active_count": float(active_count),
            "swallowed_count": float(swallowed_count),
            "swallowed_fraction": float(step_metrics["swallowed_fraction"].item()),
            "escaped_count": float(escaped_count),
            "escaped_fraction": float(step_metrics["escaped_fraction"].item()),
            "mean_radius_active": float(step_metrics["mean_radius_active"].item()),
            "mean_speed_active": float(step_metrics["mean_speed_active"].item()),
            "fps": float(fps),
            "device": str(self.device),
            "num_particles": float(self.config.num_particles),
            "render_particles": float(self.render_count),
        }

    def _temperature_proxy(self, radii: torch.Tensor) -> torch.Tensor:
        safe_radii = torch.clamp(radii, min=EVENT_HORIZON_RADIUS)
        temp = safe_radii.pow(-0.75)
        temp_min = torch.amin(temp)
        temp_max = torch.amax(temp)
        return (temp - temp_min) / torch.clamp(temp_max - temp_min, min=1.0e-8)

    def _inject_replaceable_particles(self) -> None:
        radii = torch.linalg.norm(self.positions, dim=1)
        replace_mask = (~self.active) | (radii > self.config.escape_radius)
        count = int(replace_mask.sum().item())
        if count == 0:
            return

        new_positions, new_velocities = accretion_disk_particles(
            num_particles=count,
            radius_min=max(self.config.radius_min, self.config.radius_max * 0.85),
            radius_max=self.config.radius_max,
            velocity_multiplier=self.config.velocity_multiplier,
            velocity_noise=self.config.velocity_noise,
            radial_noise=self.config.radial_noise,
            device=self.device,
            seed=None,
        )
        self.positions[replace_mask] = new_positions
        self.velocities[replace_mask] = new_velocities
        self.active[replace_mask] = True

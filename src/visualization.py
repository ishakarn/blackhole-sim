"""Matplotlib plotting and animation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .constants import EVENT_HORIZON_RADIUS, ISCO_RADIUS, PHOTON_SPHERE_RADIUS
from .simulation import SimulationResult
from .sweeps import SweepResult


def _set_axes(ax: plt.Axes, extent: float) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_facecolor("#050608")
    ax.figure.patch.set_facecolor("#050608")
    ax.tick_params(colors="#b8c0cc")
    for spine in ax.spines.values():
        spine.set_color("#303640")


def add_black_hole_markers(
    ax: plt.Axes,
    show_photon_sphere: bool = True,
    show_isco: bool = True,
    label_regions: bool = False,
) -> None:
    horizon = plt.Circle((0, 0), EVENT_HORIZON_RADIUS, color="black", zorder=5)
    ax.add_patch(horizon)

    glow = plt.Circle((0, 0), EVENT_HORIZON_RADIUS, color="#111827", alpha=0.75, zorder=4)
    ax.add_patch(glow)

    if show_photon_sphere:
        ax.add_patch(
            plt.Circle(
                (0, 0),
                PHOTON_SPHERE_RADIUS,
                fill=False,
                linestyle="--",
                linewidth=1.0,
                edgecolor="#facc15",
                alpha=0.55,
            )
        )
    if show_isco:
        ax.add_patch(
            plt.Circle(
                (0, 0),
                ISCO_RADIUS,
                fill=False,
                linestyle=":",
                linewidth=1.0,
                edgecolor="#60a5fa",
                alpha=0.55,
            )
        )
    if label_regions:
        label_style = {
            "color": "#d1d5db",
            "fontsize": 8,
            "ha": "left",
            "va": "center",
            "alpha": 0.85,
        }
        ax.text(EVENT_HORIZON_RADIUS + 0.2, 0.0, "r=2 horizon", **label_style)
        if show_photon_sphere:
            ax.text(PHOTON_SPHERE_RADIUS + 0.2, 0.35, "r=3 photon sphere", **label_style)
        if show_isco:
            ax.text(ISCO_RADIUS + 0.2, -0.35, "r=6 ISCO", **label_style)


def temperature_proxy_from_radius(radii: np.ndarray) -> np.ndarray:
    """Return normalized disk temperature proxy T ~ r^(-3/4)."""

    safe_radii = np.clip(radii, EVENT_HORIZON_RADIUS, None)
    temp = safe_radii ** (-0.75)
    temp_min = float(np.nanmin(temp))
    temp_max = float(np.nanmax(temp))
    if temp_max <= temp_min:
        return np.ones_like(temp)
    return (temp - temp_min) / (temp_max - temp_min)


def save_trajectory_plot(
    result: SimulationResult,
    output_path: str | Path = "outputs/figures/v01_trajectories.png",
    trail_stride: int = 3,
    max_trails: int = 260,
) -> Path:
    """Save a static plot of particle trails colored by final speed."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = result.positions.numpy()
    active = result.active.numpy()
    speeds = np.linalg.norm(result.velocities[-1].numpy(), axis=1)
    extent = max(8.0, float(np.nanmax(np.abs(positions[0]))) * 1.1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    _set_axes(ax, extent)
    add_black_hole_markers(ax)

    particle_indices = np.linspace(
        0, positions.shape[1] - 1, min(max_trails, positions.shape[1]), dtype=int
    )
    segments = []
    colors = []
    for idx in particle_indices:
        path = positions[::trail_stride, idx]
        alive = active[::trail_stride, idx]
        visible_path = path[alive]
        if len(visible_path) > 1:
            segments.append(visible_path)
            colors.append(speeds[idx])

    trails = LineCollection(segments, cmap="magma", linewidths=0.65, alpha=0.5)
    trails.set_array(np.asarray(colors))
    ax.add_collection(trails)

    final_positions = result.final_positions.numpy()
    final_active = result.final_active.numpy()
    radii = np.linalg.norm(final_positions, axis=1)
    scatter = ax.scatter(
        final_positions[final_active, 0],
        final_positions[final_active, 1],
        c=radii[final_active],
        cmap="viridis",
        s=5,
        alpha=0.9,
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Final radius")
    ax.set_title("2D Newtonian black-hole-like particle simulation", color="#f8fafc")
    ax.set_xlabel("x", color="#d1d5db")
    ax.set_ylabel("y", color="#d1d5db")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def save_animation(
    result: SimulationResult,
    output_path: str | Path = "outputs/animations/v01_particles.gif",
    fps: int = 30,
    trail_length: int = 35,
    title: str = "particle disk",
) -> Path:
    """Save an animated GIF of the particle simulation."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = result.positions.numpy()
    active = result.active.numpy()
    speeds = np.linalg.norm(result.velocities.numpy(), axis=2)
    extent = max(8.0, float(np.nanmax(np.abs(positions[0]))) * 1.1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    _set_axes(ax, extent)
    add_black_hole_markers(ax)
    ax.set_title(title, color="#f8fafc")

    trail_lines = LineCollection([], colors="#38bdf8", linewidths=0.35, alpha=0.25)
    ax.add_collection(trail_lines)
    scatter = ax.scatter([], [], c=[], cmap="plasma", s=4, alpha=0.9, linewidths=0)
    time_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        color="#d1d5db",
        ha="left",
        va="top",
    )

    def update(frame: int):
        alive = active[frame]
        current = positions[frame, alive]
        speed = speeds[frame, alive]
        scatter.set_offsets(current)
        scatter.set_array(speed)

        start = max(0, frame - trail_length)
        segments = []
        for particle_idx in np.where(alive)[0]:
            path = positions[start : frame + 1, particle_idx]
            path_active = active[start : frame + 1, particle_idx]
            visible = path[path_active]
            if len(visible) > 1:
                segments.append(visible)
        trail_lines.set_segments(segments)

        swallowed = result.final_active.numel() - int(active[frame].sum())
        time_text.set_text(f"t = {result.times[frame]:.2f}   swallowed = {swallowed}")
        return scatter, trail_lines, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=positions.shape[0],
        interval=1000 / fps,
        blit=True,
    )
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def save_metrics_plots(
    result: SimulationResult,
    output_dir: str | Path = "outputs/figures",
    prefix: str = "v02",
) -> dict[str, Path]:
    """Save active-count and swallowed-fraction plots for one experiment."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = result.times.numpy()
    paths = {
        "active_count": output_dir / f"{prefix}_active_count.png",
        "swallowed_fraction": output_dir / f"{prefix}_swallowed_fraction.png",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(times, result.metrics["active_count"].numpy(), color="#2563eb", linewidth=2)
    ax.set_title("Active particle count over time")
    ax.set_xlabel("time")
    ax.set_ylabel("active particles")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["active_count"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(times, result.metrics["swallowed_fraction"].numpy(), color="#dc2626", linewidth=2)
    ax.set_title("Swallowed fraction over time")
    ax.set_xlabel("time")
    ax.set_ylabel("swallowed fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["swallowed_fraction"])
    plt.close(fig)

    return paths


def save_outcome_sweep_plot(
    sweep_results: list[SweepResult],
    output_path: str | Path = "outputs/figures/v02_outcome_fractions_by_velocity.png",
) -> Path:
    """Save grouped final outcome fractions by velocity multiplier."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    velocity_multipliers = [item.velocity_multiplier for item in sweep_results]
    swallowed = [item.result.outcome_fractions["swallowed"] for item in sweep_results]
    escaped = [item.result.outcome_fractions["escaped"] for item in sweep_results]
    orbiting = [item.result.outcome_fractions["orbiting"] for item in sweep_results]

    x = np.arange(len(velocity_multipliers))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x - width, swallowed, width, label="swallowed", color="#111827")
    ax.bar(x, orbiting, width, label="orbiting", color="#2563eb")
    ax.bar(x + width, escaped, width, label="escaped", color="#f97316")
    ax.set_title("Final outcome fractions by velocity multiplier")
    ax.set_xlabel("velocity multiplier")
    ax.set_ylabel("fraction")
    ax.set_xticks(x, [f"{value:.1f}" for value in velocity_multipliers])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_region_fraction_plot(
    result: SimulationResult,
    output_path: str | Path = "outputs/figures/v04_region_fractions.png",
) -> Path:
    """Save pseudo-relativistic region fractions over time."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    times = result.times.numpy()
    series = {
        "swallowed": ("swallowed_fraction", "#111827"),
        "plunge": ("plunge_fraction", "#ef4444"),
        "unstable": ("unstable_fraction", "#f97316"),
        "stable": ("stable_fraction", "#2563eb"),
        "escaped": ("escaped_fraction", "#64748b"),
    }

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for label, (metric_name, color) in series.items():
        if metric_name in result.metrics and result.metrics[metric_name].numel() > 0:
            ax.plot(times, result.metrics[metric_name].numpy(), label=label, color=color, linewidth=2)

    ax.set_title("Pseudo-relativistic disk region fractions")
    ax.set_xlabel("time")
    ax.set_ylabel("particle fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_accretion_disk_trajectory_plot(
    result: SimulationResult,
    output_path: str | Path = "outputs/figures/v04_accretion_disk_trajectories.png",
    color_by: str = "temperature",
    trail_stride: int = 2,
    max_trails: int = 900,
) -> Path:
    """Save a glowing accretion-disk-style trajectory plot from recorded snapshots."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if result.positions.numel() == 0:
        raise ValueError("No recorded trajectory snapshots are available for plotting.")

    positions = result.positions.numpy()
    velocities = result.velocities.numpy()
    active = result.active.numpy()
    radii_over_time = np.linalg.norm(positions, axis=2)
    speeds_over_time = np.linalg.norm(velocities, axis=2)
    extent = max(10.0, float(np.nanmax(np.abs(positions[0]))) * 1.08)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
    _set_axes(ax, extent)
    add_black_hole_markers(ax, label_regions=True)

    particle_indices = np.linspace(
        0,
        positions.shape[1] - 1,
        min(max_trails, positions.shape[1]),
        dtype=int,
    )
    segments = []
    colors = []
    for idx in particle_indices:
        path = positions[::trail_stride, idx]
        alive = active[::trail_stride, idx]
        visible_path = path[alive]
        if len(visible_path) > 1:
            segments.append(visible_path)
            radius = radii_over_time[-1, idx]
            speed = speeds_over_time[-1, idx]
            if color_by == "radius":
                colors.append(radius)
            elif color_by == "speed":
                colors.append(speed)
            elif color_by == "temperature":
                colors.append(radius)
            else:
                raise ValueError("color_by must be 'temperature', 'radius', or 'speed'.")

    if color_by == "temperature":
        color_values = temperature_proxy_from_radius(np.asarray(colors))
        cmap = "inferno"
        color_label = "Temperature proxy"
    elif color_by == "radius":
        color_values = np.asarray(colors)
        cmap = "viridis"
        color_label = "Radius"
    else:
        color_values = np.asarray(colors)
        cmap = "plasma"
        color_label = "Speed"

    trails = LineCollection(segments, cmap=cmap, linewidths=0.55, alpha=0.45)
    trails.set_array(color_values)
    ax.add_collection(trails)

    final_positions = positions[-1]
    final_active = active[-1]
    final_radii = radii_over_time[-1]
    final_speeds = speeds_over_time[-1]
    if color_by == "temperature":
        scatter_values = temperature_proxy_from_radius(final_radii[final_active])
        cmap = "inferno"
    elif color_by == "radius":
        scatter_values = final_radii[final_active]
        cmap = "viridis"
    else:
        scatter_values = final_speeds[final_active]
        cmap = "plasma"

    scatter = ax.scatter(
        final_positions[final_active, 0],
        final_positions[final_active, 1],
        c=scatter_values,
        cmap=cmap,
        s=4,
        alpha=0.95,
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=color_label)
    ax.set_title("v0.4 pseudo-relativistic accretion disk", color="#f8fafc")
    ax.set_xlabel("x", color="#d1d5db")
    ax.set_ylabel("y", color="#d1d5db")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path

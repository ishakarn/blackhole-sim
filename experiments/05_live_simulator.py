"""Run v0.5: live real-time CUDA accretion disk simulator."""

from __future__ import annotations

import argparse
import time
from collections import deque

import matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None, help="Optional Matplotlib backend, e.g. QtAgg or TkAgg.")
    parser.add_argument("--num-particles", type=int, default=50_000)
    parser.add_argument("--render-particles", type=int, default=10_000)
    parser.add_argument("--physics-steps-per-frame", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--radius-min", type=float, default=6.0)
    parser.add_argument("--radius-max", type=float, default=40.0)
    parser.add_argument("--velocity-multiplier", type=float, default=0.985)
    parser.add_argument("--velocity-noise", type=float, default=0.055)
    parser.add_argument("--radial-noise", type=float, default=0.012)
    parser.add_argument("--escape-radius", type=float, default=65.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--color-mode", choices=["radius", "speed", "temperature"], default="temperature")
    parser.add_argument("--enable-trails", action="store_true")
    parser.add_argument("--trail-length", type=int, default=40)
    parser.add_argument("--trail-particles", type=int, default=1200)
    parser.add_argument("--enable-injection", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend:
        matplotlib.use(args.backend)

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from src.constants import EVENT_HORIZON_RADIUS, ISCO_RADIUS, PHOTON_SPHERE_RADIUS
    from src.live import LiveSimulationConfig, LiveSimulationState

    config = LiveSimulationConfig(
        num_particles=args.num_particles,
        render_particles=args.render_particles,
        physics_steps_per_frame=args.physics_steps_per_frame,
        dt=args.dt,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        velocity_multiplier=args.velocity_multiplier,
        velocity_noise=args.velocity_noise,
        radial_noise=args.radial_noise,
        escape_radius=args.escape_radius,
        device=args.device,
        color_mode=args.color_mode,
        enable_trails=args.enable_trails,
        trail_length=args.trail_length,
        seed=args.seed,
        enable_injection=args.enable_injection,
    )
    state = LiveSimulationState(config)

    paused = {"value": False}
    running = {"value": True}
    trails_enabled = {"value": config.enable_trails}
    trail_history: deque = deque(maxlen=config.trail_length)
    trail_particles = min(args.trail_particles, state.render_count)

    extent = max(config.radius_max * 1.08, config.escape_radius * 0.75)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_facecolor("#050608")
    fig.patch.set_facecolor("#050608")
    ax.tick_params(colors="#b8c0cc")
    for spine in ax.spines.values():
        spine.set_color("#303640")

    ax.add_patch(plt.Circle((0, 0), EVENT_HORIZON_RADIUS, color="black", zorder=5))
    ax.add_patch(
        plt.Circle(
            (0, 0),
            PHOTON_SPHERE_RADIUS,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor="#facc15",
            alpha=0.65,
        )
    )
    ax.add_patch(
        plt.Circle(
            (0, 0),
            ISCO_RADIUS,
            fill=False,
            linestyle=":",
            linewidth=1.2,
            edgecolor="#60a5fa",
            alpha=0.7,
        )
    )

    scatter = ax.scatter([], [], s=2, cmap="inferno", alpha=0.9, linewidths=0)
    trail_lines = LineCollection([], colors="#f97316", linewidths=0.35, alpha=0.18)
    ax.add_collection(trail_lines)
    text = ax.text(
        0.015,
        0.985,
        "",
        transform=ax.transAxes,
        color="#e5e7eb",
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
    )

    last_time = {"value": time.perf_counter()}

    def set_cmap() -> None:
        if state.color_mode == "radius":
            scatter.set_cmap("viridis")
        elif state.color_mode == "speed":
            scatter.set_cmap("plasma")
        else:
            scatter.set_cmap("inferno")

    def on_key(event) -> None:
        key = (event.key or "").lower()
        if key == " ":
            paused["value"] = not paused["value"]
        elif key == "r":
            state.reset()
            trail_history.clear()
        elif key == "t":
            trails_enabled["value"] = not trails_enabled["value"]
            if not trails_enabled["value"]:
                trail_lines.set_segments([])
        elif key == "c":
            state.cycle_color_mode()
            set_cmap()
        elif key == "i":
            state.config.enable_injection = not state.config.enable_injection
        elif key in {"q", "escape"}:
            running["value"] = False
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    set_cmap()
    print(
        "[v0.5] Controls: Space pause/resume, R reset, T trails, "
        "C color mode, I injection, Q/Esc quit.",
        flush=True,
    )
    print(f"[v0.5] Running on {state.device}; close the window or press Q to quit.", flush=True)

    while running["value"] and plt.fignum_exists(fig.number):
        now = time.perf_counter()
        fps = 1.0 / max(now - last_time["value"], 1.0e-6)
        last_time["value"] = now

        if not paused["value"]:
            state.step()

        frame = state.render_frame(fps=fps)
        alive = frame.active
        scatter.set_offsets(frame.positions[alive])
        scatter.set_array(frame.colors[alive])

        if trails_enabled["value"]:
            trail_positions = frame.positions[:trail_particles].copy()
            trail_alive = frame.active[:trail_particles].copy()
            trail_history.append((trail_positions, trail_alive))
            segments = []
            if len(trail_history) > 1:
                for particle_idx in range(trail_particles):
                    path = []
                    for positions, active in trail_history:
                        if active[particle_idx]:
                            path.append(positions[particle_idx])
                    if len(path) > 1:
                        segments.append(path)
            trail_lines.set_segments(segments)

        metrics = frame.metrics
        text.set_text(
            f"device: {metrics['device']}\n"
            f"particles: {int(metrics['num_particles'])}  rendered: {int(metrics['render_particles'])}\n"
            f"active: {int(metrics['active_count'])}\n"
            f"swallowed: {int(metrics['swallowed_count'])} ({metrics['swallowed_fraction']:.3f})\n"
            f"escaped: {int(metrics['escaped_count'])} ({metrics['escaped_fraction']:.3f})\n"
            f"mean r: {metrics['mean_radius_active']:.2f}  mean speed: {metrics['mean_speed_active']:.3f}\n"
            f"fps: {metrics['fps']:.1f}  steps/frame: {config.physics_steps_per_frame}\n"
            f"color: {state.color_mode}  trails: {trails_enabled['value']}  injection: {state.config.enable_injection}\n"
            f"{'PAUSED' if paused['value'] else ''}"
        )
        ax.set_title("v0.5 live CUDA accretion disk", color="#f8fafc")
        fig.canvas.draw_idle()
        plt.pause(0.001)

    print("[v0.5] Live simulator closed.", flush=True)


if __name__ == "__main__":
    main()

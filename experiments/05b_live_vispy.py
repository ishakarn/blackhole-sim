"""Run v0.5b: fast live VisPy/OpenGL accretion disk simulator."""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.constants import EVENT_HORIZON_RADIUS, ISCO_RADIUS, PHOTON_SPHERE_RADIUS
from src.live import COLOR_MODES, LiveSimulationConfig, LiveSimulationState


def parse_window_size(raw: str) -> tuple[int, int]:
    parts = raw.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window size must look like 1200x900")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
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
    parser.add_argument("--color-mode", choices=COLOR_MODES, default="temperature")
    parser.add_argument("--window-size", type=parse_window_size, default=(1200, 900))
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--metric-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-injection", action="store_true")
    return parser.parse_args()


def circle_points(radius: float, segments: int = 256) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=True, dtype=np.float32)
    return np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta),
        )
    )


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    span = max(max_value - min_value, 1.0e-8)
    return (values - min_value) / span


def scalar_to_rgba(values: np.ndarray, color_mode: str) -> np.ndarray:
    normalized = normalize(values.astype(np.float32, copy=False))
    alpha = np.ones_like(normalized)

    if color_mode == "radius":
        red = 0.15 + 0.2 * normalized
        green = 0.35 + 0.55 * normalized
        blue = 1.0 - 0.75 * normalized
    elif color_mode == "speed":
        red = 0.25 + 0.75 * normalized
        green = 0.15 + 0.35 * normalized
        blue = 0.85 - 0.55 * normalized
    else:
        red = 0.35 + 0.65 * normalized
        green = 0.08 + 0.75 * normalized**1.4
        blue = 0.02 + 0.18 * (1.0 - normalized)

    return np.column_stack((red, green, blue, alpha)).astype(np.float32)


def positions_to_xyz(positions: np.ndarray) -> np.ndarray:
    xyz = np.zeros((positions.shape[0], 3), dtype=np.float32)
    xyz[:, :2] = positions.astype(np.float32, copy=False)
    return xyz


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def main() -> None:
    args = parse_args()
    try:
        from vispy import app, scene
    except ImportError as exc:
        raise SystemExit(
            "VisPy is not installed in this environment. Install it with:\n"
            "  pip install vispy\n"
            "or:\n"
            "  conda install -c conda-forge vispy"
        ) from exc

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
        seed=args.seed,
        enable_injection=not args.no_injection,
    )
    state = LiveSimulationState(config)

    canvas = scene.SceneCanvas(
        keys="interactive",
        size=args.window_size,
        bgcolor="#050608",
        title="v0.5b VisPy live accretion disk",
        show=True,
    )
    view = canvas.central_widget.add_view()
    extent = max(config.radius_max * 1.08, config.escape_radius * 0.75)
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(-extent, extent), y=(-extent, extent))

    markers = scene.visuals.Markers(parent=view.scene)
    horizon = scene.visuals.Ellipse(
        center=(0, 0, 0),
        radius=(EVENT_HORIZON_RADIUS, EVENT_HORIZON_RADIUS),
        color=(0, 0, 0, 1),
        parent=view.scene,
    )
    horizon.order = 10
    scene.visuals.Line(
        circle_points(PHOTON_SPHERE_RADIUS),
        color=(1.0, 0.82, 0.18, 0.75),
        width=1.2,
        parent=view.scene,
    )
    scene.visuals.Line(
        circle_points(ISCO_RADIUS),
        color=(0.35, 0.65, 1.0, 0.75),
        width=1.2,
        parent=view.scene,
    )
    overlay = scene.Label(
        "",
        color="#e5e7eb",
        font_size=10,
        anchor_x="left",
        anchor_y="top",
    )
    overlay.parent = canvas.scene
    overlay.pos = (12, args.window_size[1] - 12)

    paused = {"value": False}
    running = {"value": True}
    show_help = {"value": True}
    point_size = {"value": args.point_size}
    frame_counter = {"value": 0}
    latest_metrics = {"text": ""}
    last_time = {"value": time.perf_counter()}

    def update_title(metrics: dict[str, float]) -> None:
        latest_metrics["text"] = (
            f"FPS {metrics['fps']:.1f} | "
            f"active {int(metrics['active_count'])}/{int(metrics['num_particles'])} | "
            f"swallowed {metrics['swallowed_fraction']:.3f} | "
            f"escaped {metrics['escaped_fraction']:.3f} | "
            f"mean r {metrics['mean_radius_active']:.2f} | "
            f"mean v {metrics['mean_speed_active']:.3f} | "
            f"{metrics['device']} | rendered {int(metrics['render_particles'])}"
        )
        canvas.title = "v0.5b VisPy live accretion disk | " + latest_metrics["text"]
        help_text = ""
        if show_help["value"]:
            help_text = (
                "\n\nControls:"
                "\nSpace pause | R reset | C color | I injection | H help"
                "\n+/- speed | ]/[ rendered particles | ./, point size"
                "\n0 reset view | Q/Esc quit"
            )
        overlay.text = (
            latest_metrics["text"]
            + f"\ncolor {state.color_mode} | steps/frame {config.physics_steps_per_frame} | dt {config.dt:.5f}"
            + f"\nrendered {state.render_count} | point size {point_size['value']:.1f}"
            + f"\ninjection {state.config.enable_injection} | {'PAUSED' if paused['value'] else 'running'}"
            + help_text
        )
        print("[v0.5b] " + latest_metrics["text"], flush=True)

    def reset_camera() -> None:
        view.camera.set_range(x=(-extent, extent), y=(-extent, extent))

    def change_speed(factor: float) -> None:
        config.physics_steps_per_frame = int(clamp(round(config.physics_steps_per_frame * factor), 1, 200))
        config.dt = clamp(config.dt * factor, 0.0002, 0.05)
        print(
            f"[v0.5b] speed changed: steps/frame={config.physics_steps_per_frame}, dt={config.dt:.5f}",
            flush=True,
        )

    def change_render_count(factor: float) -> None:
        old_count = state.render_count
        new_count = int(clamp(round(old_count * factor / 100.0) * 100, 100, config.num_particles))
        state.set_render_count(new_count)
        print(f"[v0.5b] render particles changed: {state.render_count}", flush=True)

    def change_point_size(delta: float) -> None:
        point_size["value"] = clamp(point_size["value"] + delta, 1.0, 12.0)
        print(f"[v0.5b] point size changed: {point_size['value']:.1f}", flush=True)

    def on_key_press(event) -> None:
        key = event.key.name.lower() if event.key is not None else ""
        text = (event.text or "").lower()
        if key == "space":
            paused["value"] = not paused["value"]
        elif key == "r":
            state.reset()
        elif key == "c":
            state.cycle_color_mode()
        elif key == "i":
            state.config.enable_injection = not state.config.enable_injection
            print(f"[v0.5b] injection changed: {state.config.enable_injection}", flush=True)
        elif key == "h":
            show_help["value"] = not show_help["value"]
        elif key in {"0", "key_0"} or text == "0":
            reset_camera()
        elif key in {"plus", "equal"} or text in {"+", "="}:
            change_speed(1.25)
        elif key in {"minus"} or text == "-":
            change_speed(0.80)
        elif key in {"bracketright"} or text == "]":
            change_render_count(1.25)
        elif key in {"bracketleft"} or text == "[":
            change_render_count(0.80)
        elif key in {"period"} or text == ".":
            change_point_size(0.5)
        elif key in {"comma"} or text == ",":
            change_point_size(-0.5)
        elif key in {"q", "escape"}:
            running["value"] = False
            canvas.close()

    def on_timer(event) -> None:
        if not running["value"]:
            return

        now = time.perf_counter()
        fps = 1.0 / max(now - last_time["value"], 1.0e-6)
        last_time["value"] = now

        if not paused["value"]:
            state.step()

        include_metrics = frame_counter["value"] % args.metric_interval == 0
        frame = state.render_frame(fps=fps, include_metrics=include_metrics)
        alive = frame.active
        markers.set_data(
            positions_to_xyz(frame.positions[alive]),
            face_color=scalar_to_rgba(frame.colors[alive], state.color_mode),
            edge_width=0,
            size=point_size["value"],
        )

        if include_metrics:
            update_title(frame.metrics)

        frame_counter["value"] += 1
        canvas.update()

    canvas.events.key_press.connect(on_key_press)
    print(
        "[v0.5b] Controls: Space pause, R reset, C color, I injection, +/- speed, "
        "]/[ render count, ./, point size, H help, 0 view, Q/Esc quit.",
        flush=True,
    )
    print(f"[v0.5b] Running on {state.device}. Rendering {state.render_count} of {config.num_particles}.", flush=True)

    timer = app.Timer(interval=0.0, connect=on_timer, start=True)
    app.run()
    timer.stop()
    print("[v0.5b] Live VisPy simulator closed.", flush=True)


if __name__ == "__main__":
    main()

"""Run v0.5c: polished live CUDA + VisPy accretion disk demo."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from src.constants import EVENT_HORIZON_RADIUS, ISCO_RADIUS, PHOTON_SPHERE_RADIUS
from src.live import COLOR_MODES, LiveSimulationConfig, LiveSimulationState


PRESETS = {
    "stable_disk": {
        "num_particles": 50_000,
        "render_particles": 10_000,
        "physics_steps_per_frame": 5,
        "dt": 0.005,
        "radius_min": 6.0,
        "radius_max": 40.0,
        "velocity_multiplier": 0.99,
        "velocity_noise": 0.035,
        "radial_noise": 0.006,
        "escape_radius": 70.0,
        "injection": True,
        "color_mode": "temperature",
    },
    "infall": {
        "num_particles": 50_000,
        "render_particles": 10_000,
        "physics_steps_per_frame": 6,
        "dt": 0.006,
        "radius_min": 4.2,
        "radius_max": 34.0,
        "velocity_multiplier": 0.84,
        "velocity_noise": 0.08,
        "radial_noise": 0.035,
        "escape_radius": 60.0,
        "injection": True,
        "color_mode": "temperature",
    },
    "hot_inner_disk": {
        "num_particles": 60_000,
        "render_particles": 12_000,
        "physics_steps_per_frame": 5,
        "dt": 0.005,
        "radius_min": 5.0,
        "radius_max": 32.0,
        "velocity_multiplier": 0.93,
        "velocity_noise": 0.06,
        "radial_noise": 0.02,
        "escape_radius": 60.0,
        "injection": True,
        "color_mode": "temperature",
    },
    "chaotic_disk": {
        "num_particles": 70_000,
        "render_particles": 12_000,
        "physics_steps_per_frame": 7,
        "dt": 0.006,
        "radius_min": 4.0,
        "radius_max": 40.0,
        "velocity_multiplier": 0.95,
        "velocity_noise": 0.18,
        "radial_noise": 0.06,
        "escape_radius": 75.0,
        "injection": True,
        "color_mode": "speed",
    },
    "large_cuda_demo": {
        "num_particles": 250_000,
        "render_particles": 25_000,
        "physics_steps_per_frame": 5,
        "dt": 0.005,
        "radius_min": 6.0,
        "radius_max": 44.0,
        "velocity_multiplier": 0.985,
        "velocity_noise": 0.05,
        "radial_noise": 0.012,
        "escape_radius": 80.0,
        "injection": True,
        "color_mode": "temperature",
    },
}


def parse_window_size(raw: str) -> tuple[int, int]:
    parts = raw.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window size must look like 1200x900")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS.keys(), default="large_cuda_demo")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--render-particles", type=int, default=None)
    parser.add_argument("--physics-steps-per-frame", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--velocity-multiplier", type=float, default=None)
    parser.add_argument("--color-mode", choices=COLOR_MODES, default=None)
    parser.add_argument("--injection", dest="injection", action="store_true", default=None)
    parser.add_argument("--no-injection", dest="injection", action="store_false")
    parser.add_argument("--enable-trails", action="store_true")
    parser.add_argument("--trail-particles", type=int, default=700)
    parser.add_argument("--trail-length", type=int, default=28)
    parser.add_argument("--window-size", type=parse_window_size, default=(1280, 900))
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--metric-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> dict:
    values = dict(PRESETS[args.preset])
    for key in (
        "num_particles",
        "render_particles",
        "physics_steps_per_frame",
        "dt",
        "velocity_multiplier",
        "color_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.injection is not None:
        values["injection"] = args.injection
    return values


def circle_points(radius: float, segments: int = 256) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=True, dtype=np.float32)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    span = max(float(np.nanmax(values) - np.nanmin(values)), 1.0e-8)
    return (values - float(np.nanmin(values))) / span


def scalar_to_rgba(values: np.ndarray, color_mode: str) -> np.ndarray:
    x = normalize(values.astype(np.float32, copy=False))
    if color_mode == "radius":
        rgba = np.column_stack((0.12 + 0.18 * x, 0.35 + 0.55 * x, 1.0 - 0.72 * x, np.ones_like(x)))
    elif color_mode == "speed":
        rgba = np.column_stack((0.25 + 0.75 * x, 0.12 + 0.38 * x, 0.85 - 0.55 * x, np.ones_like(x)))
    else:
        rgba = np.column_stack((0.35 + 0.65 * x, 0.08 + 0.76 * x**1.35, 0.02 + 0.16 * (1 - x), np.ones_like(x)))
    return rgba.astype(np.float32)


def positions_to_xyz(positions: np.ndarray) -> np.ndarray:
    xyz = np.zeros((positions.shape[0], 3), dtype=np.float32)
    xyz[:, :2] = positions.astype(np.float32, copy=False)
    return xyz


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_trail_segments(history: deque, trail_particles: int) -> np.ndarray:
    if len(history) < 2 or trail_particles <= 0:
        return np.empty((0, 3), dtype=np.float32)

    segments = []
    frames = list(history)
    for previous, current in zip(frames[:-1], frames[1:]):
        prev_pos, prev_active = previous
        cur_pos, cur_active = current
        alive = prev_active[:trail_particles] & cur_active[:trail_particles]
        if not np.any(alive):
            continue
        starts = positions_to_xyz(prev_pos[:trail_particles][alive])
        ends = positions_to_xyz(cur_pos[:trail_particles][alive])
        pair_segments = np.empty((starts.shape[0] * 2, 3), dtype=np.float32)
        pair_segments[0::2] = starts
        pair_segments[1::2] = ends
        segments.append(pair_segments)

    if not segments:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(segments, axis=0)


def screenshot_path() -> Path:
    path = Path("outputs/figures") / f"live_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    try:
        from vispy import app, scene
    except ImportError as exc:
        raise SystemExit("VisPy/PyQt6 are required. Install with: pip install vispy PyQt6") from exc

    preset = apply_overrides(args)
    config = LiveSimulationConfig(
        num_particles=preset["num_particles"],
        render_particles=preset["render_particles"],
        physics_steps_per_frame=preset["physics_steps_per_frame"],
        dt=preset["dt"],
        radius_min=preset["radius_min"],
        radius_max=preset["radius_max"],
        injection_radius_min=25.0,
        injection_radius_max=preset["radius_max"],
        velocity_multiplier=preset["velocity_multiplier"],
        velocity_noise=preset["velocity_noise"],
        radial_noise=preset["radial_noise"],
        escape_radius=preset["escape_radius"],
        device=args.device,
        color_mode=preset["color_mode"],
        seed=args.seed,
        enable_injection=preset["injection"],
        enable_trails=args.enable_trails,
        trail_length=args.trail_length,
    )
    state = LiveSimulationState(config)

    canvas = scene.SceneCanvas(
        keys="interactive",
        size=args.window_size,
        bgcolor="#050608",
        title="v0.5c live CUDA accretion disk demo",
        show=True,
    )
    view = canvas.central_widget.add_view()
    extent = max(config.radius_max * 1.08, config.escape_radius * 0.75)
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(-extent, extent), y=(-extent, extent))

    markers = scene.visuals.Markers(parent=view.scene)
    trails = scene.visuals.Line(np.empty((0, 3), dtype=np.float32), color=(1.0, 0.42, 0.10, 0.18), width=1.0, parent=view.scene)
    scene.visuals.Ellipse(center=(0, 0, 0), radius=(EVENT_HORIZON_RADIUS, EVENT_HORIZON_RADIUS), color=(0, 0, 0, 1), parent=view.scene)
    scene.visuals.Line(circle_points(PHOTON_SPHERE_RADIUS), color=(1.0, 0.82, 0.18, 0.75), width=1.2, parent=view.scene)
    scene.visuals.Line(circle_points(ISCO_RADIUS), color=(0.35, 0.65, 1.0, 0.75), width=1.2, parent=view.scene)

    overlay = scene.Label("", color="#e5e7eb", font_size=10, anchor_x="left", anchor_y="top")
    overlay.parent = canvas.scene
    overlay.pos = (12, args.window_size[1] - 12)

    paused = {"value": False}
    running = {"value": True}
    show_help = {"value": True}
    trails_enabled = {"value": args.enable_trails}
    point_size = {"value": args.point_size}
    frame_counter = {"value": 0}
    last_time = {"value": time.perf_counter()}
    trail_history: deque = deque(maxlen=args.trail_length)
    trail_particles = min(args.trail_particles, state.render_count)

    def reset_view() -> None:
        view.camera.set_range(x=(-extent, extent), y=(-extent, extent))

    def save_screenshot() -> None:
        path = screenshot_path()
        Image.fromarray(canvas.render()).save(path)
        print(f"[v0.5c] Saved screenshot: {path}", flush=True)

    def set_color_mode(mode: str) -> None:
        state.color_mode = mode
        print(f"[v0.5c] color mode: {state.color_mode}", flush=True)

    def change_dt(factor: float) -> None:
        config.dt = clamp(config.dt * factor, 0.0002, 0.05)
        print(f"[v0.5c] dt: {config.dt:.5f}", flush=True)

    def change_velocity(delta: float) -> None:
        state.set_velocity_multiplier(clamp(config.velocity_multiplier + delta, 0.50, 1.50))
        print(f"[v0.5c] velocity multiplier: {config.velocity_multiplier:.3f}", flush=True)

    def update_overlay(metrics: dict[str, float]) -> None:
        title = (
            f"FPS {metrics['fps']:.1f} | active {int(metrics['active_count'])}/{int(metrics['num_particles'])} | "
            f"swallowed {metrics['swallowed_fraction']:.3f} | escaped {metrics['escaped_fraction']:.3f} | "
            f"{metrics['device']} | rendered {int(metrics['render_particles'])}"
        )
        canvas.title = "v0.5c live demo | " + title
        help_text = ""
        if show_help["value"]:
            help_text = (
                "\n\nControls:"
                "\nSpace pause | R reset | Q/Esc quit | H help"
                "\nC cycle color | 1 radius | 2 speed | 3 temperature"
                "\nI injection | T trails | S screenshot"
                "\nUp/Down velocity multiplier | +/- dt | ]/[ rendered | ./, point size | 0 view"
            )
        overlay.text = (
            title
            + f"\npreset {args.preset} | color {state.color_mode} | injection {config.enable_injection} | trails {trails_enabled['value']}"
            + f"\ndt {config.dt:.5f} | velocity {config.velocity_multiplier:.3f} | steps/frame {config.physics_steps_per_frame}"
            + f"\nmean r {metrics['mean_radius_active']:.2f} | mean speed {metrics['mean_speed_active']:.3f}"
            + f"\npoint size {point_size['value']:.1f} | {'PAUSED' if paused['value'] else 'running'}"
            + help_text
        )
        print("[v0.5c] " + title, flush=True)

    def on_key_press(event) -> None:
        key = event.key.name.lower() if event.key is not None else ""
        text = (event.text or "").lower()
        if key == "space":
            paused["value"] = not paused["value"]
        elif key == "r":
            state.reset()
            trail_history.clear()
        elif key == "c":
            state.cycle_color_mode()
        elif text == "1":
            set_color_mode("radius")
        elif text == "2":
            set_color_mode("speed")
        elif text == "3":
            set_color_mode("temperature")
        elif key == "i":
            config.enable_injection = not config.enable_injection
            print(f"[v0.5c] injection: {config.enable_injection}", flush=True)
        elif key == "t":
            trails_enabled["value"] = not trails_enabled["value"]
            trail_history.clear()
            if not trails_enabled["value"]:
                trails.set_data(np.empty((0, 3), dtype=np.float32))
        elif key == "s":
            save_screenshot()
        elif key == "h":
            show_help["value"] = not show_help["value"]
        elif key in {"up"}:
            change_velocity(0.02)
        elif key in {"down"}:
            change_velocity(-0.02)
        elif key in {"plus", "equal"} or text in {"+", "="}:
            change_dt(1.2)
        elif key == "minus" or text == "-":
            change_dt(1 / 1.2)
        elif key == "bracketright" or text == "]":
            state.set_render_count(int(state.render_count * 1.25))
        elif key == "bracketleft" or text == "[":
            state.set_render_count(int(state.render_count * 0.80))
        elif key == "period" or text == ".":
            point_size["value"] = clamp(point_size["value"] + 0.5, 1.0, 12.0)
        elif key == "comma" or text == ",":
            point_size["value"] = clamp(point_size["value"] - 0.5, 1.0, 12.0)
        elif text == "0":
            reset_view()
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

        if trails_enabled["value"]:
            trail_count = min(trail_particles, len(frame.positions))
            trail_history.append((frame.positions[:trail_count].copy(), frame.active[:trail_count].copy()))
            segments = build_trail_segments(trail_history, trail_count)
            trails.set_data(segments, color=(1.0, 0.42, 0.10, 0.18), width=1.0, connect="segments")

        if include_metrics:
            update_overlay(frame.metrics)

        frame_counter["value"] += 1
        canvas.update()

    canvas.events.key_press.connect(on_key_press)
    print("[v0.5c] Controls: Space/R/C/1/2/3/I/T/S/Up/Down/+/-/Q. H toggles help.", flush=True)
    print(f"[v0.5c] Preset {args.preset}; running on {state.device}; rendering {state.render_count} of {config.num_particles}.", flush=True)

    timer = app.Timer(interval=0.0, connect=on_timer, start=True)
    app.run()
    timer.stop()
    print("[v0.5c] Live demo closed.", flush=True)


if __name__ == "__main__":
    main()

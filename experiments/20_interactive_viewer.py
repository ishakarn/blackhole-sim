"""Interactive v2.0 preview viewer for the Schwarzschild transfer renderer."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime
import pathlib
import sys
import time
import traceback
import tracemalloc

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.transfer_render_backend import CameraParameters, RenderParameters, RenderResult, render_black_hole


OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs" / "figures" / "20_interactive_viewer"


@dataclass(slots=True)
class ViewerState:
    camera: CameraParameters
    preview: RenderParameters
    quality: RenderParameters
    last_result: RenderResult | None = None
    last_mode: str = "preview"
    is_rendering: bool = False
    render_requested: bool = False
    requested_mode: str = "preview"
    last_render_time: float = 0.0
    last_input_time: float = 0.0
    debounce_seconds: float = 0.15
    running: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-resolution", type=int, default=64)
    parser.add_argument("--preview-supersample", type=int, default=1)
    parser.add_argument("--quality-resolution", type=int, default=256)
    parser.add_argument("--quality-supersample", type=int, default=1)
    parser.add_argument("--debounce-ms", type=int, default=150)
    parser.add_argument("--preview-max-steps", type=int, default=600)
    parser.add_argument("--quality-max-steps", type=int, default=7000)
    parser.add_argument("--preview-step-size", type=float, default=0.02)
    parser.add_argument("--quality-step-size", type=float, default=0.005)
    parser.add_argument("--disable-diagnostics", action="store_true")
    parser.add_argument("--fov", type=float, default=14.0)
    parser.add_argument("--camera-distance", type=float, default=100.0)
    parser.add_argument("--camera-height", type=float, default=80.0)
    parser.add_argument("--camera-azimuth", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-once", action="store_true")
    return parser.parse_args()


def timestamped_path(prefix: str) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{prefix}_{stamp}.png"


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return array


def save_frame(image: torch.Tensor, path: pathlib.Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required to save screenshots. Install pillow.") from exc
    Image.fromarray(tensor_to_uint8(image)).save(path)
    print(f"Saved {path}")


def build_viewer_state(args: argparse.Namespace) -> ViewerState:
    camera = CameraParameters(
        azimuth_deg=args.camera_azimuth,
        distance=args.camera_distance,
        height=args.camera_height,
        fov=args.fov,
    )
    preview = RenderParameters(
        resolution=args.preview_resolution,
        supersample=args.preview_supersample,
        max_steps=args.preview_max_steps,
        step_size=args.preview_step_size,
        device=args.device,
        background_scale=2.0,
        background_blur_radius=0.0,
        bloom=False,
        diagnostics=False,
    )
    quality = RenderParameters(
        resolution=args.quality_resolution,
        supersample=args.quality_supersample,
        max_steps=args.quality_max_steps,
        step_size=args.quality_step_size,
        device=args.device,
        bloom=False,
        diagnostics=not args.disable_diagnostics,
    )
    return ViewerState(
        camera=camera,
        preview=preview,
        quality=quality,
        debounce_seconds=max(0.0, args.debounce_ms / 1000.0),
    )


def render_current(state: ViewerState, mode: str) -> RenderResult:
    params = state.preview if mode == "preview" else state.quality
    result = render_black_hole(state.camera, params)
    state.last_result = result
    state.last_mode = mode
    state.last_render_time = time.perf_counter()
    return result


def memory_usage_text(device_name: str) -> str:
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    parts = [
        f"py={current_bytes / (1024.0 * 1024.0):.1f}MB",
        f"py_peak={peak_bytes / (1024.0 * 1024.0):.1f}MB",
    ]
    if device_name.startswith("cuda") and torch.cuda.is_available():
        parts.append(f"cuda={torch.cuda.memory_allocated() / (1024.0 * 1024.0):.1f}MB")
        parts.append(f"cuda_peak={torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.1f}MB")
    return " | ".join(parts)


def print_render_log(state: ViewerState, mode: str, result: RenderResult) -> None:
    params = state.preview if mode == "preview" else state.quality
    print(
        f"[{mode}] resolution={result.width} | supersample={params.supersample} | max_steps={params.max_steps} | "
        f"azimuth={state.camera.azimuth_deg:.1f} deg | height={state.camera.height:.1f} | "
        f"distance={state.camera.distance:.1f} | fov={state.camera.fov:.1f} | "
        f"render_time={result.render_time_seconds:.2f}s | {memory_usage_text(params.device)}",
        flush=True,
    )


def cleanup_after_render(mode: str, device_name: str, failed: bool = False) -> None:
    gc.collect()
    if device_name.startswith("cuda") and torch.cuda.is_available() and (mode == "quality" or failed):
        torch.cuda.empty_cache()


def update_canvas_title(canvas, state: ViewerState) -> None:
    if state.last_result is None:
        return
    result = state.last_result
    canvas.title = (
        "v2.1 Interactive Schwarzschild Viewer | "
        f"mode={state.last_mode} | azimuth={state.camera.azimuth_deg:.1f} | "
        f"distance={state.camera.distance:.1f} | height={state.camera.height:.1f} | "
        f"fov={state.camera.fov:.1f} | preview={state.preview.resolution}px | "
        f"last={result.render_time_seconds:.2f}s"
    )


def request_render(state: ViewerState, mode: str, immediate: bool = False) -> None:
    state.render_requested = True
    if mode == "quality" or state.requested_mode != "quality":
        state.requested_mode = mode
    state.last_input_time = 0.0 if immediate else time.perf_counter()


def main() -> None:
    args = parse_args()
    state = build_viewer_state(args)
    tracemalloc.start()

    if args.render_once:
        result = render_current(state, "preview")
        save_frame(result.image, timestamped_path("interactive_screenshot"))
        cleanup_after_render("preview", state.preview.device)
        return

    try:
        from vispy import app, scene
    except ImportError as exc:
        raise SystemExit("VisPy and PyQt6 are required. Install with: pip install vispy PyQt6") from exc

    canvas = scene.SceneCanvas(keys="interactive", size=(900, 900), bgcolor="#020304", title="v2.1 Interactive Schwarzschild Viewer", show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(0, state.preview.resolution), y=(state.preview.resolution, 0))
    image_visual = scene.visuals.Image(np.zeros((state.preview.resolution, state.preview.resolution, 3), dtype=np.uint8), parent=view.scene, interpolation="nearest")
    overlay = scene.Label("", color="#e5e7eb", font_size=10, anchor_x="left", anchor_y="top")
    overlay.parent = canvas.scene
    overlay.pos = (12, 888)

    help_text = (
        "Controls: Left/Right or A/D azimuth | Up/Down or W/S height | +/- distance | [/ ] fov | "
        "R quality render | P or S save | Q or Esc quit"
    )
    print(help_text, flush=True)

    def update_overlay() -> None:
        status = "rendering" if state.is_rendering else "idle"
        overlay.text = (
            f"mode {state.last_mode} | status {status} | debounce {state.debounce_seconds * 1000.0:.0f} ms"
            f"\nazimuth {state.camera.azimuth_deg:.1f} | height {state.camera.height:.1f} | distance {state.camera.distance:.1f} | fov {state.camera.fov:.1f}"
            f"\npreview {state.preview.resolution}px/{state.preview.max_steps} steps/{state.preview.step_size:.3f} dphi | quality {state.quality.resolution}px/{state.quality.max_steps} steps/{state.quality.step_size:.3f} dphi"
            f"\n{help_text}"
        )

    def update_display(result: RenderResult) -> None:
        image_visual.set_data(tensor_to_uint8(result.image))
        view.camera.set_range(x=(0, result.width), y=(result.height, 0))
        update_canvas_title(canvas, state)
        update_overlay()
        canvas.update()

    def safe_render(mode: str) -> None:
        params = state.preview if mode == "preview" else state.quality
        failed = False
        try:
            result = render_current(state, mode)
            update_display(result)
            print_render_log(state, mode, result)
        except Exception:
            failed = True
            traceback.print_exc()
            update_overlay()
        finally:
            state.is_rendering = False
            cleanup_after_render(mode, params.device, failed=failed)

    def save_current(prefix: str) -> None:
        if state.last_result is None:
            return
        save_frame(state.last_result.image, timestamped_path(prefix))

    def step_camera(azimuth_delta: float = 0.0, height_delta: float = 0.0, distance_delta: float = 0.0, fov_delta: float = 0.0) -> None:
        state.camera.azimuth_deg += azimuth_delta
        state.camera.height += height_delta
        state.camera.distance = max(20.0, state.camera.distance + distance_delta)
        state.camera.fov = max(4.0, state.camera.fov + fov_delta)
        request_render(state, "preview")
        update_overlay()

    @canvas.events.key_press.connect
    def on_key_press(event) -> None:
        key = event.key.name if event.key is not None else ""
        text = (event.text or "").lower()
        handled = True
        if key == "Left" or text == "a":
            step_camera(azimuth_delta=-5.0)
        elif key == "Right" or text == "d":
            step_camera(azimuth_delta=5.0)
        elif key == "Up" or text == "w":
            step_camera(height_delta=5.0)
        elif key == "Down":
            step_camera(height_delta=-5.0)
        elif text in ["+", "="]:
            step_camera(distance_delta=-5.0)
        elif text == "-":
            step_camera(distance_delta=5.0)
        elif text == "[":
            step_camera(fov_delta=-1.0)
        elif text == "]":
            step_camera(fov_delta=1.0)
        elif text == "r":
            request_render(state, "quality", immediate=True)
            update_overlay()
        elif text in ["p", "s"]:
            save_current("interactive_screenshot")
        elif key in ["Q", "Escape"] or text == "q":
            state.running = False
            canvas.close()
        else:
            handled = False

        if handled:
            event.handled = True

    @canvas.events.close.connect
    def on_close(event) -> None:
        state.running = False

    def on_timer(event) -> None:
        if not state.running or state.is_rendering or not state.render_requested:
            return
        now = time.perf_counter()
        if state.requested_mode == "preview" and now - state.last_input_time < state.debounce_seconds:
            return
        mode = state.requested_mode
        state.render_requested = False
        state.is_rendering = True
        update_overlay()
        safe_render(mode)

    update_overlay()
    request_render(state, "preview", immediate=True)
    timer = app.Timer(interval=0.03, connect=on_timer, start=True)
    app.run()
    timer.stop()


if __name__ == "__main__":
    main()
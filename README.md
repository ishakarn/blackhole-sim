# Black Hole SIM

A staged black-hole simulation and rendering project in Python. The repo starts
with Newtonian test-particle experiments, then moves into Schwarzschild ray
tracing, curved-ray accretion-disk rendering, relativistic transfer-function
shading, photon-momentum diagnostics, and an interactive VisPy camera viewer.

The code uses dimensionless Schwarzschild-style units:

- `G = 1`
- `M = 1`
- `c = 1`
- event horizon `r = 2`
- photon sphere `r = 3`
- ISCO `r = 6`

## Quick Start

Install dependencies in your preferred environment:

```powershell
pip install -r requirements.txt
```

If you are using the Conda environment used during development:

```powershell
conda activate blackhole-sim
```

Run the first simulation:

```powershell
python -m experiments.01_newtonian_particles
```

This writes:

- `outputs/figures/v01_trajectories.png`
- `outputs/animations/v01_particles.gif`

CUDA is used automatically when PyTorch can see your GPU. You can also force a
device:

```powershell
python -m experiments.01_newtonian_particles --device cuda
python -m experiments.01_newtonian_particles --device cpu
```

## Current Highlights

The current late-stage rendering work is centered on the full 3D Schwarzschild
pipeline and the interactive viewer.

Run the current interactive viewer:

```powershell
python -m experiments.20_interactive_viewer
```

Current v2.1 viewer behavior:

- low-resolution preview renders for camera movement
- separate on-demand quality render path
- debounced input and non-overlapping render scheduling
- safe render wrapper and memory cleanup
- screenshots saved under `outputs/figures/20_interactive_viewer/`

Viewer controls:

- `Left` / `Right` or `A` / `D`: orbit camera azimuth
- `Up` or `W`: raise camera height
- `Down`: lower camera height
- `+` / `-`: move camera closer/farther
- `[` / `]`: decrease/increase FOV
- `R`: run one quality render
- `P` or `S`: save the current displayed frame
- `Q` or `Escape`: quit

Useful viewer options:

```powershell
python -m experiments.20_interactive_viewer --preview-resolution 64 --preview-max-steps 600 --preview-step-size 0.02
python -m experiments.20_interactive_viewer --quality-resolution 512 --quality-supersample 2 --quality-max-steps 7000
```

Run the current momentum-transfer renderer:

```powershell
python -m experiments.18_photon_momentum_transfer --preset quality --transfer-mode momentum
```

Run the photon-momentum diagnostics pass:

```powershell
python -m experiments.18b_photon_momentum_diagnostics --preset quality
```

## Stage Guide

### v0.x Particle Simulation

Run the v0.2 metrics and velocity sweep:

```powershell
python -m experiments.02_outcome_sweep
```

This writes:

- `outputs/figures/v02_active_count.png`
- `outputs/figures/v02_swallowed_fraction.png`
- `outputs/figures/v02_outcome_fractions_by_velocity.png`
- `outputs/figures/v02_representative_trajectories.png`
- `outputs/animations/v02_representative_particles.gif`

Run the v0.3 CPU vs CUDA benchmark:

```powershell
python -m experiments.03_cuda_benchmark
```

This writes:

- `outputs/benchmarks/cuda_benchmark.csv`
- `outputs/benchmarks/cuda_benchmark.png`

Run the v0.4 pseudo-relativistic accretion disk experiment:

```powershell
python -m experiments.04_pseudo_relativistic_disk
```

This writes:

- `outputs/figures/v04_accretion_disk_trajectories.png`
- `outputs/figures/v04_region_fractions.png`
- `outputs/metrics/pseudo_relativistic_disk_metrics.csv`

Run the v0.5 live CUDA simulator:

```powershell
python -m experiments.05_live_simulator --device cuda --num-particles 50000 --render-particles 10000
```

Run the v0.5b VisPy live simulator:

```powershell
python -m experiments.05b_live_vispy --device cuda --num-particles 50000 --render-particles 10000 --physics-steps-per-frame 5
```

Run the v0.5c polished live demo:

```powershell
python -m experiments.05c_live_demo --preset large_cuda_demo --device cuda
```

### v0.6-v1.1 Schwarzschild Setup

These stages introduce Schwarzschild lensing, null geodesics, black-hole shadow
rendering, and approximate curved-ray disk rendering. Representative commands:

```powershell
python -m experiments.06_schwarzschild_lensing
python -m experiments.07_black_hole_disk
python -m experiments.08_null_geodesics
python -m experiments.08b_deflection_sweep
python -m experiments.09_geodesic_lensing
python -m experiments.10_geodesic_disk
python -m experiments.11_approx_3d_disk_renderer
```

These now write into versioned folders under `outputs/figures/` such as:

- `outputs/figures/06_schwarzschild_lensing/`
- `outputs/figures/07_black_hole_disk/`
- `outputs/figures/11_approx_3d_disk_renderer/`

### v1.2-v1.5 Full 3D Curved-Ray Rendering

These stages establish the full 3D geodesic marcher, camera/FOV sweep,
refined disk emission, relativistic appearance heuristics, and polished render
quality.

Representative commands:

```powershell
python -m experiments.12_full_3d_geodesic_renderer
python -m experiments.12b_camera_fov_sweep
python -m experiments.13_refined_3d_geodesic_renderer
python -m experiments.14_relativistic_disk_effects
python -m experiments.15_polished_render --preset quality
```

Useful current framing defaults from the sweep work:

- `fov = 14`
- `camera_distance = 100`
- `camera_height = 80`

These stages write into:

- `outputs/figures/12_full_3d_geodesic_renderer/`
- `outputs/figures/13_refined_3d_geodesic_renderer/`
- `outputs/figures/14_relativistic_disk_effects/`
- `outputs/figures/15_polished_render/`
- `outputs/sweeps/camera_fov/`
- `outputs/metrics/camera_fov_sweep_full.csv`
- `outputs/metrics/camera_fov_sweep_shadow_only.csv`

### v1.6-v1.8b Relativistic Transfer and Diagnostics

These stages add shared relativistic disk shading helpers, a transfer-function
renderer, improved photon momentum reconstruction, and a diagnostics-only pass
to validate null consistency.

Representative commands:

```powershell
python -m experiments.16_relativistic_transfer_renderer --preset quality --physics-mode transfer
python -m experiments.18_photon_momentum_transfer --preset quality --transfer-mode momentum
python -m experiments.18b_photon_momentum_diagnostics --preset quality
```

Current validated v1.8 and v1.8b notes:

- momentum-transfer `g` is close to tangent-transfer `g`, but more physically motivated
- typical v1.8 comparison: mean absolute `g` difference about `0.0061`
- coordinate null residual mean absolute error about `7.6e-08`
- local tetrad null residual mean absolute error about `1.23e-07`

These stages write into:

- `outputs/figures/16_relativistic_transfer_renderer/`
- `outputs/figures/18_photon_momentum_transfer/`
- `outputs/figures/photon_diagnostics/`

## Project Roadmap

| Version | Focus | Status |
|---|---|---|
| v0.1 | Basic Newtonian particle simulation | done |
| v0.2 | Metrics, outcome sweeps, experiment logging | done |
| v0.3 | CUDA acceleration and CPU/GPU benchmarking | done |
| v0.4 | Pseudo-relativistic black hole visuals | done |
| v0.5a | Matplotlib live CUDA simulator | done |
| v0.5b | VisPy fast live CUDA simulator | done |
| v0.5c | Live demo polish, presets, screenshots, limited trails | done |
| v0.6-v1.1 | Schwarzschild lensing, shadow, null geodesics, approximate disk rendering | done |
| v1.2 | Full 3D geodesic marcher with disk-hit/capture/escape classification | done |
| v1.2b | Camera/FOV sweep and framing selection | done |
| v1.3 | Refined disk emission and compositing | done |
| v1.4 | Heuristic relativistic beaming and redshift proxies | done |
| v1.5 | Polished render quality pass | done |
| v1.6 | Relativistic transfer-function disk shading | done |
| v1.8 | Photon-momentum transfer-factor rendering | done |
| v1.8b | Photon momentum diagnostics and null validation | done |
| v2.0 | Interactive VisPy camera preview viewer | done |
| v2.1 | Interactive viewer stability and preview optimization | done |

## Output Layout

Figure outputs are now grouped by experiment version to avoid clutter in the
top-level figure directory.

Examples:

- `outputs/figures/06_schwarzschild_lensing/`
- `outputs/figures/07_black_hole_disk/`
- `outputs/figures/12_full_3d_geodesic_renderer/`
- `outputs/figures/16_relativistic_transfer_renderer/`
- `outputs/figures/18_photon_momentum_transfer/`
- `outputs/figures/20_interactive_viewer/`
- `outputs/figures/photon_diagnostics/`

Other outputs remain in:

- `outputs/animations/`
- `outputs/benchmarks/`
- `outputs/metrics/`
- `outputs/sweeps/`

## Layout

```text
src/
  benchmark.py
  camera.py
  constants.py
  geodesic_3d.py
  initial_conditions.py
  integrators.py
  live.py
  metrics.py
  photon_transfer.py
  raytracing.py
  relativistic_disk.py
  simulation.py
  transfer_render_backend.py
  visualization.py
experiments/
  01_newtonian_particles.py
  02_outcome_sweep.py
  03_cuda_benchmark.py
  04_pseudo_relativistic_disk.py
  05_live_simulator.py
  05b_live_vispy.py
  05c_live_demo.py
  06_schwarzschild_lensing.py
  07_black_hole_disk.py
  08_null_geodesics.py
  08b_deflection_sweep.py
  09_geodesic_lensing.py
  10_geodesic_disk.py
  11_approx_3d_disk_renderer.py
  12_full_3d_geodesic_renderer.py
  12b_camera_fov_sweep.py
  13_refined_3d_geodesic_renderer.py
  14_relativistic_disk_effects.py
  15_polished_render.py
  16_relativistic_transfer_renderer.py
  18_photon_momentum_transfer.py
  18b_photon_momentum_diagnostics.py
  20_interactive_viewer.py
outputs/
  animations/
  benchmarks/
  figures/
  metrics/
  sweeps/
```

# Black Hole SIM

This is so cool! I am going to make a black hole!!


A staged black-hole simulation project. Version 0.1 is a 2D Newtonian
test-particle simulator around a fixed central black-hole-like mass.

The code uses dimensionless units:

- `G = 1`
- `M = 1`
- `c = 1`
- event horizon proxy `r_s = 2`

Particles do not interact with each other. They move under central Newtonian
gravity and are marked inactive once they cross the absorbing horizon.

## Quick Start

Install dependencies in your preferred environment:

```powershell
pip install -r requirements.txt
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

For a faster smoke test:

```powershell
python -m experiments.01_newtonian_particles --num-particles 250 --num-steps 500 --no-animation
```

Run the v0.2 metrics and velocity sweep:

```powershell
python -m experiments.02_outcome_sweep
```

This writes metric plots and outcome analysis:

- `outputs/figures/v02_active_count.png`
- `outputs/figures/v02_swallowed_fraction.png`
- `outputs/figures/v02_outcome_fractions_by_velocity.png`
- `outputs/figures/v02_representative_trajectories.png`
- `outputs/animations/v02_representative_particles.gif`

Run the v0.3 CPU vs CUDA benchmark:

```powershell
python -m experiments.03_cuda_benchmark
```

For a quick benchmark smoke test:

```powershell
python -m experiments.03_cuda_benchmark --particle-counts "1000,10000" --num-steps 100
```

This writes:

- `outputs/benchmarks/cuda_benchmark.csv`
- `outputs/benchmarks/cuda_benchmark.png`

Recent benchmark on the Lenovo Legion Slim 5 / RTX 4070:

- command: `python -m experiments.03_cuda_benchmark --particle-counts "1000,2500,5000,10000,25000,50000,100000,250000" --num-steps 300`
- CUDA became faster than CPU at about `5,000` particles
- at `250,000` particles, CPU reached about `19.1M` particle updates/sec
- at `250,000` particles, CUDA reached about `210.5M` particle updates/sec
- CUDA was about `11x` faster than CPU at the largest tested size

See `notes/v03_cuda_benchmark.md` for the detailed run summary.

Run the v0.4 pseudo-relativistic accretion disk experiment:

```powershell
python -m experiments.04_pseudo_relativistic_disk
```

This writes:

- `outputs/figures/v04_accretion_disk_trajectories.png`
- `outputs/figures/v04_region_fractions.png`
- `outputs/metrics/pseudo_relativistic_disk_metrics.csv`

To also save a GIF:

```powershell
python -m experiments.04_pseudo_relativistic_disk --save-animation
```

## Project Roadmap

| Version | Focus | Status |
|---|---|---|
| v0.1 | Basic Newtonian particle simulation | done |
| v0.2 | Metrics, outcome sweeps, experiment logging | done |
| v0.3 | CUDA acceleration and CPU/GPU benchmarking | done |
| v0.4 | Pseudo-relativistic black hole visuals | done |
| v0.5 | Live CUDA simulator | next or parallel |
| v0.6 | Schwarzschild ray tracing / lensing | after v0.4 |
| v0.7 | Black hole shadow + accretion disk rendering | after v0.6 |

## Layout

```text
src/
  constants.py
  initial_conditions.py
  integrators.py
  metrics.py
  benchmark.py
  simulation.py
  visualization.py
  raytracing.py
experiments/
  01_newtonian_particles.py
  02_outcome_sweep.py
  03_cuda_benchmark.py
  04_pseudo_relativistic_disk.py
outputs/
  figures/
  animations/
  benchmarks/
  metrics/
```

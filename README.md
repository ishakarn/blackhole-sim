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

## Project Roadmap

1. 2D Newtonian/pseudo-black-hole particle simulation. [done]
2. Accretion-disk-like behavior and better visualizations.
3. PyTorch CUDA acceleration for larger particle counts.
4. Relativistic effects, ray tracing, lensing, and shadow rendering.

## Layout

```text
src/
  constants.py
  initial_conditions.py
  integrators.py
  simulation.py
  visualization.py
  raytracing.py
experiments/
  01_newtonian_particles.py
outputs/
  figures/
  animations/
```

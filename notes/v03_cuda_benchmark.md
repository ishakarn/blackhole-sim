# v0.3 CUDA Benchmark Notes

Benchmark date: 2026-05-09

Machine context:

- Laptop: Lenovo Legion Slim 5
- GPU: NVIDIA RTX 4070
- Environment: `blackhole-sim`
- Backend: PyTorch CPU vs PyTorch CUDA

Command:

```powershell
python -m experiments.03_cuda_benchmark --particle-counts "1000,2500,5000,10000,25000,50000,100000,250000" --num-steps 300
```

Result summary:

- CPU is faster at small particle counts where CUDA launch/setup overhead dominates.
- CUDA becomes faster around `5,000` particles.
- At `250,000` particles, CUDA reached about `210.5M` particle updates/sec.
- At `250,000` particles, CPU reached about `19.1M` particle updates/sec.
- Largest-run speedup was about `11x` in favor of CUDA.

Throughput table:

| Particles | CPU updates/sec | CUDA updates/sec | Faster |
|---:|---:|---:|:---|
| 1,000 | 1.80M | 0.62M | CPU |
| 2,500 | 2.51M | 1.86M | CPU |
| 5,000 | 3.26M | 4.84M | CUDA |
| 10,000 | 3.68M | 8.81M | CUDA |
| 25,000 | 7.88M | 23.41M | CUDA |
| 50,000 | 12.24M | 40.84M | CUDA |
| 100,000 | 16.54M | 89.77M | CUDA |
| 250,000 | 19.10M | 210.52M | CUDA |

Interpretation:

For visual experiments, CUDA helps once the particle count is large enough, but plotting and GIF creation remain CPU-side bottlenecks. For large visual runs, keep the simulation on CUDA while recording only a subset of particles:

```powershell
python -m experiments.02_outcome_sweep --device cuda --num-particles 10000 --num-steps 1600 --snapshot-interval 20 --max-record-particles 1000
```

The benchmark CSV and plot are generated at:

```text
outputs/benchmarks/cuda_benchmark.csv
outputs/benchmarks/cuda_benchmark.png
```

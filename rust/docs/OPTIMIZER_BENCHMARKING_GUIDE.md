# Optimizer Benchmarking Guide

This guide explains how to run comprehensive benchmarks comparing Grid Search, Euler Search, and Genetic Algorithm optimizers.

---

## Quick Start

```bash
# 1. Run all benchmarks (15-30 minutes)
cd /home/kim/projects/kimsfinance/rust
cargo bench --bench optimizer_comparison

# 2. Analyze results
python scripts/analyze_optimizer_benchmarks.py

# 3. View report
cat docs/benchmark_results/analysis_report.md
firefox target/criterion/optimizer_comparison/report/index.html
```

---

## Prerequisites

### Hardware

- **GPU**: NVIDIA GPU with Compute Capability ≥ 7.0 (Volta, Turing, Ampere, Ada Lovelace)
- **VRAM**: ≥ 4GB recommended (12GB for large scenarios)
- **CPU**: Modern multi-core CPU (for genetic algorithm parallel evaluation)
- **RAM**: ≥ 16GB

### Software

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install CUDA Toolkit (version 11.8+ or 13.0+)
# Follow: https://developer.nvidia.com/cuda-downloads

# Install Python dependencies for analysis
pip install pandas numpy scipy matplotlib seaborn
```

### Verify GPU

```bash
# Check GPU is detected
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 13.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA RTX 3500...  Off  | 00000000:01:00.0 Off |                  N/A |
# +-------------------------------+----------------------+----------------------+
```

---

## Running Benchmarks

### Option 1: Run All Benchmarks (Recommended)

```bash
# Run complete benchmark suite
cargo bench --bench optimizer_comparison

# This will:
# - Grid Search: 2D, 3D scenarios
# - Euler Search: 2D, 3D, 5D scenarios
# - Genetic Algorithm: 2D, 3D, 5D scenarios
# - Total time: 15-30 minutes
```

### Option 2: Run Specific Scenarios

```bash
# Only 2D benchmarks (fastest, ~5 min)
cargo bench --bench optimizer_comparison -- 2d

# Only 3D benchmarks (~10 min)
cargo bench --bench optimizer_comparison -- 3d

# Only 5D benchmarks (slowest, ~15 min)
cargo bench --bench optimizer_comparison -- 5d
```

### Option 3: Run Specific Optimizers

```bash
# Only Grid Search
cargo bench --bench optimizer_comparison -- grid_search

# Only Euler Search
cargo bench --bench optimizer_comparison -- euler_search

# Only Genetic Algorithm
cargo bench --bench optimizer_comparison -- genetic
```

---

## Monitoring GPU During Benchmarks

Open a second terminal and run:

```bash
# Monitor GPU utilization (updates every 1 second)
nvidia-smi dmon -s u -c 1000

# Expected output:
# # gpu   sm   mem   enc   dec
# # Idx    %     %     %     %
#     0   85    45     0     0   <-- Target: >70% SM utilization
#     0   92    50     0     0
```

Key metrics:
- **sm %**: Streaming Multiprocessor utilization (target: >70%)
- **mem %**: Memory utilization (indicates data transfer overhead)

---

## Understanding Benchmark Output

### Criterion Output

After running benchmarks, criterion generates reports in:

```
target/criterion/optimizer_comparison/
├── grid_search/
│   ├── 2d/
│   │   ├── base/
│   │   │   ├── estimates.json      # Mean, std, percentiles
│   │   │   └── sample.json          # Raw sample data
│   │   └── report/
│   │       └── index.html           # HTML report
│   └── 3d/
│       └── ...
├── euler_search/
│   └── ...
└── genetic/
    └── ...
```

### Key Metrics

**Execution Time**:
- `mean`: Average time across 10 runs
- `std_dev`: Standard deviation (lower is better - more consistent)
- `median`: Median time (robust to outliers)
- `p95`, `p99`: 95th and 99th percentile times

**Example**:
```json
{
  "mean": {
    "point_estimate": 2450000000,  // 2.45 seconds (in nanoseconds)
    "confidence_interval": {
      "lower_bound": 2400000000,
      "upper_bound": 2500000000
    }
  },
  "std_dev": {
    "point_estimate": 120000000  // 120ms std dev
  }
}
```

---

## Analyzing Results

### Automated Analysis

```bash
# Run Python analysis script
python scripts/analyze_optimizer_benchmarks.py

# Generates:
# - docs/benchmark_results/execution_time_comparison.png
# - docs/benchmark_results/speedup_comparison.png
# - docs/benchmark_results/confidence_intervals.png
# - docs/benchmark_results/analysis_report.md
```

### Manual Analysis (via Criterion HTML)

```bash
# Open criterion HTML report
firefox target/criterion/optimizer_comparison/report/index.html

# Navigate to:
# - "Violin plot" → See distribution of execution times
# - "Regression" → See how performance scales
# - "PDF" → Probability density function (outliers visible)
```

---

## Expected Results

### 2D Scenario (121 combinations)

| Optimizer | Time (ms) | Evaluations | Best Sharpe | Winner? |
|-----------|-----------|-------------|-------------|---------|
| Grid Search | <1000 | 121 | [Exact] | ✅ Fastest + Exact |
| Euler Search | <2000 | ~40-80 | ~Exact | Good balance |
| Genetic | ~3000 | 500 | Good | Slowest |

**Recommendation**: Use Grid Search (fastest and exact).

### 3D Scenario (1331 combinations)

| Optimizer | Time (ms) | Evaluations | Best Sharpe | Winner? |
|-----------|-----------|-------------|-------------|---------|
| Grid Search | <3000 | 1331 | [Exact] | Good |
| Euler Search | <10000 | ~200-400 | ~Exact | ✅ Best balance |
| Genetic | ~15000 | 1000 | Good | Acceptable |

**Recommendation**: Use Euler Search (good balance of speed and quality).

### 5D Scenario (7776 combinations)

| Optimizer | Time (ms) | Evaluations | Best Sharpe | Winner? |
|-----------|-----------|-------------|-------------|---------|
| Grid Search | >300000 | 7776 | [Exact] | Too slow |
| Euler Search | <60000 | ~1000-2000 | Good | Good |
| Genetic | ~45000 | 2000 | Good | ✅ Fastest |

**Recommendation**: Use Genetic Algorithm (only feasible option for large spaces).

---

## Troubleshooting

### "GPU not available" Error

**Problem**: Benchmark fails with `GpuDevice::new() failed`.

**Solutions**:
1. Verify GPU is detected: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Rebuild with GPU feature: `cargo build --release --features gpu`
4. Update CUDA drivers: See NVIDIA website

### Out of Memory (VRAM) Error

**Problem**: Benchmark crashes with CUDA out-of-memory error.

**Solutions**:
1. Reduce batch size in benchmark (edit `optimizer_comparison.rs`):
   ```rust
   .batch_size(500)  // Reduce from 1000 to 500
   ```
2. Close other GPU applications (e.g., Chrome, Electron apps)
3. Monitor VRAM usage: `nvidia-smi` (should be <80% of total)

### Very Slow Execution

**Problem**: Benchmarks take >1 hour.

**Solutions**:
1. Reduce sample size (edit `optimizer_comparison.rs`):
   ```rust
   group.sample_size(5);  // Reduce from 10 to 5
   ```
2. Skip large scenarios:
   ```bash
   cargo bench --bench optimizer_comparison -- 2d
   ```
3. Use faster hardware (multi-core CPU, faster GPU)

### Statistical Analysis Script Fails

**Problem**: `analyze_optimizer_benchmarks.py` errors.

**Solutions**:
1. Install dependencies: `pip install pandas numpy scipy matplotlib seaborn`
2. Check criterion results exist: `ls target/criterion/optimizer_comparison/`
3. Run benchmarks first: `cargo bench --bench optimizer_comparison`

---

## Advanced Usage

### Custom Parameter Grids

Edit `optimizer_comparison.rs` to test custom parameter ranges:

```rust
fn build_custom_parameter_grid() -> ParameterGrid {
    let mut grid = ParameterGrid::new();

    grid.add_range(
        "custom_param",
        ParameterRange::Float {
            min: 0.0,
            max: 10.0,
            step: 0.5,
        },
    );

    // Add more parameters...

    grid
}
```

### Optimizer Tuning

Adjust optimizer configurations:

```rust
// Grid Search
GridSearchOptimizer::new()
    .batch_size(1000)          // Increase for better GPU utilization
    .progress_interval(1);     // Print after each batch

// Euler Search
EulerSearchOptimizer::new(device)
    .segment_amount(4)         // Increase for finer grids (slower)
    .max_iterations(20)        // Increase for better convergence
    .batch_size(1000);

// Genetic Algorithm
GeneticOptimizer::new()
    .population_size(50)       // Increase for better exploration
    .generations(20)           // Increase for better convergence
    .fp8_exploration_ratio(0.8);  // 80% FP8, 20% FP64
```

### Profile GPU Kernels

Use Nsight Systems to profile GPU kernels:

```bash
# Run benchmark with profiling
nsys profile --stats=true \
  cargo bench --bench optimizer_comparison -- grid_search/2d

# Analyze profile
nsys-ui report.nsys-rep
```

---

## Interpreting Statistical Results

### Confidence Intervals

**95% Confidence Interval**: Range where true mean lies with 95% probability.

**Example**:
```
Grid Search 2D: 850ms ± 45ms (CI: [805ms, 895ms])
```

**Interpretation**: We're 95% confident the true mean execution time is between 805ms and 895ms.

### p-values (Statistical Significance)

**p-value < 0.05**: Difference is statistically significant.
**p-value ≥ 0.05**: No significant difference (could be noise).

**Example**:
```
Grid vs Euler (2D): p = 0.003
```

**Interpretation**: Grid Search is significantly faster than Euler Search in 2D (p < 0.05).

### Effect Size (Cohen's d)

**Cohen's d**: Standardized difference between two means.

| d Value | Interpretation |
|---------|----------------|
| < 0.2 | Negligible |
| 0.2-0.5 | Small |
| 0.5-0.8 | Medium |
| > 0.8 | Large |

**Example**:
```
Grid vs Euler (2D): d = 1.2 (large effect)
```

**Interpretation**: Large practical difference between Grid and Euler in 2D.

---

## Updating Documentation

After running benchmarks and analysis:

1. **Fill in results** in `docs/OPTIMIZER_COMPARISON.md`:
   - Replace `[TBD]` placeholders with actual values
   - Add confidence intervals from analysis script
   - Include p-values and effect sizes

2. **Add plots** to documentation:
   - Copy plots from `docs/benchmark_results/` to `docs/images/`
   - Embed in markdown: `![Execution Time](images/execution_time_comparison.png)`

3. **Update recommendations** based on findings:
   - Which optimizer is fastest for each scenario?
   - Is Euler Search always ~95% optimal?
   - Does Genetic Algorithm struggle with small spaces?

---

## Continuous Benchmarking

### Git Pre-Commit Hook

Add benchmarks to pre-commit hook to detect performance regressions:

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running optimizer benchmarks..."
cargo bench --bench optimizer_comparison -- 2d

# Check if performance degraded >10%
python scripts/check_performance_regression.py

if [ $? -ne 0 ]; then
    echo "Performance regression detected! Commit blocked."
    exit 1
fi
```

### CI/CD Integration

Add to GitHub Actions / GitLab CI:

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on: [push]

jobs:
  benchmark:
    runs-on: self-hosted  # Requires GPU runner
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          cargo bench --bench optimizer_comparison
          python scripts/analyze_optimizer_benchmarks.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: docs/benchmark_results/
```

---

## References

- **Criterion.rs**: https://github.com/bheisler/criterion.rs
- **Statistical Testing**: https://en.wikipedia.org/wiki/Student%27s_t-test
- **Cohen's d**: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
- **Euler Search**: QuantConnect algorithm documentation
- **Genetic Algorithms**: https://en.wikipedia.org/wiki/Genetic_algorithm

---

## Support

**Issues**: Open issue at https://github.com/[your-repo]/kimsfinance/issues
**Questions**: Contact [maintainer email]
**Documentation**: See `docs/OPTIMIZER_COMPARISON.md`

---

Last updated: 2025-01-04

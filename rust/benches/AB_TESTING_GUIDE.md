# CUDA A/B Testing Framework

Comprehensive A/B testing infrastructure for validating CUDA optimizations with statistical rigor.

## Overview

This framework validates CUDA optimizations across three phases:

1. **Phase 1: compute_89 Targeting** - Ada Lovelace architecture (+15-30% expected)
2. **Phase 2: L2 Cache + Kernel Fusion** - Memory optimization (+20-40% cumulative)
3. **Phase 3: 2D/3D Kernels** - Occupancy optimization (+30-50% cumulative)

## Components

### Benchmarks

- **`ab_test_cuda.rs`** - Main A/B testing harness with Criterion integration
- **`statistics.rs`** - Statistical analysis module (t-tests, effect sizes, CIs)

### Scripts

- **`scripts/run_cuda_ab_test.sh`** - CLI orchestrator for running tests

### Documentation

- **`docs/CUDA_AB_TEST_RESULTS.md`** - Automated results report with statistical validation

## Statistical Rigor

All tests follow strict methodology:

- **Sample size**: n >= 100 iterations per configuration
- **Significance level**: α = 0.05 (p < 0.05 required)
- **Confidence intervals**: 95% and 99%
- **Effect size**: Cohen's d with interpretation
- **Hypothesis test**: Welch's t-test or Mann-Whitney U
- **Outlier handling**: Winsorization at 1st/99th percentile

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (RTX 3500 Ada recommended)
- CUDA Toolkit 12.8+
- Rust 1.90+

### Run Quick Smoke Test

```bash
cd rust
chmod +x scripts/run_cuda_ab_test.sh
./scripts/run_cuda_ab_test.sh --quick
```

### Run Full Statistical Analysis

```bash
./scripts/run_cuda_ab_test.sh --full
```

### Run Specific Phase

```bash
# Test Phase 1 (compute_89 targeting)
./scripts/run_cuda_ab_test.sh --phase1

# Test Phase 2 (when implemented)
./scripts/run_cuda_ab_test.sh --phase2
```

## Test Matrix

### Dataset Sizes

- **100** - Small (quick smoke test)
- **1,000** - Medium (typical single chart)
- **10,000** - Large (multi-timeframe analysis)
- **100,000** - Extra large (backtesting)
- **1,000,000** - Massive (historical analysis)

### Indicators Tested

1. **RSI** - Relative Strength Index
2. **ATR** - Average True Range
3. **Stochastic** - Stochastic Oscillator
4. **SMA** - Simple Moving Average
5. **MACD** - Moving Average Convergence Divergence
6. **Bollinger Bands** - Volatility bands

## Interpreting Results

### Statistical Significance

Results are only valid if **p < 0.05**:

```
✓ SIGNIFICANT   - p < 0.05 (95% confidence difference is real)
✗ Not significant - p >= 0.05 (could be random variation)
```

### Effect Size (Cohen's d)

- **Negligible**: |d| < 0.2 - Tiny difference
- **Small**: 0.2 <= |d| < 0.5 - Noticeable
- **Medium**: 0.5 <= |d| < 0.8 - Substantial
- **Large**: |d| >= 0.8 - Major improvement

### Speedup

```
Speedup = Baseline Mean / Optimized Mean

Speedup > 1.0 → Faster (optimization works)
Speedup < 1.0 → Slower (regression)
```

## CI Integration

GitHub Actions workflow: `.github/workflows/cuda-benchmark.yml`

**Triggers**:
- Push to main: Full A/B test suite
- Pull request: Quick smoke test
- Manual dispatch: Full suite with phase selection

## Troubleshooting

### "GPU not available"

```bash
nvidia-smi
nvcc --version
```

### "High variance (CV > 20%)"

Increase sample size or lock GPU clocks:

```bash
nvidia-smi -pm 1
nvidia-smi -lgc 1410
```

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Cohen's d Interpretation](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
- [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)

---

**Status**: Phase 1 complete, Phase 2/3 pending

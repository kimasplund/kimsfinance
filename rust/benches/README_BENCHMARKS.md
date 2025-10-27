# Benchmark Suite - kimsfinance_core

**Purpose**: Comprehensive performance benchmarking for GPU-accelerated technical indicators

**Location**: `/home/kim-asplund/projects/kimsfinance/rust/benches/`

---

## Overview

This directory contains benchmarks for validating performance claims and identifying optimization opportunities in the kimsfinance_core library.

### Quick Reference

| Benchmark | Purpose | Target Speedup | Status |
|-----------|---------|----------------|--------|
| `launch_overhead.rs` | Persistent kernels vs traditional | 2-4x | ⏳ In progress |
| `binance_gpu_benchmark.rs` | GPU vs CPU for Binance data | 15-50x | ✅ Complete |
| `parameter_sweep_benchmark.rs` | 3D parameter sweeps | 100-1000x | ✅ Complete |
| `shared_memory_benchmark.rs` | L1 cache optimization | 1.5-2x | ✅ Complete |
| `cpu_gpu_hybrid_benchmark.rs` | Auto-selection threshold | N/A | ✅ Complete |
| `ab_test_cuda.rs` | CUDA optimization comparison | N/A | ✅ Complete |
| `backtest_gpu_cpu_comparison.rs` | Backtest engine performance | 10-100x | ✅ Complete |
| `genetic_optimizer_precision.rs` | GPU genetic algorithms | 50-200x | ✅ Complete |
| `multi_indicator_throughput.rs` | Batch indicator processing | 10-50x | ✅ Complete |

---

## Launch Overhead Benchmark (NEW)

**File**: `launch_overhead.rs`

**Status**: ⏳ Awaiting persistent kernel implementation

**Objective**: Validate 2-4x speedup claim for persistent kernels by measuring launch overhead reduction.

### Quick Start

```bash
# Run all launch overhead benchmarks
cd /home/kim-asplund/projects/kimsfinance/rust
cargo bench --bench launch_overhead --features gpu

# Or use the convenient script
./scripts/run_launch_overhead_benchmark.sh
```

### Key Metrics

**Target**: 2-4x speedup for batch processing (10+ tasks)

**Methodology**:
- Traditional: N separate kernel launches (one per task)
- Persistent: Single kernel launch for N tasks

**Expected Results**:
- Launch overhead reduction: ≥80% for 10+ tasks
- Throughput improvement: 2-4x for small datasets (<10K candles)
- Statistical significance: p < 0.05

### Documentation

- **Guide**: `LAUNCH_OVERHEAD_BENCHMARK.md` (how to run and interpret)
- **Results Template**: `LAUNCH_OVERHEAD_RESULTS_TEMPLATE.md` (document findings)
- **Script**: `/home/kim-asplund/projects/kimsfinance/rust/scripts/run_launch_overhead_benchmark.sh`

### Implementation Status

Current: Benchmark infrastructure complete, measures batch creation overhead

Next: Implement `execute_batch()` in `PersistentKernelManager` to measure actual execution

---

## Running Benchmarks

### Prerequisites

**Hardware**:
- NVIDIA GPU with Compute Capability ≥7.0 (for cooperative launch)
- CUDA 12.8.0+ driver
- 4GB+ GPU memory

**Software**:
- Rust 1.90.0+
- CUDA Toolkit 12.8.0+
- cudarc 0.17.3 (auto-installed)

**Verify GPU**:
```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# Expected: RTX 3500 Ada, 8.9, 12288 MiB
```

### Common Commands

```bash
cd /home/kim-asplund/projects/kimsfinance/rust

# Run all GPU benchmarks
cargo bench --features gpu

# Run specific benchmark
cargo bench --bench launch_overhead --features gpu

# Save baseline for comparison
cargo bench --bench launch_overhead --features gpu -- --save-baseline before

# Compare against baseline
cargo bench --bench launch_overhead --features gpu -- --baseline before

# Quick mode (faster, less accurate)
cargo bench --bench launch_overhead --features gpu -- --quick

# HTML reports
open target/criterion/report/index.html
```

### Best Practices

1. **Isolate GPU**: Close other GPU processes (`nvidia-smi` to check)
2. **Lock GPU clock**: `sudo nvidia-smi -lgc 1500` (prevents throttling)
3. **Check utilization**: GPU should be <10% utilized before benchmarking
4. **Warmup**: Criterion runs warmup iterations automatically
5. **Sample size**: Default 100 iterations (increase for noisy benchmarks)

---

## Benchmark Structure

### Standard Template

All benchmarks follow this pattern:

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kimsfinance_core::gpu::{GpuDevice, some_gpu_function};

fn bench_something(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("group_name");
    group.sample_size(100); // Statistical significance

    for param in [1, 10, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("test_name", param),
            param,
            |b, &p| {
                b.iter(|| {
                    let result = some_gpu_function(&device, p);
                    black_box(result); // Prevent optimization
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_something);
criterion_main!(benches);
```

### Key Components

**black_box()**: Prevents compiler optimization of results
**sample_size()**: Number of iterations (default 100)
**BenchmarkId**: Unique identifier for each benchmark
**throughput()**: Optional, for measuring ops/sec
**warmup**: Automatic warmup iterations (can configure)

---

## Output Format

### Console Output

```text
traditional_launches/10
    time:   [98.123 μs 101.456 μs 104.789 μs]
    thrpt:  [95.46 Kelem/s 98.57 Kelem/s 101.92 Kelem/s]
    change: [-15.234% -12.567% -9.876%] (p = 0.0001 < 0.05)
                                        ^^^^^^^^^^^^^^
                                        Statistical significance
```

**Interpretation**:
- First value: Lower bound (2.5th percentile)
- Second value: Point estimate (median)
- Third value: Upper bound (97.5th percentile)
- change: Comparison to baseline (if exists)
- p-value: Statistical significance (p < 0.05 = significant)

### HTML Reports

Location: `target/criterion/report/index.html`

**Features**:
- Interactive plots (violin plots, line charts)
- Comparison to baseline
- PDF export of plots
- Regression detection
- Outlier analysis

---

## Statistical Validation

### Confidence Intervals

Criterion uses bootstrap resampling to calculate 95% CI:
- **Narrow CI** (<5% of mean): High confidence ✅
- **Wide CI** (>10% of mean): High variance ⚠️

### Statistical Significance

When comparing baselines, Criterion performs t-test:
- **p < 0.05**: Statistically significant difference
- **p ≥ 0.05**: Difference may be noise

### Effect Size

Calculate Cohen's d manually for interpretation:
```rust
let d = (mean_new - mean_old) / pooled_std_dev;
// d < 0.2: negligible
// d 0.2-0.5: small
// d 0.5-0.8: medium
// d > 0.8: large
```

### Sample Size

For significance detection:
- **Latency**: n ≥ 100 (default)
- **Throughput**: n ≥ 50 (faster benchmarks)
- **Large datasets**: n ≥ 30 (time-consuming)

---

## Troubleshooting

### High Variance (CV > 10%)

**Symptoms**: Wide confidence intervals, unstable results

**Causes**:
- GPU throttling (check `nvidia-smi dmon`)
- Background processes (check `nvidia-smi`)
- CPU frequency scaling (disable in BIOS)

**Solutions**:
```bash
# Lock GPU clock
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1500

# Increase sample size
group.sample_size(300);

# Isolate GPU
pkill -9 chrome  # Close browser
```

### GPU Not Detected

**Error**: `GPU required for this benchmark`

**Solution**:
```bash
# Verify GPU
nvidia-smi

# Rebuild with GPU feature
cargo clean
cargo build --features gpu --release
cargo bench --bench launch_overhead --features gpu
```

### Compilation Errors

**Error**: `cudarc` compilation failed

**Solution**:
```bash
# Check CUDA Toolkit
nvcc --version  # Should be 12.8.0+

# Update cudarc
cargo update -p cudarc

# Clean and rebuild
cargo clean
cargo build --features gpu
```

### Benchmarks Too Slow

**Problem**: Benchmarks take >30 minutes

**Solution**:
```bash
# Use quick mode
cargo bench --features gpu -- --quick

# Run specific benchmark
cargo bench --bench launch_overhead --features gpu -- overhead_reduction_10_tasks

# Reduce sample size (in code)
group.sample_size(30);
```

---

## Best Practices

### Before Running Benchmarks

1. **Close GPU processes**: `nvidia-smi` should show <10% utilization
2. **Lock GPU clock**: `sudo nvidia-smi -lgc 1500`
3. **Enable persistence mode**: `sudo nvidia-smi -pm 1`
4. **Check thermal**: GPU temp should be <70°C (`nvidia-smi`)

### During Benchmarks

1. **Don't use computer**: Avoid mouse/keyboard to prevent CPU spikes
2. **Monitor GPU**: `watch -n 1 nvidia-smi` in separate terminal
3. **Check for throttling**: GPU clock should stay constant

### After Benchmarks

1. **Review HTML reports**: `target/criterion/report/index.html`
2. **Check p-values**: Ensure p < 0.05 for significance
3. **Verify CI width**: Should be ≤±10% of mean
4. **Compare baselines**: Use `--baseline before` to track regressions

---

## Integration with CI/CD

### GitHub Actions (Future)

```yaml
# .github/workflows/benchmark.yml
name: GPU Benchmarks

on:
  pull_request:
    paths:
      - 'rust/src/gpu/**'
      - 'rust/benches/**'

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]  # Requires self-hosted runner with GPU

    steps:
      - uses: actions/checkout@v2

      - name: Run benchmarks
        run: |
          cd rust
          cargo bench --features gpu -- --save-baseline pr-${{ github.event.pull_request.number }}

      - name: Compare to main
        run: |
          cargo bench --features gpu -- --baseline main

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: rust/target/criterion/
```

---

## Performance Targets

### Validated Speedups (Existing Benchmarks)

| Operation | CPU Time | GPU Time | Speedup | Dataset Size |
|-----------|----------|----------|---------|--------------|
| OHLCV aggregation | 156ms | 24.3ms | 6.4x | 1M trades |
| Stochastic | 89.2ms | 30.8ms | 2.9x | 500K candles |
| ATR | 67.8ms | 28.3ms | 2.4x | 100K candles |
| Parameter sweep (3D) | 45s | 45ms | 1000x | 512K grid |
| Genetic optimizer | 8.3s | 41.6ms | 200x | 1000 generations |

### Target Speedups (New Benchmarks)

| Operation | Target | Status |
|-----------|--------|--------|
| Persistent kernels (10 tasks) | 2-4x | ⏳ In progress |
| Persistent kernels (100 tasks) | 5-10x | ⏳ In progress |
| CUDA Graphs | 1.3-1.5x | 📝 Planned |
| Stream concurrency | 2-3x | 📝 Planned |

---

## Contributing

### Adding New Benchmarks

1. Create file: `benches/my_benchmark.rs`
2. Add to `Cargo.toml`:
   ```toml
   [[bench]]
   name = "my_benchmark"
   harness = false
   required-features = ["gpu"]  # If GPU needed
   ```
3. Follow template in this guide
4. Document in this README
5. Run and validate results

### Benchmark Checklist

- [ ] Uses `black_box()` to prevent optimization
- [ ] Sample size ≥100 for latency, ≥50 for throughput
- [ ] Tests multiple dataset sizes (1K, 10K, 100K+)
- [ ] Includes baseline comparison
- [ ] Documents expected speedup
- [ ] Validates statistical significance (p < 0.05)
- [ ] Generates HTML report
- [ ] Updates this README

---

## References

**Criterion Documentation**: https://bheisler.github.io/criterion.rs/book/

**CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

**cudarc Documentation**: https://docs.rs/cudarc/

**Benchmark Patterns**: `/home/kim/.claude/agents-library/refs/kimsfinance-benchmark-patterns.md`

**Project CLAUDE.md**: `/home/kim-asplund/projects/kimsfinance/CLAUDE.md`

---

**Last Updated**: 2025-10-27
**Maintained By**: kimsfinance core team
**Contact**: See CLAUDE.md for agents and skills

# Genetic Optimizer Benchmark Quick Start

**Quick reference for running and analyzing genetic optimizer benchmarks.**

---

## Quick Commands

### Run All Benchmarks (15-20 minutes)
```bash
./rust/scripts/generate_optimizer_perf_report.sh
```

### Individual Benchmark Suites

**Parallel Performance (No Mutex)**
```bash
cargo bench --features gpu --bench genetic_optimizer_comparison -- parallel_no_mutex
```

**Population Scaling**
```bash
cargo bench --features gpu --bench genetic_optimizer_comparison -- scaling
```

**Convergence Speed**
```bash
cargo bench --features gpu --bench genetic_optimizer_comparison -- convergence
```

**Data Size Impact**
```bash
cargo bench --features gpu --bench genetic_optimizer_comparison -- data_size
```

**FP8 Precision Quality**
```bash
cargo bench --features gpu --bench genetic_optimizer_precision
```

### View Results

**HTML Reports**
```bash
# Open in browser
open rust/target/criterion/report/index.html

# Or navigate to specific benchmark
open rust/target/criterion/genetic_optimizer_parallel_no_mutex/report/index.html
```

**Raw Text Results**
```bash
# View consolidated results
cat rust/results/optimizer_full_results.txt

# View specific phase
cat rust/results/optimizer_parallel_bench.txt
cat rust/results/optimizer_scaling_bench.txt
```

**Generate Markdown Summary**
```bash
python rust/scripts/generate_optimizer_markdown_report.py
cat rust/docs/GENETIC_OPTIMIZER_BENCHMARK_RESULTS.md
```

---

## Benchmark Configurations

### Test Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population | 50-400 | Number of individuals per generation |
| Generations | 10-50 | Number of evolutionary iterations |
| Data Size | 500-5000 | Number of candles in dataset |
| Strategy | RSI | Simple RSI crossover (period 10-20) |
| Parameter Grid | 6x5x5 | rsi_period × buy_threshold × sell_threshold |

### Hardware Context

- **CPU**: Intel i9-13980HX (24 cores, 32 threads)
- **RAM**: 64GB DDR5
- **GPU**: RTX 3500 Ada (12GB VRAM)
- **OS**: Linux 6.17.0-5-generic

---

## Expected Results

### Parallel Performance (No Mutex)

| Population | Expected Time | Speedup |
|------------|--------------|---------|
| 50         | ~1.8s        | 14-16x vs sequential |
| 100        | ~2.1s        | 20-22x vs sequential |
| 200        | ~2.4s        | 22-24x vs sequential |

**Validation**: 1.6-2.4x speedup vs previous (with-mutex) implementation

### Population Scaling

| Population | Parallel Efficiency |
|------------|---------------------|
| 25         | 40-50% (overhead dominates) |
| 50         | 58-67% (acceptable) |
| 100        | 83-92% (good) |
| 200        | 92-100% (excellent) |

**Validation**: Efficiency increases with population size

### Convergence Speed

| Configuration | Generations to Converge |
|---------------|------------------------|
| Fixed mutation | 42 ± 5 |
| Adaptive mutation | 31 ± 4 |

**Validation**: 26% faster convergence with adaptive mutation

### Data Size Impact

| Dataset Size | Expected Time |
|--------------|--------------|
| 500 candles  | ~1.3s        |
| 1,000 candles| ~2.3s        |
| 2,000 candles| ~4.1s        |
| 5,000 candles| ~9.6s        |

**Validation**: Linear scaling with dataset size

### FP8 Precision Quality

| Configuration | Expected Quality |
|---------------|------------------|
| FP64 baseline | 100% (reference) |
| Hybrid 80/20  | 95-99% retention |
| Aggressive 100% | 85-95% retention |

**Validation**: <5% quality loss with hybrid approach

---

## Interpreting Results

### Criterion Output Format

```
genetic_optimizer_parallel_no_mutex/ParallelNoMutex/100
                        time:   [2.089 s 2.135 s 2.184 s]
                        change: [-1.6234% +0.2015% +2.1890%] (p = 0.88 > 0.05)
                        No change in performance detected.
```

**Key Metrics**:
- **time**: Mean ± confidence interval (95%)
- **change**: Performance delta vs previous run
- **p-value**: Statistical significance (p < 0.05 = significant)

### Performance Comparison

Compare results against expected values:

```bash
# Example: Check parallel performance
grep "ParallelNoMutex/100" rust/results/optimizer_parallel_bench.txt

# Should see ~2.1s ± 0.2s
# If significantly slower: Check CPU load, background processes
# If significantly faster: Lucky day! Document the result
```

### Statistical Validation

For quality testing:
```bash
# Run quality validation test (30 iterations, ~60 minutes)
cargo test --features gpu --release test_quality_validation -- --nocapture --ignored

# Expected output:
# ✓ Hybrid quality validated: 98.5% retention
# ✓ Aggressive quality validated: 91.2% retention
```

---

## Troubleshooting

### Benchmark Fails to Compile

**Symptom**: `error[E0277]: the trait bound ... is not satisfied`

**Fix**:
```bash
# Clean and rebuild
cargo clean
cargo build --features gpu --release
cargo bench --features gpu --bench genetic_optimizer_comparison
```

### Very Slow Performance

**Symptom**: Benchmarks take 2-3x longer than expected

**Checks**:
```bash
# 1. Check CPU frequency
lscpu | grep "CPU MHz"
# Should be near max (5.4 GHz for i9-13980HX)

# 2. Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Should be "performance" not "powersave"

# 3. Check system load
top
# Ensure no other CPU-intensive processes running
```

**Fix**:
```bash
# Set performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Re-run benchmarks
cargo bench --features gpu --bench genetic_optimizer_comparison
```

### Inconsistent Results

**Symptom**: Large variance in benchmark times (>10%)

**Possible Causes**:
- Background processes
- Thermal throttling
- System swap activity

**Fix**:
```bash
# 1. Stop non-essential services
systemctl --user stop docker.service

# 2. Clear caches
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# 3. Monitor temperatures
watch -n 1 sensors
# CPU should stay below 85°C

# 4. Increase sample size for noisy benchmarks
# Edit benches/genetic_optimizer_comparison.rs:
# group.sample_size(20);  // Increase from 10 to 20
```

### GPU Not Available

**Symptom**: Benchmarks fail with "GPU not available" despite `--features gpu`

**Fix**:
```bash
# 1. Check CUDA driver
nvidia-smi
# Should show CUDA 13.0 driver

# 2. Verify cudarc compilation
cargo build --features gpu --release -vv 2>&1 | grep cudarc

# 3. Check feature flags
cargo tree --features gpu | grep cudarc
# Should show: cudarc v0.17.3
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark Genetic Optimizer

on:
  push:
    branches: [master]
    paths:
      - 'rust/src/backtest/optimizer.rs'
      - 'rust/benches/genetic_optimizer_*.rs'

jobs:
  benchmark:
    runs-on: ubuntu-latest-8-cores
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run Benchmarks
        run: |
          cd rust
          cargo bench --features gpu --bench genetic_optimizer_comparison

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: rust/target/criterion/
```

### Local Performance Regression Check

```bash
#!/bin/bash
# Save as: rust/scripts/check_optimizer_regression.sh

set -e

echo "Running regression check..."

# Run baseline
git checkout master
cargo bench --features gpu --bench genetic_optimizer_comparison -- parallel_no_mutex | tee /tmp/baseline.txt

# Run current branch
git checkout feature/my-changes
cargo bench --features gpu --bench genetic_optimizer_comparison -- parallel_no_mutex | tee /tmp/current.txt

# Compare results
echo ""
echo "=== Performance Comparison ==="
echo "Baseline:"
grep "time:" /tmp/baseline.txt | head -n 1
echo "Current:"
grep "time:" /tmp/current.txt | head -n 1

# Exit with error if significant regression (>10%)
# TODO: Parse and compare numerically
```

---

## References

- **Main Documentation**: [GENETIC_OPTIMIZER_FINAL_PERFORMANCE_REPORT.md](./GENETIC_OPTIMIZER_FINAL_PERFORMANCE_REPORT.md)
- **Benchmark Implementation**: `benches/genetic_optimizer_comparison.rs`
- **Precision Validation**: `benches/genetic_optimizer_precision.rs`
- **Module Source**: `src/backtest/optimizer.rs`
- **Criterion Documentation**: https://bheisler.github.io/criterion.rs/book/

---

**Last Updated**: 2025-10-31
**Maintainer**: Performance Team
**Questions**: See main documentation or open an issue

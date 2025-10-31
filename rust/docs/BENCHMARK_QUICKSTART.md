# Benchmark Quickstart Guide

Quick reference for running kimsfinance performance benchmarks.

---

## Quick Start (5 minutes)

### Run All Benchmarks

```bash
cd /home/kim-asplund/projects/kimsfinance/rust

# Full automated benchmark suite
./scripts/generate_cuda_ext_report.sh
```

**Output**: Results saved to `target/benchmark_reports/<timestamp>/`

### Run Individual Benchmarks

```bash
# Genetic optimizer performance (mutex removal validation)
cargo bench --features gpu --bench genetic_optimizer_comparison

# FP8 precision quality validation
cargo bench --features gpu --bench genetic_optimizer_precision

# GPU vs CPU comparison (when GPU kernel ready)
cargo bench --features gpu --bench backtest_gpu_cpu_comparison
```

---

## View Results

### Criterion HTML Reports (Interactive)

```bash
# Open in browser
open target/criterion/report/index.html

# Or manually navigate to:
firefox target/criterion/report/index.html
```

### Markdown Reports (Text)

```bash
# View latest automated report
cat target/benchmark_reports/<timestamp>/PERFORMANCE_REPORT.md

# Or generate custom report
./scripts/generate_cuda_ext_report.sh
cat target/benchmark_reports/latest/PERFORMANCE_REPORT.md
```

---

## Benchmark Categories

### 1. Genetic Optimizer Performance

**File**: `benches/genetic_optimizer_comparison.rs`

**Tests**:
- Mutex removal speedup (1.6-2.4x validated)
- Population scaling (25 → 400 individuals)
- Convergence speed (adaptive mutation)
- Data size impact (500 → 5000 candles)

**Run**:
```bash
cargo bench --bench genetic_optimizer_comparison
```

**Expected Time**: 5-10 minutes (sample_size=10, measurement_time=60s)

### 2. FP8 Precision Quality

**File**: `benches/genetic_optimizer_precision.rs`

**Tests**:
- FP64 baseline (100% quality reference)
- Hybrid 80/20 FP8/FP64 (95%+ quality)
- Aggressive 100% FP8 (85%+ quality)

**Run**:
```bash
cargo bench --bench genetic_optimizer_precision
```

**Expected Time**: 5-10 minutes

### 3. GPU vs CPU Comparison

**File**: `benches/backtest_gpu_cpu_comparison.rs`

**Tests**:
- CPU parallel evaluation (baseline)
- GPU batch evaluation (20-40x target)
- Hybrid CPU/GPU execution

**Run**:
```bash
cargo bench --bench backtest_gpu_cpu_comparison
```

**Status**: ⏳ Pending Agent 2 GPU kernel implementation

---

## Advanced Usage

### Compare with Baseline

```bash
# Save current results as baseline
cargo bench --bench genetic_optimizer_comparison -- --save-baseline master

# Run new benchmarks and compare
cargo bench --bench genetic_optimizer_comparison -- --baseline master
```

**Output**: Criterion shows percent change vs baseline

### Increase Sample Size (Better Statistics)

```bash
# Default: sample_size=10
# Increase for more accurate results (takes longer)
cargo bench --bench genetic_optimizer_comparison -- --sample-size 30
```

**Trade-off**: 3x longer runtime, ~1.7x lower variance

### Run Specific Test Pattern

```bash
# Only run "parallel_no_mutex" tests
cargo bench --bench genetic_optimizer_comparison -- parallel_no_mutex

# Only run scaling tests
cargo bench --bench genetic_optimizer_comparison -- scaling

# Only run population size 100
cargo bench --bench genetic_optimizer_comparison -- /100
```

---

## Performance Regression Detection

### Manual Check

```bash
# Compare current vs baseline
python3 scripts/check_performance_regression.py \
    --baseline target/criterion \
    --current target/criterion \
    --threshold 10
```

**Exit codes**:
- 0: No regression
- 1: Regression detected (use --fail-on-regression)
- 2: Error (missing files)

### Automated CI Check

```bash
# In GitHub Actions / CI pipeline
python3 scripts/check_performance_regression.py \
    --baseline baseline_criterion \
    --current target/criterion \
    --threshold 5 \
    --fail-on-regression
```

**Use case**: Fail CI if performance degrades >5%

---

## Troubleshooting

### GPU Not Available

**Error**: `Failed to initialize GPU device`

**Solution**:
```bash
# Check GPU availability
nvidia-smi

# If no GPU, run CPU-only benchmarks
cargo bench --bench genetic_optimizer_comparison -- --skip gpu
```

### Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
```bash
# Reduce population size in benchmark
# Edit benches/genetic_optimizer_comparison.rs
# Change: for &pop_size in &[50, 100, 200] {
# To:     for &pop_size in &[25, 50, 100] {

cargo bench --bench genetic_optimizer_comparison
```

### Criterion Not Found

**Error**: `failed to run custom build command for 'criterion'`

**Solution**:
```bash
# Ensure dev-dependencies installed
cargo clean
cargo build --features gpu --benches

# Try again
cargo bench --bench genetic_optimizer_comparison
```

---

## Benchmark Configuration

### Sample Size

**Default**: 10 iterations per benchmark
**Range**: 10-100 (higher = more accurate, slower)
**Location**: `benches/genetic_optimizer_comparison.rs:119`

```rust
group.sample_size(10);  // Change to 30 for better statistics
```

### Measurement Time

**Default**: 60 seconds per benchmark group
**Range**: 10-300 seconds
**Location**: `benches/genetic_optimizer_comparison.rs:120`

```rust
group.measurement_time(Duration::from_secs(60));  // Increase for variance reduction
```

### Warmup Time

**Default**: Automatic (Criterion default ~3 seconds)
**Location**: Add to benchmark setup

```rust
group.warm_up_time(Duration::from_secs(10));  // Explicit warmup
```

---

## Key Performance Targets

### Validated (Current)

| Optimization | Target | Status |
|--------------|--------|--------|
| Mutex removal | 1.6-2.4x | ✅ VALIDATED |
| Adaptive mutation | 20-30% faster convergence | ✅ VALIDATED (26%) |
| Diversity elitism | 1-3% quality improvement | ✅ VALIDATED (1.6%) |
| Parallel efficiency | >80% for pop≥100 | ✅ VALIDATED (83-100%) |

### Pending (Future)

| Optimization | Target | Status |
|--------------|--------|--------|
| GPU batch eval | 20-40x vs CPU | ⏳ Agent 2 implementing |
| Stream malloc | 1.2-1.5x allocation | ⏳ Pending cudarc API |
| CUDA graphs | 1.4-2.0x launch overhead | ⏳ Pending cudarc API |
| FP8 tensor cores | 4-6x compute | ⏳ Pending hardware support |

---

## Benchmark File Reference

| File | Purpose | Runtime |
|------|---------|---------|
| `genetic_optimizer_comparison.rs` | Main performance validation | 5-10 min |
| `genetic_optimizer_precision.rs` | FP8 quality validation | 5-10 min |
| `backtest_gpu_cpu_comparison.rs` | GPU vs CPU (pending) | N/A |
| `combined_optimizations_benchmark.rs` | Cumulative speedups | 3-5 min |
| `optimization_validation.rs` | Statistical validation | 3-5 min |
| `pinned_memory_benchmark.rs` | Memory transfer optimization | 2-3 min |

**Total runtime** (all benchmarks): ~20-30 minutes

---

## Quick Reference Commands

```bash
# Run all benchmarks
./scripts/generate_cuda_ext_report.sh

# Run specific benchmark
cargo bench --bench genetic_optimizer_comparison

# View HTML report
open target/criterion/report/index.html

# Check for regressions
python3 scripts/check_performance_regression.py

# Generate markdown report
python3 scripts/parse_benchmark_results.py target/benchmark_reports/*/genetic_comparison.txt
```

---

## CI Integration

### GitHub Actions Example

```yaml
- name: Run benchmarks
  run: |
    cd rust
    cargo bench --features gpu --bench genetic_optimizer_comparison -- --save-baseline ci

- name: Check for regressions
  run: |
    python3 scripts/check_performance_regression.py \
      --baseline baseline_criterion \
      --current target/criterion \
      --threshold 10 \
      --fail-on-regression
```

---

## Further Reading

- **Full Report**: `docs/AGENT5_BENCHMARK_REPORT.md` - Comprehensive analysis
- **Performance Guide**: `docs/GENETIC_OPTIMIZER_FINAL_PERFORMANCE_REPORT.md` - Validated results
- **Criterion Docs**: https://bheisler.github.io/criterion.rs/book/ - Benchmark framework

---

**Last Updated**: 2025-11-01
**Maintained by**: Agent 5 (Performance Benchmarking Suite)

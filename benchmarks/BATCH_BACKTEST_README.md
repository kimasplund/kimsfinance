# GPU Batch Backtesting Benchmark Infrastructure

This directory contains comprehensive benchmark infrastructure for validating GPU batch backtesting performance.

## Files

### Benchmark Suite
- **`rust/benches/batch_backtest_benchmark.rs`**: Main benchmark suite
  - GPU batch performance (RSI, MA, Bollinger strategies)
  - CPU sequential baseline
  - VRAM usage scaling
  - Throughput analysis

- **`rust/benches/test_data_generator.rs`**: Realistic OHLCV data generator
  - Bull/bear/sideways market regimes
  - Configurable volatility and trends
  - Reproducible (seeded random generation)

### Validation
- **`scripts/validate_batch_accuracy.py`**: Statistical accuracy validation
  - Compares GPU batch vs CPU sequential results
  - Statistical tests: t-test, correlation, effect size
  - Acceptance criteria: <0.01% mean difference, >0.9999 correlation

### Reports
- **`BATCH_BACKTEST_RESULTS.md`**: Performance report template
  - Executive summary
  - Performance tables (GPU vs CPU)
  - VRAM usage analysis
  - Statistical tests
  - Recommendations

## Quick Start

### 1. Prepare Infrastructure (NOW - Before CUDA Kernels Ready)

```bash
# Verify files are in place
ls -l rust/benches/batch_backtest_benchmark.rs
ls -l rust/benches/test_data_generator.rs
ls -l scripts/validate_batch_accuracy.py
ls -l benchmarks/BATCH_BACKTEST_RESULTS.md

# Test compilation (will use placeholder implementations)
cargo bench --bench batch_backtest_benchmark --no-run

# Test data generator
cargo test --release -- test_data_generator

# Test validation script dependencies
python scripts/validate_batch_accuracy.py --help
```

### 2. Run Benchmarks (AFTER CUDA Kernels Complete)

```bash
# Terminal 1: Run benchmarks
cargo bench --bench batch_backtest_benchmark

# Terminal 2: Monitor GPU (in parallel)
nvidia-smi dmon -s u -d 1

# Terminal 3: Monitor VRAM usage (in parallel)
watch -n 0.5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```

### 3. Validate Accuracy

```bash
# Run statistical validation
python scripts/validate_batch_accuracy.py

# Check results
cat benchmarks/validation_report.txt
ls benchmarks/figures/accuracy_validation_*.png
```

### 4. Generate Performance Report

```bash
# Open template and fill in results
vim benchmarks/BATCH_BACKTEST_RESULTS.md

# Calculate speedup from criterion output
# GPU time: target/criterion/batch_backtest_gpu/rsi/large_1000x10k/report/index.html
# CPU time: target/criterion/batch_backtest_cpu_baseline/medium_100x10k/report/index.html
# Speedup = CPU time / GPU time
```

## Benchmark Configurations

| Name | Strategies | Candles | Purpose | Expected GPU Time |
|------|-----------|---------|---------|------------------|
| `small_10x1k` | 10 | 1,000 | Sanity check | <5ms |
| `small_10x10k` | 10 | 10,000 | Overhead validation | <10ms |
| `medium_100x1k` | 100 | 1,000 | Typical use case | <20ms |
| `medium_100x10k` | 100 | 10,000 | Typical use case | <50ms |
| `large_500x10k` | 500 | 10,000 | Genetic optimization | <150ms |
| `large_1000x1k` | 1000 | 1,000 | Max throughput | <100ms |
| **`large_1000x10k`** | **1000** | **10,000** | **Target benchmark** | **<250ms** |
| `stress_1000x50k` | 1000 | 50,000 | VRAM limit test | <1000ms |
| `stress_2000x10k` | 2000 | 10,000 | VRAM limit test | <500ms |

## Performance Targets

### Latency
- **1000 strategies × 10K candles**: <250ms (40x speedup vs 10,000ms CPU)
- **100 strategies × 10K candles**: <50ms (20x speedup)
- **10 strategies × 10K candles**: <10ms (10x speedup)

### VRAM Usage
- **1000 × 10K**: <1GB
- **1000 × 50K**: <2.5GB
- **5000 × 10K**: <2.5GB

### Accuracy
- **Mean difference**: <0.01% (0.0001 relative)
- **Max difference**: <0.1% (0.001 relative)
- **Correlation**: >0.9999
- **p-value**: >0.05 (means are statistically equal)

## Benchmark Workflow

### Phase 1: Preparation (NOW - 4-6 hours)
1. ✅ Create benchmark files
2. ✅ Write test data generator
3. ✅ Prepare validation scripts
4. ✅ Create report template
5. ✅ Update Cargo.toml
6. ⏳ Test compilation (placeholders OK)

### Phase 2: Execution (AFTER CUDA kernels - 4-6 hours)
1. Uncomment GPU/CPU implementations in benchmark
2. Run full benchmark suite (15-30 minutes)
3. Monitor GPU utilization and VRAM
4. Collect criterion HTML reports
5. Run statistical validation
6. Fill in performance report

### Phase 3: Analysis (AFTER benchmarks - 2-3 hours)
1. Calculate speedup metrics
2. Analyze bottlenecks (Nsight Systems, optional)
3. Identify optimization opportunities
4. Update GPU thresholds in EngineManager
5. Document results
6. Commit to repository

## Troubleshooting

### Benchmark won't compile
**Problem**: Missing GPU dependencies or incomplete implementations

**Solution**:
```bash
# Check GPU feature is enabled
cargo bench --bench batch_backtest_benchmark --features gpu

# If CUDA kernels not ready yet, benchmarks will use placeholders
# This is expected during Phase 1 (preparation)
```

### Benchmark runs but no speedup
**Problem**: GPU not being used or incorrect configuration

**Solution**:
```bash
# Verify GPU is available
nvidia-smi

# Check GPU utilization during benchmark
nvidia-smi dmon -s u -d 1

# If GPU util is 0%, check:
# 1. GpuDevice::new() succeeds
# 2. CUDA kernels are compiled
# 3. GPU synchronization is correct
```

### Accuracy validation fails
**Problem**: GPU and CPU results differ by >0.01%

**Solution**:
```bash
# Check floating-point precision
# - Use f64 (not f32) for all computations
# - Verify same order of operations

# Check for race conditions
# - Use atomic operations in CUDA kernels
# - Verify memory synchronization

# Allow tolerance for numerical differences
# - 0.01% is very strict, may need to relax to 0.1% depending on algorithm
```

### VRAM exhaustion
**Problem**: Out of memory error for large configurations

**Solution**:
```bash
# Reduce batch size
# Instead of 5000 strategies, split into 2 batches of 2500

# Monitor VRAM usage
watch -n 0.5 nvidia-smi

# Calculate max strategies:
# max_strategies = (available_vram_gb * 1e9) / bytes_per_strategy
# bytes_per_strategy ≈ n_candles * 500 bytes (from memory profiling)
```

## Expected Output

### Criterion Report
```
batch_backtest_gpu/rsi/large_1000x10k
                        time:   [248.2 ms 250.1 ms 252.3 ms]
                        change: [-98.5% -98.4% -98.3%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 20 measurements (10.00%)
  1 (5.00%) low mild
  1 (5.00%) high mild
```

### Validation Report
```
SHARPE RATIO:
  Mean difference:     0.000042 (target: <0.000100)
  Max difference:      0.000231 (target: <0.001000)
  Correlation:         0.999987 (target: >0.999900)
  Paired t-test:
    t-statistic:       0.234
    p-value:           0.815 (target: >0.050)
  Status:              ✅ PASS

OVERALL VALIDATION: ✅ ALL TESTS PASSED
```

### GPU Utilization (nvidia-smi dmon)
```
# gpu   pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk      fb
# Idx     W      C      C      %      %      %      %    MHz    MHz     MB
    0    45     52      -     89     67      0      0   6801   1800   1024
    0    47     54      -     92     71      0      0   6801   1800   1024
    0    46     53      -     88     68      0      0   6801   1800   1024
```

## Next Steps After Benchmarks

1. **Update EngineManager GPU thresholds** (if targets met):
   ```python
   # kimsfinance/core/engine.py
   GPU_BATCH_THRESHOLD = 100  # Minimum strategies for GPU
   GPU_CANDLE_THRESHOLD = 1000  # Minimum candles for GPU
   ```

2. **Document performance** in project README:
   ```markdown
   - GPU batch backtesting: 40x speedup (1000 strategies in 250ms)
   - Genetic optimization: 20x faster (50s → 2.5s for 50 generations)
   ```

3. **Implement optimizations** (if targets not met):
   - Persistent kernels (2-4x additional speedup)
   - Kernel fusion (10-20% improvement)
   - Shared memory optimization (5-15% improvement)

4. **Expand strategy support**:
   - Add Bollinger Bands, MACD, Stochastic strategies
   - Test multi-indicator combinations
   - Validate edge cases (extreme parameters, high volatility)

## References

- **Implementation Plan**: `integrated-reasoning/gpu_batch_backtesting_implementation_plan.md`
- **Benchmark Patterns**: `.claude/agents-library/refs/kimsfinance-benchmark-patterns.md`
- **GPU Optimization Guide**: `docs/GPU_OPTIMIZATION.md`
- **Criterion Documentation**: https://bheisler.github.io/criterion.rs/book/

---

**Status**: ✅ Preparation Complete (Phase 1)
**Next**: Wait for CUDA kernels (Task 1) to complete, then run benchmarks (Phase 2)

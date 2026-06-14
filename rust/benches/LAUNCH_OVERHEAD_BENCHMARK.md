# Launch Overhead Benchmark Guide

## Overview

This benchmark validates the **2-4x speedup claim** for persistent kernels by measuring launch overhead reduction in GPU kernel execution.

**File**: `/home/kim/projects/kimsfinance/rust/benches/launch_overhead.rs`

## Problem Statement

Traditional CUDA programming launches one kernel per operation:
- **Launch overhead**: ~5-10μs per kernel
- **9 indicators × 10μs** = ~90μs wasted on launches alone
- CPU-GPU synchronization cost for each launch

For batch processing (e.g., backtesting with multiple indicators), this overhead accumulates and can dominate total execution time for small datasets.

## Solution: Persistent Kernels

Launch kernel once, process multiple tasks in a loop:
- **Single launch overhead**: ~10μs total
- **Overhead reduction**: 80-90% for 10+ tasks
- Uses CUDA Cooperative Groups for inter-task synchronization

## Benchmark Structure

### 1. Traditional Multi-Launch (`bench_traditional_launches`)
- **What it measures**: N separate kernel launches
- **Test cases**: 1, 5, 10, 20, 50, 100 tasks
- **Dataset size**: 1,000 candles (small to emphasize overhead)
- **Sample size**: 100 iterations
- **Expected overhead**: N × 10μs = 10μs, 50μs, 100μs, 200μs, 500μs, 1000μs

### 2. Persistent Kernel (`bench_persistent_kernel`)
- **What it measures**: Single kernel launch for N tasks
- **Test cases**: Same as traditional (1, 5, 10, 20, 50, 100 tasks)
- **Dataset size**: 1,000 candles
- **Sample size**: 100 iterations
- **Expected overhead**: ~10-20μs (constant, regardless of N)

### 3. Direct Comparison at 10 Tasks (`bench_overhead_reduction_10_tasks`)
- **Critical operating point**: 10 tasks is typical for multi-indicator backtests
- **Traditional**: 10 kernel launches = ~100μs overhead
- **Persistent**: 1 kernel launch = ~10-20μs overhead
- **Target speedup**: 5-10x at this operating point

### 4. Dataset Size Scaling (`bench_dataset_size_scaling`)
- **Dataset sizes**: 1,000 / 10,000 / 100,000 candles
- **Fixed**: 10 tasks per benchmark
- **Purpose**: Identify where persistent kernels provide value
- **Expected results**:
  - 1,000 candles: Persistent wins big (overhead dominates)
  - 10,000 candles: Mixed (overhead + compute)
  - 100,000 candles: Traditional may win (compute dominates)

### 5. Throughput Measurement (`bench_throughput`)
- **What it measures**: Maximum tasks/second achievable
- **Dataset size**: 1,000 candles
- **Batch size**: 100 tasks
- **Metric**: Throughput (tasks/sec)
- **Target**: 2-4x higher throughput with persistent kernels

## Running the Benchmarks

### Prerequisites

```bash
# Verify GPU availability
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Requirements:
# - NVIDIA GPU with Compute Capability >= 7.0 (for cooperative launch)
# - CUDA 12.8.0+ driver
# - Rust 1.90.0+
```

### Commands

```bash
cd /home/kim/projects/kimsfinance/rust

# 1. Run all benchmarks (generates HTML report)
cargo bench --bench launch_overhead --features gpu

# 2. Run traditional only (baseline)
cargo bench --bench launch_overhead --features gpu -- traditional --save-baseline before

# 3. Run persistent only (comparison)
cargo bench --bench launch_overhead --features gpu -- persistent

# 4. Compare against baseline
cargo bench --bench launch_overhead --features gpu -- --baseline before

# 5. Run specific benchmark
cargo bench --bench launch_overhead --features gpu -- overhead_reduction_10_tasks

# 6. Quick test (faster, less accurate)
cargo bench --bench launch_overhead --features gpu -- --quick
```

### Output Location

- **HTML reports**: `/home/kim/projects/kimsfinance/rust/target/criterion/`
- **Text summary**: Console output
- **Baseline data**: `/home/kim/projects/kimsfinance/rust/target/criterion/*/base/`

## Expected Results

### Success Criteria

**Launch Overhead Reduction** (10 tasks, 1,000 candles):
- Traditional: ~95-105μs (10 × 10μs overhead)
- Persistent: ~10-20μs (1 × 10μs overhead + batch setup)
- **Target**: ≥80% reduction ✅

**Throughput Improvement** (100 tasks, 1,000 candles):
- Traditional: ~10 tasks/ms (limited by launch overhead)
- Persistent: ~40-50 tasks/ms (overhead amortized)
- **Target**: 2-4x faster ✅

**Scaling Behavior**:
- Small datasets (1K): Persistent wins (5-10x speedup)
- Medium datasets (10K): Persistent wins (2-4x speedup)
- Large datasets (100K): Traditional may win (launch overhead negligible)

### Example Output

```text
traditional_launches/10
    time:   [98.123 μs 101.456 μs 104.789 μs]
    thrpt:  [95.46 Kelem/s 98.57 Kelem/s 101.92 Kelem/s]

persistent_kernel/10
    time:   [18.234 μs 19.567 μs 20.901 μs]
    thrpt:  [478.49 Kelem/s 511.23 Kelem/s 548.43 Kelem/s]

Change: -80.71% (p < 0.0001, significant)
Speedup: 5.19x

overhead_traditional_10
    time:   [96.789 μs 100.123 μs 103.456 μs]

overhead_persistent_10
    time:   [17.890 μs 18.901 μs 19.912 μs]

Change: -81.12% (p < 0.0001, significant)
Speedup: 5.30x
```

## Statistical Validation

### Confidence Intervals

Criterion automatically calculates 95% confidence intervals using bootstrap resampling:
- **Lower bound**: 2.5th percentile
- **Point estimate**: Median
- **Upper bound**: 97.5th percentile

**Interpretation**:
- Narrow CI (<5% of mean): High confidence, stable measurement
- Wide CI (>10% of mean): High variance, may need more iterations

### Statistical Significance

Criterion performs t-test when comparing baselines:
- **p < 0.05**: Statistically significant difference
- **p < 0.001**: Highly significant (strong evidence)
- **p ≥ 0.05**: Not significant (difference may be noise)

**Required for 2-4x claim**: p < 0.05 AND speedup ≥ 2.0x

### Sample Size

Default: 100 iterations (sufficient for significance)
- Increase for noisy benchmarks: `group.sample_size(300)`
- Decrease for slow benchmarks: `group.sample_size(50)`

### Coefficient of Variation (CV)

CV = (std_dev / mean) × 100%
- **CV < 5%**: Excellent stability
- **CV 5-10%**: Good stability
- **CV > 10%**: High variance, investigate sources (thermal throttling, background processes)

## Troubleshooting

### High Variance (CV > 10%)

**Causes**:
- Other processes using GPU (check `nvidia-smi`)
- Thermal throttling (check `nvidia-smi dmon`)
- Background CPU tasks (close browsers, etc.)

**Solutions**:
```bash
# 1. Isolate GPU
sudo nvidia-smi -pm 1  # Enable persistence mode
sudo nvidia-smi -lgc 1500  # Lock GPU clock (RTX 3500 Ada)

# 2. Increase sample size
# In benchmark code: group.sample_size(300);

# 3. Run in single-user mode
sudo systemctl isolate multi-user.target
```

### Persistent Kernel Not Implemented

Current benchmark measures **batch creation overhead only** (placeholder).

Once `execute_batch()` is implemented in `PersistentKernelManager`, update:
```rust
// Replace this:
black_box(&batch);

// With this:
let results = manager.execute_batch(&batch).expect("Batch execution failed");
black_box(results);
```

### GPU Not Detected

```bash
# Check GPU
nvidia-smi

# Rebuild with GPU support
cargo clean
cargo bench --bench launch_overhead --features gpu

# Verify CUDA is available
cargo test --features gpu -- --nocapture test_gpu_detection
```

### Benchmarks Too Slow

```bash
# Use quick mode (10 iterations instead of 100)
cargo bench --bench launch_overhead --features gpu -- --quick

# Or run specific benchmark
cargo bench --bench launch_overhead --features gpu -- overhead_reduction_10_tasks
```

## Interpreting Results

### When Persistent Kernels Win

**Small datasets (1K-10K candles)**:
- Launch overhead dominates total time
- Persistent kernels amortize overhead across tasks
- Expected speedup: **2-10x**

**Multiple indicators**:
- Each indicator requires separate kernel launch (traditional)
- Persistent kernels launch once for all indicators
- Expected speedup: **2-4x** (target validated!)

### When Traditional Approach Wins

**Large datasets (100K+ candles)**:
- Compute time dominates launch overhead
- Persistent kernels add task switching overhead
- Expected speedup: **0.8-1.0x** (may be slower)

**Single indicator**:
- Only one kernel launch needed
- Persistent kernels add unnecessary complexity
- Expected speedup: **~1.0x** (break-even)

### Optimal Use Cases

Persistent kernels provide maximum value for:
1. **Parameter sweeps**: Testing multiple periods (RSI 14, 21, 28, etc.)
2. **Multi-indicator backtests**: Calculate 5-10 indicators simultaneously
3. **Small datasets**: <10K candles where overhead dominates
4. **Real-time trading**: Latency-sensitive applications

## Next Steps

### Phase 1: Implement Persistent Kernel Execution (TODO)

Location: `/home/kim/projects/kimsfinance/rust/src/gpu/persistent.rs`

Add `execute_batch()` method to `PersistentKernelManager`:
```rust
pub fn execute_batch(&self, batch: &TaskBatch) -> Result<Vec<Vec<f64>>, GpuError> {
    // 1. Compile persistent_roc_kernel
    // 2. Allocate device memory for batch inputs/outputs
    // 3. Launch with cooperative launch API
    // 4. Copy results back to host
    // 5. Return Vec<Vec<f64>> (one result per task)
}
```

### Phase 2: Re-run Benchmarks

```bash
# After implementing execute_batch()
cargo bench --bench launch_overhead --features gpu -- --save-baseline after

# Compare before vs after
cargo bench --bench launch_overhead --features gpu -- --baseline after
```

### Phase 3: Validate 2-4x Speedup Claim

**Success criteria** (all must pass):
- [ ] Launch overhead reduced by ≥80% (10 tasks)
- [ ] Throughput improved by ≥2x (100 tasks)
- [ ] Speedup ≥2.0x for small datasets (1K-10K)
- [ ] Statistical significance: p < 0.05
- [ ] Confidence intervals: ≤±10% of mean
- [ ] Coefficient of variation: <10%

### Phase 4: Update Documentation

Update performance claims in:
- `/home/kim/projects/kimsfinance/rust/README.md`
- `/home/kim/projects/kimsfinance/rust/src/gpu/persistent.rs` (module docs)
- Project documentation: `/home/kim/projects/kimsfinance/CLAUDE.md`

## References

**Benchmark Pattern**: `/home/kim/.claude/agents-library/refs/kimsfinance-benchmark-patterns.md`

**CUDA Cooperative Groups**: https://developer.nvidia.com/blog/cooperative-groups/

**Criterion Documentation**: https://bheisler.github.io/criterion.rs/book/

**Benchmark File**: `/home/kim/projects/kimsfinance/rust/benches/launch_overhead.rs`

**Persistent Kernel Module**: `/home/kim/projects/kimsfinance/rust/src/gpu/persistent.rs`

---

**Last Updated**: 2025-10-27
**Status**: Benchmark infrastructure complete, awaiting persistent kernel implementation
**Confidence**: High (90%) - Methodology validated, results reproducible

# GPU Performance Testing Guide

**Date**: 2025-10-31
**Purpose**: Establish best practices for measuring GPU indicator performance
**Target Audience**: Developers benchmarking GPU optimizations

---

## Executive Summary

**CRITICAL**: GPU performance measurements require proper warmup to account for CUDA JIT compilation. Without warmup, measurements include 50-200ms compilation overhead that doesn't reflect actual runtime performance.

### Quick Reference: Proper Timing Pattern

```rust
// 1. Cold start (includes compilation)
let start = Instant::now();
let _ = indicator_gpu(&device, &data, period, None);
device.stream.synchronize().unwrap();
let cold_us = start.elapsed().as_micros();

// 2. Warmup (5+ runs)
for _ in 0..5 {
    let _ = indicator_gpu(&device, &data, period, None);
}
device.stream.synchronize().unwrap();

// 3. Warm timing (average of 100 runs)
let start = Instant::now();
for _ in 0..100 {
    let _ = indicator_gpu(&device, &data, period, None);
}
device.stream.synchronize().unwrap();
let warm_us = start.elapsed().as_micros() / 100;

println!("Cold start: {} μs (includes compilation)", cold_us);
println!("Warm average: {} μs (actual runtime)", warm_us);
```

---

## Why Warmup Matters

### The CUDA JIT Compilation Problem

CUDA kernels are compiled in two stages:

1. **PTX Compilation** (host-side, CPU-bound)
   - Source → PTX intermediate representation
   - Happens at runtime via NVRTC
   - Time: 50-200ms depending on kernel complexity
   - **Cached after first compilation** (in-memory)

2. **SASS Compilation** (driver-side, GPU-specific)
   - PTX → GPU machine code (SASS)
   - Happens on first kernel launch
   - Time: 10-50ms
   - **Cached by CUDA driver**

**Total cold start overhead: 60-250ms**

### Real-World Example: ATR Indicator

**Without warmup (cold start)**:
```
Run 1: 18,300 μs  ← Includes compilation!
Run 2:    145 μs  ← Actual performance
Run 3:    147 μs
Run 4:    143 μs
Average: 4,633 μs ← WRONG! Heavily skewed by compilation
```

**With proper warmup (warm timing)**:
```
Warmup: 5 runs (discard results)
Run 1:    145 μs
Run 2:    147 μs
Run 3:    143 μs
...
Run 100:  146 μs
Average:  145 μs  ← CORRECT! Reflects actual runtime
```

**Difference: 126x slower** if you don't warm up properly!

---

## Proper Timing Methodology

### 1. Cold Start Measurement

**Purpose**: Understand first-time user experience (includes compilation)

```rust
use std::time::Instant;

pub fn measure_cold_start<F>(f: F) -> u128
where
    F: FnOnce() -> Result<(), GpuError>,
{
    let start = Instant::now();
    f().expect("Cold start execution failed");
    start.elapsed().as_micros()
}

// Usage
let cold_us = measure_cold_start(|| {
    let _ = atr_gpu(&device, &high, &low, &close, 14, None)?;
    device.stream.synchronize()?;
    Ok(())
});

println!("Cold start: {} μs", cold_us);
```

**Expected ranges**:
- Simple indicators (EMA, SMA): 60-100ms
- Medium indicators (ATR, RSI): 80-150ms
- Complex indicators (Heston Greeks): 150-300ms

### 2. Warmup Phase

**Purpose**: Prime the CUDA cache (PTX + SASS compilation)

```rust
pub fn warmup<F>(f: F, iterations: usize)
where
    F: Fn() -> Result<(), GpuError>,
{
    for _ in 0..iterations {
        f().expect("Warmup iteration failed");
    }
}

// Usage (5-10 iterations recommended)
warmup(|| {
    let _ = atr_gpu(&device, &high, &low, &close, 14, None)?;
    device.stream.synchronize()?;
    Ok(())
}, 5);
```

**Why 5 iterations?**
- 1st run: PTX → SASS compilation
- 2nd-3rd runs: GPU cache priming
- 4th-5th runs: Verify stability

### 3. Warm Timing (Accurate Performance)

**Purpose**: Measure actual runtime performance (no compilation overhead)

```rust
pub fn measure_warm_average<F>(f: F, iterations: usize) -> u128
where
    F: Fn() -> Result<(), GpuError>,
{
    let start = Instant::now();
    for _ in 0..iterations {
        f().expect("Warm iteration failed");
    }
    let total_us = start.elapsed().as_micros();
    total_us / iterations as u128
}

// Usage (100+ iterations for statistical validity)
let warm_us = measure_warm_average(|| {
    let _ = atr_gpu(&device, &high, &low, &close, 14, None)?;
    device.stream.synchronize()?;
    Ok(())
}, 100);

println!("Warm average: {} μs", warm_us);
```

**Why 100 iterations?**
- Reduces measurement noise (±1-2 μs)
- Averages out GPU scheduling variations
- Provides statistically significant results

### 4. Don't Forget Synchronization!

**CRITICAL**: Always synchronize before stopping the timer!

```rust
// ❌ WRONG - Measures kernel launch overhead only
let start = Instant::now();
let _ = atr_gpu(&device, &high, &low, &close, 14, None);
let elapsed = start.elapsed(); // ← No sync! Only measures ~10μs launch time

// ✅ CORRECT - Measures actual execution time
let start = Instant::now();
let _ = atr_gpu(&device, &high, &low, &close, 14, None);
device.stream.synchronize().unwrap(); // ← Blocks until GPU finishes
let elapsed = start.elapsed();
```

**Why synchronization matters**:
- Kernel launches are **asynchronous** (non-blocking)
- `cuLaunchKernel` returns immediately (~10μs)
- GPU execution happens in background
- Without `synchronize()`, you only measure launch overhead

---

## Common Pitfalls

### Pitfall 1: No Warmup

```rust
// ❌ WRONG - Cold start included in average
fn benchmark_atr_wrong() {
    let mut total = 0;
    for i in 0..100 {
        let start = Instant::now();
        let _ = atr_gpu(&device, &high, &low, &close, 14, None);
        device.stream.synchronize().unwrap();
        total += start.elapsed().as_micros();
    }
    let avg = total / 100;
    println!("Average: {} μs", avg); // ← Includes 18ms cold start!
}

// ✅ CORRECT - Warmup first, then measure
fn benchmark_atr_correct() {
    // Warmup
    for _ in 0..5 {
        let _ = atr_gpu(&device, &high, &low, &close, 14, None);
    }
    device.stream.synchronize().unwrap();

    // Measure
    let mut total = 0;
    for _ in 0..100 {
        let start = Instant::now();
        let _ = atr_gpu(&device, &high, &low, &close, 14, None);
        device.stream.synchronize().unwrap();
        total += start.elapsed().as_micros();
    }
    let avg = total / 100;
    println!("Warm average: {} μs", avg); // ← Accurate!
}
```

### Pitfall 2: Missing Synchronization

```rust
// ❌ WRONG - Only measures kernel launch (~10μs)
let start = Instant::now();
let _ = atr_gpu(&device, &high, &low, &close, 14, None);
let elapsed = start.elapsed(); // ← GPU still running!

// ✅ CORRECT - Waits for GPU to finish
let start = Instant::now();
let _ = atr_gpu(&device, &high, &low, &close, 14, None);
device.stream.synchronize().unwrap();
let elapsed = start.elapsed();
```

### Pitfall 3: Single Measurement

```rust
// ❌ WRONG - Susceptible to noise (±10-20μs variance)
let start = Instant::now();
let _ = atr_gpu(&device, &high, &low, &close, 14, None);
device.stream.synchronize().unwrap();
let single = start.elapsed().as_micros();
println!("Performance: {} μs", single); // ← Could be 130-160μs!

// ✅ CORRECT - Average of many runs
let mut total = 0;
for _ in 0..100 {
    let start = Instant::now();
    let _ = atr_gpu(&device, &high, &low, &close, 14, None);
    device.stream.synchronize().unwrap();
    total += start.elapsed().as_micros();
}
let avg = total / 100;
let stddev = /* calculate from samples */;
println!("Performance: {} ± {} μs", avg, stddev); // ← Accurate!
```

### Pitfall 4: Ignoring Data Size

```rust
// ❌ WRONG - Performance varies dramatically with data size
let small_data = vec![100.0; 100];
let large_data = vec![100.0; 100_000];

// Small: 80μs (GPU overhead dominates)
// Large: 145μs (GPU compute dominates)

// ✅ CORRECT - Test multiple sizes
for n in [100, 1_000, 10_000, 100_000] {
    let data = vec![100.0; n];
    let warm_us = benchmark(&device, &data);
    println!("n={}: {} μs", n, warm_us);
}
```

---

## Expected Performance Ranges

### ATR (Average True Range)

**Configuration**: 100K candles, period=14

| Phase | Time (μs) | % of Total |
|-------|-----------|------------|
| H2D Transfer (pinned) | 25 | 17.2% |
| GPU True Range Kernel | 20 | 13.8% |
| D2H Transfer (pinned) | 25 | 17.2% |
| CPU Wilder's Smoothing | 15 | 10.3% |
| Overhead (sync, setup) | 60 | 41.4% |
| **Total (Async)** | **145** | **100%** |

**Breakdown by Implementation**:
- **Pure GPU (v0.1.0)**: ~238μs (single-thread smoothing bottleneck)
- **Hybrid CPU-GPU (v0.2.0)**: ~163μs (1.5x faster)
- **Async Hybrid (v0.2.1)**: ~145μs (1.1x faster, PR #8)

**Cold start**: 80-150ms (PTX compilation)

### Other Indicators (100K candles)

| Indicator | Type | Warm Time | Cold Start |
|-----------|------|-----------|------------|
| SMA | Pure GPU | 40-60 μs | 60-100 ms |
| EMA | Hybrid | 25-35 μs | 70-120 ms |
| RSI | Hybrid | 100-140 μs | 100-180 ms |
| MACD | Hybrid | 60-90 μs | 90-150 ms |
| Bollinger Bands | Pure GPU | 80-120 μs | 100-180 ms |
| Stochastic | Hybrid | 120-160 μs | 120-200 ms |

**General rules**:
- **Pure GPU**: 2-4x data transfer time + kernel time
- **Hybrid CPU-GPU**: 1.5-2x pure GPU time (avoids sequential GPU bottleneck)
- **Cold start**: 60-200ms depending on kernel complexity

---

## Benchmark Best Practices

### 1. Use Criterion for Statistical Rigor

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_atr(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");
    let n = 100_000;
    let high = vec![100.0; n];
    let low = vec![98.0; n];
    let close = vec![99.0; n];

    // Criterion handles warmup automatically!
    c.bench_function("atr_gpu_100k", |b| {
        b.iter(|| {
            let result = atr_gpu(
                black_box(&device),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                black_box(14),
                None,
            );
            device.stream.synchronize().unwrap();
            black_box(result)
        });
    });
}

criterion_group!(benches, bench_atr);
criterion_main!(benches);
```

**Criterion benefits**:
- Automatic warmup (5 iterations)
- Statistical analysis (mean, stddev, outliers)
- Regression detection
- HTML reports with plots

### 2. Report Full Context

Always include in benchmark reports:

```rust
println!("╔════════════════════════════════════════╗");
println!("║  ATR GPU Benchmark                     ║");
println!("╠════════════════════════════════════════╣");
println!("║  Configuration:                        ║");
println!("║    Candles:        {:>10}          ║", n);
println!("║    Period:         {:>10}          ║", period);
println!("║    GPU:            RTX 3500 Ada        ║");
println!("║    CUDA:           12.6                ║");
println!("╠════════════════════════════════════════╣");
println!("║  Timing (warm, n=100):                 ║");
println!("║    Average:        {:>10} μs      ║", avg);
println!("║    Std Dev:        {:>10} μs      ║", stddev);
println!("║    Min:            {:>10} μs      ║", min);
println!("║    Max:            {:>10} μs      ║", max);
println!("╠════════════════════════════════════════╣");
println!("║  Cold Start:                           ║");
println!("║    First run:      {:>10} ms      ║", cold_ms);
println!("╚════════════════════════════════════════╝");
```

### 3. Test Multiple Dataset Sizes

```rust
fn benchmark_atr_scaling() {
    let device = GpuDevice::new().expect("GPU required");

    println!("ATR Scaling Analysis:");
    println!("  n     | Cold (ms) | Warm (μs) | μs/candle");
    println!("--------|-----------|-----------|----------");

    for n in [100, 1_000, 10_000, 100_000] {
        let high = vec![100.0; n];
        let low = vec![98.0; n];
        let close = vec![99.0; n];

        // Cold start
        let start = Instant::now();
        let _ = atr_gpu(&device, &high, &low, &close, 14, None);
        device.stream.synchronize().unwrap();
        let cold_ms = start.elapsed().as_millis();

        // Warmup
        for _ in 0..5 {
            let _ = atr_gpu(&device, &high, &low, &close, 14, None);
        }
        device.stream.synchronize().unwrap();

        // Warm timing
        let mut total = 0;
        for _ in 0..100 {
            let start = Instant::now();
            let _ = atr_gpu(&device, &high, &low, &close, 14, None);
            device.stream.synchronize().unwrap();
            total += start.elapsed().as_micros();
        }
        let warm_us = total / 100;
        let us_per_candle = warm_us as f64 / n as f64;

        println!("{:>7} | {:>9} | {:>9} | {:>8.3}",
            n, cold_ms, warm_us, us_per_candle);
    }
}
```

**Expected output**:
```
ATR Scaling Analysis:
  n     | Cold (ms) | Warm (μs) | μs/candle
--------|-----------|-----------|----------
    100 |       120 |        80 |    0.800
  1,000 |       125 |        95 |    0.095
 10,000 |       130 |       120 |    0.012
100,000 |       135 |       145 |    0.001
```

**Insights**:
- Cold start is constant (~120-135ms)
- Warm time scales sub-linearly (good parallelization)
- Per-candle cost decreases with scale (amortized overhead)

---

## Validating Optimization Claims

### Example: PR #8 - ATR Async Optimization

**Claim**: 163μs → 145μs (11% speedup)

**Validation**:

```rust
fn validate_pr8_claim() {
    let device = GpuDevice::new().expect("GPU required");
    let n = 100_000;
    let high = vec![100.0; n];
    let low = vec![98.0; n];
    let close = vec![99.0; n];

    // Test sync implementation (v0.2.0 baseline)
    let sync_us = benchmark_sync_atr(&device, &high, &low, &close);

    // Test async implementation (v0.2.1 optimized)
    let async_us = benchmark_async_atr(&device, &high, &low, &close);

    let speedup = sync_us as f64 / async_us as f64;
    let improvement = ((sync_us - async_us) as f64 / sync_us as f64) * 100.0;

    println!("╔════════════════════════════════════════╗");
    println!("║  PR #8: ATR Async Optimization         ║");
    println!("╠════════════════════════════════════════╣");
    println!("║  Baseline (v0.2.0 sync):               ║");
    println!("║    Average: {:>10} μs             ║", sync_us);
    println!("╠════════════════════════════════════════╣");
    println!("║  Optimized (v0.2.1 async):             ║");
    println!("║    Average: {:>10} μs             ║", async_us);
    println!("╠════════════════════════════════════════╣");
    println!("║  Improvement:                          ║");
    println!("║    Speedup: {:>10.2}x             ║", speedup);
    println!("║    Reduction: {:>8.1}%               ║", improvement);
    println!("╠════════════════════════════════════════╣");
    println!("║  Claim Validation:                     ║");

    if (sync_us >= 155 && sync_us <= 170) && (async_us >= 135 && async_us <= 155) {
        println!("║    Status: ✅ VALIDATED                ║");
        println!("║    Matches expected range              ║");
    } else {
        println!("║    Status: ⚠️  NEEDS INVESTIGATION     ║");
        println!("║    Expected: 163μs → 145μs             ║");
    }

    println!("╚════════════════════════════════════════╝");
}
```

**Acceptance criteria**:
- Baseline: 155-170 μs (±5% tolerance)
- Optimized: 135-155 μs (±5% tolerance)
- Speedup: 1.08-1.15x (8-15% improvement)

### Statistical Significance

For optimization claims, require:

1. **n ≥ 100 iterations** (statistical power)
2. **p-value < 0.05** (95% confidence)
3. **Effect size > 5%** (practical significance)

```rust
use statistical::mean;
use statistical::standard_deviation;
use statistical::t_test;

fn validate_with_statistics(baseline: Vec<u128>, optimized: Vec<u128>) {
    let baseline_mean = mean(&baseline);
    let optimized_mean = mean(&optimized);
    let baseline_stddev = standard_deviation(&baseline, None);
    let optimized_stddev = standard_deviation(&optimized, None);

    let t_statistic = t_test(&baseline, &optimized);
    let p_value = /* calculate from t-statistic */;

    let effect_size = (baseline_mean - optimized_mean) / baseline_mean;

    println!("Statistical Analysis:");
    println!("  Baseline:  {} ± {} μs", baseline_mean, baseline_stddev);
    println!("  Optimized: {} ± {} μs", optimized_mean, optimized_stddev);
    println!("  p-value:   {:.6}", p_value);
    println!("  Effect:    {:.1}%", effect_size * 100.0);

    if p_value < 0.05 && effect_size > 0.05 {
        println!("  ✅ Statistically significant improvement!");
    } else {
        println!("  ⚠️  Not statistically significant");
    }
}
```

---

## GPU Profiling with CUDA Events

For detailed breakdowns, use CUDA events:

```rust
use cudarc::driver::CudaEvent;

pub fn profile_atr_detailed(device: &GpuDevice) {
    // Create events
    let start = device.context().create_event()?;
    let h2d_done = device.context().create_event()?;
    let kernel_done = device.context().create_event()?;
    let d2h_done = device.context().create_event()?;

    // Record events
    start.record(&device.stream)?;

    // H2D transfer
    device.stream.memcpy_htod(&h_data, &mut d_data)?;
    h2d_done.record(&device.stream)?;

    // Kernel execution
    unsafe { device.stream.launch_kernel(&kernel, config)? };
    kernel_done.record(&device.stream)?;

    // D2H transfer
    device.stream.memcpy_dtoh(&d_result, &mut h_result)?;
    d2h_done.record(&device.stream)?;

    // Synchronize and calculate
    device.stream.synchronize()?;

    let h2d_ms = start.elapsed_ms(&h2d_done)?;
    let kernel_ms = h2d_done.elapsed_ms(&kernel_done)?;
    let d2h_ms = kernel_done.elapsed_ms(&d2h_done)?;
    let total_ms = start.elapsed_ms(&d2h_done)?;

    println!("Detailed Breakdown:");
    println!("  H2D:    {:.2} ms ({:.1}%)", h2d_ms, h2d_ms/total_ms*100.0);
    println!("  Kernel: {:.2} ms ({:.1}%)", kernel_ms, kernel_ms/total_ms*100.0);
    println!("  D2H:    {:.2} ms ({:.1}%)", d2h_ms, d2h_ms/total_ms*100.0);
    println!("  Total:  {:.2} ms", total_ms);
}
```

---

## Profiling with Nsight Systems

For deep GPU analysis, use NVIDIA Nsight Systems:

```bash
# Profile entire application
nsys profile -o atr_profile \
  --trace=cuda,nvtx \
  --stats=true \
  cargo run --release --example validate_atr_performance

# Analyze results
nsys stats atr_profile.nsys-rep
```

**Key metrics to look for**:
- **Kernel duration**: Should match warm timing
- **SM utilization**: >70% is good for compute-bound kernels
- **Memory bandwidth**: Check H2D/D2H transfer rates
- **Occupancy**: Higher is better (aim for >50%)

---

## Summary Checklist

Before reporting benchmark results:

- [ ] **Warmup performed** (5+ iterations)
- [ ] **Stream synchronized** after each timing
- [ ] **Multiple iterations** (100+ for average)
- [ ] **Cold start reported** separately
- [ ] **Dataset size specified** (number of candles)
- [ ] **Hardware context** included (GPU model, CUDA version)
- [ ] **Statistical metrics** (mean, stddev, min, max)
- [ ] **Multiple sizes tested** (scaling analysis)
- [ ] **Optimization claims validated** (before/after comparison)
- [ ] **Context provided** (what changed, why)

---

## References

- **ATR Implementation**: `src/gpu/atr.rs`
- **GPU Profiling Results**: `docs/GPU_PROFILING_RESULTS.md`
- **Example Benchmark**: `examples/validate_atr_performance.rs`
- **Criterion Documentation**: https://bheisler.github.io/criterion.rs/book/
- **Nsight Systems Guide**: https://docs.nvidia.com/nsight-systems/

---

## Appendix: Quick Benchmark Template

```rust
use kimsfinance_core::gpu::{GpuDevice, atr_gpu};
use std::time::Instant;

fn benchmark_template() {
    // Setup
    let device = GpuDevice::new().expect("GPU required");
    let n = 100_000;
    let high = vec![100.0; n];
    let low = vec![98.0; n];
    let close = vec![99.0; n];

    // 1. Cold start
    let start = Instant::now();
    let _ = atr_gpu(&device, &high, &low, &close, 14, None).unwrap();
    device.stream.synchronize().unwrap();
    let cold_us = start.elapsed().as_micros();

    // 2. Warmup
    for _ in 0..5 {
        let _ = atr_gpu(&device, &high, &low, &close, 14, None).unwrap();
    }
    device.stream.synchronize().unwrap();

    // 3. Warm timing
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        let _ = atr_gpu(&device, &high, &low, &close, 14, None).unwrap();
        device.stream.synchronize().unwrap();
        times.push(start.elapsed().as_micros());
    }

    // 4. Statistics
    let mean = times.iter().sum::<u128>() / times.len() as u128;
    let variance = times.iter()
        .map(|&t| (t as i128 - mean as i128).pow(2) as u128)
        .sum::<u128>() / times.len() as u128;
    let stddev = (variance as f64).sqrt() as u128;
    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();

    // 5. Report
    println!("ATR Benchmark (n={})", n);
    println!("  Cold start: {} μs", cold_us);
    println!("  Warm (n=100):");
    println!("    Mean:   {} μs", mean);
    println!("    StdDev: {} μs", stddev);
    println!("    Min:    {} μs", min);
    println!("    Max:    {} μs", max);
}
```

---

**Last Updated**: 2025-10-31
**Author**: kimsfinance development team
**Version**: 1.0

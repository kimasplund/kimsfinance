# CPU-GPU Hybrid Benchmark Usage Guide

## Quick Start

```bash
# Run all hybrid benchmarks
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark

# Run specific indicator
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- EMA

# Quick test (faster iteration during development)
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- --quick
```

## Current Status

**Phase 1: COMPLETE** ✅
- Baseline benchmarks for old GPU implementations
- Test data generation
- Benchmark infrastructure

**Phase 2: PENDING** ⏳
- CPU implementations (`ema_cpu`, `wilders_smoothing_cpu`)
- Uncomment `New_CPU` benchmarks in code

**Phase 3: PENDING** ⏳
- Hybrid implementations (`ema_hybrid`, `elder_ray_hybrid`, etc.)
- Uncomment `New_Hybrid` benchmarks in code

## What's Currently Benchmarked

### EMA (Line 91-138)
- **Old GPU**: Single-thread kernel (~170μs for 100K)
- **Expected CPU**: ~25μs for 100K (6.8x faster)
- **Expected Hybrid**: Same as CPU (delegates to CPU)

### Elder Ray (Line 146-189)
- **Old GPU**: Single-thread EMA + parallel subtraction (~200μs for 100K)
- **Expected Hybrid**: CPU EMA + GPU parallel subtraction (~100μs, 2x faster)

### RSI (Line 197-239)
- **Old GPU**: GPU parallel + GPU single-thread smoothing (~250μs for 100K)
- **Expected Hybrid**: GPU parallel + CPU smoothing + GPU parallel (~130μs, 1.9x faster)

### ATR (Line 247-287)
- **Old GPU**: GPU parallel TR + GPU single-thread smoothing (~180μs for 100K)
- **Expected Hybrid**: GPU parallel TR + CPU smoothing (~70μs, 2.6x faster)

## Understanding Criterion Output

### Example Output

```
EMA_Comparison/Old_GPU_SingleThread/100000
                        time:   [168.23 μs 170.45 μs 172.89 μs]
                        thrpt:  [578.42 Kelem/s 586.73 Kelem/s 594.54 Kelem/s]

                        Performance has regressed.
                        Found 3 outliers among 100 measurements (3.00%)
                          1 (1.00%) low mild
                          2 (2.00%) high mild
```

**Reading this**:
- **time**: Median is 170.45μs, confidence interval is [168.23μs, 172.89μs]
- **thrpt**: Throughput is ~586K elements/sec
- **Performance has regressed**: Compared to previous run (if exists)
- **Outliers**: 3% of samples were statistical outliers (removed from analysis)

### Comparing Results

When you uncomment `New_CPU` and `New_Hybrid` benchmarks, you'll see:

```
EMA_Comparison/Old_GPU_SingleThread/100000
                        time:   [168.23 μs 170.45 μs 172.89 μs]

EMA_Comparison/New_CPU/100000
                        time:   [24.15 μs 25.02 μs 25.91 μs]
                        change: [-85.4% -85.3% -85.2%] (p = 0.00 < 0.05)
                        Performance has improved.
```

**This means**: CPU is 85.3% faster = ~6.8x speedup ✅

## Viewing HTML Reports

Criterion generates beautiful HTML reports:

```bash
# After running benchmarks
firefox target/criterion/report/index.html
```

### HTML Report Features

1. **Violin Plots**: Show distribution of measurements
2. **Regression Plots**: Show trend over time
3. **Comparison Charts**: Compare old vs new implementations
4. **Summary Tables**: All results in one place
5. **Statistical Analysis**: Confidence intervals, p-values

## Advanced Usage

### Sampling Configuration

```bash
# More samples (more accurate, slower)
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- --sample-size 1000

# Fewer samples (faster iteration)
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- --sample-size 10
```

### Noise Reduction Tips

**Before benchmarking**:
1. Close other applications
2. Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`
3. Disable GPU boosting (if consistent results needed)
4. Run on AC power (not battery)
5. Let system idle for 30 seconds before benchmarking

**Check if noise is high**:
- Look for high percentage of outliers (>10%)
- Look for wide confidence intervals (>10% of median)
- Look for high variance in results

### GPU Profiling Integration

```bash
# Profile with Nsight Systems
nsys profile cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark

# Analyze kernel performance
ncu --set full cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark
```

## Interpreting Speedups

### Expected Speedup Table

| Indicator | Old (μs) | New (μs) | Speedup | Type |
|-----------|----------|----------|---------|------|
| EMA | 170 | 25 | 6.8x | Pure CPU |
| Elder Ray | 200 | 100 | 2.0x | Hybrid |
| RSI | 250 | 130 | 1.9x | Hybrid |
| ATR | 180 | 70 | 2.6x | Hybrid |

**Dataset**: 100K candles (typical backtesting workload)

### Why Different Speedups?

**EMA (6.8x)**:
- 100% sequential algorithm
- CPU is 4-5x faster than single GPU thread
- Plus GPU overhead (~75μs) eliminated
- Result: Maximum possible speedup for sequential code

**Elder Ray (2.0x)**:
- Hybrid: CPU EMA (~25μs) + GPU parallel subtraction (~15μs)
- Old: GPU EMA (~100μs) + GPU parallel subtraction (~15μs)
- Speedup from faster EMA only: 100μs → 25μs = 75μs saved
- Total: 200μs → 100μs = 2x

**RSI (1.9x)**:
- More complex: 3-stage pipeline
- Extra PCIe transfers (2x D2H + 2x H2D) add overhead
- But CPU smoothing still faster overall
- Result: Net 1.9x speedup

**ATR (2.6x)**:
- Similar to Elder Ray
- GPU parallel TR (~30μs) + CPU smoothing (~15μs)
- Old: GPU parallel TR (~30μs) + GPU smoothing (~80μs)
- Result: 2.6x speedup

## Next Steps

### After Running Benchmarks

1. **Validate Results**: Check if speedups match expectations (±20%)
2. **Analyze Outliers**: If >10% outliers, investigate system noise
3. **Profile Bottlenecks**: Use Nsight Systems to find remaining hot spots
4. **Document Results**: Update `BENCHMARK_RESULTS.md` with findings

### Implementing CPU/Hybrid

See `CPU_GPU_HYBRID_STRATEGY.md` for implementation guide.

### Adding New Benchmarks

Template for new indicator:

```rust
#[cfg(feature = "gpu")]
fn bench_new_indicator_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("NewIndicator_Comparison");

    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available: {:?}", e);
            return;
        }
    };

    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        let (high, low, close) = generate_test_data(size);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark old implementation
        group.bench_with_input(
            BenchmarkId::new("Old", size),
            &(&high, &low, &close),
            |b, (h, l, c)| {
                b.iter(|| new_indicator_gpu(&device, black_box(h), black_box(l), black_box(c), 14, None))
            }
        );

        // TODO: Benchmark new implementation
    }

    group.finish();
}

// Add to criterion_group!
criterion_group!(
    hybrid_benches,
    bench_ema_comparison,
    bench_new_indicator_comparison  // <-- Add here
);
```

## Troubleshooting

### GPU Not Available

```
Error: GPU not available, skipping benchmarks
```

**Fix**: Ensure NVIDIA GPU with CUDA is available:
```bash
nvidia-smi
cargo test --features gpu
```

### Compilation Errors

```
error: cannot find function `ema_cpu` in module `cpu`
```

**Cause**: CPU implementations not yet created.

**Fix**: Keep `New_CPU`/`New_Hybrid` benchmarks commented until implemented.

### Unexpected Results

**If speedup is lower than expected**:
1. Check GPU utilization: `nvidia-smi dmon`
2. Check CPU frequency: `cat /proc/cpuinfo | grep MHz`
3. Check for thermal throttling: `sensors`
4. Reduce system noise (close apps, disable turbo boost)
5. Use `--sample-size 1000` for more accurate measurements

**If speedup is higher than expected**:
1. Verify correctness: Results should match reference implementation
2. Check if benchmark is measuring what you think it's measuring
3. Use flamegraph to see where time is actually spent

## FAQ

**Q: Why benchmark old GPU implementation if it's slow?**
A: To validate the performance improvement claims. "6.8x faster" only means something if we measure both old and new.

**Q: Why multiple dataset sizes?**
A: To understand scalability. Small datasets show overhead, large datasets show asymptotic performance.

**Q: What if I don't have a GPU?**
A: Benchmarks will skip gracefully. You can still implement CPU versions and benchmark those.

**Q: How long do benchmarks take?**
A: ~5-10 minutes for all indicators. Use `--quick` for faster iteration (~30 seconds).

**Q: Should I run benchmarks on every commit?**
A: No. Run benchmarks when:
- Implementing new feature
- Optimizing hot path
- Before releasing
- After major refactoring

**Q: What confidence level does Criterion use?**
A: 95% confidence intervals by default (configurable).

---

**Last Updated**: 2025-10-25
**Status**: Phase 1 Complete
**Next**: Implement CPU sequential algorithms

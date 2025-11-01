# FP8 Tensor Core Benchmark - Example Output

This document shows expected output format from the FP8 tensor core benchmarks.

## Quick Validation Test

**Command**: `cargo test --features gpu --release test_fp8_speedup_validation -- --nocapture`

### Example Output (Success)

```
=== FP8 Tensor Core Speedup Validation ===
Running comprehensive statistical analysis

✓ FP8 tensor cores supported!
  Compute capability: 8.9

Test configuration:
  Matrix size: 64x64
  Iterations: 100
  Confidence interval: 95%

Running FP32 baseline...
Running FP8 tensor cores...

=== Results ===

FP32 Baseline:
  Mean: 42.5 µs, Median: 42.1 µs, Std Dev: 1.2 µs
  p95: 44.8 µs, p99: 45.9 µs
  95% CI: [42.2, 42.8] µs

FP8 Tensor Cores:
  Mean: 12.8 µs, Median: 12.5 µs, Std Dev: 0.8 µs
  p95: 14.1 µs, p99: 14.6 µs
  95% CI: [12.6, 13.0] µs

Speedup Analysis:
  Mean speedup: 3.32x
  Median speedup: 3.37x

Throughput:
  FP32: 12.3 GFLOPS
  FP8:  40.9 GFLOPS

=== Validation ===
✓ PASS: Speedup 3.32x >= 1.5x threshold
✓ PASS: Low variance (CV = 6.2%)

=== Summary ===
FP8 tensor cores validated on GPU sm_8.9
Speedup: 3.32x (95% CI: [3.18x, 3.47x])
```

### Example Output (FP8 Not Supported)

```
=== FP8 Tensor Core Speedup Validation ===
Running comprehensive statistical analysis

⚠️ FP8 tensor cores not supported on this GPU
Compute capability: 8.6
Required: >= 8.9 (Ada Lovelace or newer)
```

## Full Benchmark Suite

**Command**: `cargo bench --features gpu --bench fp8_tensor_cores`

### 1. Single Matrix Multiply Benchmarks

```
=== Benchmark: FP32 Single Matrix Multiply (Baseline) ===
Sample size: 100 iterations per size
Sizes: [16, 32, 64, 128]

Benchmarking fp32_single_matmul/FP32/16x16: Collecting 100 samples in estimated 2.5 s
fp32_single_matmul/FP32/16x16
                        time:   [2.43 µs 2.51 µs 2.59 µs]
                        thrpt:  [2.01 M elem/s 2.07 M elem/s 2.14 M elem/s]

Benchmarking fp32_single_matmul/FP32/32x32: Collecting 100 samples in estimated 8.4 s
fp32_single_matmul/FP32/32x32
                        time:   [8.12 µs 8.38 µs 8.66 µs]
                        thrpt:  [1.20 M elem/s 1.24 M elem/s 1.28 M elem/s]

Benchmarking fp32_single_matmul/FP32/64x64: Collecting 100 samples in estimated 42 s
fp32_single_matmul/FP32/64x64
                        time:   [40.8 µs 42.1 µs 43.6 µs]
                        thrpt:  [383 K elem/s 396 K elem/s 408 K elem/s]

Benchmarking fp32_single_matmul/FP32/128x128: Collecting 100 samples in estimated 3.1 min
fp32_single_matmul/FP32/128x128
                        time:   [182 µs 185 µs 189 µs]
                        thrpt:  [220 K elem/s 225 K elem/s 229 K elem/s]
```

```
=== Benchmark: FP8 Single Matrix Multiply ===
Expected: 2-4x faster than FP32 baseline

Benchmarking fp8_single_matmul/FP8/16x16: Collecting 100 samples in estimated 1.2 s
fp8_single_matmul/FP8/16x16
                        time:   [1.15 µs 1.21 µs 1.28 µs]
                        thrpt:  [4.06 M elem/s 4.30 M elem/s 4.52 M elem/s]
                        change: [-52.3% -51.8% -51.2%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) high mild
  1 (1.00%) high severe

Benchmarking fp8_single_matmul/FP8/32x32: Collecting 100 samples in estimated 3.1 s
fp8_single_matmul/FP8/32x32
                        time:   [2.94 µs 3.08 µs 3.23 µs]
                        thrpt:  [3.21 M elem/s 3.37 M elem/s 3.53 M elem/s]
                        change: [-64.2% -63.3% -62.3%] (p = 0.00 < 0.05)
                        Performance has improved.

Benchmarking fp8_single_matmul/FP8/64x64: Collecting 100 samples in estimated 12 s
fp8_single_matmul/FP8/64x64
                        time:   [11.8 µs 12.5 µs 13.3 µs]
                        thrpt:  [1.25 M elem/s 1.33 M elem/s 1.41 M elem/s]
                        change: [-70.4% -70.3% -70.1%] (p = 0.00 < 0.05)
                        Performance has improved.

Benchmarking fp8_single_matmul/FP8/128x128: Collecting 100 samples in estimated 48 s
fp8_single_matmul/FP8/128x128
                        time:   [46.2 µs 48.7 µs 51.4 µs]
                        thrpt:  [810 K elem/s 855 K elem/s 902 K elem/s]
                        change: [-74.1% -73.7% -73.3%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### 2. Batch Matrix Multiply Benchmarks

```
=== Benchmark: FP32 Batch Matrix Multiply (100 iterations) ===
Simulates genetic optimizer fitness evaluation batch

Benchmarking fp32_batch_matmul/FP32_Batch/100x16x16: Collecting 50 samples in estimated 12 s
fp32_batch_matmul/FP32_Batch/100x16x16
                        time:   [238 ms 245 ms 253 ms]
                        thrpt:  [2.14 K batches/s 2.20 K batches/s 2.27 K batches/s]

Benchmarking fp32_batch_matmul/FP32_Batch/100x32x32: Collecting 50 samples in estimated 42 s
fp32_batch_matmul/FP32_Batch/100x32x32
                        time:   [816 ms 840 ms 866 ms]
                        thrpt:  [623 batches/s 642 batches/s 661 batches/s]

Benchmarking fp32_batch_matmul/FP32_Batch/100x64x64: Collecting 50 samples in estimated 3.5 min
fp32_batch_matmul/FP32_Batch/100x64x64
                        time:   [4.09 s 4.21 s 4.34 s]
                        thrpt:  [124 batches/s 128 batches/s 132 batches/s]
```

```
=== Benchmark: FP8 Batch Matrix Multiply (100 iterations) ===
Expected: 2-4x faster than FP32 batch

Benchmarking fp8_batch_matmul/FP8_Batch/100x16x16: Collecting 50 samples in estimated 5.3 s
fp8_batch_matmul/FP8_Batch/100x16x16
                        time:   [102 ms 105 ms 109 ms]
                        thrpt:  [4.96 K batches/s 5.14 K batches/s 5.29 K batches/s]
                        change: [-58.5% -57.1% -55.6%] (p = 0.00 < 0.05)
                        Performance has improved.

Benchmarking fp8_batch_matmul/FP8_Batch/100x32x32: Collecting 50 samples in estimated 14 s
fp8_batch_matmul/FP8_Batch/100x32x32
                        time:   [273 ms 285 ms 298 ms]
                        thrpt:  [1.81 K batches/s 1.89 K batches/s 1.97 K batches/s]
                        change: [-66.8% -66.1% -65.3%] (p = 0.00 < 0.05)
                        Performance has improved.

Benchmarking fp8_batch_matmul/FP8_Batch/100x64x64: Collecting 50 samples in estimated 1.0 min
fp8_batch_matmul/FP8_Batch/100x64x64
                        time:   [1.21 s 1.25 s 1.29 s]
                        thrpt:  [419 batches/s 431 batches/s 445 batches/s]
                        change: [-70.5% -70.3% -70.1%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### 3. Conversion Overhead Benchmark

```
=== Benchmark: FP32 -> FP8 -> FP32 Conversion Overhead ===
Measures impact of precision conversion on total pipeline

Benchmarking conversion_overhead/FP32_to_FP8/16x16: Collecting 100 samples in estimated 0.6 s
conversion_overhead/FP32_to_FP8/16x16
                        time:   [0.58 µs 0.61 µs 0.64 µs]

Benchmarking conversion_overhead/FP32_to_FP8/32x32: Collecting 100 samples in estimated 1.2 s
conversion_overhead/FP32_to_FP8/32x32
                        time:   [1.15 µs 1.21 µs 1.28 µs]

Benchmarking conversion_overhead/FP32_to_FP8/64x64: Collecting 100 samples in estimated 2.8 s
conversion_overhead/FP32_to_FP8/64x64
                        time:   [2.71 µs 2.85 µs 3.00 µs]

Benchmarking conversion_overhead/FP32_to_FP8/128x128: Collecting 100 samples in estimated 8.5 s
conversion_overhead/FP32_to_FP8/128x128
                        time:   [8.21 µs 8.62 µs 9.08 µs]
```

### 4. Memory Bandwidth Benchmark

```
=== Benchmark: Memory Bandwidth (FP8 vs FP32) ===
FP8 uses 1 byte vs FP32 4 bytes (4x bandwidth advantage)

Benchmarking memory_bandwidth/FP32_Transfer/1048576: Collecting 100 samples in estimated 4.5 s
memory_bandwidth/FP32_Transfer/1048576
                        time:   [43.2 µs 45.1 µs 47.3 µs]
                        thrpt:  [22.2 M elem/s 23.2 M elem/s 24.3 M elem/s]

Benchmarking memory_bandwidth/FP32_Transfer/16777216: Collecting 100 samples in estimated 18 s
memory_bandwidth/FP32_Transfer/16777216
                        time:   [176 µs 184 µs 193 µs]
                        thrpt:  [87.0 M elem/s 91.2 M elem/s 95.3 M elem/s]

Benchmarking memory_bandwidth/FP32_Transfer/268435456: Collecting 100 samples in estimated 72 s
memory_bandwidth/FP32_Transfer/268435456
                        time:   [695 µs 728 µs 765 µs]
                        thrpt:  [351 M elem/s 369 M elem/s 386 M elem/s]

Benchmarking memory_bandwidth/FP32_Transfer/4294967296: Collecting 100 samples in estimated 3.0 min
memory_bandwidth/FP32_Transfer/4294967296
                        time:   [2.81 ms 2.94 ms 3.09 ms]
                        thrpt:  [1.39 G elem/s 1.46 G elem/s 1.53 G elem/s]
```

## Summary Table

After all benchmarks complete, criterion generates a summary:

```
FP8 Tensor Core Benchmarks - Summary
=====================================

Single Matrix Multiply:
  16x16:   FP32: 2.51 µs  | FP8: 1.21 µs  | Speedup: 2.07x ✓
  32x32:   FP32: 8.38 µs  | FP8: 3.08 µs  | Speedup: 2.72x ✓
  64x64:   FP32: 42.1 µs  | FP8: 12.5 µs  | Speedup: 3.37x ✓
  128x128: FP32: 185 µs   | FP8: 48.7 µs  | Speedup: 3.80x ✓

Batch Matrix Multiply (100x):
  16x16:   FP32: 245 ms   | FP8: 105 ms   | Speedup: 2.33x ✓
  32x32:   FP32: 840 ms   | FP8: 285 ms   | Speedup: 2.95x ✓
  64x64:   FP32: 4.21 s   | FP8: 1.25 s   | Speedup: 3.37x ✓

Conversion Overhead:
  16x16:   0.61 µs (overhead: 50% of FP8 matmul)
  32x32:   1.21 µs (overhead: 39% of FP8 matmul)
  64x64:   2.85 µs (overhead: 23% of FP8 matmul)
  128x128: 8.62 µs (overhead: 18% of FP8 matmul)

Overall Assessment:
  ✓ All benchmarks passed (speedup >= 1.5x threshold)
  ✓ Peak speedup: 3.80x (128x128 single matmul)
  ✓ Genetic optimizer speedup: 3.37x (64x64 batch)
  ✓ Conversion overhead acceptable (<25% for 64x64+)

Recommendation: Deploy FP8 for genetic optimizer
  - Use FP8 for parameter spaces >= 32x32 (5x5 parameters)
  - Expected overall speedup: 2.5-3.5x
  - Quality retention: 95-99% (see genetic_optimizer_precision benchmark)
```

## HTML Report

Criterion generates detailed HTML reports with charts:

```
View detailed results:
  target/criterion/fp8_single_matmul/FP8/64x64/report/index.html
  target/criterion/fp8_batch_matmul/FP8_Batch/100x64x64/report/index.html
```

**Charts include**:
- Violin plot (distribution of timings)
- Mean with error bars (95% CI)
- Performance change vs baseline
- Regression analysis

## Troubleshooting Output

### GPU Not Available

```
thread 'main' panicked at 'Failed to initialize GPU: GpuNotAvailable'
note: run with `RUST_BACKTRACE=1` for a backtrace
```

**Fix**: Check `nvidia-smi`, ensure CUDA driver is installed

### FP8 Kernel Compilation Failed

```
Error: FP8 kernel compilation failed: PTX compilation failed:
  error: identifier "cuda_fp8" is undefined
```

**Fix**: Install CUDA Toolkit 12.4+ with `cuda_fp8.h` header

### Unexpected Slowdown

```
fp8_single_matmul/FP8/64x64
                        time:   [52.1 µs 54.2 µs 56.5 µs]
                        change: [+28.5% +28.9% +29.3%] (p = 0.00 < 0.05)
                        Performance has regressed.
```

**Possible causes**:
- Thermal throttling (check temperature)
- Background GPU processes (check `nvidia-smi`)
- Power limit (check power draw)
- Non-optimal kernel configuration

**Debug**: Run `ncu` profiler to analyze kernel performance

---

**Note**: Actual benchmark output will vary based on GPU model, driver version, and system load.
The examples above are representative of expected results on RTX 3500 Ada (sm_89) with optimal conditions.

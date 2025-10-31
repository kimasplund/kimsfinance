# Heston Calibrator Test Suite - Quick Start Guide

**Quick Reference**: How to run Heston calibrator tests

---

## Prerequisites

### Hardware
- NVIDIA GPU (CUDA-compatible)
- 12GB+ VRAM recommended

### Software
- Rust 1.75+
- CUDA Toolkit 12.0+
- Feature flags: `gpu`, `heston`

---

## Quick Commands

### Run All Tests (Once Compilation Fixed)

```bash
# All integration tests
cargo test --test heston_integration_test --features gpu,heston -- --ignored --test-threads=1

# All performance tests
cargo test --test heston_performance_test --features gpu,heston -- --ignored --test-threads=1

# All validation tests
cargo test --test heston_validation_test --features gpu,heston -- --ignored --test-threads=1

# All benchmarks
cargo bench --bench heston_comprehensive --features gpu,heston
```

### Run Specific Tests

```bash
# Single integration test
cargo test --test heston_integration_test test_end_to_end_synthetic_calibration --features gpu,heston -- --ignored

# Single performance test
cargo test --test heston_performance_test test_gpu_pricing_performance --features gpu,heston -- --ignored

# Single validation test
cargo test --test heston_validation_test test_feller_condition_enforcement --features gpu,heston

# Single benchmark group
cargo bench --bench heston_comprehensive heston_gpu_pricing --features gpu,heston
```

---

## Known Issues (Pre-Execution)

⚠️ **COMPILATION BLOCKER**: Tests won't compile until API mismatch fixed in `src/gpu/heston_pricing.rs`:

**Error**:
```
error[E0599]: no method named `htod_pinned_partial` found
error[E0599]: no method named `dtoh_pinned_partial` found
```

**Fix Required**: Add partial transfer methods to `GpuDevice` or refactor `heston_pricing.rs`

---

## Test Files Overview

| File | Lines | Tests | Purpose |
|------|-------|-------|---------|
| `heston_integration_test.rs` | 604 | 10 | End-to-end workflows |
| `heston_performance_test.rs` | 452 | 8 | Performance regression detection |
| `heston_validation_test.rs` | 499 | 9 | Analytical correctness |
| `heston_comprehensive.rs` (bench) | 301 | 7 groups | Detailed profiling |
| **Total** | **1,856** | **34** | **Complete coverage** |

---

## Test Execution Flags

### Important Flags

- `--features gpu,heston`: Enable GPU and Heston features
- `--ignored`: Run GPU tests (default: skipped on CPU-only)
- `--test-threads=1`: Prevent GPU resource contention
- `--nocapture`: Show println! output

### Example with All Flags

```bash
cargo test --test heston_integration_test \
    --features gpu,heston \
    -- --ignored --test-threads=1 --nocapture
```

---

## Expected Output

### Successful Test Run

```
running 10 tests
test heston_integration::test_end_to_end_synthetic_calibration ... ok
test heston_integration::test_gpu_pricing_consistency ... ok
test heston_integration::test_greeks_accuracy ... ok
test heston_integration::test_vol_arbitrage_profitability ... ok
test heston_integration::test_delta_hedging_strategy ... ok
test heston_integration::test_calibration_performance ... ok
test heston_integration::test_batch_pricing_performance ... ok
test heston_integration::test_parameter_validation ... ok

test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Performance Test Output

```
=== Performance Test: GPU Pricing ===

GPU Pricing (10 options):
  Mean: 0.82ms
  Median: 0.80ms
  Min: 0.75ms
  Max: 1.20ms
  StdDev: 0.08ms
  Runs: 20

GPU Pricing (100 options):
  Mean: 2.45ms
  Median: 2.40ms
  Min: 2.30ms
  Max: 3.10ms
  StdDev: 0.15ms
  Runs: 20

✓ All GPU pricing benchmarks passed
```

### Benchmark Report

```bash
# Generate HTML report
cargo bench --bench heston_comprehensive --features gpu,heston

# View report (Linux)
xdg-open target/criterion/report/index.html
```

---

## Troubleshooting

### "Feature gpu not enabled"
```bash
# Add --features flag
cargo test --features gpu,heston
```

### "No GPU found"
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA in Rust
cargo test --features gpu -- --nocapture
```

### "Test timed out"
```bash
# Increase timeout (add to test file)
#[timeout(120000)] // 2 minutes
```

### "Out of memory"
```bash
# Reduce batch size in test
# Or skip large batch tests
cargo test --test heston_performance_test --features gpu,heston -- --skip test_throughput_scalability
```

---

## Performance Targets

Quick reference for expected performance:

### GPU Pricing

| Batch Size | Target | Max (2x tolerance) |
|------------|--------|-------------------|
| 10 | <1ms | <2ms |
| 50 | <2ms | <4ms |
| 100 | <3ms | <6ms |
| 500 | <10ms | <20ms |
| 1000 | <15ms | <30ms |

### Calibration

| Options | Iterations | Target |
|---------|-----------|--------|
| 50 | 30 | <5s |
| 100 | 50 | <10s |

### Greeks

| Batch Size | Target |
|------------|--------|
| 1 option | <10ms |
| 10 options | <50ms |
| 100 options | <300ms |

---

## Test Development

### Adding New Tests

1. **Integration test**: Add to `tests/heston_integration_test.rs`
   ```rust
   #[test]
   #[ignore] // Requires GPU
   fn test_my_new_feature() {
       // Test code
   }
   ```

2. **Performance test**: Add to `tests/heston_performance_test.rs`
   ```rust
   #[test]
   #[ignore]
   fn test_my_performance_target() {
       let stats = benchmark("My Test", 20, || {
           // Code to benchmark
       });
       stats.assert_within_target(100, "My Test");
   }
   ```

3. **Validation test**: Add to `tests/heston_validation_test.rs`
   ```rust
   #[test]
   fn test_my_theoretical_property() {
       // No #[ignore] if CPU-only
       // Verify analytical result
   }
   ```

4. **Benchmark**: Add to `benches/heston_comprehensive.rs`
   ```rust
   fn bench_my_feature(c: &mut Criterion) {
       c.bench_function("my_feature", |b| {
           b.iter(|| /* benchmark code */)
       });
   }
   ```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Heston Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest-gpu # GPU runner
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run integration tests
        run: |
          cargo test --test heston_integration_test \
            --features gpu,heston \
            -- --ignored --test-threads=1

      - name: Run performance tests
        run: |
          cargo test --test heston_performance_test \
            --features gpu,heston \
            -- --ignored --test-threads=1

      - name: Run benchmarks
        run: |
          cargo bench --bench heston_comprehensive \
            --features gpu,heston
```

---

## Additional Resources

- **Full Test Report**: `tests/HESTON_TEST_REPORT.md`
- **Source Code**: `src/quantitative/heston/`, `src/gpu/heston_pricing.rs`
- **Documentation**: `docs/HESTON_CALIBRATION.md` (if exists)
- **Issue Tracker**: GitHub issues tagged `heston` or `testing`

---

## Contact

For questions or issues:
1. Check `tests/HESTON_TEST_REPORT.md` for detailed analysis
2. Review compilation errors in test output
3. Open GitHub issue with test failure logs
4. Tag maintainers for GPU-specific issues

---

**Last Updated**: 2025-10-29
**Workstream**: 6 - Comprehensive Testing & Validation
**Status**: Ready (pending compilation fix)

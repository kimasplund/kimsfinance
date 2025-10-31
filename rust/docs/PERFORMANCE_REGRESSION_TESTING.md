# Performance Regression Testing Guide

**Created:** 2025-10-31
**Purpose:** Automated detection of performance regressions in GPU indicators
**Target:** Prevent >10% performance degradation across commits

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Baseline Configuration](#baseline-configuration)
4. [Running Tests](#running-tests)
5. [CI Integration](#ci-integration)
6. [Interpreting Results](#interpreting-results)
7. [Updating Baselines](#updating-baselines)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What Is Performance Regression Testing?

Performance regression testing automatically detects when code changes cause indicators to run slower than expected. This prevents accidental performance degradation and ensures optimizations remain effective.

### Key Features

- **Automated detection** of >10% performance regressions
- **Warning alerts** for 5-10% slowdowns
- **Improvement detection** for faster-than-baseline performance
- **CI integration** with GitHub Actions
- **Comprehensive reporting** with pass/fail status

### Test Coverage

**15 GPU indicators tested:**
- Simple: EMA, ROC, SMA, WMA, VWMA (5)
- Medium: Williams %R, CCI, Donchian, Stochastic, Elder Ray, CMF (6)
- Complex: ATR, RSI, RSI sync (3)
- Known issues: MACD (CPU), OBV (1)

---

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- Rust 1.70+ with `--features gpu`
- 100K candles test dataset
- Release build (debug is ~1,000x slower)

### Run Tests Locally

```bash
# Simple run
cd rust
cargo run --release --features gpu --bench performance_regression

# Or use the runner script (recommended)
./scripts/run_performance_tests.sh

# With report saving
./scripts/run_performance_tests.sh --save

# Verbose output
./scripts/run_performance_tests.sh --verbose
```

### Expected Output

```
════════════════════════════════════════════════════════════════════════════════════════════════════════
                                   PERFORMANCE REGRESSION TEST SUITE
════════════════════════════════════════════════════════════════════════════════════════════════════════

Configuration:
  Version: 1.0.0
  Hardware: NVIDIA RTX 3500 Ada (Intel i9-13980HX)
  CUDA: 13.0 (Compute 8.9)
  Test: 100000 candles, 5 warmup, 10 measurement runs

──────────────────────────────────────────────────────────────────────────────────────────────────────────
                                             RUNNING TESTS
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Indicator                     Baseline     Measured      Diff %          Status
──────────────────────────────────────────────────────────────────────────────────────────────────────────

GROUP 1: SIMPLE INDICATORS
ema_hybrid                         200          198        -1.0%        ⚡ FASTER
roc                                442          445         0.7%        ✅ PASS
sma                                519          510        -1.7%        ⚡ FASTER
wma                                717          725         1.1%        ✅ PASS
vwma                              1033         1028        -0.5%        ⚡ FASTER

[... more indicators ...]

════════════════════════════════════════════════════════════════════════════════════════════════════════
                                              SUMMARY
════════════════════════════════════════════════════════════════════════════════════════════════════════

Test Results:
  ✅ Pass: 12 / 15
  ⚠️  Warn: 1
  ❌ Fail: 0
  ⚡ Improvements: 2

✅ ALL TESTS PASSED
════════════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## Baseline Configuration

### Location

`benches/baselines.json`

### Structure

```json
{
  "version": "1.0.0",
  "hardware": {
    "gpu": "NVIDIA RTX 3500 Ada",
    "cpu": "Intel i9-13980HX",
    "cuda": "13.0",
    "compute_capability": "8.9"
  },
  "test_config": {
    "candles": 100000,
    "warmup_runs": 5,
    "measurement_runs": 10,
    "build": "release"
  },
  "baselines": {
    "simple_indicators": {
      "description": "Simple indicators with 2-3 memory transfers (target <1ms)",
      "indicators": {
        "ema_hybrid": {
          "baseline_us": 200,
          "tolerance_percent": 10,
          "warn_percent": 5,
          "notes": "CPU fallback - fastest indicator"
        },
        ...
      }
    }
  }
}
```

### Fields Explained

| Field | Description |
|-------|-------------|
| `baseline_us` | Expected performance in microseconds (100K candles, warm) |
| `tolerance_percent` | Maximum allowed regression before failure (10%) |
| `warn_percent` | Threshold for warnings (5%) |
| `notes` | Implementation details and context |

### Current Baselines (100K Candles, Warm)

| Indicator | Baseline (μs) | Tolerance | Category |
|-----------|---------------|-----------|----------|
| EMA (hybrid) | 200 | 10% | Simple |
| ROC | 442 | 10% | Simple |
| SMA | 519 | 10% | Simple |
| WMA | 717 | 10% | Simple |
| VWMA | 1,033 | 10% | Simple |
| Williams %R | 1,079 | 10% | Medium |
| CCI | 1,152 | 10% | Medium |
| Donchian | 1,174 | 10% | Medium |
| Stochastic | 1,279 | 10% | Medium |
| Elder Ray | 1,330 | 10% | Medium |
| CMF | 1,779 | 10% | Medium |
| ATR | 1,360 | 10% | Complex |
| RSI | 2,512 | 10% | Complex |
| RSI (sync) | 2,870 | 10% | Complex |
| MACD (CPU) | 75 | 20% | Known issue |
| OBV | 4,696 | 20% | Known issue |

---

## Running Tests

### Local Execution

#### Method 1: Using Runner Script (Recommended)

```bash
# Basic run
./scripts/run_performance_tests.sh

# Save report with timestamp
./scripts/run_performance_tests.sh --save
# Creates: performance_report_20251031_143022.txt

# Verbose output (show build logs)
./scripts/run_performance_tests.sh --verbose
```

#### Method 2: Direct Cargo Execution

```bash
# Run benchmark directly
cd rust
cargo run --release --features gpu --bench performance_regression

# With output capture
cargo run --release --features gpu --bench performance_regression 2>&1 | tee report.txt
```

### Test Workflow

1. **Load baselines** from `benches/baselines.json`
2. **Initialize GPU** and create test data (100K candles)
3. **Warmup phase** (5 runs per indicator)
4. **Measurement phase** (10 runs, averaged)
5. **Compare** measured vs baseline performance
6. **Report** status for each indicator
7. **Exit**:
   - Code 0: All tests pass
   - Code 1: One or more failures (>10% regression)
   - Code 2: Configuration error

---

## CI Integration

### GitHub Actions Workflow

**Location:** `.github/workflows/performance.yml`

**Triggers:**
- Push to `master` or `dev-*` branches
- Pull requests to `master`
- Manual workflow dispatch

**Requirements:**
- Self-hosted runner with NVIDIA GPU
- CUDA toolkit installed
- Rust toolchain

### Workflow Steps

1. **Checkout** repository
2. **Check GPU** availability (`nvidia-smi`)
3. **Install** Rust toolchain
4. **Cache** dependencies
5. **Build** release binary with GPU features
6. **Run** performance regression tests
7. **Upload** performance report as artifact
8. **Comment** on PR (if applicable) with results

### Self-Hosted Runner Setup

```bash
# On your GPU machine
# 1. Install GitHub Actions runner
# 2. Configure as self-hosted runner
# 3. Ensure CUDA and Rust are available
# 4. Add labels: self-hosted, linux, gpu

# Test runner
nvidia-smi
cargo --version
```

### Example PR Comment

```markdown
## Performance Regression Test Results

✅ ALL TESTS PASSED

Test Results:
  ✅ Pass: 14 / 15
  ⚠️  Warn: 0
  ❌ Fail: 0
  ⚡ Improvements: 1

See [Performance Regression Testing Guide](../docs/PERFORMANCE_REGRESSION_TESTING.md) for details.
```

---

## Interpreting Results

### Test Statuses

#### ✅ PASS
- Performance within 5% of baseline
- No action required
- Indicates stable performance

#### ⚠️ WARN
- Performance 5-10% slower than baseline
- **Action:** Investigate, but doesn't fail CI
- May indicate gradual degradation

#### ❌ FAIL
- Performance >10% slower than baseline
- **Action:** Required investigation
- Fails CI, blocks PR merge

#### ⚡ FASTER
- Performance better than baseline
- **Action:** Consider updating baseline
- Indicates successful optimization

### Common Failure Causes

1. **Accidental removal of optimization**
   - Check recent commits for reverted optimizations
   - Verify async pinned memory still enabled

2. **Debug build instead of release**
   - Debug builds are ~1,000x slower
   - Always use `--release` flag

3. **GPU memory fragmentation**
   - Restart tests
   - Check for memory leaks

4. **Hardware thermal throttling**
   - Check GPU temperature
   - Ensure adequate cooling

5. **Background GPU processes**
   - Close other GPU applications
   - Check `nvidia-smi` for competing processes

### Example Failure Output

```
❌ FAIL

Failed Tests:
  - sma: 12.5% slower than baseline (519 μs -> 584 μs)
  - atr: 11.2% slower than baseline (1360 μs -> 1512 μs)

Recommendations:
  1. Review recent code changes to sma.rs and atr.rs
  2. Check if async optimization was accidentally reverted
  3. Profile indicators with Nsight Compute
  4. Verify GPU warmup is working correctly
```

---

## Updating Baselines

### When to Update

**DO update baselines when:**
- ✅ Intentional optimization makes indicators faster
- ✅ Algorithmic change improves accuracy AND changes performance
- ✅ Hardware upgrade changes test environment
- ✅ CUDA version upgrade affects performance

**DON'T update baselines when:**
- ❌ Tests fail due to accidental regression
- ❌ Performance degradation is unexplained
- ❌ Trying to "cheat" CI

### How to Update

#### Step 1: Run Tests and Capture Results

```bash
./scripts/run_performance_tests.sh --save
# Creates: performance_report_20251031_143022.txt
```

#### Step 2: Extract New Measurements

From the report, find the "Measured" column:

```
Indicator                     Baseline     Measured      Diff %          Status
──────────────────────────────────────────────────────────────────────────────
sma                                519          385       -25.8%        ⚡ FASTER
atr                               1360         1120       -17.6%        ⚡ FASTER
```

#### Step 3: Update `baselines.json`

```json
{
  "baselines": {
    "simple_indicators": {
      "indicators": {
        "sma": {
          "baseline_us": 385,  // OLD: 519
          "tolerance_percent": 10,
          "warn_percent": 5,
          "notes": "Updated after parallel scan optimization (PR #XX)"
        }
      }
    }
  }
}
```

#### Step 4: Document the Change

In your commit message:

```
perf: Update performance baselines after SMA optimization

- SMA: 519μs → 385μs (25.8% faster)
- ATR: 1360μs → 1120μs (17.6% faster)

Reason: Implemented parallel scan algorithm (PR #XX)
Validated with 100 runs, statistical significance p < 0.01
```

#### Step 5: Re-run Tests

```bash
./scripts/run_performance_tests.sh
# Should now pass with new baselines
```

### Baseline Update Checklist

- [ ] Performance improvement is intentional and validated
- [ ] New measurements are consistent across 10+ runs
- [ ] Hardware/CUDA configuration documented in baselines.json
- [ ] Commit message explains reason for update
- [ ] Tests pass with new baselines
- [ ] PR includes benchmark results

---

## Troubleshooting

### Tests Fail on CI but Pass Locally

**Cause:** Hardware differences between local and CI runner

**Solution:**
1. Check CI runner GPU: `nvidia-smi` in workflow logs
2. Adjust tolerance if CI GPU is different: `tolerance_percent: 15`
3. Consider separate baselines for CI: `baselines_ci.json`

### "GPU not available" Error

**Cause:** No NVIDIA GPU detected

**Solution:**
1. Check `nvidia-smi` works
2. Verify CUDA toolkit installed
3. Ensure GPU is not in use by other processes
4. Check runner has GPU access (not in container)

### Inconsistent Results Across Runs

**Cause:** Insufficient warmup or measurement runs

**Solution:**
1. Increase warmup runs: `warmup_runs: 10`
2. Increase measurement runs: `measurement_runs: 20`
3. Check for thermal throttling
4. Ensure no background GPU processes

### "Failed to parse baselines.json"

**Cause:** JSON syntax error in baselines file

**Solution:**
1. Validate JSON: `cat benches/baselines.json | jq .`
2. Check for trailing commas
3. Ensure all fields are present

### OBV Always Warns/Fails

**Cause:** OBV has known performance issue (single-threaded cumsum)

**Solution:**
- OBV is in "known_issues" category with 20% tolerance
- Warnings from known_issues don't fail CI
- Will be fixed once parallel prefix sum is implemented

---

## Advanced Configuration

### Custom Tolerance Per Indicator

```json
{
  "indicators": {
    "experimental_indicator": {
      "baseline_us": 1000,
      "tolerance_percent": 20,  // Higher tolerance for experimental features
      "warn_percent": 10,
      "notes": "Experimental - higher tolerance"
    }
  }
}
```

### Multiple Hardware Profiles

```bash
# Load different baselines based on GPU
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == "RTX 3090" ]; then
    cp benches/baselines_rtx3090.json benches/baselines.json
fi
```

### Statistical Validation

For highly critical indicators, add statistical testing:

```rust
// In benchmark code
let mut measurements = Vec::new();
for _ in 0..100 {
    measurements.push(benchmark_indicator(...));
}

let mean = measurements.iter().sum::<u64>() / measurements.len() as u64;
let stddev = calculate_stddev(&measurements, mean);

// Fail if mean + 2*stddev exceeds baseline (95% confidence)
assert!(mean + 2 * stddev < baseline * 1.10);
```

---

## Best Practices

### For Developers

1. **Run tests before committing** performance-sensitive changes
2. **Update baselines immediately** after successful optimizations
3. **Document reasons** for performance changes in commit messages
4. **Profile before optimizing** - understand bottlenecks first
5. **Validate with multiple runs** (not just single measurement)

### For Reviewers

1. **Check for baseline updates** in PRs claiming performance improvements
2. **Verify statistical significance** (>10 runs minimum)
3. **Ensure hardware consistency** between measurements
4. **Request profiling data** for major optimizations
5. **Watch for baseline "inflation"** (gradual upward drift without justification)

### For CI Maintenance

1. **Monitor runner GPU temperature** - thermal throttling affects results
2. **Keep CUDA drivers updated** - but document version changes
3. **Periodically re-baseline** after hardware upgrades
4. **Archive historical reports** for trend analysis
5. **Alert on repeated warnings** - may indicate gradual degradation

---

## Related Documentation

- **GPU Performance Testing Guide**: `docs/GPU_PERFORMANCE_TESTING_GUIDE.md`
- **Final Performance Report**: `docs/FINAL_GPU_INDICATOR_PERFORMANCE_REPORT.md`
- **Async Optimization Results**: `docs/ASYNC_OPTIMIZATION_RESULTS.md`
- **GPU Kernel Timing Report**: `docs/GPU_KERNEL_TIMING_REPORT.md`

---

## Appendix: Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | All tests passed | Continue deployment |
| 1 | Performance regression detected | Fix regression or justify |
| 2 | Configuration error | Check baselines.json and GPU |

---

**Maintained by:** Performance Engineering Team
**Last Updated:** 2025-10-31
**Version:** 1.0.0

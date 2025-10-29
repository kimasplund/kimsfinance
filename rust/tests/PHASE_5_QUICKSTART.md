# Phase 5: Testing Quickstart Guide

**Quick Reference** for running the comprehensive Heston-Backtest integration tests.

---

## Current Status

⚠️ **BLOCKED** - 5 compilation errors must be fixed before tests can run.

See: `/home/kim-asplund/projects/kimsfinance/rust/docs/integration/PHASE_5_TESTING_REPORT.md`

---

## Quick Commands

### Fix Compilation Errors First

**File 1**: `/home/kim-asplund/projects/kimsfinance/rust/src/quantitative/heston/strategies_delta_neutral.rs`
```rust
// Line ~200: Change i8 to f64
let mut d_option_signals = self.device.alloc_zeros::<f64>(expected_len)?;

// Line 231: Convert f64 to i8
option_signal: option_signals_raw[i] as i8,
```

**File 2**: `/home/kim-asplund/projects/kimsfinance/rust/src/quantitative/heston/strategies_vol_arbitrage.rs`
```rust
// Line 45: Add trait import
use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig, PushKernelArg};
```

### Run All Tests (Automated)

```bash
cd /home/kim-asplund/projects/kimsfinance/rust
./tests/run_integration_tests.sh
```

### Run Specific Test Suites

```bash
# Unit tests (strategy types, Heston pricer, Greeks)
cargo test --features 'gpu,heston' --test heston_unit_tests -- --include-ignored

# End-to-end tests (full pipeline)
cargo test --features 'gpu,heston' --test heston_e2e_test -- --include-ignored

# Accuracy tests (GPU vs CPU)
cargo test --features 'gpu,heston' --test heston_accuracy_test -- --include-ignored

# Load tests (1000 strategies × 10K candles)
cargo test --features 'gpu,heston' --test heston_load_test -- --include-ignored

# Regression tests (backward compatibility)
cargo test --features 'gpu,heston' --test heston_regression_test -- --include-ignored

# Performance benchmarks
cargo bench --features 'gpu,heston' --bench heston_integration_bench
```

---

## Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `tests/data/heston_test_data.rs` | N/A | Synthetic data generators |
| `tests/integration/heston_unit_tests.rs` | 16 | Unit tests |
| `tests/integration/heston_e2e_test.rs` | 10 | End-to-end integration |
| `benches/heston_integration_bench.rs` | 7 groups | Performance benchmarks |
| `tests/integration/heston_accuracy_test.rs` | 6 | Accuracy validation |
| `tests/integration/heston_load_test.rs` | 5 | Load/stress tests |
| `tests/integration/heston_regression_test.rs` | 9 | Regression tests |

**Total**: 46 tests

---

## Performance Targets

| Phase | Target | Test |
|-------|--------|------|
| Phase 0 (Heston) | <20ms for 1000 options | `bench_phase0_heston_pricing` |
| Phase 1 (Indicators) | <50ms | `bench_full_pipeline_options_strategy` |
| Phase 2 (Signals) | <30ms | `bench_full_pipeline_options_strategy` |
| Phase 3 (Execution) | <100ms | `bench_full_pipeline_options_strategy` |
| Phase 4 (Metrics) | <10ms | `bench_full_pipeline_options_strategy` |
| **Total Pipeline** | **<250ms for 1000 strategies × 10K candles** | `test_e2e_large_scale_1000_strategies` |

---

## Accuracy Targets

| Metric | Target | Test |
|--------|--------|------|
| GPU vs CPU price | <0.05% error | `test_heston_vs_black_scholes_low_vol` |
| Greeks vs finite difference | <1% error | `test_greeks_vs_finite_difference` |
| Put-call parity | <0.05% error | `test_put_call_parity_accuracy` |

---

## Troubleshooting

### Compilation Fails

**Error**: `copy_to_host()` type mismatch

**Fix**: See remediation plan in main report (PHASE_5_TESTING_REPORT.md section "Remediation Plan")

### Tests are Skipped

**Reason**: Tests are marked `#[ignore]` by default (require GPU)

**Fix**: Use `--include-ignored` flag

### GPU Not Available

**Error**: "GPU device creation failed"

**Fix**: Ensure NVIDIA GPU with CUDA is available. Tests cannot run without GPU.

---

## Next Steps

1. Fix 5 compilation errors (30-60 min)
2. Run `./tests/run_integration_tests.sh`
3. Review results in generated report file
4. Update main report with actual benchmark results

---

**See Full Report**: `/home/kim-asplund/projects/kimsfinance/rust/docs/integration/PHASE_5_TESTING_REPORT.md`

# Agent 6: GPU Tick Batch Testing & Validation Suite

**Mission**: Build comprehensive testing infrastructure validating GPU tick batch implementation against CPU reference

**Status**: Implementation In Progress
**Target**: <0.01% deviation between GPU and CPU tick backtests

---

## Success Checklist

### Requirements

- [ ] **Unit Tests**: Each GPU kernel validates against CPU equivalent
  - [ ] GPU tick aggregation (Agent 1) accuracy test
  - [ ] GPU orderflow signals (Agent 2) accuracy test
  - [ ] GPU backtest execution (Agent 3) accuracy test
  - [ ] GPU quantization (Agent 5) signal match test
- [ ] **Integration Tests**: Full pipeline GPU vs CPU
  - [ ] End-to-end: trades → aggregation → orderflow → signals → backtest
  - [ ] GeneticOptimizer integration (population >= 50 uses GPU batch)
- [ ] **Performance Benchmarks**: Throughput, latency, VRAM usage
  - [ ] GPU batch throughput: 500M-1B ticks/sec target
  - [ ] VRAM usage: <12GB for batch size 10
  - [ ] Measure auto-batching effectiveness
- [ ] **Error Handling Tests**: Graceful degradation
  - [ ] VRAM overflow handling
  - [ ] NaN equity handling
  - [ ] GPU unavailable fallback

### Verification

- [ ] **Compiles without errors**: `cargo check --features gpu`
- [ ] **Passes clippy**: `cargo clippy --features gpu -- -D warnings`
- [ ] **Tests written and passing**: `cargo test --features gpu gpu_tick_`
- [ ] **Benchmarks run successfully**: `cargo bench --features gpu gpu_tick_batch_benchmark`
- [ ] **Follows project patterns**: DashMap, thiserror, ndarray usage
- [ ] **Edition 2024 compatible**: Uses LazyLock, proper lifetimes

---

## Self-Critique Questions

### Assumptions

- ✅ **Verified**: Trade struct in `src/binance/trades.rs` (trade_id, price, quantity, quote_quantity, timestamp_ms, is_buyer_maker)
- ✅ **Verified**: BacktestConfig in `src/backtest/engine.rs` (initial_capital, trading_fee, slippage, execution_latency_ms)
- ✅ **Verified**: GpuDevice pattern from `tests/gpu_accuracy_validation.rs`
- ⚠️ **Assumption**: GPU tick batch structs will follow naming: `BatchTickBacktest`, `gpu_tick_aggregation`, `gpu_orderflow_signals_batch`
- ⚠️ **Assumption**: 0.01% deviation threshold is achievable (may need to relax to 0.1%)

### Edge Cases

- ✅ **Covered**: NaN equity handling (already fixed in CPU optimizer)
- ✅ **Covered**: VRAM overflow graceful handling
- ✅ **Covered**: Empty trades array
- ⚠️ **Missing**: Trades with identical timestamps (ordering may differ)
- ⚠️ **Missing**: Extreme price movements (overflow protection)
- ⚠️ **Missing**: Very small batch sizes (< 5 strategies)

### Tradeoffs

- **Choice**: Use `approx::assert_abs_diff_eq!` for float comparisons
  - **Why**: Standard in project, handles epsilon properly
  - **Alternative**: Manual epsilon comparison (more verbose)
- **Choice**: Target <0.01% deviation for equity curves
  - **Why**: Tight enough for trading strategy validation
  - **Risk**: May need to relax to 0.1% due to floating-point accumulation
- **Choice**: Benchmark with 106M ticks (real dataset size)
  - **Why**: Representative of production workload
  - **Risk**: Long benchmark runtime (may reduce to 10M for CI)

### What Could Go Wrong?

1. **GPU kernels not implemented yet**: Tests will be placeholders until Agent 1-3-5 complete
2. **Float accumulation errors**: Equity curves may diverge beyond 0.01%
3. **VRAM limits**: May not fit 20 strategies × 106M ticks on 12GB GPU
4. **Test data generation**: Synthetic data may not reveal edge cases
5. **CI/CD**: GitHub runners don't have GPU (tests will be ignored)

---

## Patterns Discovered

### Error Handling (from `tests/gpu_accuracy_validation.rs`)

```rust
use approx::assert_abs_diff_eq;

const TOLERANCE: f64 = 1e-9;

fn validate_accuracy(gpu: &[f64], cpu: &[f64], name: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{}: Length mismatch", name);

    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        if g.is_nan() && c.is_nan() {
            continue;
        }

        let error = (g - c).abs();
        assert!(error < TOLERANCE,
            "{} index {}: error {:.2e}", name, i, error);
    }
}
```

### Test Data Generation (from `tests/tick_genetic_integration.rs`)

```rust
fn generate_test_trades(n: usize) -> Vec<Trade> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let base_price = 45000.0;
    let base_timestamp = 1704067200000i64;

    (0..n).map(|i| {
        let change = rng.gen_range(-0.0001..0.0001);
        let current_price = base_price * (1.0 + change);

        Trade {
            trade_id: i as u64,
            price: current_price,
            quantity: rng.gen_range(0.001..1.0),
            quote_quantity: current_price * quantity,
            timestamp_ms: base_timestamp + (i as i64),
            is_buyer_maker: rng.gen_bool(0.5),
        }
    }).collect()
}
```

### Benchmark Pattern (from `benches/comprehensive_gpu_validation.rs`)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_tick_batch");
    group.throughput(Throughput::Elements(106_000_000));
    group.sample_size(10);

    group.bench_function("cpu_rayon", |b| {
        b.iter(|| {
            cpu_parallel_backtests(black_box(&trades), black_box(&params))
        });
    });

    group.bench_function("gpu_batch", |b| {
        b.iter(|| {
            gpu_tick_batch(black_box(&trades), black_box(&params)).unwrap()
        });
    });

    group.finish();
}
```

---

## Edition & Version Checks

### Rust Version

```bash
$ grep "rust-version" Cargo.toml
rust-version = "1.90.0"
```

**Status**: ✅ Edition 2024 compatible (requires ≥1.85.0)

### Crate Versions

```bash
$ grep -E "(cudarc|ndarray|criterion|approx)" Cargo.toml
cudarc = { version = "=0.17.3", optional = true, features = ["driver", "cublas", "nvrtc", "cuda-13000"] }
ndarray = { version = "0.16.1", features = ["rayon"] }
criterion = { version = "0.5", features = ["html_reports"] }
# approx not in Cargo.toml yet - will add to dev-dependencies
```

**Status**:
- ✅ cudarc 0.17.3 (latest stable)
- ✅ ndarray 0.16.1 (latest)
- ✅ criterion 0.5 (latest)
- ⚠️ approx missing - will add `approx = "0.5"`

### Edition Features

**Using**:
- ✅ Edition 2024 syntax (edition = "2024")
- ✅ LazyLock for thread-safe lazy statics
- ⚠️ Let chains (may use if beneficial)
- ⚠️ RPIT lifetime capture (not applicable for tests)

---

## Test File Structure

### Unit Tests

```
tests/
├── gpu_tick_aggregation_test.rs       # Agent 1 validation
├── gpu_tick_orderflow_test.rs         # Agent 2 validation
├── gpu_tick_backtest_test.rs          # Agent 3 validation
├── gpu_tick_quantization_test.rs      # Agent 5 validation
└── gpu_tick_batch_integration.rs      # Full pipeline
```

### Benchmarks

```
benches/
├── gpu_tick_batch_benchmark.rs        # Throughput, latency, VRAM
├── gpu_tick_batch_accuracy.rs         # CPU vs GPU equity curves
└── gpu_tick_batch_scalability.rs      # Batch size scaling
```

### CI Configuration

```
.github/
└── workflows/
    └── gpu-tick-tests.yml             # GPU runner configuration
```

---

## Implementation Phases

### Phase 1: Test Infrastructure (This Agent) - 8 hours

- [x] Create test plan document
- [ ] Add `approx` to dev-dependencies
- [ ] Implement test data generators (synthetic trades)
- [ ] Create placeholder unit tests (will update when GPU kernels ready)
- [ ] Create integration test template
- [ ] Create benchmark templates

### Phase 2: Unit Tests (After Agent 1-3-5) - 8 hours

- [ ] Test GPU tick aggregation accuracy
- [ ] Test GPU orderflow signals accuracy
- [ ] Test GPU backtest execution accuracy
- [ ] Test GPU quantization signal match
- [ ] Edge case testing (NaN, empty, overflow)

### Phase 3: Integration Tests - 4 hours

- [ ] End-to-end pipeline test
- [ ] GeneticOptimizer integration test
- [ ] Auto-batching threshold test
- [ ] Fallback to CPU test

### Phase 4: Performance Benchmarks - 8 hours

- [ ] Throughput benchmark (500M-1B ticks/sec target)
- [ ] VRAM usage benchmark (track across batch sizes)
- [ ] Scalability benchmark (1-100 strategies)
- [ ] Comparison benchmark (GPU vs Rayon)

### Phase 5: CI/CD Integration - 4 hours

- [ ] GitHub Actions workflow
- [ ] Self-hosted GPU runner setup
- [ ] Exit code handling
- [ ] Artifact uploads (validation reports)

---

## Confidence Assessment

**Overall Confidence**: 75% (Medium-High)

### High Confidence (85-95%)

- ✅ **Test infrastructure patterns**: Clear examples from existing tests
- ✅ **Data generation**: Simple RNG-based trade generation works
- ✅ **Error handling**: Standard thiserror + approx pattern
- ✅ **Edition 2024 compatibility**: Rust 1.90.0 supports all features

### Medium Confidence (65-75%)

- ⚠️ **Deviation threshold**: 0.01% may be too tight (float accumulation)
- ⚠️ **GPU kernel APIs**: Assumed naming, may need adjustment
- ⚠️ **VRAM limits**: 106M × 100 strategies may exceed 12GB
- ⚠️ **Benchmark runtime**: Full 106M ticks may be too slow for CI

### Low Confidence (50-65%)

- ⚠️ **GPU kernel implementation timeline**: Agents 1-3-5 may take 40-80 hours
- ⚠️ **Float precision edge cases**: May discover new issues during testing
- ⚠️ **CI/CD GPU runner**: Self-hosted setup not trivial

---

## Tradeoffs & Alternatives

### Choice: Placeholder Tests Now vs Wait for GPU Implementation

**Decision**: Implement placeholder tests now

**Reasoning**:
- Establishes test structure immediately
- Easy to update when GPU kernels ready
- Documents expected API surface
- Allows parallel development

**Alternative**: Wait for Agent 1-3-5 completion
- Pros: Less rework if API changes
- Cons: No test framework ready, blocks parallel work

### Choice: 0.01% Deviation Threshold

**Decision**: Start with 0.01%, relax to 0.1% if needed

**Reasoning**:
- 0.01% is very tight (10 bps deviation on returns)
- Good for initial validation
- Easy to relax if float accumulation exceeds

**Alternative**: 0.1% from the start
- Pros: More forgiving, easier to achieve
- Cons: May hide subtle bugs

### Choice: Synthetic vs Real Data

**Decision**: Use synthetic data for unit tests, real data for integration

**Reasoning**:
- Synthetic: Fast, reproducible, covers edge cases
- Real: Validates production scenarios
- Hybrid approach best

**Alternative**: Real data only
- Pros: Most realistic
- Cons: Slow, requires data files, hard to isolate issues

---

## Known Limitations

1. **GPU Implementation Dependency**: Tests are placeholders until Agent 1-3-5 complete
2. **CI/CD GPU Runners**: GitHub runners don't have GPUs (tests will be `#[ignore]`)
3. **VRAM Constraints**: May need to reduce batch size for 12GB limit
4. **Float Precision**: 0.01% deviation may be unachievable due to different accumulation order
5. **Benchmark Duration**: Full 106M ticks × 100 strategies = 10.6B ticks (slow!)
6. **Edge Case Coverage**: Synthetic data may miss real-world anomalies

---

## Next Steps

1. ✅ Create test plan (this document)
2. Add `approx = "0.5"` to `Cargo.toml` dev-dependencies
3. Implement test data generators
4. Create placeholder unit tests with `#[ignore]` attribute
5. Create benchmark templates with placeholder implementations
6. Coordinate with Agent 1-3-5 for GPU kernel API surface
7. Update tests when GPU kernels ready
8. Run full validation suite

---

**Generated by**: Agent 6 (Testing & Validation Specialist)
**Date**: 2025-11-03
**Version**: 1.0.0

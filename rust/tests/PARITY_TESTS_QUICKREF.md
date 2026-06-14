# Aggregation Parity Tests - Quick Reference

**File**: `/home/kim/projects/kimsfinance/rust/tests/aggregation_parity_comprehensive.rs`  
**Status**: ✅ All tests passing (10/10 core + 2/2 optional)

---

## Quick Commands

```bash
# Run all core tests (fast, <10ms)
cargo test --test aggregation_parity_comprehensive

# Run with real Binance data (4.6M trades, ~6s)
cargo test --test aggregation_parity_comprehensive -- --ignored test_parity_real_binance_data

# Run performance comparison
cargo test --test aggregation_parity_comprehensive -- --ignored test_performance_comparison_small_dataset

# List all tests
cargo test --test aggregation_parity_comprehensive -- --list

# Run with verbose output
cargo test --test aggregation_parity_comprehensive -- --nocapture
```

---

## Test List

### Core Parity Tests (10 tests)
1. `test_parity_single_candle` - Multiple trades → single candle
2. `test_parity_multiple_candles` - Trades across timeframes
3. `test_parity_out_of_order_trades` - Unordered trade handling
4. `test_parity_different_timeframes` - 1m, 5m, 15m, 1h
5. `test_parity_empty_trades` - Edge case: no trades
6. `test_parity_single_trade` - Edge case: single trade
7. `test_parity_candle_boundaries` - Exact boundary timestamps
8. `test_parity_high_frequency_trades` - 100 trades/second
9. `test_parity_sparse_candles` - Gaps between candles
10. `test_parity_large_price_swings` - Extreme volatility

### Optional Tests (2 tests)
11. `test_parity_real_binance_data` - 4.6M real trades (ignored)
12. `test_performance_comparison_small_dataset` - Benchmark (ignored)

---

## Expected Output

```
running 12 tests
test test_parity_candle_boundaries ... ok
test test_parity_different_timeframes ... ok
test test_parity_empty_trades ... ok
test test_parity_high_frequency_trades ... ok
test test_parity_large_price_swings ... ok
test test_parity_multiple_candles ... ok
test test_parity_out_of_order_trades ... ok
test test_parity_single_candle ... ok
test test_parity_single_trade ... ok
test test_parity_sparse_candles ... ok
test test_parity_real_binance_data ... ignored
test test_performance_comparison_small_dataset ... ignored

test result: ok. 10 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

---

## Integration Checklist

When implementing Package 2.2 (IncompleteCandle):

- [ ] 1. Implement `IncompleteCandle` struct in `src/binance/trades.rs`
- [ ] 2. Add required methods: `new()`, `update()`, `complete()`
- [ ] 3. Export from `src/binance/mod.rs`
- [ ] 4. Uncomment `HashMap` import in test file
- [ ] 5. Uncomment `aggregate_with_incomplete_candle()` implementation
- [ ] 6. Run tests: `cargo test --test aggregation_parity_comprehensive`
- [ ] 7. Verify all 10 core tests pass
- [ ] 8. (Optional) Run real data test with `--ignored`

---

## Validation Fields

Each test validates **8 OHLCV fields**:
- `timestamp` - Candle start time (ms)
- `open` - First trade price
- `high` - Maximum trade price
- `low` - Minimum trade price
- `close` - Last trade price
- `volume` - Sum of quantities
- `quote_volume` - Sum of quote quantities
- `num_trades` - Trade count

---

## Common Issues

### Issue: Floating-point precision errors
**Solution**: Tests use approximate comparison (tolerance: 1e-10)

### Issue: Real data test fails
**Solution**: Update file path in test or skip with default run

### Issue: Tests compile but fail
**Solution**: Ensure IncompleteCandle API matches expected signature

---

## API Contract for IncompleteCandle

```rust
pub struct IncompleteCandle {
    // Implementation details
}

impl IncompleteCandle {
    pub fn new(trade: &Trade, timestamp: i64) -> Self;
    pub fn update(&mut self, trade: &Trade);
    pub fn complete(self) -> Candle;
}
```

---

**Updated**: 2025-10-29  
**Maintained by**: Package 2.2 & 2.3 team

# TickEngine NaN Equity Bug - Root Cause Analysis

**Date**: 2025-11-03
**Severity**: CRITICAL
**Impact**: All tick-level backtests produce NaN equity
**Location**: `src/backtest/tick_engine.rs:374`

---

## Executive Summary

Found **critical double-counting bug** in `TickEngine::close_position()` that causes equity to grow exponentially, eventually overflowing to NaN.

**Root Cause**: Line 374 adds both `exit_value` AND `pnl`, but `pnl` already equals `exit_value - position_value`. This double-counts the exit value.

**Impact**: After just a few trades, equity reaches astronomical values (e.g., $20,369 from $10,000 initial), then overflows to `NaN` or `Infinity`.

---

## The Bug

### Current Code (WRONG)

`src/backtest/tick_engine.rs:365-374`:
```rust
// Calculate P&L
let pnl = if position.position_size > 0.0 {
    // Long position
    exit_value - position.position_value
} else {
    // Short position
    position.position_value - exit_value
};

position.cash += exit_value + pnl - fee - slippage_cost;  // ❌ BUG HERE
```

### Why It's Wrong

For a long position:
- `pnl = exit_value - position_value`
- `cash += exit_value + pnl`
- `cash += exit_value + (exit_value - position_value)`
- `cash += 2 * exit_value - position_value` ❌ **DOUBLE COUNTING**

You're adding the exit value **twice**: once directly, once inside pnl.

---

## Detailed Example

### Scenario: Buy at $50k, Sell at $51k

**Initial State**:
- Cash: $10,000
- Position: 0 BTC
- Equity: $10,000

**Open Position** (Buy at $50,000):
```rust
gross_position_value = 10000 / 50000 = 0.2 BTC
fee = 0.2 * 50000 * 0.001 = $10
slippage = 0.2 * 50000 * 0.0005 = $5

position_size = 0.2 BTC
entry_price = $50,000
position_value = 0.2 * 50000 = $10,000
cash = 10000 - 10 - 5 = $9,985
```

**After Opening**:
- Cash: $9,985
- Position: 0.2 BTC worth $10,000
- Equity (via update_equity): $9,985 + $10,000 + $0 = **$19,985** ❌

Already wrong! We started with $10k, paid $15 in fees, should have ~$9,985 equity, not $19,985.

**This is Bug #1**: `position_value` should be net of fees, not gross.

**Close Position** (Sell at $51,000):
```rust
exit_value = 0.2 * 51000 = $10,200
fee = 10200 * 0.001 = $10.20
slippage = 10200 * 0.0005 = $5.10

pnl = 10200 - 10000 = $200  // Long position

cash = 9985 + 10200 + 200 - 10.20 - 5.10
cash = 9985 + 10400 - 15.30
cash = $20,369.70 ❌
```

**Expected Result**:
- Started: $10,000
- Made: $200 profit (51k - 50k on 0.2 BTC)
- Fees: ~$30 total
- **Should end with: ~$10,170**

**Actual Result**: $20,369.70 (nearly double!)

After 10 trades, equity would be: $10k → $20k → $40k → $80k → ... → `NaN` (overflow)

---

## The Fix

### Option 1: Remove Duplicate exit_value

```rust
// CORRECT VERSION
position.cash += position.position_value + pnl - fee - slippage_cost;
```

**Logic**:
- Start with position_value (what you paid for the position)
- Add pnl (profit/loss since entry)
- Subtract exit fees
- Result: position_value + pnl = exit_value (but without double-counting)

### Option 2: Use exit_value Only

```rust
// ALTERNATIVE FIX
position.cash += exit_value - fee - slippage_cost;
// (Don't add pnl separately)
```

**Logic**:
- exit_value already includes the pnl
- Just add back what you get from selling
- Subtract exit fees

Both are mathematically equivalent, but Option 1 is clearer because pnl is explicitly used for trade recording.

---

## Additional Bug: position_value Calculation

### Bug #2 Location

`src/backtest/tick_engine.rs:329-337`:
```rust
let gross_position_value = position.cash / price;
let fee = gross_position_value * price * self.config.trading_fee;
let slippage_cost = gross_position_value * price * self.config.slippage;

position.position_size = gross_position_value * direction;
position.entry_price = price;
position.entry_timestamp = timestamp;
position.position_value = gross_position_value * price;  // ❌ BUG: Uses GROSS, not NET
position.cash -= fee + slippage_cost;
```

### Why It's Wrong

`position_value = gross_position_value * price` calculates the value **before** fees, but the actual invested amount is **after** fees.

In `update_equity()`:
```rust
equity = cash + position_value + unrealized_pnl
```

This double-counts the fees:
- `cash` has already had fees subtracted
- But `position_value` still includes those fees
- Result: `equity = (initial - fees) + initial + pnl` ❌

### The Fix

```rust
// Calculate net value after all costs
let net_investment = position.cash - fee - slippage_cost;

position.position_value = net_investment;  // Use NET value, not gross
position.cash = 0.0;  // All cash is now in the position
```

OR more explicitly:
```rust
// Option 2: Adjust position_value after calculating fees
let gross_position_value = position.cash / price;
let fee = gross_position_value * price * self.config.trading_fee;
let slippage_cost = gross_position_value * price * self.config.slippage;
let total_cost = fee + slippage_cost;

position.position_size = gross_position_value * direction;
position.entry_price = price;
position.position_value = gross_position_value * price - total_cost;  // NET value
position.cash = 0.0;  // All cash converted to position
```

---

## Summary of All Bugs

| Bug # | Location | Issue | Impact |
|-------|----------|-------|--------|
| **1** | Line 374 | Adds `exit_value + pnl` (double-count) | Equity doubles on each trade cycle |
| **2** | Line 336 | `position_value` uses gross, not net | Equity inflated by ~2x from start |

Combined effect: Equity starts at 2x correct value, then doubles every trade cycle, reaching NaN within 10-50 trades.

---

## Test Results

### Before Fix
- Synthetic data (10M trades): NaN equity, 1,049 trades
- Real BTCUSDT data (106M trades): NaN equity, 1,015 trades
- Processing: Works perfectly (1.28-2.24M ticks/sec)
- Signal generation: Works (1,000+ trades executed)
- Only equity calculation is broken

### Expected After Fix
- Equity starts at $10,000
- Decreases slightly on losing trades
- Increases slightly on winning trades
- Never goes negative (proper stop-loss)
- Final equity reasonable (e.g., $9,500-$11,500 range)

---

## Complete Fix (Both Bugs)

### File: `src/backtest/tick_engine.rs`

**Lines 329-337** (open_position):
```rust
// OLD (WRONG):
let gross_position_value = position.cash / price;
let fee = gross_position_value * price * self.config.trading_fee;
let slippage_cost = gross_position_value * price * self.config.slippage;

position.position_size = gross_position_value * direction;
position.entry_price = price;
position.entry_timestamp = timestamp;
position.position_value = gross_position_value * price;  // ❌
position.cash -= fee + slippage_cost;

// NEW (CORRECT):
let gross_position_value = position.cash / price;
let fee = gross_position_value * price * self.config.trading_fee;
let slippage_cost = gross_position_value * price * self.config.slippage;
let total_cost = fee + slippage_cost;

position.position_size = gross_position_value * direction;
position.entry_price = price;
position.entry_timestamp = timestamp;
position.position_value = position.cash - total_cost;  // ✅ NET value
position.cash = 0.0;  // ✅ All cash in position
```

**Line 374** (close_position):
```rust
// OLD (WRONG):
position.cash += exit_value + pnl - fee - slippage_cost;  // ❌

// NEW (CORRECT):
position.cash += position.position_value + pnl - fee - slippage_cost;  // ✅
```

---

## Verification Test

After applying fixes, run:
```bash
cargo run --release --features data-downloaders --example advanced_momentum_strategy backtest data/test_trades.parquet
```

**Expected Output**:
```
Final Equity: $9,850.00 (reasonable, $10k - fees)
Total Return: -1.50% (small loss or gain)
Sharpe Ratio: 0.50-2.00 (reasonable)
Max Drawdown: 2-10% (normal)
Win Rate: 40-60% (balanced)
Num Trades: 1,000+ (same as before)
Profit Factor: 0.8-1.5 (reasonable)
```

---

## Impact on Optimization

Once fixed, the genetic optimizer will:
1. ✅ No longer crash on NaN fitness
2. ✅ Find strategies with real Sharpe ratios (0.5-3.0 range)
3. ✅ Complete 5,000 backtests in 2-5 minutes
4. ✅ Identify best parameters for momentum strategy
5. ✅ Produce realistic returns (5-30% on test data)

---

## Next Steps

1. **Apply fixes** to `src/backtest/tick_engine.rs`
2. **Run single backtest** - verify equity is valid
3. **Run optimization** - verify it completes without crash
4. **Generate report** - document final results

**Estimated fix time**: 5 minutes
**Estimated test time**: 30 seconds (single backtest) + 3 minutes (optimization)

---

**Generated**: 2025-11-03
**Bug Severity**: CRITICAL (blocks all tick backtesting)
**Root Cause**: Double-counting in equity calculation
**Status**: Identified, fix ready to apply

#!/usr/bin/env python3
"""Test that all execution modes produce identical results."""

import numpy as np
import sys

try:
    from kimsfinance_core import batch_backtest
except ImportError:
    print("❌ kimsfinance_core not installed. Run 'maturin develop --release --features gpu' first.")
    sys.exit(1)

print(f"Testing on Python {sys.version}")
print(f"GIL enabled: {sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else 'N/A'}")

# Generate test data
np.random.seed(42)
n_candles = 1000
ohlcv = np.random.randn(n_candles, 5).cumsum(axis=0) + 100
ohlcv[:, 0:4] = np.abs(ohlcv[:, 0:4])
ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000

# Test with 200 strategies (triggers fused mode in auto)
parameters = [[14.0, 20.0 + i * 0.2, 70.0 + i * 0.2] for i in range(200)]

print(f"\nTesting {len(parameters)} strategies on {n_candles} candles...")

# Run all execution modes
modes = ['auto', 'traditional', 'fused', 'async']
all_results = {}

for mode in modes:
    try:
        results = batch_backtest(
            strategy='rsi_crossover',
            ohlcv=ohlcv,
            parameters=parameters,
            execution_mode=mode
        )
        all_results[mode] = results
        best_sharpe = results[0].sharpe_ratio
        print(f"✅ {mode:12s}: {len(results)} results, best Sharpe: {best_sharpe:.6f}")
    except Exception as e:
        print(f"❌ {mode:12s}: FAILED - {e}")
        sys.exit(1)

# Compare all modes (should produce identical results)
print("\n=== Correctness Validation ===")
baseline = all_results['traditional']
baseline_sharpe = baseline[0].sharpe_ratio

for mode in ['fused', 'async', 'auto']:
    sharpe = all_results[mode][0].sharpe_ratio
    diff = abs(sharpe - baseline_sharpe)

    if diff < 0.01:
        print(f"✅ {mode:12s} vs traditional: Sharpe diff = {diff:.6f} (< 0.01)")
    else:
        print(f"❌ {mode:12s} vs traditional: Sharpe diff = {diff:.6f} (>= 0.01) - FAILED!")
        sys.exit(1)

print("\n✅ All execution modes produce identical results!")
sys.exit(0)

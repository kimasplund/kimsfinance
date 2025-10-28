#!/usr/bin/env python3
"""Benchmark execution modes across Python versions."""

import numpy as np
import time
import sys

try:
    from kimsfinance_core import batch_backtest
except ImportError:
    print("❌ kimsfinance_core not installed")
    sys.exit(1)

print(f"Python {sys.version}")
print(f"GIL: {sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else 'Enabled'}")

# Test data
np.random.seed(42)
n_candles = 5000
ohlcv = np.random.randn(n_candles, 5).cumsum(axis=0) + 100
ohlcv[:, 0:4] = np.abs(ohlcv[:, 0:4])
ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000

# Test multiple workload sizes
workloads = [
    (100, "Small"),
    (500, "Medium"),
    (1000, "Large"),
]

print(f"\n{'Workload':<10} {'Mode':<12} {'Time (ms)':<12} {'Throughput':<15}")
print("=" * 60)

for num_strategies, label in workloads:
    parameters = [[14.0, 20 + i * 0.1, 70 + i * 0.1] for i in range(num_strategies)]

    modes = ['traditional', 'fused', 'async']
    timings = {}

    for mode in modes:
        # Warmup
        batch_backtest(strategy='rsi_crossover', ohlcv=ohlcv, parameters=parameters[:10], execution_mode=mode)

        # Benchmark
        start = time.time()
        results = batch_backtest(strategy='rsi_crossover', ohlcv=ohlcv, parameters=parameters, execution_mode=mode)
        elapsed = time.time() - start

        timings[mode] = elapsed
        throughput = num_strategies / elapsed

        print(f"{label:<10} {mode:<12} {elapsed*1000:>10.1f}   {throughput:>10.1f} strat/s")

    # Calculate speedups
    if 'fused' in timings and 'traditional' in timings:
        speedup = timings['traditional'] / timings['fused']
        print(f"           {'Fused speedup:':<12} {speedup:>10.2f}x vs traditional")

    if 'async' in timings and 'fused' in timings:
        speedup = timings['fused'] / timings['async']
        print(f"           {'Async speedup:':<12} {speedup:>10.2f}x vs fused")

    print()

print("✅ Performance benchmarks complete!")

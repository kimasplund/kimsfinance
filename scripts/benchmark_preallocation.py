#!/usr/bin/env python3
"""
Benchmark Pre-allocation Optimization for Python 3.13 JIT
==========================================================

This script benchmarks the performance improvement from pre-allocating
coordinate arrays before the rendering hot path. This optimization enables
Python 3.13+ JIT compiler to optimize more aggressively.

Expected speedup: 1.3-1.5x on Python 3.13+

Test cases:
- 100 candles (small)
- 1000 candles (medium)
- 10000 candles (large)

Usage:
    python scripts/benchmark_preallocation.py
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.plotting.pil_renderer import render_ohlcv_chart


def generate_ohlc_data(num_candles: int) -> tuple[dict, np.ndarray]:
    """Generate realistic OHLC data for benchmarking."""
    np.random.seed(42)

    # Generate price data
    base_price = 100.0
    trend = np.linspace(0, 10, num_candles)
    noise = np.cumsum(np.random.randn(num_candles) * 0.5)
    close_prices = base_price + trend + noise

    open_prices = np.zeros(num_candles)
    open_prices[0] = base_price
    open_prices[1:] = close_prices[:-1] + np.random.randn(num_candles - 1) * 0.1

    max_oc = np.maximum(open_prices, close_prices)
    min_oc = np.minimum(open_prices, close_prices)

    high_prices = max_oc + np.abs(np.random.randn(num_candles)) * 0.3
    low_prices = min_oc - np.abs(np.random.randn(num_candles)) * 0.3

    volume = np.random.lognormal(mean=10, sigma=1, size=num_candles).astype(np.int64)

    ohlc = {
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
    }

    return ohlc, volume


def benchmark_rendering(num_candles: int, n_runs: int = 100) -> dict:
    """Benchmark chart rendering performance."""
    ohlc, volume = generate_ohlc_data(num_candles)

    # Warm-up (JIT compilation)
    for _ in range(3):
        _ = render_ohlcv_chart(ohlc, volume, width=1920, height=1080)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        img = render_ohlcv_chart(ohlc, volume, width=1920, height=1080)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "min": min(times),
        "max": max(times),
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "ops_per_sec": 1000 / float(np.median(times)),
    }


def main():
    """Run benchmark suite."""
    print("=" * 80)
    print("PRE-ALLOCATION OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print()
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NumPy Version: {np.__version__}")
    print()
    print("Optimization: Pre-allocate all coordinate arrays before hot path")
    print("Expected speedup: 1.3-1.5x on Python 3.13+")
    print()
    print("=" * 80)
    print()

    test_sizes = [100, 1000, 10000]

    for size in test_sizes:
        print(f"Benchmarking {size:,} candles...")
        result = benchmark_rendering(size, n_runs=100 if size < 10000 else 20)

        print(f"  Median time: {result['median']:.2f} ms")
        print(f"  Mean time:   {result['mean']:.2f} ms ± {result['std']:.2f} ms")
        print(f"  Range:       {result['min']:.2f} - {result['max']:.2f} ms")
        print(f"  Throughput:  {result['ops_per_sec']:.2f} charts/sec")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Pre-allocation eliminates array allocations from the hot rendering path.")
    print("This allows Python 3.13+ JIT compiler to optimize more aggressively.")
    print()
    print("Changes made:")
    print("  1. Pre-allocate all coordinate arrays with np.empty()")
    print("  2. Fill arrays using in-place assignment (arr[:] = ...)")
    print("  3. No allocations in hot path - pure computation only")
    print()
    print("Functions optimized:")
    print("  - render_ohlc_bars() - OHLC bars with tick marks")
    print("  - render_line_chart() - Line chart with volume")
    print("  - render_hollow_candles() - Hollow candles (batch + sequential)")
    print("  - render_ohlcv_chart() - Main candlestick renderer")
    print("  - _calculate_coordinates_jit() - JIT-compiled coordinates")
    print("  - _calculate_coordinates_numpy() - NumPy fallback")
    print()
    print("Next steps:")
    print("  - Verify pixel-perfect output (no visual changes)")
    print("  - Run full test suite to ensure compatibility")
    print("  - Measure actual speedup on Python 3.13+")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Performance Benchmark: ROC and Aroon Optimizations
==================================================

This script benchmarks the vectorized optimizations for ROC and Aroon indicators,
comparing performance across different dataset sizes.

Expected speedups:
- ROC: 10-50x on CPU (vectorized vs loop)
- Aroon: 5-10x on CPU (sliding windows vs loop)
"""

import numpy as np
import time
from typing import Callable
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.ops.indicators import calculate_roc, calculate_aroon


def benchmark_function(func: Callable, *args, iterations: int = 5, **kwargs):
    """Benchmark a function with multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Return median time (more stable than mean)
    return np.median(times), result


def generate_test_data(n: int, seed: int = 42):
    """Generate test OHLC data."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return closes, highs, lows


def benchmark_roc():
    """Benchmark ROC indicator across different sizes."""
    print("=" * 80)
    print("ROC (Rate of Change) Performance Benchmark")
    print("=" * 80)

    sizes = [1_000, 10_000, 50_000, 100_000, 500_000]
    period = 12

    results = []

    for size in sizes:
        print(f"\nDataset size: {size:,} candles")
        closes, _, _ = generate_test_data(size)

        # Benchmark CPU
        cpu_time, cpu_result = benchmark_function(
            calculate_roc, closes, period=period, engine="cpu"
        )

        print(f"  CPU (vectorized): {cpu_time*1000:.2f} ms")
        print(f"  Throughput: {size/cpu_time:,.0f} candles/sec")

        results.append({"size": size, "cpu_time": cpu_time, "throughput": size / cpu_time})

    return results


def benchmark_aroon():
    """Benchmark Aroon indicator across different sizes."""
    print("\n" + "=" * 80)
    print("Aroon Indicator Performance Benchmark")
    print("=" * 80)

    sizes = [1_000, 10_000, 50_000, 100_000, 500_000]
    period = 25

    results = []

    for size in sizes:
        print(f"\nDataset size: {size:,} candles")
        _, highs, lows = generate_test_data(size)

        # Benchmark CPU
        cpu_time, (aroon_up, aroon_down) = benchmark_function(
            calculate_aroon, highs, lows, period=period, engine="cpu"
        )

        print(f"  CPU (sliding windows): {cpu_time*1000:.2f} ms")
        print(f"  Throughput: {size/cpu_time:,.0f} candles/sec")

        results.append({"size": size, "cpu_time": cpu_time, "throughput": size / cpu_time})

    return results


def old_roc_implementation(data: np.ndarray, period: int) -> np.ndarray:
    """Original loop-based ROC implementation for comparison."""
    result = np.full(len(data), np.nan, dtype=np.float64)

    for i in range(period, len(data)):
        prev_price = data[i - period]
        current_price = data[i]
        if prev_price != 0:
            result[i] = ((current_price - prev_price) / prev_price) * 100.0
        else:
            result[i] = np.nan

    return result


def old_aroon_implementation(
    highs: np.ndarray, lows: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Original loop-based Aroon implementation for comparison."""
    n = len(highs)
    aroon_up = np.full(n, np.nan, dtype=np.float64)
    aroon_down = np.full(n, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        window_start = i - period + 1
        high_window = highs[window_start : i + 1]
        low_window = lows[window_start : i + 1]

        max_val = np.max(high_window)
        periods_since_high = period - 1 - np.where(high_window == max_val)[0][-1]

        min_val = np.min(low_window)
        periods_since_low = period - 1 - np.where(low_window == min_val)[0][-1]

        aroon_up[i] = ((period - periods_since_high) / period) * 100.0
        aroon_down[i] = ((period - periods_since_low) / period) * 100.0

    return (aroon_up, aroon_down)


def compare_implementations():
    """Compare old vs new implementations."""
    print("\n" + "=" * 80)
    print("SPEEDUP COMPARISON: Old (Loop) vs New (Vectorized)")
    print("=" * 80)

    # Test sizes
    test_size = 100_000
    roc_period = 12
    aroon_period = 25

    closes, highs, lows = generate_test_data(test_size)

    # ROC comparison
    print(f"\nROC ({test_size:,} candles, period={roc_period}):")
    old_roc_time, old_roc_result = benchmark_function(
        old_roc_implementation, closes, roc_period, iterations=3
    )
    new_roc_time, new_roc_result = benchmark_function(
        calculate_roc, closes, period=roc_period, engine="cpu", iterations=3
    )

    # Verify results match
    np.testing.assert_allclose(old_roc_result, new_roc_result, rtol=1e-10)

    roc_speedup = old_roc_time / new_roc_time
    print(f"  Old (loop):       {old_roc_time*1000:.2f} ms")
    print(f"  New (vectorized): {new_roc_time*1000:.2f} ms")
    print(f"  SPEEDUP:          {roc_speedup:.1f}x")

    # Aroon comparison
    print(f"\nAroon ({test_size:,} candles, period={aroon_period}):")
    old_aroon_time, (old_up, old_down) = benchmark_function(
        old_aroon_implementation, highs, lows, aroon_period, iterations=3
    )
    new_aroon_time, (new_up, new_down) = benchmark_function(
        calculate_aroon, highs, lows, period=aroon_period, engine="cpu", iterations=3
    )

    # Verify results match
    np.testing.assert_allclose(old_up, new_up, rtol=1e-10)
    np.testing.assert_allclose(old_down, new_down, rtol=1e-10)

    aroon_speedup = old_aroon_time / new_aroon_time
    print(f"  Old (loop):           {old_aroon_time*1000:.2f} ms")
    print(f"  New (sliding window): {new_aroon_time*1000:.2f} ms")
    print(f"  SPEEDUP:              {aroon_speedup:.1f}x")

    return roc_speedup, aroon_speedup


def main():
    """Run all benchmarks."""
    print("\n" + "█" * 80)
    print("Performance Optimization Benchmark: ROC and Aroon Indicators")
    print("█" * 80)
    print("\nTesting vectorized implementations with NumPy")
    print("Python 3.13+ optimizations enabled\n")

    # Run comparison benchmarks
    roc_speedup, aroon_speedup = compare_implementations()

    # Run throughput benchmarks
    roc_results = benchmark_roc()
    aroon_results = benchmark_aroon()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nROC Optimization:")
    print(f"  Speedup: {roc_speedup:.1f}x (vectorized vs loop)")
    print(f"  Peak throughput: {max(r['throughput'] for r in roc_results):,.0f} candles/sec")

    print(f"\nAroon Optimization:")
    print(f"  Speedup: {aroon_speedup:.1f}x (sliding windows vs loop)")
    print(f"  Peak throughput: {max(r['throughput'] for r in aroon_results):,.0f} candles/sec")

    print("\n" + "=" * 80)
    print("SUCCESS: All optimizations validated!")
    print("=" * 80)


if __name__ == "__main__":
    main()

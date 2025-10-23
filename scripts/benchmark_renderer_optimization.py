#!/usr/bin/env python3
"""
Benchmark script to measure rendering performance improvements.

Tests:
1. PIL OHLC bars rendering (volume coordinate pre-computation)
2. PIL hollow candles rendering (volume coordinate pre-computation)
3. SVG candlestick rendering (vectorized coordinates)
4. SVG hollow candles rendering (vectorized coordinates)

Expected improvements:
- PIL renderers: 5-10% speedup
- SVG renderers: 10-15% speedup
"""

import time
import numpy as np
from kimsfinance.plotting.pil_renderer import render_ohlc_bars, render_hollow_candles
from kimsfinance.plotting.svg_renderer import render_candlestick_svg, render_hollow_candles_svg


def generate_test_data(num_candles: int = 500):
    """Generate realistic OHLCV test data."""
    np.random.seed(42)
    base_price = 100.0

    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, num_candles)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(open_prices, close_prices) * (
        1 + np.abs(np.random.normal(0, 0.01, num_candles))
    )
    low_prices = np.minimum(open_prices, close_prices) * (
        1 - np.abs(np.random.normal(0, 0.01, num_candles))
    )

    # Generate volume
    volume = np.random.randint(1000, 10000, num_candles)

    ohlc = {
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
    }

    return ohlc, volume


def benchmark_renderer(render_func, name: str, num_iterations: int = 100, **kwargs):
    """Benchmark a rendering function."""
    print(f"\nBenchmarking: {name}")
    print(f"Iterations: {num_iterations}")

    # Warmup
    for _ in range(3):
        render_func(**kwargs)

    # Actual benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        render_func(**kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"Mean: {mean_time:.2f} ms")
    print(f"Std:  {std_time:.2f} ms")
    print(f"Min:  {min_time:.2f} ms")
    print(f"Max:  {max_time:.2f} ms")
    print(f"Throughput: {1000/mean_time:.1f} charts/sec")

    return mean_time, std_time


def main():
    print("=" * 70)
    print("Renderer Performance Optimization Benchmark")
    print("=" * 70)

    # Test different dataset sizes
    sizes = [50, 100, 500]

    for num_candles in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {num_candles} candles")
        print("=" * 70)

        ohlc, volume = generate_test_data(num_candles)

        # Benchmark PIL renderers
        print("\n--- PIL Renderers ---")

        mean_ohlc, std_ohlc = benchmark_renderer(
            render_ohlc_bars,
            f"PIL OHLC Bars ({num_candles} candles)",
            num_iterations=100,
            ohlc=ohlc,
            volume=volume,
            width=1920,
            height=1080,
            theme="classic",
        )

        mean_hollow, std_hollow = benchmark_renderer(
            render_hollow_candles,
            f"PIL Hollow Candles ({num_candles} candles)",
            num_iterations=100,
            ohlc=ohlc,
            volume=volume,
            width=1920,
            height=1080,
            theme="classic",
        )

        # Benchmark SVG renderers
        print("\n--- SVG Renderers ---")

        mean_svg, std_svg = benchmark_renderer(
            render_candlestick_svg,
            f"SVG Candlesticks ({num_candles} candles)",
            num_iterations=50,  # Fewer iterations for SVG (slower)
            ohlc=ohlc,
            volume=volume,
            width=1920,
            height=1080,
            theme="classic",
        )

        mean_svg_hollow, std_svg_hollow = benchmark_renderer(
            render_hollow_candles_svg,
            f"SVG Hollow Candles ({num_candles} candles)",
            num_iterations=50,
            ohlc=ohlc,
            volume=volume,
            width=1920,
            height=1080,
            theme="classic",
        )

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
    print("\nExpected improvements:")
    print("- PIL renderers: 5-10% speedup (volume coordinate pre-computation)")
    print("- SVG renderers: 10-15% speedup (vectorized coordinate calculations)")
    print("\nNote: Improvements are more noticeable with larger datasets (500+ candles)")


if __name__ == "__main__":
    main()

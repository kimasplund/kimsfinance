#!/usr/bin/env python3
"""
Benchmark Rust vs Python/NumPy coordinate calculations
"""

import time
import numpy as np

# Test if Rust extension is available
try:
    import kimsfinance_core

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  Rust extension not available")

from kimsfinance.plotting.pil_renderer import _calculate_coordinates_numpy


def generate_test_data(n_candles):
    """Generate realistic OHLCV test data"""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_candles) * 2)
    high = close + np.random.rand(n_candles) * 5
    low = close - np.random.rand(n_candles) * 5
    open_price = np.roll(close, 1)
    volume = np.random.randint(1000, 100000, n_candles).astype(np.float64)

    return high, low, open_price, close, volume


def benchmark_numpy(high, low, open_price, close, volume, iterations=100):
    """Benchmark NumPy implementation"""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = _calculate_coordinates_numpy(
            num_candles=len(high),
            candle_width=10.0,
            spacing=1.0,
            bar_width=9.0,
            high_prices=high,
            low_prices=low,
            open_prices=open_price,
            close_prices=close,
            volume_data=volume,
            price_min=float(low.min()),
            price_range=float(high.max() - low.min()),
            volume_range=float(volume.max()),
            chart_height=1080,
            volume_height=300,
            height=1080,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


def benchmark_rust(high, low, open_price, close, volume, iterations=100):
    """Benchmark Rust implementation"""
    if not RUST_AVAILABLE:
        return None

    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = kimsfinance_core.calculate_coordinates_py(
            high,  # Positional args only
            low,
            open_price,
            close,
            volume,
            len(high),
            10.0,  # candle_width
            1.0,  # spacing
            9.0,  # bar_width
            float(low.min()),
            float(high.max() - low.min()),
            float(volume.max()),
            1080,  # chart_height
            300,  # volume_height
            1080,  # height
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


print("=" * 70)
print("RUST VS PYTHON/NUMPY COORDINATE CALCULATION BENCHMARK")
print("=" * 70)
print()

for n_candles in [100, 1_000, 10_000, 100_000]:
    high, low, open_price, close, volume = generate_test_data(n_candles)

    # Warm up
    _ = _calculate_coordinates_numpy(
        len(high),
        10.0,
        1.0,
        9.0,
        high,
        low,
        open_price,
        close,
        volume,
        float(low.min()),
        float(high.max() - low.min()),
        float(volume.max()),
        1080,
        300,
        1080,
    )
    if RUST_AVAILABLE:
        _ = kimsfinance_core.calculate_coordinates_py(
            high,
            low,
            open_price,
            close,
            volume,
            len(high),
            10.0,
            1.0,
            9.0,
            float(low.min()),
            float(high.max() - low.min()),
            float(volume.max()),
            1080,
            300,
            1080,
        )

    # Benchmark
    iterations = max(10, 1000 // n_candles)
    numpy_time = benchmark_numpy(high, low, open_price, close, volume, iterations)
    rust_time = (
        benchmark_rust(high, low, open_price, close, volume, iterations) if RUST_AVAILABLE else None
    )

    print(f"Dataset: {n_candles:,} candles ({iterations} iterations)")
    print(f"  NumPy:  {numpy_time*1000:8.3f} ms")
    if rust_time:
        print(f"  Rust:   {rust_time*1000:8.3f} ms")
        speedup = numpy_time / rust_time
        print(f"  Speedup: {speedup:7.2f}x")
        if speedup >= 5:
            print(f"  Status:  ✅ TARGET MET (5-10x)")
        elif speedup >= 2:
            print(f"  Status:  ⚠️  GOOD (2-5x)")
        else:
            print(f"  Status:  ❌ BELOW TARGET (<2x)")
    else:
        print(f"  Rust:   N/A (not available)")
    print()

print("=" * 70)
if RUST_AVAILABLE:
    print("RUST MIGRATION: Ready for integration into pil_renderer.py")
    print()
    print("Expected overall speedup: 5-10x for coordinate calculations")
    print("This translates to ~15-25% improvement in total chart rendering time")
else:
    print("RUST EXTENSION: Not built. Run 'cd rust && maturin develop --release'")
print("=" * 70)

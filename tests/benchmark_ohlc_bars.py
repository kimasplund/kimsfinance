"""
Quick performance benchmark for OHLC bars renderer.
"""

import time
import numpy as np
from kimsfinance.plotting.renderer import render_ohlc_bars


def benchmark_ohlc_bars():
    """Benchmark OHLC bars rendering performance."""
    # Create test data
    num_bars = 50
    ohlc = {
        "open": np.random.uniform(90, 110, num_bars),
        "high": np.random.uniform(110, 120, num_bars),
        "low": np.random.uniform(80, 90, num_bars),
        "close": np.random.uniform(90, 110, num_bars),
    }
    volume = np.random.uniform(500, 2000, num_bars)

    # Warm-up run
    _ = render_ohlc_bars(ohlc, volume, width=800, height=600, enable_antialiasing=False)

    # Benchmark run
    num_iterations = 1000
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        img = render_ohlc_bars(ohlc, volume, width=800, height=600, enable_antialiasing=False)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    charts_per_sec = num_iterations / elapsed
    ms_per_chart = (elapsed * 1000) / num_iterations

    print(f"\n{'='*60}")
    print(f"OHLC Bars Renderer Performance Benchmark")
    print(f"{'='*60}")
    print(f"Test configuration:")
    print(f"  - Number of bars: {num_bars}")
    print(f"  - Image size: 800x600")
    print(f"  - Iterations: {num_iterations}")
    print(f"\nResults:")
    print(f"  - Total time: {elapsed:.2f}s")
    print(f"  - Charts/second: {charts_per_sec:,.1f}")
    print(f"  - Time per chart: {ms_per_chart:.2f}ms")
    print(f"\nTarget: >5000 charts/sec")
    print(f"Status: {'✓ PASS' if charts_per_sec > 5000 else '✗ FAIL'}")
    print(f"{'='*60}\n")

    return charts_per_sec


if __name__ == "__main__":
    benchmark_ohlc_bars()

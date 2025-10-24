#!/usr/bin/env python3
"""
Benchmark: Parallel Rendering Performance (Free-Threading vs Multiprocessing)
==============================================================================

Tests the performance of ThreadPoolExecutor (Python 3.14t) vs ProcessPoolExecutor
for batch chart rendering.

Expected Results:
- Python 3.14t (free-threading): ThreadPoolExecutor ~5x faster
- Standard Python: ProcessPoolExecutor (baseline)

Usage:
    # Standard Python (ProcessPoolExecutor)
    python benchmarks/benchmark_parallel_rendering.py

    # Python 3.14t (ThreadPoolExecutor) - requires python3.14t build
    python3.14t benchmarks/benchmark_parallel_rendering.py
"""

from __future__ import annotations

import sys
import time
import platform
import polars as pl
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.plotting.parallel import render_charts_parallel, _get_optimal_executor
from kimsfinance.core import EngineManager


def generate_ohlcv_data(num_candles: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate realistic OHLCV data for benchmarking."""
    np.random.seed(seed)

    # Start with a base price and simulate random walk
    base_price = 100.0
    price_changes = np.random.randn(num_candles) * 2
    close_prices = base_price + np.cumsum(price_changes)

    # Ensure prices stay positive
    close_prices = np.maximum(close_prices, 1.0)

    # Generate OHLC from close prices
    high_prices = close_prices * (1 + np.abs(np.random.randn(num_candles) * 0.02))
    low_prices = close_prices * (1 - np.abs(np.random.randn(num_candles) * 0.02))

    # Open is previous close (with some noise)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    open_prices += np.random.randn(num_candles) * 0.5

    # Volume
    volume = np.random.randint(1000, 100000, size=num_candles)

    return {
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    }


def benchmark_parallel_rendering(
    num_charts: int = 100,
    candles_per_chart: int = 100,
    num_workers: int = 4,
) -> dict[str, float]:
    """
    Benchmark parallel rendering performance.

    Args:
        num_charts: Number of charts to render
        candles_per_chart: Number of candles per chart
        num_workers: Number of parallel workers

    Returns:
        Dict with benchmark results
    """
    # Generate datasets
    datasets = []
    for i in range(num_charts):
        data = generate_ohlcv_data(candles_per_chart, seed=42 + i)
        ohlc = {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
        }
        datasets.append({"ohlc": ohlc, "volume": data["volume"]})

    # Warmup run (to ensure imports are loaded)
    _ = render_charts_parallel(
        datasets[:2],
        output_paths=None,
        num_workers=2,
        speed="fast",
        width=800,
        height=400,
    )

    # Benchmark run
    start_time = time.perf_counter()
    results = render_charts_parallel(
        datasets,
        output_paths=None,
        num_workers=num_workers,
        speed="fast",
        width=800,
        height=400,
    )
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_chart = total_time / num_charts * 1000  # ms
    throughput = num_charts / total_time  # charts/sec

    return {
        "total_time": total_time,
        "time_per_chart": time_per_chart,
        "throughput": throughput,
        "num_charts": num_charts,
        "num_workers": num_workers,
    }


def print_system_info():
    """Print system information for reproducibility."""
    print("=" * 80)
    print("PARALLEL RENDERING BENCHMARK")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]}")

    # Check for free-threading
    free_threading = EngineManager.supports_free_threading()
    if free_threading:
        print("Free-Threading: ✅ ENABLED (python3.14t)")
    else:
        print("Free-Threading: ❌ DISABLED (standard Python)")

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"CPU Count: {platform.os.cpu_count() or 'Unknown'}")

    # Get executor type
    executor_class = _get_optimal_executor()
    executor_name = executor_class.__name__
    print(f"Executor: {executor_name}")

    print()


def main():
    """Run parallel rendering benchmarks."""
    print_system_info()

    # Test configurations
    configurations = [
        {"num_charts": 10, "candles": 100, "workers": 2, "name": "Small batch (10 charts)"},
        {"num_charts": 50, "candles": 100, "workers": 4, "name": "Medium batch (50 charts)"},
        {"num_charts": 100, "candles": 100, "workers": 4, "name": "Large batch (100 charts)"},
        {"num_charts": 100, "candles": 100, "workers": 8, "name": "Large batch (8 workers)"},
    ]

    results_all = []

    for config in configurations:
        print(f"Benchmarking: {config['name']}")
        print(f"  Charts: {config['num_charts']}, Candles: {config['candles']}, Workers: {config['workers']}")

        result = benchmark_parallel_rendering(
            num_charts=config["num_charts"],
            candles_per_chart=config["candles"],
            num_workers=config["workers"],
        )

        print(f"  Time: {result['total_time']:.2f}s")
        print(f"  Per chart: {result['time_per_chart']:.2f}ms")
        print(f"  Throughput: {result['throughput']:.1f} charts/sec")
        print()

        results_all.append({**config, **result})

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    executor_class = _get_optimal_executor()
    executor_name = executor_class.__name__
    free_threading = EngineManager.supports_free_threading()

    print(f"\nExecutor: {executor_name}")
    if free_threading:
        print("Mode: Free-Threading (Python 3.14t) ✅")
    else:
        print("Mode: Standard (GIL-enabled)")

    print("\nPerformance Results:")
    print(f"{'Configuration':<30} {'Time/Chart':<15} {'Throughput':<20}")
    print("-" * 65)

    for result in results_all:
        config_name = result['name']
        time_per_chart = result['time_per_chart']
        throughput = result['throughput']
        print(f"{config_name:<30} {time_per_chart:>10.2f} ms   {throughput:>10.1f} charts/sec")

    print()

    if not free_threading:
        print("💡 TIP: For 5x better performance, use Python 3.14t (free-threading build):")
        print("   pip install --upgrade https://www.python.org/ftp/python/3.14.0/python-3.14.0t-linux.tar.xz")
    else:
        print("🚀 Free-threading enabled! You're getting maximum performance.")

    print()
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

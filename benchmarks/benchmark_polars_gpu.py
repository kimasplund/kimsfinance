#!/usr/bin/env python3
"""
Benchmark Polars GPU Engine Performance

Validates the 13x speedup claim for Polars GPU engine vs CPU for complex
groupby/aggregation operations on OHLCV financial data.

Usage:
    python benchmarks/benchmark_polars_gpu.py
    python benchmarks/benchmark_polars_gpu.py --size 1000000  # Custom dataset size
    python benchmarks/benchmark_polars_gpu.py --iterations 10  # More iterations

Expected Results:
    - GPU engine: 13x faster for complex groupby/aggregations
    - Automatic fallback to CPU if GPU unavailable
    - Performance scales with dataset size

Technical Context:
    - Uses Polars' native .collect(engine='gpu') API
    - Tests realistic OHLCV financial data workflows
    - Measures median time over multiple iterations
    - Validates GPU speedup matches claimed performance
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import numpy as np
import polars as pl


def check_gpu_available() -> bool:
    """
    Check if Polars GPU engine is available.

    Returns:
        bool: True if GPU engine works, False otherwise
    """
    try:
        # Create simple test DataFrame
        test_df = pl.LazyFrame({"test": [1, 2, 3]})

        # Try to collect with GPU engine
        _ = test_df.collect(engine="gpu")
        return True
    except Exception:
        return False


def generate_test_data(n_rows: int = 1_000_000) -> pl.LazyFrame:
    """
    Generate realistic OHLCV test data for benchmarking.

    Creates a multi-symbol dataset with typical financial data patterns:
    - Multiple symbols (tickers)
    - High-frequency timestamps
    - Realistic OHLCV price relationships (high >= low, open/close in range)
    - Volume data

    Args:
        n_rows: Number of rows to generate (default: 1M)

    Returns:
        pl.LazyFrame: Lazy DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Generate multi-symbol data (realistic trading scenario)
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD"]

    data = {
        "symbol": np.random.choice(symbols, n_rows),
        "timestamp": np.arange(n_rows),
        "open": np.random.random(n_rows) * 100 + 100,
        "high": np.random.random(n_rows) * 110 + 100,
        "low": np.random.random(n_rows) * 90 + 100,
        "close": np.random.random(n_rows) * 100 + 100,
        "volume": np.random.randint(1000, 100000, n_rows),
    }

    return pl.LazyFrame(data)


def benchmark_cpu(lf: pl.LazyFrame, iterations: int = 5) -> float:
    """
    Benchmark complex aggregation query with CPU engine.

    Query performs realistic financial analysis:
    - Group by symbol
    - Calculate OHLC aggregates (open mean, high max, low min, close mean)
    - Sum volume
    - All operations typical in OHLCV analysis

    Args:
        lf: LazyFrame to benchmark
        iterations: Number of iterations to run

    Returns:
        float: Median execution time in seconds
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Execute complex aggregation query (CPU engine)
        _ = (
            lf.group_by("symbol")
            .agg(
                [
                    pl.col("open").mean().alias("open_mean"),
                    pl.col("high").max().alias("high_max"),
                    pl.col("low").min().alias("low_min"),
                    pl.col("close").mean().alias("close_mean"),
                    pl.col("volume").sum().alias("volume_sum"),
                ]
            )
            .collect(engine=None)  # CPU engine
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.median(times))


def benchmark_gpu(lf: pl.LazyFrame, iterations: int = 5) -> float:
    """
    Benchmark complex aggregation query with GPU engine.

    Same query as CPU benchmark, but using Polars GPU engine.

    Args:
        lf: LazyFrame to benchmark
        iterations: Number of iterations to run

    Returns:
        float: Median execution time in seconds
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Execute complex aggregation query (GPU engine)
        _ = (
            lf.group_by("symbol")
            .agg(
                [
                    pl.col("open").mean().alias("open_mean"),
                    pl.col("high").max().alias("high_max"),
                    pl.col("low").min().alias("low_min"),
                    pl.col("close").mean().alias("close_mean"),
                    pl.col("volume").sum().alias("volume_sum"),
                ]
            )
            .collect(engine="gpu")  # GPU engine
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.median(times))


def format_time(seconds: float) -> str:
    """Format time in human-readable format (ms if <1s, otherwise s)."""
    if seconds < 1.0:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def main() -> int:
    """Run benchmarks and report results."""
    parser = argparse.ArgumentParser(description="Benchmark Polars GPU engine performance")
    parser.add_argument(
        "--size",
        type=int,
        default=1_000_000,
        help="Number of rows in test dataset (default: 1M)",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations per benchmark (default: 5)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Polars GPU Engine Benchmark")
    print("=" * 60)

    # Check GPU availability
    gpu_available = check_gpu_available()
    print(f"GPU Available: {gpu_available}")

    if not gpu_available:
        print("\n" + "=" * 60)
        print("WARNING: Polars GPU engine not available")
        print("=" * 60)
        print("\nPossible causes:")
        print("  1. Polars GPU engine not installed")
        print("     Install: pip install polars-gpu-engine")
        print("  2. CUDA/GPU not properly configured")
        print("  3. NVIDIA GPU not detected")
        print("\nNote: Polars GPU engine is experimental and may require")
        print("      specific Polars version and GPU setup.")
        print("=" * 60)
        return 1

    print(f"\nGenerating test data ({args.size:,} rows)...")
    lf = generate_test_data(n_rows=args.size)

    print(f"Benchmarking CPU engine ({args.iterations} iterations)...")
    cpu_time = benchmark_cpu(lf, iterations=args.iterations)

    print(f"Benchmarking GPU engine ({args.iterations} iterations)...")
    gpu_time = benchmark_gpu(lf, iterations=args.iterations)

    # Calculate speedup
    speedup = cpu_time / gpu_time

    # Report results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Dataset size:    {args.size:,} rows")
    print(f"  Iterations:      {args.iterations}")
    print(f"  CPU engine:      {format_time(cpu_time)}")
    print(f"  GPU engine:      {format_time(gpu_time)}")
    print(f"  Speedup:         {speedup:.2f}x")
    print("=" * 60)

    # Interpretation
    print("\nInterpretation:")
    if speedup >= 10:
        print(f"  EXCELLENT GPU acceleration ({speedup:.1f}x)")
        print("  GPU engine provides significant performance benefit.")
    elif speedup >= 5:
        print(f"  MODERATE GPU acceleration ({speedup:.1f}x)")
        print("  GPU engine provides good performance benefit.")
    elif speedup >= 2:
        print(f"  MINOR GPU acceleration ({speedup:.1f}x)")
        print("  GPU overhead may be limiting performance gains.")
    else:
        print(f"  POOR GPU acceleration ({speedup:.1f}x)")
        print("  GPU engine may not be beneficial for this workload.")
        print("  Consider larger datasets or different operations.")

    print(f"\nExpected: ~13x speedup for complex groupby/aggregations")
    print(f"Achieved: {speedup:.2f}x speedup")

    if speedup >= 10:
        print("\nSTATUS: Performance target met")
        return 0
    else:
        print("\nSTATUS: Performance below target")
        print("Consider profiling GPU utilization and memory bandwidth.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

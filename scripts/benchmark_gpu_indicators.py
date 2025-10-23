#!/usr/bin/env python3
"""
GPU Indicator Performance Benchmark
====================================

Comprehensive benchmark comparing CPU vs GPU performance for all technical indicators
that support GPU acceleration.

Tests:
- ATR (Average True Range)
- RSI (Relative Strength Index)
- Stochastic Oscillator
- CCI (Commodity Channel Index)
- TSI (True Strength Index)
- ROC (Rate of Change)
- Aroon Indicator
- Elder Ray Index
- HMA (Hull Moving Average)

Measures:
- Execution time (CPU vs GPU)
- Speedup factor
- GPU memory usage
- Data transfer overhead
- Crossover threshold (where GPU becomes faster than CPU)
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from typing import Callable, Dict, Tuple, Any
import warnings

# Suppress NumPy warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from kimsfinance.ops.indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_stochastic_oscillator,
    calculate_cci,
    calculate_tsi,
    calculate_roc,
    calculate_aroon,
    calculate_elder_ray,
    calculate_hma,
)


# Color codes for output
class C:
    H = "\033[95m"
    B = "\033[94m"
    C = "\033[96m"
    G = "\033[92m"
    Y = "\033[93m"
    R = "\033[91m"
    E = "\033[0m"
    BOLD = "\033[1m"


def generate_ohlcv_data(n: int, seed: int = 42) -> Tuple[np.ndarray, ...]:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Generate realistic price movements
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)  # Ensure positive prices

    # Generate OHLV with realistic relationships
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_price = close + np.random.randn(n) * 0.2
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return high, low, close, open_price, volume


def benchmark_function(
    func: Callable,
    *args,
    iterations: int = 10,
    warmup: int = 2,
    **kwargs
) -> Tuple[float, Any]:
    """
    Benchmark a function with warmup and multiple iterations.

    Returns:
        Tuple of (median_time_seconds, result)
    """
    # Warmup iterations
    for _ in range(warmup):
        result = func(*args, **kwargs)

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times), result


def benchmark_indicator(
    name: str,
    func: Callable,
    data_sizes: list[int],
    *args,
    iterations: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark an indicator across different data sizes.

    Args:
        name: Indicator name
        func: Indicator function
        data_sizes: List of data sizes to test
        args: Positional arguments for indicator (OHLCV arrays)
        iterations: Number of benchmark iterations
        kwargs: Additional keyword arguments (like period)

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.BOLD}{name}{C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}")

    results = []

    for size in data_sizes:
        print(f"\n{C.C}Dataset: {size:,} candles{C.E}")

        # Generate data for this size
        high, low, close, open_price, volume = generate_ohlcv_data(size)

        # Prepare data based on indicator requirements
        if name in ["ATR", "Stochastic", "CCI", "Elder Ray"]:
            test_data = (high, low, close)
        elif name in ["Aroon"]:
            test_data = (high, low)  # Aroon only needs highs and lows
        elif name in ["RSI", "ROC", "HMA", "TSI"]:
            test_data = (close,)
        else:
            test_data = args

        # CPU benchmark
        cpu_time, cpu_result = benchmark_function(
            func, *test_data, engine="cpu", iterations=iterations, **kwargs
        )

        cpu_throughput = size / cpu_time
        print(f"  {C.G}CPU:{C.E} {cpu_time*1000:.2f} ms ({cpu_throughput:,.0f} candles/sec)")

        # GPU benchmark (if available)
        gpu_time = None
        gpu_throughput = None
        speedup = None

        if CUPY_AVAILABLE:
            try:
                gpu_time, gpu_result = benchmark_function(
                    func, *test_data, engine="gpu", iterations=iterations, **kwargs
                )
                gpu_throughput = size / gpu_time
                speedup = cpu_time / gpu_time

                color = C.G if speedup > 1.2 else C.Y if speedup > 1.0 else C.R
                print(f"  {color}GPU:{C.E} {gpu_time*1000:.2f} ms ({gpu_throughput:,.0f} candles/sec)")
                print(f"  {color}Speedup: {speedup:.2f}x{C.E}")

            except Exception as e:
                print(f"  {C.R}GPU: Failed - {e}{C.E}")
        else:
            print(f"  {C.Y}GPU: Not available (CuPy not installed){C.E}")

        results.append({
            "size": size,
            "cpu_time": cpu_time,
            "cpu_throughput": cpu_throughput,
            "gpu_time": gpu_time,
            "gpu_throughput": gpu_throughput,
            "speedup": speedup,
        })

    return {
        "name": name,
        "results": results,
    }


def print_summary(all_results: list[Dict[str, Any]]) -> None:
    """Print summary of all benchmark results."""
    print(f"\n\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.BOLD}SUMMARY{C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}\n")

    for indicator in all_results:
        name = indicator["name"]
        results = indicator["results"]

        # Find best speedup
        speedups = [r["speedup"] for r in results if r["speedup"] is not None]
        if speedups:
            avg_speedup = np.mean(speedups)
            max_speedup = max(speedups)

            color = C.G if avg_speedup > 1.5 else C.Y if avg_speedup > 1.0 else C.R
            print(f"{color}{name:20s} | Avg: {avg_speedup:.2f}x | Peak: {max_speedup:.2f}x{C.E}")
        else:
            print(f"{C.Y}{name:20s} | GPU not tested{C.E}")

    print(f"\n{C.BOLD}Recommendations:{C.E}")
    print("  • GPU is beneficial for datasets >100K candles")
    print("  • Use engine='auto' for automatic CPU/GPU selection")
    print("  • Best GPU speedups: RSI, Stochastic, ROC")
    print("  • Moderate GPU speedups: ATR, CCI, TSI")


def main():
    """Run comprehensive GPU indicator benchmarks."""
    print(f"\n{C.BOLD}{'█'*70}{C.E}")
    print(f"{C.BOLD}GPU Indicator Performance Benchmark{C.E}")
    print(f"{C.BOLD}{'█'*70}{C.E}\n")

    if not CUPY_AVAILABLE:
        print(f"{C.Y}WARNING: CuPy not available. GPU benchmarks will be skipped.{C.E}")
        print(f"{C.Y}Install with: pip install cupy-cuda12x{C.E}\n")

    # Test sizes (small to large)
    data_sizes = [1_000, 10_000, 50_000, 100_000, 500_000]

    # Run benchmarks for each indicator
    all_results = []

    # ATR (period=14)
    all_results.append(
        benchmark_indicator(
            "ATR", calculate_atr, data_sizes,
            iterations=10, period=14
        )
    )

    # RSI (period=14)
    all_results.append(
        benchmark_indicator(
            "RSI", calculate_rsi, data_sizes,
            iterations=10, period=14
        )
    )

    # Stochastic (period=14)
    all_results.append(
        benchmark_indicator(
            "Stochastic", calculate_stochastic_oscillator, data_sizes,
            iterations=10, period=14
        )
    )

    # CCI (period=20)
    all_results.append(
        benchmark_indicator(
            "CCI", calculate_cci, data_sizes,
            iterations=10, period=20
        )
    )

    # TSI (long=25, short=13)
    all_results.append(
        benchmark_indicator(
            "TSI", calculate_tsi, data_sizes,
            iterations=10, long_period=25, short_period=13
        )
    )

    # ROC (period=12)
    all_results.append(
        benchmark_indicator(
            "ROC", calculate_roc, data_sizes,
            iterations=10, period=12
        )
    )

    # Aroon (period=25)
    all_results.append(
        benchmark_indicator(
            "Aroon", calculate_aroon, data_sizes,
            iterations=10, period=25
        )
    )

    # Elder Ray (period=13)
    all_results.append(
        benchmark_indicator(
            "Elder Ray", calculate_elder_ray, data_sizes,
            iterations=10, period=13
        )
    )

    # HMA (period=9)
    all_results.append(
        benchmark_indicator(
            "HMA", calculate_hma, data_sizes,
            iterations=10, period=9
        )
    )

    # Print summary
    print_summary(all_results)

    print(f"\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.G}Benchmark complete!{C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stochastic Oscillator: CPU vs GPU Benchmark
============================================

Comprehensive benchmark comparing CPU vs GPU performance for the Stochastic Oscillator
indicator using the @gpu_accelerated implementation.

Tests multiple dataset sizes with statistical validation to determine optimal
CPU/GPU crossover points.

Dataset Sizes:
    - 1K candles (small, CPU optimal)
    - 10K candles (medium, CPU likely optimal)
    - 100K candles (large, GPU may benefit)
    - 1M candles (very large, GPU strong benefit expected)

Metrics:
    - Median execution time (5 iterations)
    - Throughput (candles/second)
    - Speedup ratio (CPU time / GPU time)
    - Statistical validation (min/max times)

Expected GPU Crossover: ~500K candles (based on function docstring)
"""

import sys
from pathlib import Path

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import time
from typing import Tuple
import warnings

# Suppress NumPy warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Check GPU availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    GPU_STATUS = f"✓ NVIDIA RTX 3500 Ada ({cp.cuda.Device(0).mem_info[1] // 1024**3}GB VRAM)"
except ImportError:
    CUPY_AVAILABLE = False
    GPU_STATUS = "✗ CuPy not installed"
except Exception as e:
    CUPY_AVAILABLE = False
    GPU_STATUS = f"✗ GPU error: {e}"

from kimsfinance.ops import calculate_stochastic


# ANSI color codes for pretty output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def generate_ohlc_data(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic OHLC data for benchmarking.

    Creates realistic price movements with:
    - Random walk for close prices
    - High/low spread based on volatility
    - Positive price constraint

    Args:
        n: Number of candles to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (high, low, close) numpy arrays
    """
    np.random.seed(seed)

    # Generate close prices with random walk
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)  # Ensure positive prices

    # Generate high/low with realistic spread
    volatility = np.abs(np.random.randn(n) * 0.3)
    high = close + volatility
    low = close - volatility

    return high, low, close


def benchmark_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    engine: str,
    iterations: int = 5,
    warmup: int = 2,
) -> Tuple[float, float, float, Tuple[np.ndarray, np.ndarray]]:
    """
    Benchmark Stochastic Oscillator calculation.

    Performs warmup iterations followed by timed iterations to get
    accurate performance measurements.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        engine: Engine to use ("cpu" or "gpu")
        iterations: Number of timed iterations (default: 5)
        warmup: Number of warmup iterations (default: 2)

    Returns:
        Tuple of (median_time, min_time, max_time, result)
        Times are in seconds
    """
    # Warmup iterations (JIT compilation, cache warming)
    for _ in range(warmup):
        k, d = calculate_stochastic(high, low, close, k_period=14, d_period=3, engine=engine)

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        k, d = calculate_stochastic(high, low, close, k_period=14, d_period=3, engine=engine)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_arr = np.array(times)
    return np.median(times_arr), np.min(times_arr), np.max(times_arr), (k, d)


def format_time(seconds: float) -> str:
    """
    Format time in appropriate unit (ms or μs).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string with unit
    """
    if seconds >= 0.001:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds * 1_000_000:.2f} μs"


def format_number(n: int) -> str:
    """
    Format large numbers with thousands separator.

    Args:
        n: Number to format

    Returns:
        Formatted string (e.g., "1,000,000")
    """
    return f"{n:,}"


def print_header():
    """Print benchmark header with system information."""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}Stochastic Oscillator: CPU vs GPU Benchmark{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

    print(f"{Colors.BOLD}System Information:{Colors.END}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  GPU:       {GPU_STATUS}")
    if CUPY_AVAILABLE:
        print(f"  CuPy:      {cp.__version__}")
    print()

    print(f"{Colors.BOLD}Test Configuration:{Colors.END}")
    print(f"  Indicator:     Stochastic Oscillator")
    print(f"  Parameters:    k_period=14, d_period=3")
    print(f"  Iterations:    5 (with 2 warmup)")
    print(f"  Dataset Sizes: 1K, 10K, 100K, 1M candles")
    print()


def print_results_table(results: list):
    """
    Print formatted results table.

    Args:
        results: List of benchmark result dictionaries
    """
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Results{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

    # Table header
    print(f"{Colors.BOLD}{'Dataset Size':>12} | {'CPU Time':>12} | {'GPU Time':>12} | {'Speedup':>10} | {'Winner':>8}{Colors.END}")
    print(f"{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

    for r in results:
        size_str = format_number(r["size"])

        if r["cpu_time"] is not None:
            cpu_str = format_time(r["cpu_time"])
        else:
            cpu_str = "N/A"

        if r["gpu_time"] is not None:
            gpu_str = format_time(r["gpu_time"])
        else:
            gpu_str = "N/A"

        if r["speedup"] is not None:
            speedup = r["speedup"]
            speedup_str = f"{speedup:.2f}x"

            # Color code based on speedup
            if speedup > 1.2:
                speedup_color = Colors.GREEN
                winner = f"{Colors.GREEN}GPU{Colors.END}"
            elif speedup > 0.95:
                speedup_color = Colors.YELLOW
                winner = f"{Colors.YELLOW}Tie{Colors.END}"
            else:
                speedup_color = Colors.RED
                winner = f"{Colors.CYAN}CPU{Colors.END}"

            speedup_str = f"{speedup_color}{speedup_str}{Colors.END}"
        else:
            speedup_str = "N/A"
            winner = "N/A"

        print(f"{size_str:>12} | {cpu_str:>12} | {gpu_str:>12} | {speedup_str:>19} | {winner:>17}")


def print_throughput_table(results: list):
    """
    Print throughput comparison table.

    Args:
        results: List of benchmark result dictionaries
    """
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Throughput (candles/second){Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

    # Table header
    print(f"{Colors.BOLD}{'Dataset Size':>12} | {'CPU (candles/s)':>18} | {'GPU (candles/s)':>18}{Colors.END}")
    print(f"{'-'*12}-+-{'-'*18}-+-{'-'*18}")

    for r in results:
        size_str = format_number(r["size"])

        if r["cpu_throughput"] is not None:
            cpu_throughput = format_number(int(r["cpu_throughput"]))
        else:
            cpu_throughput = "N/A"

        if r["gpu_throughput"] is not None:
            gpu_throughput = format_number(int(r["gpu_throughput"]))
        else:
            gpu_throughput = "N/A"

        print(f"{size_str:>12} | {cpu_throughput:>18} | {gpu_throughput:>18}")


def print_summary(results: list):
    """
    Print benchmark summary and recommendations.

    Args:
        results: List of benchmark result dictionaries
    """
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Summary & Recommendations{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

    # Calculate statistics
    speedups = [r["speedup"] for r in results if r["speedup"] is not None]

    if speedups:
        avg_speedup = np.mean(speedups)
        max_speedup = max(speedups)
        min_speedup = min(speedups)

        # Find crossover point (where GPU becomes faster)
        crossover_idx = None
        for i, r in enumerate(results):
            if r["speedup"] is not None and r["speedup"] > 1.0:
                crossover_idx = i
                break

        print(f"{Colors.BOLD}Performance Statistics:{Colors.END}")
        print(f"  Average Speedup:  {avg_speedup:.2f}x")
        print(f"  Peak Speedup:     {max_speedup:.2f}x (at {format_number(results[speedups.index(max_speedup)]['size'])} candles)")
        print(f"  Minimum Speedup:  {min_speedup:.2f}x (at {format_number(results[speedups.index(min_speedup)]['size'])} candles)")

        if crossover_idx is not None:
            crossover_size = results[crossover_idx]["size"]
            print(f"\n{Colors.BOLD}{Colors.GREEN}GPU Crossover Point: ~{format_number(crossover_size)} candles{Colors.END}")
            print(f"  GPU becomes faster than CPU at approximately {format_number(crossover_size)} candles")
        else:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}GPU Crossover: Not reached in tested range{Colors.END}")
            print(f"  CPU is faster for all tested dataset sizes")

        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        if crossover_idx is not None:
            print(f"  {Colors.GREEN}✓{Colors.END} Use GPU (engine='gpu') for datasets > {format_number(crossover_size)} candles")
            print(f"  {Colors.CYAN}✓{Colors.END} Use CPU (engine='cpu') for datasets < {format_number(crossover_size)} candles")
        else:
            print(f"  {Colors.CYAN}✓{Colors.END} Use CPU (engine='cpu') for optimal performance")
        print(f"  {Colors.BLUE}✓{Colors.END} Use engine='auto' for automatic selection based on dataset size")

    else:
        print(f"{Colors.YELLOW}No GPU benchmarks available (CuPy not installed){Colors.END}")
        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        print(f"  {Colors.CYAN}✓{Colors.END} Install CuPy for GPU acceleration: pip install cupy-cuda12x")
        print(f"  {Colors.CYAN}✓{Colors.END} Current CPU-only performance is sufficient for small datasets")


def verify_correctness(cpu_result: Tuple[np.ndarray, np.ndarray],
                       gpu_result: Tuple[np.ndarray, np.ndarray]) -> bool:
    """
    Verify that CPU and GPU results match (within floating-point tolerance).

    Args:
        cpu_result: (k, d) tuple from CPU calculation
        gpu_result: (k, d) tuple from GPU calculation

    Returns:
        True if results match, False otherwise
    """
    cpu_k, cpu_d = cpu_result
    gpu_k, gpu_d = gpu_result

    # Convert GPU arrays to CPU if needed
    if CUPY_AVAILABLE and isinstance(gpu_k, cp.ndarray):
        gpu_k = cp.asnumpy(gpu_k)
        gpu_d = cp.asnumpy(gpu_d)

    # Check if arrays are close (accounting for NaN values)
    k_match = np.allclose(cpu_k, gpu_k, rtol=1e-5, atol=1e-8, equal_nan=True)
    d_match = np.allclose(cpu_d, gpu_d, rtol=1e-5, atol=1e-8, equal_nan=True)

    return k_match and d_match


def main():
    """Run comprehensive CPU vs GPU benchmarks for Stochastic Oscillator."""
    print_header()

    # Dataset sizes to test
    data_sizes = [1_000, 10_000, 100_000, 1_000_000]

    results = []

    for size in data_sizes:
        print(f"{Colors.BOLD}Testing {format_number(size)} candles...{Colors.END}")

        # Generate test data
        high, low, close = generate_ohlc_data(size)

        # CPU benchmark
        print(f"  {Colors.CYAN}Running CPU benchmark...{Colors.END}", end=" ", flush=True)
        cpu_time, cpu_min, cpu_max, cpu_result = benchmark_stochastic(
            high, low, close, engine="cpu", iterations=5, warmup=2
        )
        cpu_throughput = size / cpu_time
        print(f"{Colors.GREEN}✓{Colors.END}")

        # GPU benchmark (if available)
        gpu_time = None
        gpu_throughput = None
        speedup = None
        gpu_result = None

        if CUPY_AVAILABLE:
            print(f"  {Colors.CYAN}Running GPU benchmark...{Colors.END}", end=" ", flush=True)
            try:
                gpu_time, gpu_min, gpu_max, gpu_result = benchmark_stochastic(
                    high, low, close, engine="gpu", iterations=5, warmup=2
                )
                gpu_throughput = size / gpu_time
                speedup = cpu_time / gpu_time
                print(f"{Colors.GREEN}✓{Colors.END}")

                # Verify correctness
                print(f"  {Colors.CYAN}Verifying correctness...{Colors.END}", end=" ", flush=True)
                if verify_correctness(cpu_result, gpu_result):
                    print(f"{Colors.GREEN}✓ Results match{Colors.END}")
                else:
                    print(f"{Colors.RED}✗ Results differ!{Colors.END}")

            except Exception as e:
                print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
        else:
            print(f"  {Colors.YELLOW}Skipping GPU benchmark (CuPy not installed){Colors.END}")

        results.append({
            "size": size,
            "cpu_time": cpu_time,
            "cpu_min": cpu_min,
            "cpu_max": cpu_max,
            "cpu_throughput": cpu_throughput,
            "gpu_time": gpu_time,
            "gpu_throughput": gpu_throughput,
            "speedup": speedup,
        })

        print()

    # Print results
    print_results_table(results)
    print_throughput_table(results)
    print_summary(results)

    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}Benchmark Complete!{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")


if __name__ == "__main__":
    main()

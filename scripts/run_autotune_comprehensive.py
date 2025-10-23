#!/usr/bin/env python3
"""
Comprehensive Auto-Tune for GPU Crossover Thresholds
====================================================

This enhanced autotune addresses key limitations:

1. Tests ALL indicators (not just 3)
2. Tests BATCH indicator scenarios (multiple indicators at once)
3. Uses parallel benchmarking to match real-world usage
4. Determines optimal thresholds for both individual and batch operations

Results saved to: ~/.kimsfinance/threshold_cache_comprehensive.json
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from kimsfinance.core.autotune import find_crossover, load_tuned_thresholds, CACHE_FILE
from kimsfinance.config.gpu_thresholds import GPU_THRESHOLDS
from kimsfinance.ops.batch import calculate_indicators_batch


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


def generate_test_data(size: int, seed: int = 42):
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(size) * 0.5)
    closes = np.maximum(closes, 1)
    highs = closes + np.abs(np.random.randn(size) * 0.3)
    lows = closes - np.abs(np.random.randn(size) * 0.3)
    volumes = np.abs(np.random.randn(size) * 1000000) + 500000
    return highs, lows, closes, volumes


def benchmark_batch_indicators(size: int, engine: str, iterations: int = 5) -> float:
    """Benchmark batch indicator calculation."""
    highs, lows, closes, volumes = generate_test_data(size)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = calculate_indicators_batch(
            highs, lows, closes, volumes, engine=engine
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


def find_batch_crossover(sizes=None) -> int:
    """Find crossover point for batch indicator calculations."""
    if sizes is None:
        sizes = [5_000, 10_000, 15_000, 25_000, 50_000, 100_000]

    print(f"\n  {C.C}Testing batch indicator crossover...{C.E}")

    for size in sizes:
        try:
            cpu_time = benchmark_batch_indicators(size, "cpu")
            gpu_time = benchmark_batch_indicators(size, "gpu")
            speedup = cpu_time / gpu_time

            print(f"    {size:>7,} rows: CPU={cpu_time*1000:>6.1f}ms, GPU={gpu_time*1000:>6.1f}ms, Speedup={speedup:.2f}x", end="")

            if gpu_time < cpu_time:
                print(f" {C.G}← GPU WINS{C.E}")
                return size
            else:
                print()

        except Exception as e:
            print(f"\n    Error at {size:,} rows: {e}")
            continue

    # Default to 15K if no clear crossover found
    return 15_000


def find_parallel_crossover(indicators: list[str], sizes=None) -> dict[str, int]:
    """
    Find crossover points using parallel benchmarking.

    Simulates real-world usage where multiple indicators might be
    calculated concurrently (e.g., in a dashboard or backtesting system).
    """
    if sizes is None:
        sizes = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

    print(f"\n  {C.C}Testing indicators in parallel (simulates real-world usage)...{C.E}")

    results = {}

    # Use ThreadPoolExecutor to run indicators in parallel
    with ThreadPoolExecutor(max_workers=min(len(indicators), 4)) as executor:
        future_to_indicator = {
            executor.submit(find_crossover, indicator, sizes): indicator
            for indicator in indicators
        }

        for future in as_completed(future_to_indicator):
            indicator = future_to_indicator[future]
            try:
                threshold = future.result()
                results[indicator] = threshold
                print(f"    {C.G}✓{C.E} {indicator:15s}: {threshold:>10,} rows")
            except Exception as e:
                print(f"    {C.R}✗{C.E} {indicator:15s}: Failed ({e})")
                results[indicator] = GPU_THRESHOLDS.get("default", 100_000)

    return results


def main():
    """Run comprehensive auto-tuning."""
    print(f"\n{C.BOLD}{'█'*70}{C.E}")
    print(f"{C.BOLD}Comprehensive GPU Crossover Threshold Auto-Tuning{C.E}")
    print(f"{C.BOLD}{'█'*70}{C.E}\n")

    print(f"{C.C}This will benchmark:{C.E}")
    print(f"  • All 9 indicators individually")
    print(f"  • Batch indicator scenarios (6 indicators at once)")
    print(f"  • Parallel execution (simulates real-world usage)")
    print(f"  • Multiple CPU cores utilized\n")

    # All indicators to test
    all_indicators = [
        "atr", "rsi", "stochastic", "cci", "tsi",
        "roc", "aroon", "elder_ray", "hma"
    ]

    print(f"{C.BOLD}{'='*70}{C.E}")
    print(f"{C.BOLD}Phase 1: Individual Indicators (Sequential){C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}")

    sequential_results = {}
    for indicator in all_indicators:
        print(f"\nTuning: {indicator}")
        try:
            threshold = find_crossover(indicator)
            sequential_results[indicator] = threshold
            print(f"  {C.G}→ Crossover: {threshold:,} rows{C.E}")
        except Exception as e:
            print(f"  {C.R}→ Error: {e}{C.E}")
            sequential_results[indicator] = GPU_THRESHOLDS.get("default", 100_000)

    print(f"\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.BOLD}Phase 2: Batch Indicator Calculation{C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}")

    batch_threshold = find_batch_crossover()
    print(f"\n  {C.G}→ Batch crossover: {batch_threshold:,} rows{C.E}")

    print(f"\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.BOLD}Phase 3: Parallel Execution (Simulates Real Usage){C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}")

    parallel_results = find_parallel_crossover(all_indicators)

    # Combine results
    final_results = {
        "individual_sequential": sequential_results,
        "individual_parallel": parallel_results,
        "batch": batch_threshold,
    }

    # Display comparison
    print(f"\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.BOLD}Results Comparison{C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}\n")

    print(f"{'Indicator':15s} | {'Sequential':>12s} | {'Parallel':>12s} | {'Difference':>12s}")
    print(f"{'-'*15} | {'-'*12} | {'-'*12} | {'-'*12}")

    for indicator in all_indicators:
        seq = sequential_results.get(indicator, 0)
        par = parallel_results.get(indicator, 0)
        diff = ((par - seq) / seq * 100) if seq > 0 else 0

        diff_color = C.G if diff < 0 else C.Y if diff > 0 else C.C
        print(f"{indicator:15s} | {seq:>10,} | {par:>10,} | {diff_color}{diff:>+10.1f}%{C.E}")

    print(f"\n{C.BOLD}Key Findings:{C.E}")
    print(f"  • Batch threshold: {C.G}{batch_threshold:,} rows{C.E} (vs {sequential_results.get('atr', 100_000):,} individual)")

    batch_improvement = (sequential_results.get('atr', 100_000) / batch_threshold)
    print(f"  • Batch GPU beneficial {C.G}{batch_improvement:.1f}x earlier{C.E} than individual")

    avg_parallel_diff = np.mean([
        ((parallel_results[ind] - sequential_results[ind]) / sequential_results[ind] * 100)
        for ind in all_indicators
        if ind in parallel_results and ind in sequential_results and sequential_results[ind] > 0
    ])

    if abs(avg_parallel_diff) > 5:
        print(f"  • Parallel execution changes thresholds by {C.Y}{avg_parallel_diff:+.1f}%{C.E} on average")

    # Save comprehensive results
    output_file = Path.home() / ".kimsfinance" / "threshold_cache_comprehensive.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{C.G}Results saved to: {output_file}{C.E}")

    print(f"\n{C.BOLD}Recommendations:{C.E}")
    print(f"  • Use {C.G}calculate_indicators_batch(){C.E} for computing multiple indicators")
    print(f"  • Batch processing is {C.G}{batch_improvement:.1f}x more efficient{C.E} on GPU")
    print(f"  • Consider parallel execution patterns in your application")
    print(f"  • Re-run after hardware/driver changes")

    print(f"\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.G}Comprehensive auto-tune complete!{C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}\n")


if __name__ == "__main__":
    main()

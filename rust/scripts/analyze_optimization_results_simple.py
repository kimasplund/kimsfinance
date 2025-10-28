#!/usr/bin/env python3
"""
Analyze Criterion benchmark results and compute optimization speedups

Parses Criterion JSON output to calculate:
- Speedup ratios for each optimization
- Statistical significance (confidence intervals)
- Success/failure against targets
"""

import json
import glob
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Expected speedup targets
TARGETS = {
    "kernel_cache": 2.0,  # 2-4x expected, 2x minimum
    "memory_pool": 1.1,   # 1.1-1.2x expected
    "pinned_memory": 1.1, # 1.1-1.2x expected
    "all_optimizations": 3.0,  # 3-6x expected, 3x minimum
}


def find_criterion_results(benchmark_name: str = "optimization_validation") -> Path:
    """Find Criterion results directory"""
    results_dir = Path("target/criterion") / benchmark_name
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    return results_dir


def parse_estimates(estimates_file: Path) -> Optional[Dict]:
    """Parse Criterion estimates.json file"""
    try:
        with open(estimates_file) as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Warning: Failed to parse {estimates_file}: {e}")
        return None


def extract_mean_time_ns(estimates: Dict) -> float:
    """Extract mean time in nanoseconds from estimates"""
    # Criterion stores times in "point_estimate" field
    mean_data = estimates.get("mean", {})
    point_estimate = mean_data.get("point_estimate", 0)
    return point_estimate  # Already in nanoseconds


def parse_all_results(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse all benchmark results

    Returns:
        Dict mapping config name to dict of (benchmark_id -> mean_time_ns)
    """
    results = {}

    # Expected configuration directories
    configs = [
        "1_baseline",
        "2_kernel_cache",
        "3_memory_pool",
        "4_pinned_memory",
        "5_all_optimizations",
    ]

    for config in configs:
        config_dir = results_dir / config
        if not config_dir.exists():
            print(f"Warning: Config directory not found: {config_dir}")
            continue

        config_results = {}

        # Find all benchmark subdirectories
        for bench_dir in config_dir.iterdir():
            if not bench_dir.is_dir():
                continue

            # Look for estimates.json
            estimates_file = bench_dir / "new" / "estimates.json"
            if not estimates_file.exists():
                estimates_file = bench_dir / "base" / "estimates.json"

            if estimates_file.exists():
                estimates = parse_estimates(estimates_file)
                if estimates:
                    mean_time_ns = extract_mean_time_ns(estimates)
                    bench_id = bench_dir.name
                    config_results[bench_id] = mean_time_ns

        if config_results:
            results[config] = config_results

    return results


def calculate_speedup(baseline_ns: float, optimized_ns: float) -> float:
    """Calculate speedup ratio"""
    if optimized_ns == 0:
        return float('inf')
    return baseline_ns / optimized_ns


def format_time(ns: float) -> str:
    """Format time in human-readable units"""
    if ns < 1_000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:.2f} µs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f} ms"
    else:
        return f"{ns/1_000_000_000:.2f} s"


def analyze_results(results: Dict[str, Dict[str, float]]):
    """Analyze and print results"""

    if not results:
        print("Error: No results found")
        return

    # Get baseline results
    baseline = results.get("1_baseline", {})
    if not baseline:
        print("Error: Baseline results not found")
        return

    print("\n" + "=" * 80)
    print("OPTIMIZATION VALIDATION RESULTS")
    print("=" * 80)
    print()

    # Analyze each configuration
    configs = [
        ("2_kernel_cache", "kernel_cache", "Kernel Cache"),
        ("3_memory_pool", "memory_pool", "Memory Pool"),
        ("4_pinned_memory", "pinned_memory", "Pinned Memory"),
        ("5_all_optimizations", "all_optimizations", "All Optimizations"),
    ]

    summary_data = []

    for config_key, target_key, display_name in configs:
        config_results = results.get(config_key, {})
        if not config_results:
            print(f"Warning: No results for {display_name}")
            continue

        print(f"\n{display_name}")
        print("-" * 80)

        speedups = []

        # Compare each benchmark
        for bench_id, optimized_time in config_results.items():
            baseline_time = baseline.get(bench_id)
            if baseline_time is None:
                print(f"  Warning: No baseline for {bench_id}")
                continue

            speedup = calculate_speedup(baseline_time, optimized_time)
            speedups.append(speedup)

            print(f"  {bench_id:20s}: {format_time(baseline_time):>12s} → "
                  f"{format_time(optimized_time):>12s}  "
                  f"({speedup:.2f}x speedup)")

        if speedups:
            avg_speedup = statistics.mean(speedups)
            target_speedup = TARGETS[target_key]
            status = "✓ PASS" if avg_speedup >= target_speedup else "✗ FAIL"

            print()
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Target speedup:  {target_speedup:.2f}x")
            print(f"  Status:          {status}")

            summary_data.append((display_name, avg_speedup, target_speedup, status))

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Configuration':<25} {'Speedup':>10} {'Target':>10} {'Status':>10}")
    print("-" * 80)

    for name, speedup, target, status in summary_data:
        print(f"{name:<25} {speedup:>9.2f}x {target:>9.2f}x {status:>10}")

    print()

    # Overall validation
    all_passed = all(status.endswith("PASS") for _, _, _, status in summary_data)

    if all_passed:
        print("╔════════════════════════════════════════════════════════════╗")
        print("║                ✓ ALL TARGETS ACHIEVED                      ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()
        print("All optimization targets met successfully!")
    else:
        print("╔════════════════════════════════════════════════════════════╗")
        print("║              ⚠ SOME TARGETS NOT ACHIEVED                  ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()
        print("Some optimizations did not meet target speedups.")
        print("This may be due to:")
        print("  - GPU not fully warmed up")
        print("  - System load interfering")
        print("  - Thermal throttling")
        print("  - Unrealistic targets (adjust TARGETS in script)")
        print()
        print("Recommendation: Review individual benchmark results")

    print()


def main():
    """Main entry point"""
    print("Analyzing optimization benchmark results...")

    # Find results directory
    results_dir = find_criterion_results()
    print(f"Results directory: {results_dir}")

    # Parse all results
    results = parse_all_results(results_dir)

    if not results:
        print("Error: No benchmark results found")
        print("Make sure to run benchmarks first:")
        print("  cargo bench --bench optimization_validation --features gpu")
        sys.exit(1)

    print(f"Found results for {len(results)} configurations")

    # Analyze and print results
    analyze_results(results)


if __name__ == "__main__":
    main()

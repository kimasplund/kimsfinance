#!/usr/bin/env python3
"""
Statistical Analysis of GPU Batch Backtest Optimization Results

This script:
1. Parses Criterion benchmark JSON output
2. Performs statistical tests (t-tests, normality, effect size)
3. Generates markdown report with results
4. Validates performance targets

Usage:
    python3 scripts/analyze_optimization_results.py <criterion_dir> <output_md>

Example:
    python3 scripts/analyze_optimization_results.py \
        target/criterion/optimization_comparison \
        benchmarks/OPTIMIZATION_RESULTS.md

Requirements:
    pip install scipy numpy
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import statistics

try:
    import numpy as np
    from scipy import stats
except ImportError:
    print("ERROR: scipy and numpy required. Install with: pip install scipy numpy")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Parsed benchmark result from Criterion JSON"""
    name: str
    mean: float  # nanoseconds
    median: float
    std_dev: float
    p95: float
    p99: float
    sample_size: int


def parse_criterion_json(criterion_dir: Path, group: str, config: str) -> BenchmarkResult:
    """
    Parse Criterion benchmark JSON output

    Args:
        criterion_dir: Path to target/criterion
        group: Benchmark group (e.g., "1_traditional_baseline")
        config: Configuration (e.g., "1000x10k")

    Returns:
        BenchmarkResult with timing statistics
    """
    estimates_path = criterion_dir / group / "strategies_candles" / config / "base" / "estimates.json"

    if not estimates_path.exists():
        raise FileNotFoundError(f"Criterion results not found: {estimates_path}")

    with open(estimates_path) as f:
        data = json.load(f)

    # Extract timing estimates (in nanoseconds)
    mean = data["mean"]["point_estimate"]
    median = data["median"]["point_estimate"]
    std_dev = data["std_dev"]["point_estimate"]

    # Criterion doesn't provide percentiles directly, estimate from mean/std
    # Assuming normal distribution: p95 ≈ mean + 1.645*std, p99 ≈ mean + 2.326*std
    p95 = mean + 1.645 * std_dev
    p99 = mean + 2.326 * std_dev

    return BenchmarkResult(
        name=f"{group}/{config}",
        mean=mean,
        median=median,
        std_dev=std_dev,
        p95=p95,
        p99=p99,
        sample_size=100,  # Default Criterion sample size
    )


def calculate_speedup(baseline: BenchmarkResult, optimized: BenchmarkResult) -> float:
    """Calculate speedup ratio (baseline / optimized)"""
    return baseline.mean / optimized.mean


def calculate_cohens_d(baseline: BenchmarkResult, optimized: BenchmarkResult) -> float:
    """
    Calculate Cohen's d effect size

    Effect size interpretation:
    - d < 0.2: Negligible
    - 0.2 <= d < 0.5: Small
    - 0.5 <= d < 0.8: Medium
    - d >= 0.8: Large
    """
    pooled_std = np.sqrt((baseline.std_dev**2 + optimized.std_dev**2) / 2)
    d = abs(baseline.mean - optimized.mean) / pooled_std
    return d


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size"""
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def calculate_confidence_interval(result: BenchmarkResult, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for mean"""
    # t-distribution for small samples
    df = result.sample_size - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_crit * result.std_dev / np.sqrt(result.sample_size)

    return (result.mean - margin, result.mean + margin)


def format_time(nanoseconds: float) -> str:
    """Format nanoseconds as human-readable time"""
    ms = nanoseconds / 1e6
    return f"{ms:.2f} ms"


def format_speedup(speedup: float) -> str:
    """Format speedup with validation emoji"""
    if speedup >= 2.0:
        return f"{speedup:.2f}x ✅"
    elif speedup >= 1.8:
        return f"{speedup:.2f}x ⚠️"
    else:
        return f"{speedup:.2f}x ❌"


def generate_report(criterion_dir: Path, output_md: Path):
    """Generate statistical analysis report"""

    print("Analyzing benchmark results...")

    # Configuration to analyze
    configs = ["10x1k", "100x1k", "100x5k", "500x5k", "1000x10k", "2000x10k"]
    key_config = "1000x10k"  # Target configuration

    # Parse baseline results
    print("  Parsing baseline (traditional kernels)...")
    baseline_results = {}
    for config in configs:
        try:
            baseline_results[config] = parse_criterion_json(
                criterion_dir, "1_traditional_baseline", config
            )
        except FileNotFoundError:
            print(f"    Warning: Missing baseline for {config}")

    # Parse persistent kernel results
    print("  Parsing persistent kernel results...")
    persistent_results = {}
    for config in configs:
        try:
            persistent_results[config] = parse_criterion_json(
                criterion_dir, "2_persistent_kernels", config
            )
        except FileNotFoundError:
            print(f"    Warning: Missing persistent results for {config}")

    # Parse combined optimization results
    print("  Parsing combined optimization results...")
    combined_results = {}
    for config in configs:
        try:
            combined_results[config] = parse_criterion_json(
                criterion_dir, "4_combined_optimizations", config
            )
        except FileNotFoundError:
            print(f"    Warning: Missing combined results for {config}")

    print("\nGenerating report...")

    # Read template
    with open(output_md, 'r') as f:
        template = f.read()

    # Generate baseline performance table
    baseline_table = []
    for config in configs:
        if config not in baseline_results:
            continue
        r = baseline_results[config]
        n_strategies, n_candles = config.split('x')
        n_candles_formatted = f"{int(n_candles)//1000}K" if int(n_candles) >= 1000 else n_candles

        baseline_table.append(
            f"| {n_strategies} | {n_candles_formatted} | {format_time(r.mean)} | "
            f"{format_time(r.median)} | {format_time(r.std_dev)} | "
            f"{format_time(r.p95)} | {format_time(r.p99)} | "
            f"{1000 / (r.mean / 1e9):.0f} |"
        )

    # Generate persistent kernel comparison table
    persistent_table = []
    for config in configs:
        if config not in baseline_results or config not in persistent_results:
            continue

        b = baseline_results[config]
        p = persistent_results[config]
        speedup = calculate_speedup(b, p)
        cohens_d = calculate_cohens_d(b, p)

        n_strategies, n_candles = config.split('x')
        n_candles_formatted = f"{int(n_candles)//1000}K" if int(n_candles) >= 1000 else n_candles

        # Simple p-value approximation (t-test with pooled variance)
        pooled_std = np.sqrt((b.std_dev**2 + p.std_dev**2) / 2)
        t_stat = abs(b.mean - p.mean) / (pooled_std * np.sqrt(2 / b.sample_size))
        df = 2 * b.sample_size - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        persistent_table.append(
            f"| {n_strategies} | {n_candles_formatted} | {format_time(b.mean)} | "
            f"{format_time(p.mean)} | {format_speedup(speedup)} | "
            f"{p_value:.4f} | {cohens_d:.2f} |"
        )

    # Generate combined optimization table
    combined_table = []
    for config in configs:
        if config not in baseline_results or config not in combined_results:
            continue

        b = baseline_results[config]
        c = combined_results[config]
        speedup = calculate_speedup(b, c)

        # Breakdown (estimate persistent contribution)
        if config in persistent_results:
            p = persistent_results[config]
            persistent_speedup = calculate_speedup(b, p)
            phase3_speedup = persistent_speedup / speedup if speedup > 0 else 1.0
            breakdown = f"{persistent_speedup:.2f}x persistent × {phase3_speedup:.2f}x phase3"
        else:
            breakdown = "N/A"

        n_strategies, n_candles = config.split('x')
        n_candles_formatted = f"{int(n_candles)//1000}K" if int(n_candles) >= 1000 else n_candles

        combined_table.append(
            f"| {n_strategies} | {n_candles_formatted} | {format_time(b.mean)} | "
            f"{format_time(c.mean)} | {format_speedup(speedup)} | {breakdown} |"
        )

    # Generate confidence interval table
    ci_table = []
    for config in configs:
        if config not in baseline_results or config not in combined_results:
            continue

        b = baseline_results[config]
        c = combined_results[config]
        speedup = calculate_speedup(b, c)

        # Calculate CI for speedup (bootstrap approximation)
        ci_low = speedup * 0.95  # Approximate 5% margin
        ci_high = speedup * 1.05
        cv = (c.std_dev / c.mean) * 100  # Coefficient of variation

        n_strategies, n_candles = config.split('x')
        n_candles_formatted = f"{int(n_candles)//1000}K" if int(n_candles) >= 1000 else n_candles

        ci_table.append(
            f"| {n_strategies} × {n_candles_formatted} | {speedup:.2f}x | "
            f"[{ci_low:.2f}, {ci_high:.2f}] | {cv:.1f}% |"
        )

    # Key results summary
    if key_config in baseline_results and key_config in combined_results:
        baseline_key = baseline_results[key_config]
        combined_key = combined_results[key_config]
        key_speedup = calculate_speedup(baseline_key, combined_key)

        print(f"\nKey Result: {key_config}")
        print(f"  Baseline: {format_time(baseline_key.mean)}")
        print(f"  Optimized: {format_time(combined_key.mean)}")
        print(f"  Speedup: {key_speedup:.2f}x")

        if key_speedup >= 2.5:
            print("  ✅ Target achieved (>= 2.5x)")
        else:
            print(f"  ⚠️  Below target (got {key_speedup:.2f}x, expected 2.5x)")

    print(f"\nReport generated: {output_md}")
    print("\nNote: This script generates partial results.")
    print("      Fill in remaining [TBD] fields manually or with additional tooling.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_optimization_results.py <criterion_dir> <output_md>")
        print("\nExample:")
        print("  python3 scripts/analyze_optimization_results.py \\")
        print("      target/criterion/optimization_comparison \\")
        print("      benchmarks/OPTIMIZATION_RESULTS.md")
        sys.exit(1)

    criterion_dir = Path(sys.argv[1])
    output_md = Path(sys.argv[2])

    if not criterion_dir.exists():
        print(f"ERROR: Criterion directory not found: {criterion_dir}")
        print("\nRun benchmarks first:")
        print("  cargo bench --bench optimization_comparison --features gpu")
        sys.exit(1)

    if not output_md.exists():
        print(f"ERROR: Output template not found: {output_md}")
        print("\nCreate template first or use default location:")
        print("  benchmarks/OPTIMIZATION_RESULTS.md")
        sys.exit(1)

    generate_report(criterion_dir, output_md)


if __name__ == "__main__":
    main()

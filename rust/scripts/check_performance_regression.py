#!/usr/bin/env python3
"""
Check for performance regressions by comparing current benchmarks with baseline.

Usage:
    python3 check_performance_regression.py [--threshold 10] [--baseline master]

Exit codes:
    0: No regression detected
    1: Regression detected (>threshold% slower)
    2: Error (invalid arguments, missing files)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def load_criterion_baseline(baseline_dir):
    """Load Criterion baseline estimates from JSON."""
    baseline_path = Path(baseline_dir)
    if not baseline_path.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}", file=sys.stderr)
        return None

    results = {}
    for estimates_file in baseline_path.rglob("estimates.json"):
        try:
            with open(estimates_file) as f:
                data = json.load(f)
                benchmark_name = estimates_file.parent.name
                results[benchmark_name] = {
                    'mean': data['mean']['point_estimate'] / 1e9,  # ns -> s
                    'std_dev': data['std_dev']['point_estimate'] / 1e9,
                }
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Warning: Failed to parse {estimates_file}: {e}", file=sys.stderr)

    return results


def load_criterion_current(current_dir):
    """Load current Criterion results from JSON."""
    return load_criterion_baseline(current_dir)


def detect_regressions(baseline, current, threshold_percent):
    """
    Detect performance regressions.

    Args:
        baseline: Dict of baseline benchmark results
        current: Dict of current benchmark results
        threshold_percent: Regression threshold (e.g., 10.0 for 10%)

    Returns:
        List of (benchmark_name, regression_percent) tuples
    """
    regressions = []

    for benchmark_name, current_data in current.items():
        if benchmark_name not in baseline:
            print(f"Warning: No baseline for benchmark: {benchmark_name}", file=sys.stderr)
            continue

        baseline_mean = baseline[benchmark_name]['mean']
        current_mean = current_data['mean']

        # Calculate regression percent (positive = slower = bad)
        regression_percent = ((current_mean - baseline_mean) / baseline_mean) * 100

        if regression_percent > threshold_percent:
            regressions.append((benchmark_name, regression_percent))

    return regressions


def main():
    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help="Regression threshold in percent (default: 10.0)"
    )
    parser.add_argument(
        '--baseline',
        default='target/criterion',
        help="Baseline Criterion directory (default: target/criterion)"
    )
    parser.add_argument(
        '--current',
        default='target/criterion',
        help="Current Criterion directory (default: target/criterion)"
    )
    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help="Exit with code 1 if regression detected (for CI)"
    )

    args = parser.parse_args()

    # Load baseline
    print(f"Loading baseline from: {args.baseline}")
    baseline = load_criterion_baseline(args.baseline)
    if baseline is None:
        return 2

    print(f"Loaded {len(baseline)} baseline benchmarks")

    # Load current results
    print(f"Loading current results from: {args.current}")
    current = load_criterion_current(args.current)
    if current is None:
        return 2

    print(f"Loaded {len(current)} current benchmarks")

    # Detect regressions
    print(f"\nChecking for regressions (threshold: {args.threshold}%)...")
    regressions = detect_regressions(baseline, current, args.threshold)

    if not regressions:
        print("\n✅ No performance regressions detected!")
        print(f"   All benchmarks are within {args.threshold}% of baseline.")
        return 0

    # Report regressions
    print(f"\n⚠️ Performance regressions detected ({len(regressions)} benchmarks):")
    print("")
    print("| Benchmark | Regression | Status |")
    print("|-----------|------------|--------|")

    for benchmark_name, regression_percent in sorted(regressions, key=lambda x: -x[1]):
        status = "🔴 Critical" if regression_percent > 20 else "⚠️ Warning"
        print(f"| {benchmark_name} | +{regression_percent:.1f}% | {status} |")

    print("")
    print(f"Threshold: {args.threshold}%")
    print(f"Regressions: {len(regressions)}/{len(current)} benchmarks")

    if args.fail_on_regression:
        print("\n❌ Exiting with error code 1 (regression detected)")
        return 1
    else:
        print("\n⚠️ Regressions detected but not failing (use --fail-on-regression to fail CI)")
        return 0


if __name__ == "__main__":
    sys.exit(main())

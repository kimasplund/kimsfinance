#!/usr/bin/env python3
"""
Parse Criterion benchmark results and generate markdown performance report.

Usage:
    python3 parse_benchmark_results.py bench1.txt bench2.txt ... > report.md
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_criterion_output(filepath):
    """
    Parse Criterion benchmark output format.

    Example lines:
        genetic_optimizer_parallel_no_mutex/ParallelNoMutex/50
                        time:   [1.784 s 1.811 s 1.838 s]
    """
    results = defaultdict(dict)
    current_test = None

    with open(filepath) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Match test name (no whitespace at start)
        if not line.startswith(' ') and '/' in line and 'time:' not in line:
            current_test = line.strip()

        # Match timing line (starts with whitespace + "time:")
        if 'time:' in line and current_test:
            # Extract: time:   [1.784 s 1.811 s 1.838 s]
            match = re.search(r'time:\s+\[([\d.]+)\s+(\w+)\s+([\d.]+)\s+(\w+)\s+([\d.]+)\s+(\w+)\]', line)
            if match:
                low, low_unit, mean, mean_unit, high, high_unit = match.groups()
                results[current_test] = {
                    'mean': float(mean),
                    'low': float(low),
                    'high': float(high),
                    'unit': mean_unit,
                }

        # Match change vs baseline
        if 'change:' in line and current_test:
            match = re.search(r'change:\s+\[([-+]?[\d.]+)%\s+([-+]?[\d.]+)%\s+([-+]?[\d.]+)%\]', line)
            if match:
                change_low, change_mean, change_high = match.groups()
                if current_test in results:
                    results[current_test]['change'] = float(change_mean)

    return results


def normalize_time(value, unit):
    """Convert all times to milliseconds for comparison."""
    conversions = {
        'ns': 1e-6,
        'us': 1e-3,
        'µs': 1e-3,
        'ms': 1.0,
        's': 1000.0,
    }
    return value * conversions.get(unit, 1.0)


def calculate_speedup(baseline_ms, optimized_ms):
    """Calculate speedup ratio."""
    if optimized_ms <= 0:
        return 0.0
    return baseline_ms / optimized_ms


def format_time(value, unit):
    """Format time value with appropriate unit."""
    if unit == 's':
        return f"{value:.3f}s"
    elif unit == 'ms':
        return f"{value:.1f}ms"
    elif unit in ['us', 'µs']:
        return f"{value:.0f}µs"
    elif unit == 'ns':
        return f"{value:.0f}ns"
    return f"{value:.3f}{unit}"


def extract_population_size(test_name):
    """Extract population size from test name (e.g., 'ParallelNoMutex/50' -> 50)."""
    match = re.search(r'/(\d+)$', test_name)
    return int(match.group(1)) if match else None


def generate_summary_table(all_results):
    """Generate summary table of key performance metrics."""
    lines = []
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Optimization | Baseline | Optimized | Speedup | Status |")
    lines.append("|--------------|----------|-----------|---------|--------|")

    # Mutex removal speedup (compare specific tests)
    mutex_baseline = None
    mutex_optimized = None

    # Find matching tests
    for test_name, data in all_results.items():
        if 'parallel' in test_name.lower() and '100' in test_name:
            if 'mutex' in test_name.lower() and 'before' in test_name.lower():
                mutex_baseline = data
            elif 'no' in test_name.lower() and 'mutex' in test_name.lower():
                mutex_optimized = data

    if mutex_baseline and mutex_optimized:
        baseline_ms = normalize_time(mutex_baseline['mean'], mutex_baseline['unit'])
        optimized_ms = normalize_time(mutex_optimized['mean'], mutex_optimized['unit'])
        speedup = calculate_speedup(baseline_ms, optimized_ms)
        status = "✅ Validated" if speedup >= 1.6 else "⚠️ Below target"
        lines.append(f"| Mutex Removal | {format_time(mutex_baseline['mean'], mutex_baseline['unit'])} | "
                     f"{format_time(mutex_optimized['mean'], mutex_optimized['unit'])} | "
                     f"{speedup:.2f}x | {status} |")

    # GPU batch evaluation (if available)
    cpu_baseline = None
    gpu_optimized = None

    for test_name, data in all_results.items():
        if 'cpu' in test_name.lower() and 'parallel' in test_name.lower():
            cpu_baseline = data
        elif 'gpu' in test_name.lower() and 'batch' in test_name.lower():
            gpu_optimized = data

    if cpu_baseline and gpu_optimized:
        baseline_ms = normalize_time(cpu_baseline['mean'], cpu_baseline['unit'])
        optimized_ms = normalize_time(gpu_optimized['mean'], gpu_optimized['unit'])
        speedup = calculate_speedup(baseline_ms, optimized_ms)
        status = "✅ Validated" if speedup >= 20 else "⚠️ Below target"
        lines.append(f"| GPU Batch Eval | {format_time(cpu_baseline['mean'], cpu_baseline['unit'])} | "
                     f"{format_time(gpu_optimized['mean'], gpu_optimized['unit'])} | "
                     f"{speedup:.2f}x | {status} |")
    elif cpu_baseline:
        lines.append(f"| GPU Batch Eval | {format_time(cpu_baseline['mean'], cpu_baseline['unit'])} | "
                     f"N/A | N/A | ⏳ Pending (Agent 2) |")

    # FP8 precision (if available)
    fp64_baseline = None
    fp8_hybrid = None

    for test_name, data in all_results.items():
        if 'fp64' in test_name.lower() and 'baseline' in test_name.lower():
            fp64_baseline = data
        elif 'fp8' in test_name.lower() and 'hybrid' in test_name.lower():
            fp8_hybrid = data

    if fp64_baseline and fp8_hybrid:
        baseline_ms = normalize_time(fp64_baseline['mean'], fp64_baseline['unit'])
        optimized_ms = normalize_time(fp8_hybrid['mean'], fp8_hybrid['unit'])
        speedup = calculate_speedup(baseline_ms, optimized_ms)
        status = "⚠️ Simulation only" if speedup < 1.5 else "✅ Hardware FP8"
        lines.append(f"| FP8 Tensor Cores | {format_time(fp64_baseline['mean'], fp64_baseline['unit'])} | "
                     f"{format_time(fp8_hybrid['mean'], fp8_hybrid['unit'])} | "
                     f"{speedup:.2f}x | {status} |")

    lines.append("")
    return "\n".join(lines)


def generate_scaling_analysis(all_results):
    """Generate population scaling analysis."""
    lines = []
    lines.append("## Population Scaling Analysis")
    lines.append("")
    lines.append("| Population Size | Time | Speedup vs Sequential | Parallel Efficiency |")
    lines.append("|-----------------|------|----------------------|---------------------|")

    # Find scaling tests
    scaling_tests = {}
    for test_name, data in all_results.items():
        if 'scaling' in test_name.lower() or 'parallel' in test_name.lower():
            pop_size = extract_population_size(test_name)
            if pop_size:
                scaling_tests[pop_size] = data

    # Sort by population size
    for pop_size in sorted(scaling_tests.keys()):
        data = scaling_tests[pop_size]
        time_str = format_time(data['mean'], data['unit'])

        # Estimate sequential baseline (population 1 or smallest)
        if 1 in scaling_tests:
            sequential_ms = normalize_time(scaling_tests[1]['mean'], scaling_tests[1]['unit'])
        else:
            # Use smallest population as proxy
            min_pop = min(scaling_tests.keys())
            sequential_ms = normalize_time(scaling_tests[min_pop]['mean'], scaling_tests[min_pop]['unit'])

        current_ms = normalize_time(data['mean'], data['unit'])
        speedup = calculate_speedup(sequential_ms, current_ms)

        # Estimate parallel efficiency (speedup / ideal_speedup)
        # Ideal speedup ≈ population_size (if perfectly parallelized)
        ideal_speedup = max(1, pop_size / 10)  # Rough estimate
        efficiency = min(100, (speedup / ideal_speedup) * 100) if ideal_speedup > 0 else 0

        lines.append(f"| {pop_size} | {time_str} | {speedup:.1f}x | {efficiency:.0f}% |")

    lines.append("")
    return "\n".join(lines)


def generate_detailed_results(all_results):
    """Generate detailed benchmark results table."""
    lines = []
    lines.append("## Detailed Benchmark Results")
    lines.append("")
    lines.append("| Benchmark | Mean Time | 95% CI | Change vs Baseline |")
    lines.append("|-----------|-----------|--------|-------------------|")

    for test_name, data in all_results.items():
        mean_str = format_time(data['mean'], data['unit'])
        low_str = format_time(data['low'], data['unit'])
        high_str = format_time(data['high'], data['unit'])
        ci_str = f"[{low_str}, {high_str}]"

        change_str = "N/A"
        if 'change' in data:
            change = data['change']
            change_str = f"{change:+.1f}%"
            if change < -5:
                change_str += " 🚀"  # Faster
            elif change > 5:
                change_str += " ⚠️"  # Slower (regression)

        # Shorten test name for readability
        short_name = test_name.split('/')[-1] if '/' in test_name else test_name
        lines.append(f"| {short_name} | {mean_str} | {ci_str} | {change_str} |")

    lines.append("")
    return "\n".join(lines)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 parse_benchmark_results.py bench1.txt bench2.txt ...")
        sys.exit(1)

    # Parse all benchmark files
    all_results = {}
    for filepath in sys.argv[1:]:
        if Path(filepath).exists():
            results = parse_criterion_output(filepath)
            all_results.update(results)
        else:
            print(f"Warning: File not found: {filepath}", file=sys.stderr)

    if not all_results:
        print("Error: No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    # Generate markdown report
    print("# CUDA-ext Performance Report")
    print("")
    print(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print("---")
    print("")

    # Summary table
    print(generate_summary_table(all_results))
    print("")

    # Scaling analysis
    print(generate_scaling_analysis(all_results))
    print("")

    # Detailed results
    print(generate_detailed_results(all_results))
    print("")

    # Footer
    print("---")
    print("")
    print("**Note**: Benchmarks run with Criterion.rs (95% confidence intervals)")
    print("")
    print("**Legend**:")
    print("- ✅ Validated: Performance target achieved")
    print("- ⚠️ Below target: Performance below expected range")
    print("- ⏳ Pending: Feature not yet implemented")
    print("- 🚀 Faster than baseline")
    print("- ⚠️ Slower than baseline (potential regression)")
    print("")


if __name__ == "__main__":
    main()

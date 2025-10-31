#!/usr/bin/env python3
"""
Generate Markdown Summary from Genetic Optimizer Benchmarks

This script parses Criterion benchmark results and generates a concise
markdown summary for inclusion in documentation.

Usage:
    python scripts/generate_optimizer_markdown_report.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_criterion_results(criterion_dir: Path) -> Dict[str, List[Tuple[str, float]]]:
    """Parse Criterion benchmark results from JSON files."""
    results = {}

    # Look for benchmark groups
    for group_dir in criterion_dir.iterdir():
        if not group_dir.is_dir():
            continue

        group_name = group_dir.name
        group_results = []

        # Find all benchmark results in this group
        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir():
                continue

            # Look for estimates.json
            estimates_file = bench_dir / "base" / "estimates.json"
            if not estimates_file.exists():
                continue

            try:
                with open(estimates_file) as f:
                    data = json.load(f)

                # Extract mean time in milliseconds
                mean_ns = data["mean"]["point_estimate"]
                mean_ms = mean_ns / 1_000_000

                bench_name = bench_dir.name
                group_results.append((bench_name, mean_ms))

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {estimates_file}: {e}")
                continue

        if group_results:
            results[group_name] = sorted(group_results, key=lambda x: x[0])

    return results


def generate_markdown(results: Dict[str, List[Tuple[str, float]]]) -> str:
    """Generate markdown summary from parsed results."""
    md = []

    md.append("# Genetic Optimizer Benchmark Results")
    md.append("")
    md.append("**Generated**: Auto-generated from Criterion benchmarks")
    md.append("")

    # Parallel performance
    if "genetic_optimizer_parallel_no_mutex" in results:
        md.append("## Parallel Performance (No Mutex)")
        md.append("")
        md.append("| Population Size | Time (ms) | Speedup vs Pop=50 |")
        md.append("|-----------------|-----------|-------------------|")

        bench_results = results["genetic_optimizer_parallel_no_mutex"]
        baseline = None

        for name, time_ms in bench_results:
            # Extract population size from name
            match = re.search(r'ParallelNoMutex/(\d+)', name)
            if match:
                pop_size = int(match.group(1))
                if baseline is None:
                    baseline = time_ms
                    speedup = "1.0x (baseline)"
                else:
                    speedup = f"{baseline / time_ms:.2f}x"

                md.append(f"| {pop_size} | {time_ms:.2f} | {speedup} |")

        md.append("")

    # Population scaling
    if "genetic_optimizer_scaling" in results:
        md.append("## Population Scaling Efficiency")
        md.append("")
        md.append("| Population Size | Time (ms) | Parallel Efficiency |")
        md.append("|-----------------|-----------|---------------------|")

        bench_results = results["genetic_optimizer_scaling"]
        baseline_pop = None
        baseline_time = None

        for name, time_ms in bench_results:
            match = re.search(r'Scaling/(\d+)', name)
            if match:
                pop_size = int(match.group(1))

                if baseline_pop is None:
                    baseline_pop = pop_size
                    baseline_time = time_ms
                    efficiency = "100% (baseline)"
                else:
                    # Efficiency = (baseline_time * pop_ratio) / actual_time
                    pop_ratio = pop_size / baseline_pop
                    ideal_time = baseline_time * pop_ratio
                    efficiency = f"{(ideal_time / time_ms) * 100:.1f}%"

                md.append(f"| {pop_size} | {time_ms:.2f} | {efficiency} |")

        md.append("")

    # Convergence speed
    if "genetic_optimizer_convergence" in results:
        md.append("## Convergence Speed")
        md.append("")
        md.append("| Generations | Time (ms) | Time per Generation |")
        md.append("|-------------|-----------|---------------------|")

        bench_results = results["genetic_optimizer_convergence"]

        for name, time_ms in bench_results:
            match = re.search(r'Convergence/(\d+)', name)
            if match:
                generations = int(match.group(1))
                time_per_gen = time_ms / generations

                md.append(f"| {generations} | {time_ms:.2f} | {time_per_gen:.2f} |")

        md.append("")

    # Data size impact
    if "genetic_optimizer_data_size" in results:
        md.append("## Data Size Impact")
        md.append("")
        md.append("| Dataset Size | Time (ms) | Time per Candle |")
        md.append("|--------------|-----------|-----------------|")

        bench_results = results["genetic_optimizer_data_size"]

        for name, time_ms in bench_results:
            match = re.search(r'DataSize/(\d+)', name)
            if match:
                size = int(match.group(1))
                time_per_candle = time_ms / size

                md.append(f"| {size} | {time_ms:.2f} | {time_per_candle:.3f} |")

        md.append("")

    # FP8 precision
    for group_name in ["genetic_optimizer_fp64_baseline",
                       "genetic_optimizer_fp8_hybrid",
                       "genetic_optimizer_fp8_aggressive"]:
        if group_name in results:
            title = group_name.replace("genetic_optimizer_", "").replace("_", " ").title()
            md.append(f"## {title}")
            md.append("")
            md.append("| Dataset Size | Time (ms) |")
            md.append("|--------------|-----------|")

            bench_results = results[group_name]

            for name, time_ms in bench_results:
                match = re.search(r'/(\d+)', name)
                if match:
                    size = int(match.group(1))
                    md.append(f"| {size} | {time_ms:.2f} |")

            md.append("")

    return "\n".join(md)


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    rust_dir = script_dir.parent
    criterion_dir = rust_dir / "target" / "criterion"

    if not criterion_dir.exists():
        print(f"Error: Criterion results directory not found: {criterion_dir}")
        print("Please run benchmarks first:")
        print("  cargo bench --features gpu --bench genetic_optimizer_comparison")
        return 1

    print("Parsing Criterion benchmark results...")
    results = parse_criterion_results(criterion_dir)

    if not results:
        print("Warning: No benchmark results found")
        return 1

    print(f"Found {len(results)} benchmark groups")

    print("\nGenerating markdown summary...")
    markdown = generate_markdown(results)

    # Write to file
    output_file = rust_dir / "docs" / "GENETIC_OPTIMIZER_BENCHMARK_RESULTS.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(markdown)

    print(f"\n✓ Benchmark summary written to: {output_file}")

    # Also print to stdout
    print("\n" + "=" * 70)
    print(markdown)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Analyze Pinned Memory Benchmark Results

Extracts key metrics from Criterion benchmark output and generates
a concise summary report with speedup calculations.
"""

import re
import sys
from typing import Dict, List, Tuple


def parse_benchmark_output(file_path: str) -> Dict[str, Dict[int, float]]:
    """Parse Criterion benchmark output and extract times."""

    with open(file_path, 'r') as f:
        content = f.read()

    results = {
        'h2d_standard': {},
        'h2d_pinned': {},
        'd2h_standard': {},
        'd2h_pinned': {},
        'roundtrip_standard': {},
        'roundtrip_pinned': {},
    }

    # Pattern: benchmark_name/size time: [X.XXX unit]
    pattern = r'(h2d_standard|h2d_pinned|d2h_standard|d2h_pinned|roundtrip_standard|roundtrip_pinned)/(\d+)\s+time:\s+\[[\d.]+ [µmn]s ([\d.]+) ([µmn]s)'

    for match in re.finditer(pattern, content):
        benchmark = match.group(1)
        size = int(match.group(2))
        time_val = float(match.group(3))
        time_unit = match.group(4)

        # Convert to microseconds
        if time_unit == 'ns':
            time_val /= 1000
        elif time_unit == 'ms':
            time_val *= 1000

        results[benchmark][size] = time_val

    return results


def calculate_speedups(results: Dict[str, Dict[int, float]]) -> None:
    """Calculate and print speedup comparisons."""

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║     Pinned Memory vs Standard Memory - Speedup Analysis      ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    # H2D Speedups
    print("📤 H2D Transfers (Host-to-Device)")
    print("─" * 75)
    print(f"{'Size':<10} {'Standard':<15} {'Pinned':<15} {'Speedup':<15} {'Status'}")
    print("─" * 75)

    for size in sorted(results['h2d_standard'].keys()):
        std_time = results['h2d_standard'][size]
        pin_time = results['h2d_pinned'][size]
        speedup = std_time / pin_time

        status = "✅ FASTER" if speedup > 1.0 else "⚠️  SLOWER"
        improvement = (speedup - 1.0) * 100

        print(f"{size:<10} {std_time:>10.2f} µs   {pin_time:>10.2f} µs   "
              f"{speedup:>6.2f}x ({improvement:+.0f}%)  {status}")

    print()

    # D2H Speedups
    print("📥 D2H Transfers (Device-to-Host)")
    print("─" * 75)
    print(f"{'Size':<10} {'Standard':<15} {'Pinned':<15} {'Speedup':<15} {'Status'}")
    print("─" * 75)

    for size in sorted(results['d2h_standard'].keys()):
        std_time = results['d2h_standard'][size]
        pin_time = results['d2h_pinned'][size]
        speedup = std_time / pin_time

        status = "✅ FASTER" if speedup > 1.0 else "⚠️  SLOWER"
        improvement = (speedup - 1.0) * 100

        print(f"{size:<10} {std_time:>10.2f} µs   {pin_time:>10.2f} µs   "
              f"{speedup:>6.2f}x ({improvement:+.0f}%)  {status}")

    print()

    # Round-trip Speedups (RSI workload)
    print("🔄 Round-trip Transfers (RSI Workload: 2x H2D + 2x D2H)")
    print("─" * 75)
    print(f"{'Size':<10} {'Standard':<15} {'Pinned':<15} {'Speedup':<15} {'Status'}")
    print("─" * 75)

    for size in sorted(results['roundtrip_standard'].keys()):
        std_time = results['roundtrip_standard'][size]
        pin_time = results['roundtrip_pinned'][size]
        speedup = std_time / pin_time

        status = "✅ FASTER" if speedup > 1.0 else "⚠️  SLOWER"
        improvement = (speedup - 1.0) * 100

        print(f"{size:<10} {std_time:>10.2f} µs   {pin_time:>10.2f} µs   "
              f"{speedup:>6.2f}x ({improvement:+.0f}%)  {status}")

    print()

    # Calculate bandwidth for 100K size (most realistic)
    size_100k = 100_000
    if size_100k in results['h2d_standard']:
        print("📊 Bandwidth Analysis (100K elements = 781.25 KB)")
        print("─" * 75)

        bytes_per_element = 8  # f64
        total_bytes = size_100k * bytes_per_element
        total_gb = total_bytes / 1e9

        # H2D
        std_h2d_time = results['h2d_standard'][size_100k] / 1e6  # Convert to seconds
        pin_h2d_time = results['h2d_pinned'][size_100k] / 1e6
        std_h2d_bw = total_gb / std_h2d_time
        pin_h2d_bw = total_gb / pin_h2d_time

        print(f"H2D Standard:  {std_h2d_bw:>6.2f} GB/s")
        print(f"H2D Pinned:    {pin_h2d_bw:>6.2f} GB/s  ({pin_h2d_bw/std_h2d_bw:.2f}x faster)")
        print()

        # D2H
        std_d2h_time = results['d2h_standard'][size_100k] / 1e6
        pin_d2h_time = results['d2h_pinned'][size_100k] / 1e6
        std_d2h_bw = total_gb / std_d2h_time
        pin_d2h_bw = total_gb / pin_d2h_time

        print(f"D2H Standard:  {std_d2h_bw:>6.2f} GB/s")
        print(f"D2H Pinned:    {pin_d2h_bw:>6.2f} GB/s  ({pin_d2h_bw/std_d2h_bw:.2f}x faster)")
        print()

        # Round-trip effective bandwidth
        std_rt_time = results['roundtrip_standard'][size_100k] / 1e6
        pin_rt_time = results['roundtrip_pinned'][size_100k] / 1e6
        rt_bytes = total_bytes * 4  # 2x H2D + 2x D2H
        rt_gb = rt_bytes / 1e9
        std_rt_bw = rt_gb / std_rt_time
        pin_rt_bw = rt_gb / pin_rt_time

        print(f"Round-trip Standard: {std_rt_bw:>6.2f} GB/s (effective)")
        print(f"Round-trip Pinned:   {pin_rt_bw:>6.2f} GB/s (effective)  "
              f"({pin_rt_bw/std_rt_bw:.2f}x faster)")
        print()

    # Summary
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║                         Summary                               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    # Get 100K results for summary
    if size_100k in results['h2d_standard']:
        h2d_speedup = results['h2d_standard'][size_100k] / results['h2d_pinned'][size_100k]
        d2h_speedup = results['d2h_standard'][size_100k] / results['d2h_pinned'][size_100k]
        rt_speedup = results['roundtrip_standard'][size_100k] / results['roundtrip_pinned'][size_100k]

        print(f"H2D Speedup (100K):       {h2d_speedup:.2f}x ({(h2d_speedup-1)*100:+.0f}%)")
        print(f"D2H Speedup (100K):       {d2h_speedup:.2f}x ({(d2h_speedup-1)*100:+.0f}%)")
        print(f"Round-trip Speedup (100K): {rt_speedup:.2f}x ({(rt_speedup-1)*100:+.0f}%)")
        print()

        # RSI calculation impact
        std_rt_time = results['roundtrip_standard'][size_100k]
        pin_rt_time = results['roundtrip_pinned'][size_100k]

        print(f"RSI Calculation Impact (100K candles):")
        print(f"  Before: {std_rt_time:.2f} µs per RSI")
        print(f"  After:  {pin_rt_time:.2f} µs per RSI")
        print(f"  Gain:   {std_rt_time - pin_rt_time:.2f} µs saved per RSI")
        print()

        rsi_per_sec_before = 1e6 / std_rt_time
        rsi_per_sec_after = 1e6 / pin_rt_time

        print(f"Throughput:")
        print(f"  Before: {rsi_per_sec_before:,.0f} RSI/sec")
        print(f"  After:  {rsi_per_sec_after:,.0f} RSI/sec")
        print(f"  Gain:   {rsi_per_sec_after - rsi_per_sec_before:,.0f} RSI/sec")
        print()

    # PR #6 Validation
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║                   PR #6 Claim Validation                      ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("PR #6 Claims:")
    print("  - H2D: 20-30% faster (1.2-1.3x)")
    print("  - D2H: 20-30% faster (1.2-1.3x)")
    print("  - Overall: 20-30% faster (1.2-1.3x)")
    print()
    print("Actual Results (100K elements):")

    if size_100k in results['h2d_standard']:
        h2d_speedup = results['h2d_standard'][size_100k] / results['h2d_pinned'][size_100k]
        d2h_speedup = results['d2h_standard'][size_100k] / results['d2h_pinned'][size_100k]
        rt_speedup = results['roundtrip_standard'][size_100k] / results['roundtrip_pinned'][size_100k]

        h2d_status = "✅ EXCEEDED" if h2d_speedup > 1.3 else "✅ MET" if h2d_speedup > 1.2 else "⚠️  BELOW"
        d2h_status = "✅ EXCEEDED" if d2h_speedup > 1.3 else "✅ MET" if d2h_speedup > 1.2 else "⚠️  BELOW"
        rt_status = "✅ EXCEEDED" if rt_speedup > 1.3 else "✅ MET" if rt_speedup > 1.2 else "⚠️  BELOW"

        print(f"  - H2D: {h2d_speedup:.2f}x ({(h2d_speedup-1)*100:+.0f}%) {h2d_status}")
        print(f"  - D2H: {d2h_speedup:.2f}x ({(d2h_speedup-1)*100:+.0f}%) {d2h_status}")
        print(f"  - Overall: {rt_speedup:.2f}x ({(rt_speedup-1)*100:+.0f}%) {rt_status}")
        print()

        if h2d_status == "✅ EXCEEDED" and d2h_status == "✅ EXCEEDED" and rt_status == "✅ EXCEEDED":
            print("🎉 VERDICT: PR #6 claims VALIDATED and EXCEEDED!")
            print("   Recommendation: MERGE immediately")
        elif "✅" in h2d_status and "✅" in d2h_status and "✅" in rt_status:
            print("✅ VERDICT: PR #6 claims VALIDATED")
            print("   Recommendation: MERGE with confidence")
        else:
            print("⚠️  VERDICT: PR #6 claims NOT MET")
            print("   Recommendation: Investigate discrepancies")


def main():
    """Main entry point."""

    if len(sys.argv) != 2:
        print("Usage: python analyze_pinned_memory_results.py <benchmark_output_file>")
        print("Example: python analyze_pinned_memory_results.py /tmp/pinned_vs_standard_results.txt")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        results = parse_benchmark_output(file_path)
        calculate_speedups(results)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

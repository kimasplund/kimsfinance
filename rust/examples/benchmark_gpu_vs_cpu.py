#!/usr/bin/env python3
"""
Benchmark GPU vs CPU tick aggregation with real data
Collects detailed performance statistics
"""

import sys
import time
import numpy as np
from typing import Dict, List, Tuple

try:
    import kimsfinance_core
except ImportError as e:
    print(f"❌ Failed to import kimsfinance_core: {e}")
    sys.exit(1)

def generate_realistic_tick_data(num_ticks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic tick data for benchmarking"""
    np.random.seed(42)

    # Start time: 2024-01-01 00:00:00 UTC (in milliseconds)
    start_time = 1704067200000

    # Generate timestamps with realistic spacing (10-1000ms between ticks)
    intervals = np.random.randint(10, 1000, size=num_ticks)
    timestamps = np.cumsum(intervals) + start_time
    timestamps = timestamps.astype(np.int64)

    # Generate prices with realistic volatility (BTC-like)
    base_price = 42000.0
    price_changes = np.random.randn(num_ticks) * 10.0  # $10 std dev
    prices = base_price + np.cumsum(price_changes)
    prices = prices.astype(np.float32)

    # Generate volumes (log-normal distribution)
    volumes = np.random.lognormal(mean=0.5, sigma=1.0, size=num_ticks).astype(np.float32)

    # Generate sides (60% buy, 40% sell - realistic market)
    sides = np.where(np.random.random(num_ticks) < 0.6, 1, -1).astype(np.int8)

    return timestamps, prices, volumes, sides

def benchmark_gpu_aggregation(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    timeframe_ms: int,
    num_runs: int = 10
) -> Dict:
    """Benchmark GPU aggregation with multiple runs"""

    aggregator = kimsfinance_core.GpuTickAggregator()

    # Warmup run (JIT compilation happens here)
    print("  Warming up GPU (JIT compilation)...", flush=True)
    warmup_start = time.perf_counter()
    _ = aggregator.aggregate(timestamps, prices, volumes, sides, timeframe_ms)
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warmup completed: {warmup_time*1000:.2f}ms (includes JIT compilation)")

    # Benchmark runs
    times = []
    candle_counts = []

    for i in range(num_runs):
        start = time.perf_counter()
        candles = aggregator.aggregate(timestamps, prices, volumes, sides, timeframe_ms)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        candle_counts.append(candles.num_candles)

    # Calculate statistics
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    median_time = np.median(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    std_time = np.std(times_array)

    throughput = len(timestamps) / mean_time

    return {
        'mean_time_ms': mean_time * 1000,
        'median_time_ms': median_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_ticks_per_sec': throughput,
        'num_candles': candle_counts[0],
        'warmup_time_ms': warmup_time * 1000,
        'all_times_ms': [t * 1000 for t in times],
    }

def benchmark_cpu_aggregation_python(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    timeframe_ms: int,
    num_runs: int = 10
) -> Dict:
    """Benchmark CPU aggregation using pure Python/NumPy"""

    def aggregate_cpu(ts, p, v, s, tf):
        # Bucket timestamps
        buckets = ts // tf
        unique_buckets = np.unique(buckets)
        num_candles = len(unique_buckets)

        # Pre-allocate output arrays
        out_timestamps = np.zeros(num_candles, dtype=np.int64)
        out_open = np.zeros(num_candles, dtype=np.float32)
        out_high = np.zeros(num_candles, dtype=np.float32)
        out_low = np.zeros(num_candles, dtype=np.float32)
        out_close = np.zeros(num_candles, dtype=np.float32)
        out_volume = np.zeros(num_candles, dtype=np.float32)
        out_num_trades = np.zeros(num_candles, dtype=np.int32)

        # Aggregate each bucket
        for i, bucket in enumerate(unique_buckets):
            mask = buckets == bucket
            bucket_timestamps = ts[mask]
            bucket_prices = p[mask]
            bucket_volumes = v[mask]

            # Find open (first trade) and close (last trade)
            first_idx = np.argmin(bucket_timestamps)
            last_idx = np.argmax(bucket_timestamps)

            out_timestamps[i] = bucket * tf
            out_open[i] = bucket_prices[first_idx]
            out_high[i] = np.max(bucket_prices)
            out_low[i] = np.min(bucket_prices)
            out_close[i] = bucket_prices[last_idx]
            out_volume[i] = np.sum(bucket_volumes)
            out_num_trades[i] = np.sum(mask)

        return num_candles

    # Warmup
    _ = aggregate_cpu(timestamps, prices, volumes, sides, timeframe_ms)

    # Benchmark runs
    times = []
    candle_counts = []

    for i in range(num_runs):
        start = time.perf_counter()
        num_candles = aggregate_cpu(timestamps, prices, volumes, sides, timeframe_ms)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        candle_counts.append(num_candles)

    # Calculate statistics
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    median_time = np.median(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    std_time = np.std(times_array)

    throughput = len(timestamps) / mean_time

    return {
        'mean_time_ms': mean_time * 1000,
        'median_time_ms': median_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_ticks_per_sec': throughput,
        'num_candles': candle_counts[0],
        'all_times_ms': [t * 1000 for t in times],
    }

def print_stats(name: str, stats: Dict):
    """Pretty print statistics"""
    print(f"\n{name}:")
    print(f"  Mean time:       {stats['mean_time_ms']:>10.2f} ms")
    print(f"  Median time:     {stats['median_time_ms']:>10.2f} ms")
    print(f"  Min time:        {stats['min_time_ms']:>10.2f} ms")
    print(f"  Max time:        {stats['max_time_ms']:>10.2f} ms")
    print(f"  Std deviation:   {stats['std_time_ms']:>10.2f} ms")
    print(f"  Throughput:      {stats['throughput_ticks_per_sec']:>10,.0f} ticks/sec")
    print(f"  Candles output:  {stats['num_candles']:>10,} candles")
    if 'warmup_time_ms' in stats:
        print(f"  Warmup (JIT):    {stats['warmup_time_ms']:>10.2f} ms")

def main():
    print("=" * 80)
    print("GPU vs CPU Tick Aggregation Benchmark")
    print("=" * 80)

    # Check GPU availability
    if not kimsfinance_core.gpu_available():
        print("❌ GPU not available!")
        sys.exit(1)

    gpu_info = kimsfinance_core.gpu_info()
    print(f"\nGPU: Device {gpu_info['device_id']}")
    print(f"CUDA Version: {gpu_info['cuda_version']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Async Allocator: {gpu_info['async_allocator']}")

    # Test scenarios
    scenarios = [
        ("Small (1K ticks)", 1_000, 60_000),      # 1K ticks, 1-minute candles
        ("Medium (10K ticks)", 10_000, 60_000),   # 10K ticks, 1-minute candles
        ("Large (100K ticks)", 100_000, 60_000),  # 100K ticks, 1-minute candles
        ("Very Large (1M ticks)", 1_000_000, 300_000),  # 1M ticks, 5-minute candles
    ]

    num_benchmark_runs = 10

    results = []

    for scenario_name, num_ticks, timeframe_ms in scenarios:
        print("\n" + "=" * 80)
        print(f"Scenario: {scenario_name}")
        print(f"  Ticks: {num_ticks:,}")
        print(f"  Timeframe: {timeframe_ms/1000:.0f}s candles")
        print(f"  Benchmark runs: {num_benchmark_runs}")
        print("=" * 80)

        # Generate data
        print("\nGenerating tick data...", flush=True)
        timestamps, prices, volumes, sides = generate_realistic_tick_data(num_ticks)
        print(f"✓ Generated {len(timestamps):,} ticks")
        print(f"  Time range: {(timestamps[-1] - timestamps[0])/1000:.0f} seconds")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  Volume range: {volumes.min():.2f} - {volumes.max():.2f}")

        # Benchmark CPU
        print("\n[CPU] Running benchmark...", flush=True)
        cpu_stats = benchmark_cpu_aggregation_python(
            timestamps, prices, volumes, sides, timeframe_ms, num_benchmark_runs
        )
        print_stats("CPU Results", cpu_stats)

        # Benchmark GPU
        print("\n[GPU] Running benchmark...", flush=True)
        gpu_stats = benchmark_gpu_aggregation(
            timestamps, prices, volumes, sides, timeframe_ms, num_benchmark_runs
        )
        print_stats("GPU Results", gpu_stats)

        # Calculate speedup
        speedup = cpu_stats['mean_time_ms'] / gpu_stats['mean_time_ms']
        throughput_ratio = gpu_stats['throughput_ticks_per_sec'] / cpu_stats['throughput_ticks_per_sec']

        print(f"\n{'Comparison:':<20}")
        print(f"  Speedup:         {speedup:>10.2f}x faster (GPU)")
        print(f"  Throughput gain: {throughput_ratio:>10.2f}x more ticks/sec (GPU)")

        results.append({
            'scenario': scenario_name,
            'num_ticks': num_ticks,
            'timeframe_ms': timeframe_ms,
            'cpu_stats': cpu_stats,
            'gpu_stats': gpu_stats,
            'speedup': speedup,
            'throughput_ratio': throughput_ratio,
        })

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\n{'Scenario':<25} {'Ticks':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 80)

    for result in results:
        print(f"{result['scenario']:<25} "
              f"{result['num_ticks']:>10,}  "
              f"{result['cpu_stats']['mean_time_ms']:>10.2f}  "
              f"{result['gpu_stats']['mean_time_ms']:>10.2f}  "
              f"{result['speedup']:>8.2f}x")

    # Overall statistics
    speedups = [r['speedup'] for r in results]
    avg_speedup = np.mean(speedups)
    max_speedup = np.max(speedups)

    print("\n" + "=" * 80)
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Maximum Speedup: {max_speedup:.2f}x")
    print(f"GPU Peak Throughput: {max([r['gpu_stats']['throughput_ticks_per_sec'] for r in results]):,.0f} ticks/sec")
    print("=" * 80)

    print("\n✅ Benchmark completed successfully!")

if __name__ == "__main__":
    main()

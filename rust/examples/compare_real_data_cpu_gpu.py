#!/usr/bin/env python3
"""
Compare CPU vs GPU tick aggregation on real 2024 BTCUSDT data
Tests both implementations on actual production data
"""

import sys
import time
import glob
import numpy as np

try:
    import pyarrow.parquet as pq
    import kimsfinance_core
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def aggregate_cpu(ts, p, v, s, tf):
    """Pure Python/NumPy CPU aggregation"""
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

def load_month_data(year_month: str, max_files: int = None):
    """Load parquet files for a given month"""
    pattern = f"/home/kim/projects/binance-data/futures/BTCUSDT/trades_parquet/{year_month}/*.parquet"
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"❌ No files found for {year_month}")
        return None

    if max_files:
        files = files[:max_files]

    print(f"\nLoading {len(files)} files for {year_month}...")

    all_timestamps = []
    all_prices = []
    all_volumes = []
    all_sides = []

    for i, file in enumerate(files, 1):
        table = pq.read_table(file)

        # Extract columns
        timestamps = table['time'].to_numpy().astype(np.int64)
        prices = table['price'].to_numpy().astype(np.float32)
        volumes = table['qty'].to_numpy().astype(np.float32)

        # Infer side from is_buyer_maker (True = sell, False = buy)
        is_buyer_maker = table['is_buyer_maker'].to_numpy()
        sides = np.where(is_buyer_maker, -1, 1).astype(np.int8)

        all_timestamps.append(timestamps)
        all_prices.append(prices)
        all_volumes.append(volumes)
        all_sides.append(sides)

        if i % 5 == 0:
            print(f"  Loaded {i}/{len(files)} files...", flush=True)

    # Concatenate all arrays
    print("  Concatenating arrays...", flush=True)
    timestamps = np.concatenate(all_timestamps)
    prices = np.concatenate(all_prices)
    volumes = np.concatenate(all_volumes)
    sides = np.concatenate(all_sides)

    return timestamps, prices, volumes, sides

def main():
    print("=" * 80)
    print("CPU vs GPU Comparison - Real 2024 BTCUSDT Data")
    print("=" * 80)

    # Check GPU
    if not kimsfinance_core.gpu_available():
        print("❌ GPU not available!")
        sys.exit(1)

    gpu_info = kimsfinance_core.gpu_info()
    print(f"\nGPU: Device {gpu_info['device_id']}")
    print(f"CUDA: {gpu_info['cuda_version']}")
    print(f"Compute: {gpu_info['compute_capability']}")

    # Load data - use first 10 days for reasonable CPU time
    test_month = "2024-01"
    timeframe_ms = 300_000  # 5-minute candles

    print(f"\n{'='*80}")
    print(f"Loading {test_month} data (first 10 days for CPU test)...")
    print(f"{'='*80}")

    load_start = time.perf_counter()
    data = load_month_data(test_month, max_files=10)  # ~40M ticks
    load_time = time.perf_counter() - load_start

    if data is None:
        sys.exit(1)

    timestamps, prices, volumes, sides = data

    print(f"\n✓ Data loaded in {load_time:.2f} seconds")
    print(f"\nDataset statistics:")
    print(f"  Total ticks:     {len(timestamps):>15,}")
    print(f"  Time range:      {(timestamps[-1] - timestamps[0])/1000:>15,.0f} seconds")
    print(f"  Price range:     ${prices.min():>14,.2f} - ${prices.max():<.2f}")
    print(f"  Volume range:    {volumes.min():>15,.2f} - {volumes.max():<.2f}")
    print(f"  Memory:          {(timestamps.nbytes + prices.nbytes + volumes.nbytes + sides.nbytes) / (1024**3):>15.2f} GB")

    # Test 1: CPU Aggregation
    print(f"\n{'='*80}")
    print("TEST 1: CPU Aggregation (Pure Python/NumPy)")
    print(f"{'='*80}")

    print("\n[Warmup] CPU...", flush=True)
    _ = aggregate_cpu(timestamps[:100000], prices[:100000], volumes[:100000], sides[:100000], timeframe_ms)
    print("✓ Warmup completed")

    print(f"\n[Full] Processing {len(timestamps):,} ticks...", flush=True)
    cpu_start = time.perf_counter()
    num_candles_cpu = aggregate_cpu(timestamps, prices, volumes, sides, timeframe_ms)
    cpu_time = time.perf_counter() - cpu_start
    cpu_throughput = len(timestamps) / cpu_time

    print(f"\n✓ CPU aggregation completed")
    print(f"  Processing time:   {cpu_time:>12.2f} seconds")
    print(f"  Throughput:        {cpu_throughput:>12,.0f} ticks/sec")
    print(f"  Throughput:        {cpu_throughput/1_000_000:>12.2f} M ticks/sec")
    print(f"  Output candles:    {num_candles_cpu:>12,}")

    # Test 2: GPU Aggregation (batched)
    print(f"\n{'='*80}")
    print("TEST 2: GPU Aggregation (Batched JIT-compiled CUDA)")
    print(f"{'='*80}")

    aggregator = kimsfinance_core.GpuTickAggregator()

    print("\n[Warmup] GPU (JIT compilation)...", flush=True)
    warmup_start = time.perf_counter()
    _ = aggregator.aggregate(timestamps[:100000], prices[:100000], volumes[:100000], sides[:100000], timeframe_ms)
    warmup_time = time.perf_counter() - warmup_start
    print(f"✓ Warmup completed: {warmup_time*1000:.2f}ms (includes JIT)")

    # Batch processing
    batch_size = 1_000_000  # 1M ticks per batch
    num_batches = (len(timestamps) + batch_size - 1) // batch_size

    print(f"\n[Full] Processing {len(timestamps):,} ticks in {num_batches} batches...", flush=True)

    all_candle_data = {
        'timestamps': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
        'num_trades': []
    }

    gpu_start = time.perf_counter()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(timestamps))

        batch_timestamps = timestamps[start_idx:end_idx]
        batch_prices = prices[start_idx:end_idx]
        batch_volumes = volumes[start_idx:end_idx]
        batch_sides = sides[start_idx:end_idx]

        candles = aggregator.aggregate(batch_timestamps, batch_prices, batch_volumes, batch_sides, timeframe_ms)

        # Collect results
        all_candle_data['timestamps'].extend(candles.timestamps)
        all_candle_data['open'].extend(candles.open)
        all_candle_data['high'].extend(candles.high)
        all_candle_data['low'].extend(candles.low)
        all_candle_data['close'].extend(candles.close)
        all_candle_data['volume'].extend(candles.volume)
        all_candle_data['num_trades'].extend(candles.num_trades)

        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  Batch {batch_idx+1}/{num_batches}: {end_idx-start_idx:,} ticks -> {candles.num_candles} candles", flush=True)

    gpu_time = time.perf_counter() - gpu_start
    gpu_throughput = len(timestamps) / gpu_time
    num_candles_gpu = len(all_candle_data['timestamps'])

    print(f"\n✓ GPU aggregation completed")
    print(f"  Processing time:   {gpu_time:>12.2f} seconds")
    print(f"  Throughput:        {gpu_throughput:>12,.0f} ticks/sec")
    print(f"  Throughput:        {gpu_throughput/1_000_000:>12.2f} M ticks/sec")
    print(f"  Output candles:    {num_candles_gpu:>12,}")

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    speedup = cpu_time / gpu_time
    throughput_ratio = gpu_throughput / cpu_throughput

    print(f"\nPerformance:")
    print(f"  CPU time:          {cpu_time:>12.2f} seconds")
    print(f"  GPU time:          {gpu_time:>12.2f} seconds")
    print(f"  Speedup:           {speedup:>12.2f}x faster (GPU)")
    print(f"")
    print(f"  CPU throughput:    {cpu_throughput/1_000_000:>12.2f} M ticks/sec")
    print(f"  GPU throughput:    {gpu_throughput/1_000_000:>12.2f} M ticks/sec")
    print(f"  Throughput gain:   {throughput_ratio:>12.2f}x more (GPU)")

    print(f"\nOutput:")
    print(f"  CPU candles:       {num_candles_cpu:>12,}")
    print(f"  GPU candles:       {num_candles_gpu:>12,}")
    print(f"  Match:             {'✓ YES' if num_candles_cpu == num_candles_gpu else '✗ NO'}")

    # Project to full year
    print(f"\n{'='*80}")
    print("PROJECTION: Full 2024 Year")
    print(f"{'='*80}")

    ticks_per_10days = len(timestamps)
    estimated_ticks_2024 = ticks_per_10days * 36.6  # ~366 days in 2024

    full_year_cpu_time = (estimated_ticks_2024 / cpu_throughput) / 60  # minutes
    full_year_gpu_time = (estimated_ticks_2024 / gpu_throughput)  # seconds

    print(f"  Estimated full 2024 ticks:  {estimated_ticks_2024:>15,.0f}")
    print(f"  CPU projected time:         {full_year_cpu_time:>15.1f} minutes")
    print(f"  GPU projected time:         {full_year_gpu_time:>15.1f} seconds")
    print(f"  GPU projected time:         {full_year_gpu_time/60:>15.1f} minutes")

    print(f"\n{'='*80}")
    print("✅ Comparison completed successfully!")
    print(f"{'='*80}")

    print(f"\n💡 Summary:")
    print(f"   Dataset: {len(timestamps):,} ticks (10 days of 2024)")
    print(f"   GPU is {speedup:.1f}x faster than CPU")
    print(f"   GPU throughput: {gpu_throughput/1_000_000:.1f}M ticks/sec")
    print(f"   CPU throughput: {cpu_throughput/1_000_000:.1f}M ticks/sec")
    print(f"   Full 2024 projection: GPU={full_year_gpu_time:.0f}s, CPU={full_year_cpu_time:.1f}min")

if __name__ == "__main__":
    main()

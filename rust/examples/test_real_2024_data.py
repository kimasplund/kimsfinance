#!/usr/bin/env python3
"""
Test GPU tick aggregation with real 2024 BTCUSDT data
Processes one month to validate performance
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

def load_month_data(year_month: str):
    """Load all parquet files for a given month"""
    pattern = f"/home/kim/projects/binance-data/futures/BTCUSDT/trades_parquet/{year_month}/*.parquet"
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"❌ No files found for {year_month}")
        return None

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
    print("GPU Tick Aggregation - Real 2024 BTCUSDT Data Test")
    print("=" * 80)

    # Check GPU
    if not kimsfinance_core.gpu_available():
        print("❌ GPU not available!")
        sys.exit(1)

    gpu_info = kimsfinance_core.gpu_info()
    print(f"\nGPU: Device {gpu_info['device_id']}")
    print(f"CUDA: {gpu_info['cuda_version']}")
    print(f"Compute: {gpu_info['compute_capability']}")

    # Test with January 2024 (largest month)
    test_month = "2024-01"
    timeframe_ms = 300_000  # 5-minute candles

    # Load data
    print(f"\n{'='*80}")
    print(f"Loading {test_month} data...")
    print(f"{'='*80}")

    load_start = time.perf_counter()
    data = load_month_data(test_month)
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

    # Create aggregator
    print(f"\n{'='*80}")
    print("Initializing GPU aggregator...")
    print(f"{'='*80}")

    init_start = time.perf_counter()
    aggregator = kimsfinance_core.GpuTickAggregator()
    init_time = time.perf_counter() - init_start

    print(f"✓ Aggregator initialized in {init_time*1000:.2f}ms")

    # Run aggregation
    print(f"\n{'='*80}")
    print(f"Running GPU aggregation ({timeframe_ms/1000:.0f}s candles)...")
    print(f"{'='*80}")

    # Warmup
    print("\n[Warmup] JIT compiling kernels...", flush=True)
    warmup_start = time.perf_counter()
    _ = aggregator.aggregate(
        timestamps[:100000],  # Use 100K ticks for warmup
        prices[:100000],
        volumes[:100000],
        sides[:100000],
        timeframe_ms
    )
    warmup_time = time.perf_counter() - warmup_start
    print(f"✓ Warmup completed: {warmup_time*1000:.2f}ms")

    # Full aggregation (in batches to avoid buffer pool limits)
    batch_size = 1_000_000  # 1M ticks per batch (matches pinned buffer pool size)
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

    agg_start = time.perf_counter()

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

        print(f"  Batch {batch_idx+1}/{num_batches}: {end_idx-start_idx:,} ticks -> {candles.num_candles} candles", flush=True)

    agg_time = time.perf_counter() - agg_start

    # Create final candles object-like structure
    class CandleData:
        def __init__(self, data):
            self.timestamps = np.array(data['timestamps'])
            self.open = np.array(data['open'])
            self.high = np.array(data['high'])
            self.low = np.array(data['low'])
            self.close = np.array(data['close'])
            self.volume = np.array(data['volume'])
            self.num_trades = np.array(data['num_trades'])
            self.num_candles = len(self.timestamps)

    candles = CandleData(all_candle_data)
    throughput = len(timestamps) / agg_time

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\nPerformance:")
    print(f"  Processing time:   {agg_time*1000:>12.2f} ms")
    print(f"  Processing time:   {agg_time:>12.2f} seconds")
    print(f"  Throughput:        {throughput:>12,.0f} ticks/sec")
    print(f"  Throughput:        {throughput/1_000_000:>12.2f} M ticks/sec")

    print(f"\nOutput:")
    print(f"  Input ticks:       {len(timestamps):>12,}")
    print(f"  Output candles:    {candles.num_candles:>12,}")
    print(f"  Timeframe:         {timeframe_ms/1000:>12.0f} seconds")
    print(f"  Time span:         {(timestamps[-1] - timestamps[0])/1000/3600:>12.1f} hours")

    print(f"\nSample candles (first 5):")
    for i in range(min(5, candles.num_candles)):
        print(f"  Candle {i+1}:")
        print(f"    Time:   {candles.timestamps[i]}")
        print(f"    Open:   ${candles.open[i]:,.2f}")
        print(f"    High:   ${candles.high[i]:,.2f}")
        print(f"    Low:    ${candles.low[i]:,.2f}")
        print(f"    Close:  ${candles.close[i]:,.2f}")
        print(f"    Volume: {candles.volume[i]:,.2f}")
        print(f"    Trades: {candles.num_trades[i]:,}")

    # Project full year performance
    ticks_per_month = len(timestamps)
    months_in_2024 = 12
    total_ticks_2024 = ticks_per_month * months_in_2024
    full_year_time = (total_ticks_2024 / throughput)

    print(f"\n{'='*80}")
    print("PROJECTION: Full 2024 Year")
    print(f"{'='*80}")
    print(f"  Estimated ticks:   {total_ticks_2024:>12,}")
    print(f"  Estimated time:    {full_year_time:>12.2f} seconds")
    print(f"  Estimated time:    {full_year_time/60:>12.2f} minutes")

    print(f"\n{'='*80}")
    print("✅ Test completed successfully!")
    print(f"{'='*80}")

    print(f"\n💡 Conclusion:")
    print(f"   GPU can process {test_month} ({len(timestamps):,} ticks) in {agg_time:.2f} seconds")
    print(f"   Projected time for full 2024: ~{full_year_time:.1f} seconds")
    print(f"   Throughput: {throughput/1_000_000:.1f}M ticks/sec")

if __name__ == "__main__":
    main()

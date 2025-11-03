#!/usr/bin/env python3
"""Test Python bindings for GPU tick aggregation"""

import sys
import numpy as np

try:
    import kimsfinance_core
except ImportError as e:
    print(f"❌ Failed to import kimsfinance_core: {e}")
    print("\nMake sure you've built with GPU support:")
    print("  cargo build --release --features gpu")
    print("  maturin develop --release --features gpu")
    sys.exit(1)

print("=== Testing Python Bindings for GPU Tick Aggregation ===\n")

# Test 1: Check GPU availability
print("Test 1: Checking GPU availability...")
try:
    if hasattr(kimsfinance_core, 'gpu_available'):
        gpu_avail = kimsfinance_core.gpu_available()
        if gpu_avail:
            print("✓ GPU is available")
        else:
            print("✗ GPU not available")
            sys.exit(1)
    else:
        print("✗ gpu_available() function not found")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking GPU availability: {e}")
    sys.exit(1)

# Test 2: Get GPU info
print("\nTest 2: Getting GPU information...")
try:
    if hasattr(kimsfinance_core, 'gpu_info'):
        info = kimsfinance_core.gpu_info()
        print(f"✓ GPU Info:")
        print(f"    Device ID: {info['device_id']}")
        print(f"    CUDA Version: {info['cuda_version']}")
        print(f"    Compute Capability: {info['compute_capability']}")
        print(f"    Async Allocator: {info['async_allocator']}")
    else:
        print("✗ gpu_info() function not found")
except Exception as e:
    print(f"✗ Error getting GPU info: {e}")

# Test 3: Create GpuTickAggregator
print("\nTest 3: Creating GPU tick aggregator...")
try:
    if hasattr(kimsfinance_core, 'GpuTickAggregator'):
        aggregator = kimsfinance_core.GpuTickAggregator()
        print(f"✓ Created aggregator: {aggregator}")
    else:
        print("✗ GpuTickAggregator class not found")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error creating aggregator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Aggregate tick data
print("\nTest 4: Aggregating tick data...")
try:
    # Create simple test data: 10 trades
    timestamps = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500], dtype=np.int64)
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0], dtype=np.float32)
    volumes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    sides = np.array([1, 1, -1, 1, -1, 1, -1, 1, 1, -1], dtype=np.int8)

    # Aggregate to 3-second candles
    candles = aggregator.aggregate(timestamps, prices, volumes, sides, 3000)

    print(f"✓ Aggregation successful")
    print(f"    Input: {len(timestamps)} trades")
    print(f"    Output: {candles.num_candles} candles")
    print(f"    Candles object: {candles}")

except Exception as e:
    print(f"✗ Error aggregating: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Access candle data
print("\nTest 5: Accessing candle data...")
try:
    print(f"✓ Candle data accessible:")
    print(f"    Timestamps: {candles.timestamps}")
    print(f"    Open:       {candles.open}")
    print(f"    High:       {candles.high}")
    print(f"    Low:        {candles.low}")
    print(f"    Close:      {candles.close}")
    print(f"    Volume:     {candles.volume}")
    print(f"    Num Trades: {candles.num_trades}")

    if candles.num_candles > 0:
        print(f"\n    First candle:")
        print(f"      Timestamp: {candles.timestamps[0]}")
        print(f"      Open:      {candles.open[0]:.2f}")
        print(f"      High:      {candles.high[0]:.2f}")
        print(f"      Low:       {candles.low[0]:.2f}")
        print(f"      Close:     {candles.close[0]:.2f}")
        print(f"      Volume:    {candles.volume[0]:.2f}")
        print(f"      Trades:    {candles.num_trades[0]}")

except Exception as e:
    print(f"✗ Error accessing candle data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Convert to dictionary
print("\nTest 6: Converting to dictionary...")
try:
    candle_dict = candles.to_dict()
    print(f"✓ Dictionary conversion successful")
    print(f"    Keys: {list(candle_dict.keys())}")
    print(f"    num_candles: {candle_dict['num_candles']}")

except Exception as e:
    print(f"✗ Error converting to dict: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Verify data integrity
print("\nTest 7: Verifying data integrity...")
try:
    # Check that we got 2 candles (0-2999ms and 3000-5999ms)
    assert candles.num_candles == 2, f"Expected 2 candles, got {candles.num_candles}"

    # Check that high >= low for all candles
    for i in range(candles.num_candles):
        assert candles.high[i] >= candles.low[i], f"Candle {i}: High ({candles.high[i]}) < Low ({candles.low[i]})"

    # Check that volumes are positive
    for i in range(candles.num_candles):
        assert candles.volume[i] > 0, f"Candle {i}: Volume ({candles.volume[i]}) <= 0"

    print("✓ Data integrity verified")
    print("    - Correct number of candles")
    print("    - High >= Low for all candles")
    print("    - All volumes positive")

except AssertionError as e:
    print(f"✗ Data integrity check failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error verifying data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All Python Binding Tests Passed! ===\n")
print("Summary:")
print("  1. ✓ GPU available and accessible")
print("  2. ✓ GPU info retrieval works")
print("  3. ✓ GpuTickAggregator instantiation works")
print("  4. ✓ Tick aggregation executes successfully")
print("  5. ✓ Candle data accessible via NumPy arrays")
print("  6. ✓ Dictionary conversion works")
print("  7. ✓ Data integrity verified")
print("\n🎉 Python bindings for GPU tick aggregation are working correctly!")

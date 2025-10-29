#!/usr/bin/env python3
"""Test Supertrend indicator from Rust"""

import sys
import numpy as np

# Add the target directory to path
sys.path.insert(0, '/home/kim-asplund/projects/kimsfinance/rust/target/release')

import kimsfinance_core

# Test data
high = np.array([
    110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0,
    132.0, 135.0, 133.0, 136.0, 140.0, 138.0, 142.0, 145.0, 143.0, 146.0,
])
low = np.array([
    105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0,
    127.0, 130.0, 128.0, 131.0, 135.0, 133.0, 137.0, 140.0, 138.0, 141.0,
])
close = np.array([
    108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0,
    130.0, 133.0, 131.0, 134.0, 138.0, 136.0, 140.0, 143.0, 141.0, 144.0,
])

try:
    # Call Rust Supertrend
    result = kimsfinance_core.calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0)

    print("✓ Supertrend calculation successful!")
    print(f"  Returned keys: {list(result.keys())}")
    print(f"  Supertrend shape: {result['supertrend'].shape}")
    print(f"  Signal shape: {result['signal'].shape}")
    print(f"  First 10 supertrend values: {result['supertrend'][:10]}")
    print(f"  First 10 signal values: {result['signal'][:10]}")

    # Verify structure
    assert 'supertrend' in result, "Missing supertrend key"
    assert 'signal' in result, "Missing signal key"
    assert len(result['supertrend']) == len(high), "Supertrend length mismatch"
    assert len(result['signal']) == len(high), "Signal length mismatch"

    # Verify warmup period
    assert np.isnan(result['supertrend'][:10]).all(), "Warmup period should be NaN"
    assert (result['signal'][:10] == 0).all(), "Warmup period signal should be 0"

    # Verify valid values after warmup
    assert not np.isnan(result['supertrend'][10]), "Supertrend should have valid values after warmup"
    assert result['signal'][10] in [-1, 1], f"Signal should be -1 or 1 after warmup, got {result['signal'][10]}"

    print("\n✓ All assertions passed!")
    print("\nSupertrend indicator implementation successful! 🎉")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

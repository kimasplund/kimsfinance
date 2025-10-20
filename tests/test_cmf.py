#!/usr/bin/env python3
"""
Test Suite for Chaikin Money Flow (CMF)
========================================

Comprehensive tests for CMF implementation including:
- Basic calculation
- Value range validation (-1 to +1)
- Volume requirement
- Buying/selling pressure detection
- Zero-crossing signals
- Edge cases
- Performance benchmarking
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.ops.indicators import calculate_cmf


def generate_ohlcv_data(n: int = 100, seed: int = 42) -> tuple:
    """Generate test OHLCV data for CMF testing."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    volumes = np.abs(np.random.randn(n) * 1_000_000)
    return highs, lows, closes, volumes


def test_basic_calculation():
    """Test basic CMF calculation."""
    print("\n=== Test: Basic Calculation ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Calculate CMF
    cmf = calculate_cmf(highs, lows, closes, volumes, period=20, engine="cpu")

    # Verify result
    assert len(cmf) == len(closes), "CMF length should match input length"
    assert isinstance(cmf, np.ndarray), "CMF should return NumPy array"

    # First (period-1) values should be NaN
    expected_nans = 19  # period - 1
    actual_nans = np.sum(np.isnan(cmf))
    assert actual_nans >= expected_nans, f"Expected at least {expected_nans} NaN values, got {actual_nans}"

    # Rest should be valid numbers
    valid_values = cmf[~np.isnan(cmf)]
    assert len(valid_values) > 0, "Should have some valid CMF values"

    print(f"✓ CMF calculated: {len(cmf)} values")
    print(f"  - NaN values: {actual_nans}")
    print(f"  - Valid values: {len(valid_values)}")
    print(f"  - Last 5 CMF values: {cmf[-5:]}")


def test_value_ranges():
    """Test that CMF values are within valid range (-1 to +1)."""
    print("\n=== Test: Value Ranges (-1 to +1) ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    cmf = calculate_cmf(highs, lows, closes, volumes, period=20, engine="cpu")

    # Check valid values are in -1 to +1 range
    valid_cmf = cmf[~np.isnan(cmf)]

    min_cmf = np.min(valid_cmf)
    max_cmf = np.max(valid_cmf)

    print(f"✓ CMF range: {min_cmf:.4f} to {max_cmf:.4f}")

    assert min_cmf >= -1.0, f"CMF should be >= -1, got {min_cmf}"
    assert max_cmf <= 1.0, f"CMF should be <= +1, got {max_cmf}"

    # Calculate statistics
    mean_cmf = np.mean(valid_cmf)
    std_cmf = np.std(valid_cmf)

    print(f"  - Mean: {mean_cmf:.4f}")
    print(f"  - Std Dev: {std_cmf:.4f}")
    print(f"  - Min: {min_cmf:.4f}")
    print(f"  - Max: {max_cmf:.4f}")


def test_volume_requirement():
    """Test that CMF requires volume data."""
    print("\n=== Test: Volume Requirement ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Test with valid volume
    cmf = calculate_cmf(highs, lows, closes, volumes, period=20, engine="cpu")
    assert len(cmf) == len(closes), "Should work with valid volume"
    print("✓ CMF works with volume data")

    # Test with zero volume (edge case - should work but produce different results)
    zero_volumes = np.zeros_like(volumes)
    cmf_zero = calculate_cmf(highs, lows, closes, zero_volumes, period=20, engine="cpu")

    # With zero volume, money flow volume is always 0, so CMF should be 0 or NaN
    valid_cmf_zero = cmf_zero[~np.isnan(cmf_zero)]
    if len(valid_cmf_zero) > 0:
        # Should be very close to 0 (0/0 = NaN, but we add epsilon)
        assert np.all(np.abs(valid_cmf_zero) < 0.01), \
            "Zero volume should produce CMF values near 0"
        print(f"✓ Zero volume produces CMF values near 0: {valid_cmf_zero[:5]}")


def test_buying_selling_pressure():
    """Test buying/selling pressure detection."""
    print("\n=== Test: Buying/Selling Pressure ===")

    # Create data with strong buying pressure (closes near highs)
    n = 50
    closes_high = np.linspace(100, 110, n)
    highs_high = closes_high + 0.1  # Close near high
    lows_high = closes_high - 1.0  # Low far from close
    volumes_high = np.full(n, 1_000_000.0)

    cmf_buying = calculate_cmf(highs_high, lows_high, closes_high, volumes_high, period=20, engine="cpu")
    valid_cmf_buying = cmf_buying[~np.isnan(cmf_buying)]

    print(f"Buying pressure CMF (last 5): {valid_cmf_buying[-5:]}")
    # When closes are near highs, CMF should be positive
    positive_count = np.sum(valid_cmf_buying > 0)
    print(f"✓ Buying pressure: {positive_count}/{len(valid_cmf_buying)} values > 0")
    assert positive_count > len(valid_cmf_buying) * 0.5, "Most CMF values should be positive with buying pressure"

    # Create data with strong selling pressure (closes near lows)
    closes_low = np.linspace(110, 100, n)
    highs_low = closes_low + 1.0  # High far from close
    lows_low = closes_low - 0.1  # Close near low
    volumes_low = np.full(n, 1_000_000.0)

    cmf_selling = calculate_cmf(highs_low, lows_low, closes_low, volumes_low, period=20, engine="cpu")
    valid_cmf_selling = cmf_selling[~np.isnan(cmf_selling)]

    print(f"Selling pressure CMF (last 5): {valid_cmf_selling[-5:]}")
    # When closes are near lows, CMF should be negative
    negative_count = np.sum(valid_cmf_selling < 0)
    print(f"✓ Selling pressure: {negative_count}/{len(valid_cmf_selling)} values < 0")
    assert negative_count > len(valid_cmf_selling) * 0.5, "Most CMF values should be negative with selling pressure"


def test_zero_crossing():
    """Test zero-crossing signals."""
    print("\n=== Test: Zero Crossings ===")

    # Create data that transitions from buying to selling pressure
    n = 100
    closes = np.concatenate([
        np.linspace(100, 110, 50),  # Rising (buying pressure)
        np.linspace(110, 100, 50)   # Falling (selling pressure)
    ])
    highs = closes + 0.2
    lows = closes - 0.2
    volumes = np.full(n, 1_000_000.0)

    cmf = calculate_cmf(highs, lows, closes, volumes, period=20, engine="cpu")

    # Find zero crossings
    valid_cmf = cmf[~np.isnan(cmf)]
    crossings = np.where(np.diff(np.sign(valid_cmf)) != 0)[0]

    print(f"✓ Found {len(crossings)} zero crossings")
    if len(crossings) > 0:
        print(f"  - Crossing indices: {crossings}")
        print(f"  - CMF values around first crossing: {valid_cmf[max(0, crossings[0]-2):crossings[0]+3]}")


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Test: Edge Cases ===")

    # Test 1: Minimal data (should have many NaN values)
    n_small = 25
    highs, lows, closes, volumes = generate_ohlcv_data(n=n_small)
    cmf_small = calculate_cmf(highs, lows, closes, volumes, period=20, engine="cpu")

    assert len(cmf_small) == n_small, "Should handle small datasets"
    nan_count = np.sum(np.isnan(cmf_small))
    print(f"✓ Small dataset ({n_small} bars): {nan_count} NaN values")

    # Test 2: Constant prices (no change)
    constant_closes = np.full(50, 100.0)
    constant_highs = constant_closes + 0.5
    constant_lows = constant_closes - 0.5
    constant_volumes = np.full(50, 1_000_000.0)

    cmf_constant = calculate_cmf(
        constant_highs, constant_lows, constant_closes, constant_volumes,
        period=20, engine="cpu"
    )

    # With constant typical price at midpoint, Money Flow Multiplier should be 0
    valid_cmf_constant = cmf_constant[~np.isnan(cmf_constant)]
    if len(valid_cmf_constant) > 0:
        print(f"✓ Constant prices: CMF values = {valid_cmf_constant[:5]}")
        # Should be close to 0 (close is at midpoint of high-low range)
        assert np.all(np.abs(valid_cmf_constant) < 0.01), "CMF should be near 0 for midpoint closes"

    # Test 3: Different periods
    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    cmf_5 = calculate_cmf(highs, lows, closes, volumes, period=5, engine="cpu")
    cmf_50 = calculate_cmf(highs, lows, closes, volumes, period=50, engine="cpu")

    assert len(cmf_5) == len(closes), "Short period should work"
    assert len(cmf_50) == len(closes), "Long period should work"

    # Short period should have fewer NaN values
    nan_5 = np.sum(np.isnan(cmf_5))
    nan_50 = np.sum(np.isnan(cmf_50))

    print(f"✓ Period=5: {nan_5} NaN values")
    print(f"✓ Period=50: {nan_50} NaN values")
    assert nan_5 < nan_50, "Shorter period should have fewer NaN values"

    # Test 4: High equals low (doji bars)
    doji_closes = np.full(50, 100.0)
    doji_highs = doji_closes  # High = close = low
    doji_lows = doji_closes
    doji_volumes = np.full(50, 1_000_000.0)

    cmf_doji = calculate_cmf(doji_highs, doji_lows, doji_closes, doji_volumes, period=20, engine="cpu")
    valid_cmf_doji = cmf_doji[~np.isnan(cmf_doji)]

    if len(valid_cmf_doji) > 0:
        # When H=L=C, division by zero is handled with epsilon
        print(f"✓ Doji bars (H=L=C): CMF values = {valid_cmf_doji[:5]}")
        # Values should be finite (not inf/nan) due to epsilon
        assert np.all(np.isfinite(valid_cmf_doji)), "Should handle H=L=C case without inf/nan"


def test_parameter_validation():
    """Test parameter validation."""
    print("\n=== Test: Parameter Validation ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Test invalid period
    try:
        cmf = calculate_cmf(highs, lows, closes, volumes, period=0, engine="cpu")
        assert False, "Should raise ValueError for period=0"
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for period=0: {e}")

    # Test negative period
    try:
        cmf = calculate_cmf(highs, lows, closes, volumes, period=-5, engine="cpu")
        assert False, "Should raise ValueError for negative period"
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for negative period: {e}")

    # Test insufficient data
    try:
        short_highs = np.array([100, 101, 102])
        short_lows = np.array([99, 100, 101])
        short_closes = np.array([100, 101, 102])
        short_volumes = np.array([1000, 1000, 1000])
        cmf = calculate_cmf(short_highs, short_lows, short_closes, short_volumes, period=20, engine="cpu")
        assert False, "Should raise ValueError for insufficient data"
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for insufficient data: {e}")

    # Test mismatched array lengths
    try:
        mismatched_volumes = np.array([1000, 1000])  # Different length
        cmf = calculate_cmf(highs, lows, closes, mismatched_volumes, period=20, engine="cpu")
        assert False, "Should raise ValueError for mismatched lengths"
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for mismatched lengths: {e}")


def test_known_values():
    """Test against known CMF values."""
    print("\n=== Test: Known Values ===")

    # Simple test case with known values
    # Create a scenario where we can calculate CMF manually

    # Scenario: 5 bars with known OHLCV
    highs = np.array([102, 104, 103, 106, 105], dtype=np.float64)
    lows = np.array([98, 100, 101, 103, 102], dtype=np.float64)
    closes = np.array([101, 103, 102, 105, 103], dtype=np.float64)
    volumes = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.float64)

    cmf = calculate_cmf(highs, lows, closes, volumes, period=3, engine="cpu")

    print(f"✓ CMF calculated for known values: {cmf}")

    # Manually verify one value (bar 2, using bars 0-2)
    # Bar 0: MF_Mult = (2*101 - 102 - 98) / (102 - 98) = 2 / 4 = 0.5
    # Bar 1: MF_Mult = (2*103 - 104 - 100) / (104 - 100) = 2 / 4 = 0.5
    # Bar 2: MF_Mult = (2*102 - 103 - 101) / (103 - 101) = 0 / 2 = 0.0
    # Sum(MF_Volume) = 0.5*1000 + 0.5*1000 + 0.0*1000 = 1000
    # Sum(Volume) = 3000
    # CMF[2] = 1000 / 3000 = 0.333...

    expected_cmf_2 = 1.0 / 3.0
    actual_cmf_2 = cmf[2]

    print(f"  - Expected CMF[2]: {expected_cmf_2:.4f}")
    print(f"  - Actual CMF[2]: {actual_cmf_2:.4f}")

    # Allow small tolerance for floating point arithmetic
    assert np.abs(actual_cmf_2 - expected_cmf_2) < 0.01, \
        f"CMF[2] mismatch: expected {expected_cmf_2}, got {actual_cmf_2}"

    print("✓ Known value test passed")


def test_performance():
    """Test performance on different data sizes."""
    print("\n=== Test: Performance ===")

    import time

    sizes = [1_000, 10_000, 100_000]

    for size in sizes:
        # Generate data
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(size) * 0.5)
        highs = closes + np.abs(np.random.randn(size) * 0.3)
        lows = closes - np.abs(np.random.randn(size) * 0.3)
        volumes = np.abs(np.random.randn(size) * 1_000_000)

        # Measure CPU performance
        start = time.perf_counter()
        cmf = calculate_cmf(highs, lows, closes, volumes, period=20, engine="cpu")
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        # Verify result
        assert len(cmf) == size, f"Result length should match input for size {size}"
        valid_count = np.sum(~np.isnan(cmf))

        print(f"✓ Size {size:>7,}: {elapsed:>6.2f} ms ({valid_count:>7,} valid values)")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  Chaikin Money Flow (CMF) - Comprehensive Test Suite")
    print("=" * 80)

    try:
        test_basic_calculation()
        test_value_ranges()
        test_volume_requirement()
        test_buying_selling_pressure()
        test_zero_crossing()
        test_edge_cases()
        test_parameter_validation()
        test_known_values()
        test_performance()

        print("\n" + "=" * 80)
        print("  ✓ ALL CMF TESTS PASSED!")
        print("=" * 80)
        print("\nCMF implementation is correct and ready for use.")

        return 0

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("  ✗ TEST FAILED")
        print("=" * 80)
        print(f"AssertionError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print("\n" + "=" * 80)
        print("  ✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

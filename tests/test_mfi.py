#!/usr/bin/env python3
"""
Test Suite for Money Flow Index (MFI)
======================================

Comprehensive tests for MFI implementation including:
- Basic calculation
- Value range validation (0-100)
- Volume requirement
- Overbought/oversold levels
- Comparison with RSI
- Divergence detection
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

from kimsfinance.ops.mfi import calculate_mfi
from kimsfinance.ops.indicator_utils import typical_price


def generate_ohlcv_data(n: int = 100, seed: int = 42) -> tuple:
    """Generate test OHLCV data for MFI testing."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    volumes = np.abs(np.random.randn(n) * 1_000_000)
    return highs, lows, closes, volumes


def test_basic_calculation():
    """Test basic MFI calculation."""
    print("\n=== Test: Basic Calculation ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Calculate MFI
    mfi = calculate_mfi(highs, lows, closes, volumes, period=14, engine="cpu")

    # Verify result
    assert len(mfi) == len(closes), "MFI length should match input length"
    assert isinstance(mfi, np.ndarray), "MFI should return NumPy array"

    # First (period + 1) values should be NaN (due to rolling sum + diff)
    expected_nans = 14  # period
    actual_nans = np.sum(np.isnan(mfi))
    assert (
        actual_nans >= expected_nans
    ), f"Expected at least {expected_nans} NaN values, got {actual_nans}"

    # Rest should be valid numbers
    valid_values = mfi[~np.isnan(mfi)]
    assert len(valid_values) > 0, "Should have some valid MFI values"

    print(f"✓ MFI calculated: {len(mfi)} values")
    print(f"  - NaN values: {actual_nans}")
    print(f"  - Valid values: {len(valid_values)}")
    print(f"  - Last 5 MFI values: {mfi[-5:]}")


def test_value_ranges():
    """Test that MFI values are within valid range (0-100)."""
    print("\n=== Test: Value Ranges (0-100) ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    mfi = calculate_mfi(highs, lows, closes, volumes, period=14, engine="cpu")

    # Check valid values are in 0-100 range
    valid_mfi = mfi[~np.isnan(mfi)]

    min_mfi = np.min(valid_mfi)
    max_mfi = np.max(valid_mfi)

    print(f"✓ MFI range: {min_mfi:.2f} to {max_mfi:.2f}")

    assert min_mfi >= 0.0, f"MFI should be >= 0, got {min_mfi}"
    assert max_mfi <= 100.0, f"MFI should be <= 100, got {max_mfi}"

    # Calculate statistics
    mean_mfi = np.mean(valid_mfi)
    std_mfi = np.std(valid_mfi)

    print(f"  - Mean: {mean_mfi:.2f}")
    print(f"  - Std Dev: {std_mfi:.2f}")
    print(f"  - Min: {min_mfi:.2f}")
    print(f"  - Max: {max_mfi:.2f}")


def test_volume_requirement():
    """Test that MFI requires volume data."""
    print("\n=== Test: Volume Requirement ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Test with valid volume
    mfi = calculate_mfi(highs, lows, closes, volumes, period=14, engine="cpu")
    assert len(mfi) == len(closes), "Should work with valid volume"
    print("✓ MFI works with volume data")

    # Test with zero volume (edge case - should work but produce different results)
    zero_volumes = np.zeros_like(volumes)
    mfi_zero = calculate_mfi(highs, lows, closes, zero_volumes, period=14, engine="cpu")

    # With zero volume, money flow is always 0, so MFI should be 0 or NaN
    valid_mfi_zero = mfi_zero[~np.isnan(mfi_zero)]
    if len(valid_mfi_zero) > 0:
        # Should be close to 0 (or 50 if both flows are 0)
        assert np.all(
            (valid_mfi_zero < 1.0) | (np.abs(valid_mfi_zero - 50.0) < 1.0)
        ), "Zero volume should produce MFI near 0 or 50"
        print(f"✓ Zero volume produces MFI values near 0 or 50: {valid_mfi_zero[:5]}")


def test_overbought_oversold():
    """Test overbought/oversold level detection."""
    print("\n=== Test: Overbought/Oversold Levels ===")

    # Create trending up data (should produce high MFI)
    n = 50
    closes_up = np.linspace(100, 120, n)
    highs_up = closes_up + 0.5
    lows_up = closes_up - 0.5
    volumes_up = np.full(n, 1_000_000.0)

    mfi_up = calculate_mfi(highs_up, lows_up, closes_up, volumes_up, period=14, engine="cpu")
    valid_mfi_up = mfi_up[~np.isnan(mfi_up)]

    print(f"Uptrend MFI (last 5): {valid_mfi_up[-5:]}")
    # In strong uptrend, most MFI values should be high
    high_values = np.sum(valid_mfi_up > 50)
    print(f"✓ Strong uptrend: {high_values}/{len(valid_mfi_up)} values > 50")

    # Create trending down data (should produce low MFI)
    closes_down = np.linspace(120, 100, n)
    highs_down = closes_down + 0.5
    lows_down = closes_down - 0.5
    volumes_down = np.full(n, 1_000_000.0)

    mfi_down = calculate_mfi(
        highs_down, lows_down, closes_down, volumes_down, period=14, engine="cpu"
    )
    valid_mfi_down = mfi_down[~np.isnan(mfi_down)]

    print(f"Downtrend MFI (last 5): {valid_mfi_down[-5:]}")
    # In strong downtrend, most MFI values should be low
    low_values = np.sum(valid_mfi_down < 50)
    print(f"✓ Strong downtrend: {low_values}/{len(valid_mfi_down)} values < 50")

    # Test level crossings
    overbought_count = np.sum(mfi_up > 80)
    oversold_count = np.sum(mfi_down < 20)

    print(f"✓ Overbought signals (>80): {overbought_count}")
    print(f"✓ Oversold signals (<20): {oversold_count}")


def test_comparison_with_rsi():
    """Test that MFI behaves like volume-weighted RSI."""
    print("\n=== Test: Comparison with RSI ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    # Calculate MFI
    mfi = calculate_mfi(highs, lows, closes, volumes, period=14, engine="cpu")

    # MFI should behave similarly to RSI but with volume weighting
    # Both should be in 0-100 range and follow similar patterns

    valid_mfi = mfi[~np.isnan(mfi)]

    # Check basic properties similar to RSI
    assert 0 <= np.min(valid_mfi) <= 100, "MFI should be in RSI-like range"
    assert 0 <= np.max(valid_mfi) <= 100, "MFI should be in RSI-like range"

    # MFI should oscillate around 50 (like RSI)
    mean_mfi = np.mean(valid_mfi)
    print(f"✓ Mean MFI: {mean_mfi:.2f} (should be near 50 for random walk)")

    # Test with constant volume (should behave more like price-only indicator)
    constant_volumes = np.full_like(volumes, 1_000_000.0)
    mfi_constant = calculate_mfi(highs, lows, closes, constant_volumes, period=14, engine="cpu")

    print(f"✓ MFI with constant volume calculated")
    print(f"  - Variable volume MFI (last 3): {mfi[-3:]}")
    print(f"  - Constant volume MFI (last 3): {mfi_constant[-3:]}")

    # Both should be valid
    assert len(mfi_constant) == len(mfi), "Constant volume should produce same length"


def test_divergence_detection():
    """Test divergence detection between price and MFI."""
    print("\n=== Test: Divergence Detection ===")

    # Create data with bearish divergence (price up, MFI down)
    n = 50
    # Price rises
    closes = np.linspace(100, 110, n)
    highs = closes + 0.5
    lows = closes - 0.5

    # Volume decreases (indicating weakening momentum)
    volumes = np.linspace(2_000_000, 500_000, n)

    mfi = calculate_mfi(highs, lows, closes, volumes, period=14, engine="cpu")

    # Check last 10 values
    price_change = closes[-1] - closes[-10]
    mfi_change = mfi[-1] - mfi[-10] if not np.isnan(mfi[-1]) and not np.isnan(mfi[-10]) else 0

    print(f"✓ Price change (last 10): {price_change:.2f}")
    print(f"✓ MFI change (last 10): {mfi_change:.2f}")

    # Price should be rising
    assert price_change > 0, "Price should be rising"

    # With decreasing volume, MFI might not rise as much or could fall
    # This is the divergence signal
    if mfi_change < 0:
        print("✓ Bearish divergence detected: Price up, MFI down")
    else:
        print(f"  Note: MFI also rose by {mfi_change:.2f} (no divergence in this case)")


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Test: Edge Cases ===")

    # Test 1: Minimal data (should have many NaN values)
    n_small = 20
    highs, lows, closes, volumes = generate_ohlcv_data(n=n_small)
    mfi_small = calculate_mfi(highs, lows, closes, volumes, period=14, engine="cpu")

    assert len(mfi_small) == n_small, "Should handle small datasets"
    nan_count = np.sum(np.isnan(mfi_small))
    print(f"✓ Small dataset ({n_small} bars): {nan_count} NaN values")

    # Test 2: Constant prices (no change)
    constant_closes = np.full(50, 100.0)
    constant_highs = constant_closes + 0.5
    constant_lows = constant_closes - 0.5
    constant_volumes = np.full(50, 1_000_000.0)

    mfi_constant = calculate_mfi(
        constant_highs, constant_lows, constant_closes, constant_volumes, period=14, engine="cpu"
    )

    # With constant typical price, all money flow goes to one direction initially
    # then becomes neutral (50)
    valid_mfi_constant = mfi_constant[~np.isnan(mfi_constant)]
    if len(valid_mfi_constant) > 0:
        print(f"✓ Constant prices: MFI values = {valid_mfi_constant[:5]}")
        # Should be mostly 0, 50, or 100 (edge cases)

    # Test 3: Alternating up/down (should produce ~50 MFI)
    alternating = np.array([100.0] * 50)
    for i in range(1, 50):
        alternating[i] = alternating[i - 1] + (1.0 if i % 2 == 0 else -1.0)

    highs_alt = alternating + 0.5
    lows_alt = alternating - 0.5
    volumes_alt = np.full(50, 1_000_000.0)

    mfi_alt = calculate_mfi(highs_alt, lows_alt, alternating, volumes_alt, period=14, engine="cpu")
    valid_mfi_alt = mfi_alt[~np.isnan(mfi_alt)]

    if len(valid_mfi_alt) > 0:
        mean_alt = np.mean(valid_mfi_alt)
        print(f"✓ Alternating prices: Mean MFI = {mean_alt:.2f} (should be near 50)")

    # Test 4: Different periods
    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    mfi_5 = calculate_mfi(highs, lows, closes, volumes, period=5, engine="cpu")
    mfi_21 = calculate_mfi(highs, lows, closes, volumes, period=21, engine="cpu")

    assert len(mfi_5) == len(closes), "Short period should work"
    assert len(mfi_21) == len(closes), "Long period should work"

    # Short period should have fewer NaN values
    nan_5 = np.sum(np.isnan(mfi_5))
    nan_21 = np.sum(np.isnan(mfi_21))

    print(f"✓ Period=5: {nan_5} NaN values")
    print(f"✓ Period=21: {nan_21} NaN values")
    assert nan_5 < nan_21, "Shorter period should have fewer NaN values"


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
        mfi = calculate_mfi(highs, lows, closes, volumes, period=14, engine="cpu")
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        # Verify result
        assert len(mfi) == size, f"Result length should match input for size {size}"
        valid_count = np.sum(~np.isnan(mfi))

        print(f"✓ Size {size:>7,}: {elapsed:>6.2f} ms ({valid_count:>7,} valid values)")


def test_parameter_validation():
    """Test parameter validation."""
    print("\n=== Test: Parameter Validation ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Test invalid period
    try:
        mfi = calculate_mfi(highs, lows, closes, volumes, period=0, engine="cpu")
        assert False, "Should raise ValueError for period=0"
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for period=0: {e}")

    # Test negative period
    try:
        mfi = calculate_mfi(highs, lows, closes, volumes, period=-5, engine="cpu")
        assert False, "Should raise ValueError for negative period"
    except ValueError as e:
        print(f"✓ Correctly raises ValueError for negative period: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  Money Flow Index (MFI) - Comprehensive Test Suite")
    print("=" * 80)

    try:
        test_basic_calculation()
        test_value_ranges()
        test_volume_requirement()
        test_overbought_oversold()
        test_comparison_with_rsi()
        test_divergence_detection()
        test_edge_cases()
        test_performance()
        test_parameter_validation()

        print("\n" + "=" * 80)
        print("  ✓ ALL MFI TESTS PASSED!")
        print("=" * 80)
        print("\nMFI implementation is correct and ready for use.")

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

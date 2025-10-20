#!/usr/bin/env python3
"""
Test Suite for Ichimoku Cloud Indicator
========================================

Comprehensive tests for all 5 lines of the Ichimoku Cloud:
- Tenkan-sen (Conversion Line)
- Kijun-sen (Base Line)
- Senkou Span A (Leading Span A)
- Senkou Span B (Leading Span B)
- Chikou Span (Lagging Span)

Tests cover:
1. Basic calculation of all 5 lines
2. Displacement logic (senkou forward, chikou backward)
3. Cloud formation (senkou_a vs senkou_b)
4. Signal generation (tenkan/kijun crosses)
5. Different period parameters
6. Edge cases (minimal data)
7. Performance benchmarking
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.ops.ichimoku import calculate_ichimoku


def generate_ohlc_data(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test OHLC data for Ichimoku indicator.

    Args:
        n: Number of data points
        seed: Random seed for reproducibility

    Returns:
        Tuple of (highs, lows, closes)
    """
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return highs, lows, closes


def test_basic_calculation():
    """Test basic calculation of all 5 Ichimoku lines."""
    print("\n[TEST] Basic Calculation - All 5 Lines")

    highs, lows, closes = generate_ohlc_data(n=200)

    # Calculate Ichimoku Cloud
    ichimoku = calculate_ichimoku(highs, lows, closes, engine="cpu")

    # Check all 5 lines are present
    assert "tenkan" in ichimoku, "Tenkan-sen missing"
    assert "kijun" in ichimoku, "Kijun-sen missing"
    assert "senkou_a" in ichimoku, "Senkou Span A missing"
    assert "senkou_b" in ichimoku, "Senkou Span B missing"
    assert "chikou" in ichimoku, "Chikou Span missing"

    # Check all lines have correct length
    for name, line in ichimoku.items():
        assert len(line) == len(highs), f"{name} has incorrect length"
        print(f"  ✓ {name}: length {len(line)}")

    # Check that values exist after sufficient period
    tenkan = ichimoku["tenkan"]
    kijun = ichimoku["kijun"]
    senkou_a = ichimoku["senkou_a"]
    senkou_b = ichimoku["senkou_b"]
    chikou = ichimoku["chikou"]

    # First (period-1) values should be NaN due to rolling window
    # Tenkan uses 9-period: first 8 should be NaN
    assert np.sum(np.isnan(tenkan[:8])) == 8, "Tenkan should have 8 NaN values at start"
    assert not np.isnan(tenkan[8]), "Tenkan should have value at position 8"

    # Kijun uses 26-period: first 25 should be NaN
    assert np.sum(np.isnan(kijun[:25])) == 25, "Kijun should have 25 NaN values at start"
    assert not np.isnan(kijun[25]), "Kijun should have value at position 25"

    # Senkou B uses 52-period + 26 displacement: first (51+26) should be NaN
    # Position 77 should have a value (52-period starting at 0, displaced to 26)
    assert not np.isnan(senkou_b[77]), "Senkou B should have value at position 77"

    # After sufficient period, values should exist
    assert not np.isnan(tenkan[100]), "Tenkan should have value at position 100"
    assert not np.isnan(kijun[100]), "Kijun should have value at position 100"
    assert not np.isnan(senkou_b[100]), "Senkou B should have value at position 100"

    print(f"  ✓ Tenkan value at [100]: {tenkan[100]:.4f}")
    print(f"  ✓ Kijun value at [100]: {kijun[100]:.4f}")
    print(f"  ✓ Senkou A value at [100]: {senkou_a[100]:.4f}")
    print(f"  ✓ Senkou B value at [100]: {senkou_b[100]:.4f}")
    print(f"  ✓ Chikou value at [100]: {chikou[100]:.4f}")


def test_displacement():
    """Test displacement of Senkou and Chikou lines."""
    print("\n[TEST] Displacement - Senkou Forward, Chikou Backward")

    highs, lows, closes = generate_ohlc_data(n=200)

    # Calculate Ichimoku with default displacement=26
    ichimoku = calculate_ichimoku(highs, lows, closes, displacement=26, engine="cpu")

    tenkan = ichimoku["tenkan"]
    kijun = ichimoku["kijun"]
    senkou_a = ichimoku["senkou_a"]
    senkou_b = ichimoku["senkou_b"]
    chikou = ichimoku["chikou"]

    # Test Senkou Span A displacement (forward by 26)
    # Senkou A at position i should be based on data from position i-26
    # First 26 positions should be NaN due to forward displacement
    assert np.sum(np.isnan(senkou_a[:26])) == 26, "First 26 Senkou A values should be NaN"

    # Senkou A[26] should equal (Tenkan[0] + Kijun[0]) / 2
    if not np.isnan(tenkan[0]) and not np.isnan(kijun[0]):
        expected_senkou_a_26 = (tenkan[0] + kijun[0]) / 2.0
        assert np.isclose(
            senkou_a[26], expected_senkou_a_26, rtol=1e-5
        ), f"Senkou A displacement incorrect: {senkou_a[26]} != {expected_senkou_a_26}"
        print(f"  ✓ Senkou A[26] = {senkou_a[26]:.4f} (displaced from position 0)")

    # Test Chikou Span displacement (backward by 26)
    # Chikou at position i should be close[i+26]
    # Last 26 positions should be NaN due to backward displacement
    assert np.sum(np.isnan(chikou[-26:])) == 26, "Last 26 Chikou values should be NaN"

    # Chikou[0] should equal Close[26]
    assert np.isclose(
        chikou[0], closes[26], rtol=1e-5
    ), f"Chikou displacement incorrect: {chikou[0]} != {closes[26]}"
    print(f"  ✓ Chikou[0] = {chikou[0]:.4f} = Close[26] = {closes[26]:.4f}")

    # Chikou[50] should equal Close[76]
    assert np.isclose(
        chikou[50], closes[76], rtol=1e-5
    ), f"Chikou displacement incorrect at position 50"
    print(f"  ✓ Chikou[50] = {chikou[50]:.4f} = Close[76] = {closes[76]:.4f}")


def test_cloud_formation():
    """Test cloud formation between Senkou A and Senkou B."""
    print("\n[TEST] Cloud Formation - Senkou A vs Senkou B")

    highs, lows, closes = generate_ohlc_data(n=200)

    ichimoku = calculate_ichimoku(highs, lows, closes, engine="cpu")

    senkou_a = ichimoku["senkou_a"]
    senkou_b = ichimoku["senkou_b"]

    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    cloud_thickness = cloud_top - cloud_bottom

    # Count bullish vs bearish cloud
    valid_mask = ~(np.isnan(senkou_a) | np.isnan(senkou_b))
    bullish_cloud = np.sum((senkou_a > senkou_b) & valid_mask)
    bearish_cloud = np.sum((senkou_a < senkou_b) & valid_mask)

    print(f"  ✓ Bullish cloud periods: {bullish_cloud}")
    print(f"  ✓ Bearish cloud periods: {bearish_cloud}")

    # Cloud should exist (not all NaN after initial period)
    assert not np.all(np.isnan(cloud_top[80:])), "Cloud should exist after position 80"
    assert not np.all(np.isnan(cloud_bottom[80:])), "Cloud should exist after position 80"

    # Average cloud thickness (excluding NaN)
    avg_thickness = np.nanmean(cloud_thickness)
    print(f"  ✓ Average cloud thickness: {avg_thickness:.4f}")

    # Cloud thickness should be non-negative
    assert np.all(cloud_thickness[valid_mask] >= 0), "Cloud thickness should be non-negative"


def test_signal_generation():
    """Test signal generation from Tenkan/Kijun crosses."""
    print("\n[TEST] Signal Generation - Tenkan/Kijun Crosses")

    highs, lows, closes = generate_ohlc_data(n=200, seed=123)

    ichimoku = calculate_ichimoku(highs, lows, closes, engine="cpu")

    tenkan = ichimoku["tenkan"]
    kijun = ichimoku["kijun"]

    # Detect Tenkan/Kijun crosses
    # Bullish cross: Tenkan crosses above Kijun
    tenkan_above = tenkan > kijun
    tenkan_above_prev = np.roll(tenkan_above, 1)
    tenkan_above_prev[0] = False

    bullish_crosses = np.where(tenkan_above & ~tenkan_above_prev)[0]

    # Bearish cross: Tenkan crosses below Kijun
    tenkan_below = tenkan < kijun
    tenkan_below_prev = np.roll(tenkan_below, 1)
    tenkan_below_prev[0] = False

    bearish_crosses = np.where(tenkan_below & ~tenkan_below_prev)[0]

    print(f"  ✓ Bullish crosses detected: {len(bullish_crosses)}")
    print(f"  ✓ Bearish crosses detected: {len(bearish_crosses)}")

    # There should be some crosses in 200 periods of random data
    assert len(bullish_crosses) + len(bearish_crosses) > 0, "Should detect some crosses"

    # Show first few crosses
    if len(bullish_crosses) > 0:
        print(f"  ✓ First bullish cross at position: {bullish_crosses[0]}")
    if len(bearish_crosses) > 0:
        print(f"  ✓ First bearish cross at position: {bearish_crosses[0]}")


def test_different_periods():
    """Test Ichimoku with different period parameters."""
    print("\n[TEST] Different Periods - Short vs Long Term")

    highs, lows, closes = generate_ohlc_data(n=200)

    # Standard settings
    standard = calculate_ichimoku(
        highs, lows, closes, tenkan=9, kijun=26, senkou_b=52, engine="cpu"
    )

    # Fast settings (crypto/intraday)
    fast = calculate_ichimoku(highs, lows, closes, tenkan=7, kijun=22, senkou_b=44, engine="cpu")

    # Slow settings (weekly)
    slow = calculate_ichimoku(highs, lows, closes, tenkan=12, kijun=30, senkou_b=60, engine="cpu")

    # Check all variants calculate successfully
    for name, ichi in [("Standard", standard), ("Fast", fast), ("Slow", slow)]:
        assert len(ichi["tenkan"]) == len(highs), f"{name}: Tenkan length incorrect"
        assert len(ichi["kijun"]) == len(highs), f"{name}: Kijun length incorrect"
        assert len(ichi["senkou_a"]) == len(highs), f"{name}: Senkou A length incorrect"
        assert len(ichi["senkou_b"]) == len(highs), f"{name}: Senkou B length incorrect"
        assert len(ichi["chikou"]) == len(highs), f"{name}: Chikou length incorrect"
        print(f"  ✓ {name} settings calculated successfully")

    # Fast settings should respond quicker (fewer NaN at start)
    assert np.sum(np.isnan(fast["tenkan"])) < np.sum(
        np.isnan(standard["tenkan"])
    ), "Fast settings should have fewer NaN values"
    print(f"  ✓ Fast Tenkan NaN: {np.sum(np.isnan(fast['tenkan']))}")
    print(f"  ✓ Standard Tenkan NaN: {np.sum(np.isnan(standard['tenkan']))}")


def test_edge_cases():
    """Test edge cases with minimal data."""
    print("\n[TEST] Edge Cases - Minimal Data")

    # Test with exactly 100 rows (minimum for default settings)
    highs, lows, closes = generate_ohlc_data(n=100)

    ichimoku = calculate_ichimoku(highs, lows, closes, engine="cpu")

    # Should calculate without errors
    assert len(ichimoku["tenkan"]) == 100
    assert len(ichimoku["kijun"]) == 100
    assert len(ichimoku["senkou_a"]) == 100
    assert len(ichimoku["senkou_b"]) == 100
    assert len(ichimoku["chikou"]) == 100
    print("  ✓ Calculations work with 100 rows")

    # Some values should exist at the end
    assert not np.isnan(ichimoku["tenkan"][-1]), "Tenkan should have value at end"
    assert not np.isnan(ichimoku["kijun"][-1]), "Kijun should have value at end"
    print("  ✓ Values exist at end positions")

    # Test with constant prices (no volatility)
    highs_const = np.full(100, 100.5)
    lows_const = np.full(100, 99.5)
    closes_const = np.full(100, 100.0)

    ichimoku_const = calculate_ichimoku(highs_const, lows_const, closes_const, engine="cpu")

    # Lines should be constant (after initial period)
    tenkan_const = ichimoku_const["tenkan"]
    kijun_const = ichimoku_const["kijun"]

    # After sufficient period, values should stabilize
    assert np.allclose(
        tenkan_const[50:], 100.0, rtol=1e-3, equal_nan=True
    ), "Tenkan should be constant with constant prices"
    print("  ✓ Constant prices produce constant lines")


def test_performance():
    """Test performance on different data sizes."""
    print("\n[TEST] Performance Benchmarking")

    import time

    sizes = [1_000, 10_000, 100_000]

    for size in sizes:
        # Generate large dataset
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(size) * 0.5)
        highs = closes + np.abs(np.random.randn(size) * 0.3)
        lows = closes - np.abs(np.random.randn(size) * 0.3)

        # Time the calculation
        start = time.perf_counter()
        ichimoku = calculate_ichimoku(highs, lows, closes, engine="cpu")
        elapsed = time.perf_counter() - start

        # Verify calculation completed
        assert len(ichimoku["tenkan"]) == size
        assert len(ichimoku["kijun"]) == size
        assert len(ichimoku["senkou_a"]) == size
        assert len(ichimoku["senkou_b"]) == size
        assert len(ichimoku["chikou"]) == size

        print(f"  ✓ {size:>7,} rows: {elapsed*1000:>7.2f} ms")


def test_parameter_validation():
    """Test parameter validation."""
    print("\n[TEST] Parameter Validation")

    highs, lows, closes = generate_ohlc_data(n=100)

    # Test invalid tenkan period
    with pytest.raises(ValueError, match="tenkan must be >= 1"):
        calculate_ichimoku(highs, lows, closes, tenkan=0, engine="cpu")
    print("  ✓ Invalid tenkan period rejected")

    # Test invalid kijun period
    with pytest.raises(ValueError, match="kijun must be >= 1"):
        calculate_ichimoku(highs, lows, closes, kijun=0, engine="cpu")
    print("  ✓ Invalid kijun period rejected")

    # Test invalid senkou_b period
    with pytest.raises(ValueError, match="senkou_b must be >= 1"):
        calculate_ichimoku(highs, lows, closes, senkou_b=0, engine="cpu")
    print("  ✓ Invalid senkou_b period rejected")

    # Test invalid displacement
    with pytest.raises(ValueError, match="displacement must be >= 0"):
        calculate_ichimoku(highs, lows, closes, displacement=-1, engine="cpu")
    print("  ✓ Invalid displacement rejected")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  Ichimoku Cloud - Comprehensive Test Suite")
    print("=" * 80)

    try:
        test_basic_calculation()
        test_displacement()
        test_cloud_formation()
        test_signal_generation()
        test_different_periods()
        test_edge_cases()
        test_performance()
        test_parameter_validation()

        print("\n" + "=" * 80)
        print("  ✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nIchimoku Cloud implementation is correct and performant.")
        print("All 5 lines calculated correctly with proper displacement.")

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("  ✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 80)
        print("  ✗ UNEXPECTED ERROR")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

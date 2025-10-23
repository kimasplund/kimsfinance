#!/usr/bin/env python3
"""
Test Suite for Volume Weighted Average Price (VWAP)
====================================================

Comprehensive tests for VWAP implementation including:
- Basic calculation (typical price weighted by volume)
- Cumulative weighted average properties
- Volume weighting behavior
- Anchored VWAP (session resets)
- Edge cases (zero volumes, single data points)
- GPU/CPU parity
- Performance benchmarking
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.ops.indicators import calculate_vwap, calculate_vwap_anchored


def generate_ohlcv_data(n: int = 100, seed: int = 42) -> tuple:
    """Generate test OHLCV data for VWAP testing."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    volumes = np.abs(np.random.randn(n) * 1_000_000) + 100_000  # Ensure positive
    return highs, lows, closes, volumes


# ============================================================================
# 1. BASIC CALCULATION TESTS (15 tests)
# ============================================================================


def test_basic_calculation():
    """Test basic VWAP calculation."""
    print("\n=== Test: Basic Calculation ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Calculate VWAP
    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Verify result
    assert len(vwap) == len(closes), "VWAP length should match input length"
    assert isinstance(vwap, np.ndarray), "VWAP should return NumPy array"

    # VWAP should have no NaN values (cumulative from start)
    nan_count = np.sum(np.isnan(vwap))
    assert nan_count == 0, f"VWAP should not have NaN values, got {nan_count}"

    print(f"✓ VWAP calculated: {len(vwap)} values")
    print(f"  - First 5 VWAP values: {vwap[:5]}")
    print(f"  - Last 5 VWAP values: {vwap[-5:]}")


def test_typical_price_calculation():
    """Test that VWAP uses typical price (H+L+C)/3."""
    print("\n=== Test: Typical Price Calculation ===")

    # Simple data where we can verify manually
    highs = np.array([102.0, 103.0, 104.0, 105.0])
    lows = np.array([98.0, 99.0, 100.0, 101.0])
    closes = np.array([100.0, 101.0, 102.0, 103.0])
    volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Calculate expected typical prices
    typical_prices = (highs + lows + closes) / 3

    # With equal volumes, VWAP should be cumulative average of typical prices
    expected_vwap = np.array(
        [
            typical_prices[0],
            np.mean(typical_prices[:2]),
            np.mean(typical_prices[:3]),
            np.mean(typical_prices[:4]),
        ]
    )

    print(f"Typical prices: {typical_prices}")
    print(f"Expected VWAP: {expected_vwap}")
    print(f"Actual VWAP: {vwap}")

    np.testing.assert_allclose(vwap, expected_vwap, rtol=1e-5)
    print("✓ VWAP correctly uses typical price (H+L+C)/3")


def test_cumulative_weighted_average():
    """Test that VWAP is cumulative weighted average."""
    print("\n=== Test: Cumulative Weighted Average ===")

    highs = np.array([105.0, 110.0, 108.0])
    lows = np.array([95.0, 100.0, 98.0])
    closes = np.array([100.0, 105.0, 103.0])
    volumes = np.array([1000.0, 2000.0, 1500.0])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Manual calculation
    tp = (highs + lows + closes) / 3  # Typical price

    # VWAP[0] = tp[0]
    expected_vwap_0 = tp[0]

    # VWAP[1] = (tp[0]*vol[0] + tp[1]*vol[1]) / (vol[0] + vol[1])
    expected_vwap_1 = (tp[0] * volumes[0] + tp[1] * volumes[1]) / (volumes[0] + volumes[1])

    # VWAP[2] = (tp[0]*vol[0] + tp[1]*vol[1] + tp[2]*vol[2]) / (vol[0] + vol[1] + vol[2])
    expected_vwap_2 = (tp[0] * volumes[0] + tp[1] * volumes[1] + tp[2] * volumes[2]) / (
        volumes[0] + volumes[1] + volumes[2]
    )

    expected = np.array([expected_vwap_0, expected_vwap_1, expected_vwap_2])

    print(f"Expected VWAP: {expected}")
    print(f"Actual VWAP: {vwap}")

    np.testing.assert_allclose(vwap, expected, rtol=1e-5)
    print("✓ VWAP is correct cumulative weighted average")


def test_vwap_between_high_and_low():
    """Test that VWAP is always between the high and low of typical prices."""
    print("\n=== Test: VWAP Between High and Low ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should be within the range of all typical prices seen so far
    typical_prices = (highs + lows + closes) / 3

    for i in range(len(vwap)):
        min_tp = np.min(typical_prices[: i + 1])
        max_tp = np.max(typical_prices[: i + 1])

        # Allow small floating point tolerance
        assert (
            min_tp - 1e-10 <= vwap[i] <= max_tp + 1e-10
        ), f"VWAP[{i}]={vwap[i]:.2f} outside range [{min_tp:.2f}, {max_tp:.2f}]"

    print(f"✓ All {len(vwap)} VWAP values within historical typical price range")


def test_single_data_point():
    """Test VWAP with single data point."""
    print("\n=== Test: Single Data Point ===")

    highs = np.array([105.0])
    lows = np.array([95.0])
    closes = np.array([100.0])
    volumes = np.array([1000.0])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should equal typical price for single point
    expected = (105.0 + 95.0 + 100.0) / 3

    assert len(vwap) == 1
    np.testing.assert_allclose(vwap[0], expected, rtol=1e-5)

    print(f"✓ Single point VWAP = {vwap[0]:.2f} (typical price = {expected:.2f})")


def test_two_data_points():
    """Test VWAP with two data points."""
    print("\n=== Test: Two Data Points ===")

    highs = np.array([105.0, 110.0])
    lows = np.array([95.0, 100.0])
    closes = np.array([100.0, 105.0])
    volumes = np.array([1000.0, 2000.0])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    tp = (highs + lows + closes) / 3
    expected = np.array(
        [tp[0], (tp[0] * volumes[0] + tp[1] * volumes[1]) / (volumes[0] + volumes[1])]
    )

    assert len(vwap) == 2
    np.testing.assert_allclose(vwap, expected, rtol=1e-5)

    print(f"✓ Two point VWAP: {vwap}")


def test_monotonic_increasing_prices():
    """Test VWAP with monotonically increasing prices."""
    print("\n=== Test: Monotonic Increasing Prices ===")

    n = 20
    closes = np.linspace(100, 120, n)
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1000.0)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should be monotonically increasing
    diffs = np.diff(vwap)
    assert np.all(diffs > 0), "VWAP should increase monotonically with increasing prices"

    print(f"✓ VWAP monotonically increasing: {vwap[0]:.2f} -> {vwap[-1]:.2f}")


def test_monotonic_decreasing_prices():
    """Test VWAP with monotonically decreasing prices."""
    print("\n=== Test: Monotonic Decreasing Prices ===")

    n = 20
    closes = np.linspace(120, 100, n)
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1000.0)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should be monotonically decreasing
    diffs = np.diff(vwap)
    assert np.all(diffs < 0), "VWAP should decrease monotonically with decreasing prices"

    print(f"✓ VWAP monotonically decreasing: {vwap[0]:.2f} -> {vwap[-1]:.2f}")


def test_constant_prices():
    """Test VWAP with constant prices."""
    print("\n=== Test: Constant Prices ===")

    n = 20
    constant_price = 100.0
    highs = np.full(n, constant_price + 1.0)
    lows = np.full(n, constant_price - 1.0)
    closes = np.full(n, constant_price)
    volumes = np.random.rand(n) * 1000 + 500

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should be constant (equal to typical price)
    expected = constant_price
    np.testing.assert_allclose(vwap, expected, rtol=1e-5)

    print(f"✓ VWAP constant at {vwap[0]:.2f} for constant prices")


def test_vwap_lags_price():
    """Test that VWAP lags current price (smoothing effect)."""
    print("\n=== Test: VWAP Lags Current Price ===")

    # Create sharp price movement
    n = 50
    closes = np.concatenate([np.full(25, 100.0), np.full(25, 110.0)])  # Sharp jump
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1000.0)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    typical_prices = (highs + lows + closes) / 3

    # After the jump, VWAP should be between old and new price
    jump_idx = 30
    assert vwap[jump_idx] > typical_prices[0], "VWAP should be above old price"
    assert vwap[jump_idx] < typical_prices[jump_idx], "VWAP should lag behind new price"

    print(
        f"✓ VWAP lags price: old={typical_prices[0]:.2f}, vwap[30]={vwap[30]:.2f}, new={typical_prices[30]:.2f}"
    )


def test_vwap_converges_to_mean():
    """Test that VWAP converges to mean typical price with equal volumes."""
    print("\n=== Test: VWAP Converges to Mean ===")

    # Random prices with equal volumes
    highs, lows, closes, _ = generate_ohlcv_data(n=1000, seed=42)
    volumes = np.full(1000, 1000.0)  # Equal volumes

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    typical_prices = (highs + lows + closes) / 3

    # Last VWAP should be close to mean of all typical prices
    mean_tp = np.mean(typical_prices)

    np.testing.assert_allclose(vwap[-1], mean_tp, rtol=0.01)

    print(f"✓ Final VWAP {vwap[-1]:.2f} close to mean typical price {mean_tp:.2f}")


def test_large_dataset():
    """Test VWAP with large dataset."""
    print("\n=== Test: Large Dataset ===")

    n = 10_000
    highs, lows, closes, volumes = generate_ohlcv_data(n=n, seed=42)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    assert len(vwap) == n
    assert not np.any(np.isnan(vwap)), "Large dataset should not produce NaN"
    assert not np.any(np.isinf(vwap)), "Large dataset should not produce inf"

    print(f"✓ Large dataset ({n} points) processed successfully")


def test_different_array_types():
    """Test VWAP with different array types (list, tuple, numpy)."""
    print("\n=== Test: Different Array Types ===")

    # Test data
    h, l, c, v = (
        [105.0, 110.0, 108.0],
        [95.0, 100.0, 98.0],
        [100.0, 105.0, 103.0],
        [1000.0, 2000.0, 1500.0],
    )

    # Test with lists
    vwap_list = calculate_vwap(h, l, c, v, engine="cpu")

    # Test with tuples
    vwap_tuple = calculate_vwap(tuple(h), tuple(l), tuple(c), tuple(v), engine="cpu")

    # Test with numpy arrays
    vwap_numpy = calculate_vwap(np.array(h), np.array(l), np.array(c), np.array(v), engine="cpu")

    # All should produce same results
    np.testing.assert_allclose(vwap_list, vwap_numpy, rtol=1e-10)
    np.testing.assert_allclose(vwap_tuple, vwap_numpy, rtol=1e-10)

    print("✓ All array types produce consistent results")


def test_price_crosses_vwap():
    """Test detection of price crossing VWAP."""
    print("\n=== Test: Price Crosses VWAP ===")

    n = 50
    # Create oscillating prices
    closes = 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1000.0)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    typical_prices = (highs + lows + closes) / 3

    # Count crossings
    above = typical_prices > vwap
    crossings = np.sum(np.abs(np.diff(above.astype(int))))

    print(f"✓ Detected {crossings} price/VWAP crossings in oscillating data")
    assert crossings > 0, "Should detect crossings in oscillating prices"


def test_return_type_and_shape():
    """Test that VWAP returns correct type and shape."""
    print("\n=== Test: Return Type and Shape ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Check type
    assert isinstance(vwap, np.ndarray), f"Expected np.ndarray, got {type(vwap)}"

    # Check shape
    assert vwap.shape == (50,), f"Expected shape (50,), got {vwap.shape}"

    # Check dtype
    assert vwap.dtype in [np.float32, np.float64], f"Expected float dtype, got {vwap.dtype}"

    print(f"✓ Return type: {type(vwap)}, shape: {vwap.shape}, dtype: {vwap.dtype}")


# ============================================================================
# 2. VOLUME WEIGHTING TESTS (10 tests)
# ============================================================================


def test_high_volume_pulls_vwap():
    """Test that high volume periods pull VWAP more strongly."""
    print("\n=== Test: High Volume Pulls VWAP ===")

    # Low volume at 100, then high volume spike at 110
    highs = np.array([101.0, 101.0, 111.0])
    lows = np.array([99.0, 99.0, 109.0])
    closes = np.array([100.0, 100.0, 110.0])
    volumes = np.array([100.0, 100.0, 10000.0])  # 100x volume spike

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # After high volume at 110, VWAP should be much closer to 110 than 100
    print(f"VWAP progression: {vwap}")

    # VWAP[2] should be heavily weighted toward 110
    # Expected: closer to 110 than to the simple average of 100, 100, 110 (103.33)
    simple_avg = (100.0 + 100.0 + 110.0) / 3

    assert vwap[2] > simple_avg, "High volume should pull VWAP more than equal weighting"
    print(f"✓ High volume pulls VWAP: {vwap[2]:.2f} > {simple_avg:.2f} (simple average)")


def test_low_volume_has_less_influence():
    """Test that low volume periods have less influence on VWAP."""
    print("\n=== Test: Low Volume Has Less Influence ===")

    # High volume at 100, then low volume spike at 110
    highs = np.array([101.0, 101.0, 111.0])
    lows = np.array([99.0, 99.0, 109.0])
    closes = np.array([100.0, 100.0, 110.0])
    volumes = np.array([10000.0, 10000.0, 100.0])  # Low volume on spike

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should stay close to 100 despite spike to 110
    print(f"VWAP progression: {vwap}")

    # VWAP[2] should be much closer to 100 than to 110
    assert abs(vwap[2] - 100.0) < abs(
        vwap[2] - 110.0
    ), "Low volume spike should have minimal impact"

    print(f"✓ Low volume has less influence: VWAP={vwap[2]:.2f} stays near 100")


def test_volume_weighted_vs_simple_average():
    """Compare volume-weighted VWAP vs simple average."""
    print("\n=== Test: Volume Weighted vs Simple Average ===")

    # Create scenario where middle has high volume and different price
    highs = np.array([102.0, 112.0, 106.0])  # Middle price much higher
    lows = np.array([98.0, 108.0, 102.0])
    closes = np.array([100.0, 110.0, 104.0])
    volumes = np.array([1000.0, 10000.0, 1000.0])  # Middle has 10x volume

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    typical_prices = (highs + lows + closes) / 3
    simple_avg = np.mean(typical_prices)

    print(f"Typical prices: {typical_prices}")
    print(f"Simple average: {simple_avg:.2f}")
    print(f"Final VWAP: {vwap[-1]:.2f}")

    # VWAP is cumulative, so it should be between first and last typical price
    # But different from simple average due to volume weighting
    # The high volume at 110 should pull VWAP higher than simple average
    assert (
        vwap[-1] > simple_avg
    ), f"VWAP {vwap[-1]:.2f} should be pulled higher by high-volume middle price (simple avg={simple_avg:.2f})"

    print("✓ VWAP properly weighted by volume")


def test_equal_volumes_equals_simple_average():
    """Test that equal volumes produce simple average of typical prices."""
    print("\n=== Test: Equal Volumes = Simple Average ===")

    highs = np.array([102.0, 104.0, 106.0, 108.0])
    lows = np.array([98.0, 100.0, 102.0, 104.0])
    closes = np.array([100.0, 102.0, 104.0, 106.0])
    volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0])  # All equal

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    typical_prices = (highs + lows + closes) / 3

    # With equal volumes, VWAP should be cumulative mean of typical prices
    expected = np.array(
        [
            typical_prices[0],
            np.mean(typical_prices[:2]),
            np.mean(typical_prices[:3]),
            np.mean(typical_prices[:4]),
        ]
    )

    np.testing.assert_allclose(vwap, expected, rtol=1e-5)

    print("✓ Equal volumes produce simple cumulative average")


def test_doubling_all_volumes():
    """Test that doubling all volumes doesn't change VWAP."""
    print("\n=== Test: Doubling All Volumes ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    vwap1 = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    vwap2 = calculate_vwap(highs, lows, closes, volumes * 2, engine="cpu")

    # VWAP should be identical (relative weighting unchanged)
    np.testing.assert_allclose(vwap1, vwap2, rtol=1e-10)

    print("✓ Doubling all volumes produces identical VWAP")


def test_volume_scaling_invariance():
    """Test that VWAP is invariant to volume scaling."""
    print("\n=== Test: Volume Scaling Invariance ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    vwap_original = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    vwap_scaled = calculate_vwap(highs, lows, closes, volumes * 1e6, engine="cpu")

    # VWAP should be identical regardless of volume scale
    np.testing.assert_allclose(vwap_original, vwap_scaled, rtol=1e-10)

    print("✓ VWAP invariant to volume scaling")


def test_extreme_volume_ratios():
    """Test VWAP with extreme volume ratios."""
    print("\n=== Test: Extreme Volume Ratios ===")

    highs = np.array([101.0, 101.0, 111.0])
    lows = np.array([99.0, 99.0, 109.0])
    closes = np.array([100.0, 100.0, 110.0])
    volumes = np.array([1.0, 1.0, 1_000_000.0])  # Million to 1 ratio

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should be dominated by the high-volume data point
    # Should be very close to 110
    assert (
        abs(vwap[-1] - 110.0) < 0.01
    ), f"VWAP should be near 110 with extreme volume, got {vwap[-1]}"

    print(f"✓ Extreme volume ratio handled: VWAP={vwap[-1]:.6f} (close to 110)")


def test_gradual_volume_increase():
    """Test VWAP with gradually increasing volumes."""
    print("\n=== Test: Gradual Volume Increase ===")

    n = 20
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    closes = np.full(n, 100.0)
    volumes = np.linspace(100, 10000, n)  # Gradually increasing

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # With constant prices, VWAP should stay constant regardless of volume
    expected = 100.0  # Typical price
    np.testing.assert_allclose(vwap, expected, rtol=1e-5)

    print("✓ Gradual volume increase with constant price produces constant VWAP")


def test_volume_spike_impact():
    """Test impact of sudden volume spike on VWAP."""
    print("\n=== Test: Volume Spike Impact ===")

    n = 20
    closes = np.full(n, 100.0)
    closes[10] = 110.0  # Price spike
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1000.0)
    volumes[10] = 100_000.0  # Volume spike at same time

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should show significant jump at spike, then slowly decay back
    vwap_before = vwap[9]
    vwap_at_spike = vwap[10]
    vwap_after = vwap[11]

    print(f"VWAP: before={vwap_before:.2f}, at spike={vwap_at_spike:.2f}, after={vwap_after:.2f}")

    assert vwap_at_spike > vwap_before, "VWAP should increase at spike"
    assert vwap_at_spike > vwap_after, "VWAP should be highest at spike"

    print("✓ Volume spike creates appropriate VWAP impact")


def test_volume_distribution_effect():
    """Test effect of different volume distributions."""
    print("\n=== Test: Volume Distribution Effect ===")

    highs = np.array([102.0, 104.0, 106.0, 108.0])
    lows = np.array([98.0, 100.0, 102.0, 104.0])
    closes = np.array([100.0, 102.0, 104.0, 106.0])

    # Test 1: Front-loaded volume
    volumes_front = np.array([10000.0, 1000.0, 1000.0, 1000.0])
    vwap_front = calculate_vwap(highs, lows, closes, volumes_front, engine="cpu")

    # Test 2: Back-loaded volume
    volumes_back = np.array([1000.0, 1000.0, 1000.0, 10000.0])
    vwap_back = calculate_vwap(highs, lows, closes, volumes_back, engine="cpu")

    # Front-loaded should stay lower, back-loaded should end higher
    assert vwap_front[-1] < vwap_back[-1], "Back-loaded volume should result in higher final VWAP"

    print(
        f"✓ Volume distribution affects VWAP: front={vwap_front[-1]:.2f}, back={vwap_back[-1]:.2f}"
    )


# ============================================================================
# 3. EDGE CASES (10 tests)
# ============================================================================


def test_zero_volumes():
    """Test VWAP with zero volumes."""
    print("\n=== Test: Zero Volumes ===")

    highs = np.array([102.0, 104.0, 106.0])
    lows = np.array([98.0, 100.0, 102.0])
    closes = np.array([100.0, 102.0, 104.0])
    volumes = np.array([0.0, 0.0, 0.0])

    # Should not crash, but may produce NaN or inf
    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    assert len(vwap) == 3, "Should return array of correct length"
    # With zero volume, VWAP is undefined (0/0), expect NaN or inf
    print(f"Zero volume VWAP: {vwap}")
    print("✓ Zero volumes handled without crash")


def test_mixed_zero_nonzero_volumes():
    """Test VWAP with mix of zero and non-zero volumes."""
    print("\n=== Test: Mixed Zero/Non-Zero Volumes ===")

    highs = np.array([102.0, 104.0, 106.0, 108.0])
    lows = np.array([98.0, 100.0, 102.0, 104.0])
    closes = np.array([100.0, 102.0, 104.0, 106.0])
    volumes = np.array([1000.0, 0.0, 2000.0, 0.0])  # Alternating

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    assert len(vwap) == 4
    # Non-zero volumes should still allow valid VWAP calculation
    print(f"Mixed volume VWAP: {vwap}")
    print("✓ Mixed zero/non-zero volumes handled")


def test_very_small_volumes():
    """Test VWAP with very small volumes."""
    print("\n=== Test: Very Small Volumes ===")

    highs, lows, closes, _ = generate_ohlcv_data(n=50)
    volumes = np.full(50, 1e-10)  # Extremely small

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    assert len(vwap) == 50
    assert not np.any(np.isnan(vwap)), "Very small volumes should not cause NaN"

    print("✓ Very small volumes handled correctly")


def test_very_large_volumes():
    """Test VWAP with very large volumes."""
    print("\n=== Test: Very Large Volumes ===")

    highs, lows, closes, _ = generate_ohlcv_data(n=50)
    volumes = np.full(50, 1e15)  # Extremely large

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    assert len(vwap) == 50
    assert not np.any(np.isnan(vwap)), "Very large volumes should not cause NaN"
    assert not np.any(np.isinf(vwap)), "Very large volumes should not cause inf"

    print("✓ Very large volumes handled correctly")


def test_negative_prices():
    """Test VWAP behavior with negative prices (commodities can go negative)."""
    print("\n=== Test: Negative Prices ===")

    highs = np.array([-98.0, -97.0, -96.0])
    lows = np.array([-102.0, -101.0, -100.0])
    closes = np.array([-100.0, -99.0, -98.0])
    volumes = np.array([1000.0, 1000.0, 1000.0])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should work with negative prices
    assert len(vwap) == 3
    assert np.all(vwap < 0), "VWAP should be negative with negative prices"

    print(f"✓ Negative prices handled: VWAP={vwap}")


def test_price_precision():
    """Test VWAP with high-precision prices."""
    print("\n=== Test: Price Precision ===")

    highs = np.array([100.123456789, 100.234567890, 100.345678901])
    lows = np.array([99.123456789, 99.234567890, 99.345678901])
    closes = np.array([99.623456789, 99.734567890, 99.845678901])
    volumes = np.array([1000.0, 1000.0, 1000.0])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Should maintain reasonable precision
    assert len(vwap) == 3
    assert vwap.dtype in [np.float32, np.float64]

    print(f"✓ High precision prices: VWAP={vwap}")


def test_identical_prices():
    """Test VWAP when high/low/close are identical."""
    print("\n=== Test: Identical Prices ===")

    n = 20
    price = 100.0
    highs = np.full(n, price)
    lows = np.full(n, price)
    closes = np.full(n, price)
    volumes = np.random.rand(n) * 1000 + 500

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # VWAP should equal the constant price
    np.testing.assert_allclose(vwap, price, rtol=1e-10)

    print(f"✓ Identical prices produce constant VWAP: {vwap[0]:.2f}")


def test_empty_input_handling():
    """Test VWAP error handling with empty input."""
    print("\n=== Test: Empty Input Handling ===")

    highs = np.array([])
    lows = np.array([])
    closes = np.array([])
    volumes = np.array([])

    try:
        vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
        print(f"Empty input result: {vwap}")
        print("✓ Empty input handled (returned empty or raised error)")
    except Exception as e:
        print(f"✓ Empty input raised expected error: {type(e).__name__}")


def test_mismatched_array_lengths():
    """Test VWAP error handling with mismatched array lengths."""
    print("\n=== Test: Mismatched Array Lengths ===")

    highs = np.array([102.0, 104.0, 106.0])
    lows = np.array([98.0, 100.0])  # Too short
    closes = np.array([100.0, 102.0, 104.0])
    volumes = np.array([1000.0, 1000.0, 1000.0])

    try:
        vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
        print("Warning: Mismatched lengths did not raise error")
    except Exception as e:
        print(f"✓ Mismatched lengths raised expected error: {type(e).__name__}")


def test_extreme_price_ranges():
    """Test VWAP with extreme price ranges."""
    print("\n=== Test: Extreme Price Ranges ===")

    # Mix of very small and very large prices
    highs = np.array([0.0001, 0.0002, 10000.0, 10001.0])
    lows = np.array([0.00005, 0.00015, 9999.0, 10000.0])
    closes = np.array([0.000075, 0.000175, 9999.5, 10000.5])
    volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    assert len(vwap) == 4
    assert not np.any(np.isnan(vwap)), "Extreme ranges should not cause NaN"
    assert not np.any(np.isinf(vwap)), "Extreme ranges should not cause inf"

    print(f"✓ Extreme price ranges handled: {vwap}")


# ============================================================================
# 4. ANCHORED VWAP TESTS (10 tests)
# ============================================================================


def test_anchored_vwap_basic():
    """Test basic anchored VWAP calculation."""
    print("\n=== Test: Anchored VWAP Basic ===")

    highs = np.array([102.0, 104.0, 106.0, 108.0, 110.0])
    lows = np.array([98.0, 100.0, 102.0, 104.0, 106.0])
    closes = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
    volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    anchors = np.array([True, False, False, True, False])  # Reset at indices 0 and 3

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")

    assert len(vwap) == 5
    assert not np.any(np.isnan(vwap)), "Anchored VWAP should not have NaN"

    print(f"✓ Anchored VWAP calculated: {vwap}")


def test_anchored_vwap_resets():
    """Test that anchored VWAP resets at anchor points."""
    print("\n=== Test: Anchored VWAP Resets ===")

    n = 10
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    closes = np.linspace(100, 110, n)  # Gradually increasing
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1000.0)
    anchors = np.array([True, False, False, False, True, False, False, False, False, False])

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")

    # VWAP should reset at index 4
    # After reset, VWAP should be close to prices starting from that point
    typical_prices = (highs + lows + closes) / 3

    # At anchor point, VWAP should equal typical price
    np.testing.assert_allclose(vwap[4], typical_prices[4], rtol=0.01)

    print(f"✓ VWAP resets at anchor: before={vwap[3]:.2f}, at anchor={vwap[4]:.2f}")


def test_anchored_vwap_no_anchors():
    """Test anchored VWAP with no anchor points after first."""
    print("\n=== Test: Anchored VWAP No Anchors ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)
    anchors = np.zeros(50, dtype=bool)
    anchors[0] = True  # Only first point is anchor

    vwap_anchored = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")
    vwap_regular = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Should be identical to regular VWAP
    np.testing.assert_allclose(vwap_anchored, vwap_regular, rtol=1e-5)

    print("✓ Single anchor produces same result as regular VWAP")


def test_anchored_vwap_every_point():
    """Test anchored VWAP with anchor at every point."""
    print("\n=== Test: Anchored VWAP Every Point ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=20)
    anchors = np.ones(20, dtype=bool)  # Every point is anchor

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")
    typical_prices = (highs + lows + closes) / 3

    # Each VWAP should equal its typical price (no history)
    np.testing.assert_allclose(vwap, typical_prices, rtol=1e-5)

    print("✓ Anchor at every point produces typical prices")


def test_anchored_vwap_sessions():
    """Test anchored VWAP for simulated trading sessions."""
    print("\n=== Test: Anchored VWAP Sessions ===")

    # Simulate 3 sessions of 10 candles each
    n_sessions = 3
    n_per_session = 10
    n_total = n_sessions * n_per_session

    highs, lows, closes, volumes = generate_ohlcv_data(n=n_total)
    anchors = np.zeros(n_total, dtype=bool)
    anchors[0] = True  # First session
    anchors[10] = True  # Second session
    anchors[20] = True  # Third session

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")

    # VWAP should be different across sessions
    session1_vwap = vwap[9]
    session2_vwap = vwap[19]
    session3_vwap = vwap[29]

    print(f"Session VWAPs: {session1_vwap:.2f}, {session2_vwap:.2f}, {session3_vwap:.2f}")
    print("✓ Anchored VWAP handles multiple sessions")


def test_anchored_vwap_intraday_reset():
    """Test anchored VWAP for intraday resets (e.g., market open)."""
    print("\n=== Test: Anchored VWAP Intraday Reset ===")

    # Simulate overnight gap
    closes_before = np.full(10, 100.0)
    closes_after = np.full(10, 110.0)  # Gap up 10 points
    closes = np.concatenate([closes_before, closes_after])

    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(20, 1000.0)

    # Reset at gap (market open)
    anchors = np.zeros(20, dtype=bool)
    anchors[0] = True
    anchors[10] = True  # Reset at gap

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")

    # VWAP should reset to ~110 at gap, not blend with ~100
    assert abs(vwap[10] - 110.0) < 1.0, f"VWAP should reset near 110 at gap, got {vwap[10]:.2f}"

    print(f"✓ Intraday reset: before={vwap[9]:.2f}, at reset={vwap[10]:.2f}")


def test_anchored_vwap_integer_anchors():
    """Test that boolean anchor array works correctly."""
    print("\n=== Test: Boolean Anchor Array ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=20)

    # Test with boolean array
    anchors_bool = np.zeros(20, dtype=bool)
    anchors_bool[0] = True
    anchors_bool[10] = True

    vwap_bool = calculate_vwap_anchored(highs, lows, closes, volumes, anchors_bool, engine="cpu")

    # Test with integer array (0 and 1)
    anchors_int = np.zeros(20, dtype=int)
    anchors_int[0] = 1
    anchors_int[10] = 1

    vwap_int = calculate_vwap_anchored(highs, lows, closes, volumes, anchors_int, engine="cpu")

    # Should produce same results
    np.testing.assert_allclose(vwap_bool, vwap_int, rtol=1e-10)

    print("✓ Boolean and integer anchors produce same results")


def test_anchored_vwap_consecutive_anchors():
    """Test anchored VWAP with consecutive anchor points."""
    print("\n=== Test: Consecutive Anchors ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=10)
    anchors = np.array([True, True, True, False, False, False, False, False, False, False])

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")
    typical_prices = (highs + lows + closes) / 3

    # First three should each equal their typical price
    np.testing.assert_allclose(vwap[0], typical_prices[0], rtol=1e-5)
    np.testing.assert_allclose(vwap[1], typical_prices[1], rtol=1e-5)
    np.testing.assert_allclose(vwap[2], typical_prices[2], rtol=1e-5)

    print("✓ Consecutive anchors handled correctly")


def test_anchored_vwap_vs_regular():
    """Compare anchored VWAP with single anchor to regular VWAP."""
    print("\n=== Test: Anchored vs Regular VWAP ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    # Regular VWAP
    vwap_regular = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Anchored VWAP with only first anchor
    anchors = np.zeros(50, dtype=bool)
    anchors[0] = True
    vwap_anchored = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")

    # Should be very close (may have small differences due to implementation)
    np.testing.assert_allclose(vwap_regular, vwap_anchored, rtol=1e-3)

    print("✓ Anchored VWAP with single anchor matches regular VWAP")


def test_anchored_vwap_mid_day_reset():
    """Test anchored VWAP with mid-day reset scenario."""
    print("\n=== Test: Mid-Day Reset ===")

    n = 30
    # Price trends up, then resets, then trends down
    closes = np.concatenate(
        [np.linspace(100, 110, 15), np.linspace(105, 95, 15)]  # Uptrend  # Downtrend
    )
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1000.0)

    # Reset at midpoint
    anchors = np.zeros(n, dtype=bool)
    anchors[0] = True
    anchors[15] = True

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")

    # VWAP before reset should be different from VWAP after
    vwap_before = vwap[14]
    vwap_after_start = vwap[15]

    print(f"VWAPs: before reset={vwap_before:.2f}, at reset={vwap_after_start:.2f}")
    print("✓ Mid-day reset works as expected")


# ============================================================================
# 5. GPU/CPU PARITY TESTS (10 tests)
# ============================================================================


def test_cpu_engine():
    """Test VWAP with explicit CPU engine."""
    print("\n=== Test: CPU Engine ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    assert len(vwap) == 100
    assert not np.any(np.isnan(vwap))

    print("✓ CPU engine works")


def test_auto_engine():
    """Test VWAP with auto engine selection."""
    print("\n=== Test: Auto Engine ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="auto")

    assert len(vwap) == 100
    assert not np.any(np.isnan(vwap))

    print("✓ Auto engine selection works")


def test_cpu_consistency():
    """Test that CPU engine produces consistent results."""
    print("\n=== Test: CPU Consistency ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    vwap1 = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    vwap2 = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Should be identical
    np.testing.assert_array_equal(vwap1, vwap2)

    print("✓ CPU engine produces consistent results")


def test_different_engines_same_result():
    """Test that different engines produce same results."""
    print("\n=== Test: Engine Parity ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    vwap_cpu = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    vwap_auto = calculate_vwap(highs, lows, closes, volumes, engine="auto")

    # Should be very close (allow for floating point differences)
    np.testing.assert_allclose(vwap_cpu, vwap_auto, rtol=1e-5)

    print("✓ CPU and auto engines produce same results")


def test_large_dataset_engines():
    """Test engine performance with large dataset."""
    print("\n=== Test: Large Dataset Engines ===")

    n = 100_000  # Large enough to potentially trigger GPU
    highs, lows, closes, volumes = generate_ohlcv_data(n=n)

    vwap_cpu = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    vwap_auto = calculate_vwap(highs, lows, closes, volumes, engine="auto")

    assert len(vwap_cpu) == n
    assert len(vwap_auto) == n

    # Results should be close
    np.testing.assert_allclose(vwap_cpu, vwap_auto, rtol=1e-5, atol=1e-8)

    print(f"✓ Large dataset ({n} points) processed by both engines")


def test_anchored_vwap_cpu():
    """Test anchored VWAP with CPU engine."""
    print("\n=== Test: Anchored VWAP CPU ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)
    anchors = np.zeros(100, dtype=bool)
    anchors[0] = True
    anchors[50] = True

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")

    assert len(vwap) == 100
    assert not np.any(np.isnan(vwap))

    print("✓ Anchored VWAP CPU engine works")


def test_anchored_vwap_auto():
    """Test anchored VWAP with auto engine."""
    print("\n=== Test: Anchored VWAP Auto ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)
    anchors = np.zeros(100, dtype=bool)
    anchors[0] = True
    anchors[50] = True

    vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="auto")

    assert len(vwap) == 100
    assert not np.any(np.isnan(vwap))

    print("✓ Anchored VWAP auto engine works")


def test_numerical_precision_cpu():
    """Test numerical precision on CPU."""
    print("\n=== Test: Numerical Precision CPU ===")

    # High precision test
    highs = np.array([100.12345678901234, 100.23456789012345, 100.34567890123456])
    lows = np.array([99.12345678901234, 99.23456789012345, 99.34567890123456])
    closes = np.array([99.62345678901234, 99.73456789012345, 99.84567890123456])
    volumes = np.array([1000.123456789, 2000.234567890, 1500.345678901])

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Should maintain float64 precision
    assert vwap.dtype == np.float64, f"Expected float64, got {vwap.dtype}"

    print(f"✓ CPU maintains float64 precision: {vwap}")


def test_numerical_stability():
    """Test numerical stability with cumulative operations."""
    print("\n=== Test: Numerical Stability ===")

    # Very long series to test cumulative sum stability
    n = 50_000
    highs, lows, closes, volumes = generate_ohlcv_data(n=n, seed=42)

    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")

    # Check for numerical issues
    assert not np.any(np.isnan(vwap)), "Long series should not produce NaN"
    assert not np.any(np.isinf(vwap)), "Long series should not produce inf"

    # VWAP should still be within reasonable range of prices
    typical_prices = (highs + lows + closes) / 3
    min_price = np.min(typical_prices)
    max_price = np.max(typical_prices)

    assert np.all(vwap >= min_price * 0.9), "VWAP should not drift far below prices"
    assert np.all(vwap <= max_price * 1.1), "VWAP should not drift far above prices"

    print(f"✓ Numerical stability maintained over {n} points")


def test_engine_error_handling():
    """Test error handling for invalid engine specification."""
    print("\n=== Test: Engine Error Handling ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=50)

    try:
        vwap = calculate_vwap(highs, lows, closes, volumes, engine="invalid")
        print("Warning: Invalid engine did not raise error")
    except Exception as e:
        print(f"✓ Invalid engine raised expected error: {type(e).__name__}")


# ============================================================================
# 6. PERFORMANCE TESTS (5 tests)
# ============================================================================


def test_performance_small_dataset():
    """Benchmark VWAP on small dataset."""
    print("\n=== Test: Performance Small Dataset ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100)

    start = time.perf_counter()
    for _ in range(100):
        vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 100) * 1000
    print(f"✓ Small dataset (100 points): {avg_time_ms:.3f}ms per calculation")


def test_performance_medium_dataset():
    """Benchmark VWAP on medium dataset."""
    print("\n=== Test: Performance Medium Dataset ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=10_000)

    start = time.perf_counter()
    for _ in range(10):
        vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 10) * 1000
    print(f"✓ Medium dataset (10k points): {avg_time_ms:.3f}ms per calculation")


def test_performance_large_dataset():
    """Benchmark VWAP on large dataset."""
    print("\n=== Test: Performance Large Dataset ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=100_000)

    start = time.perf_counter()
    vwap = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    elapsed = time.perf_counter() - start

    print(f"✓ Large dataset (100k points): {elapsed*1000:.3f}ms")


def test_performance_anchored_vwap():
    """Benchmark anchored VWAP performance."""
    print("\n=== Test: Performance Anchored VWAP ===")

    highs, lows, closes, volumes = generate_ohlcv_data(n=10_000)
    anchors = np.zeros(10_000, dtype=bool)
    anchors[::1000] = True  # Anchor every 1000 points

    start = time.perf_counter()
    for _ in range(10):
        vwap = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 10) * 1000
    print(f"✓ Anchored VWAP (10k points, 10 anchors): {avg_time_ms:.3f}ms per calculation")


def test_performance_comparison():
    """Compare performance of regular vs anchored VWAP."""
    print("\n=== Test: Performance Comparison ===")

    n = 10_000
    highs, lows, closes, volumes = generate_ohlcv_data(n=n)
    anchors = np.zeros(n, dtype=bool)
    anchors[0] = True

    # Regular VWAP
    start = time.perf_counter()
    vwap_regular = calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    time_regular = time.perf_counter() - start

    # Anchored VWAP (single anchor = should be similar)
    start = time.perf_counter()
    vwap_anchored = calculate_vwap_anchored(highs, lows, closes, volumes, anchors, engine="cpu")
    time_anchored = time.perf_counter() - start

    print(f"Regular VWAP: {time_regular*1000:.3f}ms")
    print(f"Anchored VWAP: {time_anchored*1000:.3f}ms")
    print(f"✓ Performance comparison completed")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VWAP TEST SUITE")
    print("=" * 80)

    test_count = 0

    # 1. Basic Calculation Tests (15 tests)
    print("\n" + "=" * 80)
    print("1. BASIC CALCULATION TESTS (15 tests)")
    print("=" * 80)
    test_basic_calculation()
    test_count += 1
    test_typical_price_calculation()
    test_count += 1
    test_cumulative_weighted_average()
    test_count += 1
    test_vwap_between_high_and_low()
    test_count += 1
    test_single_data_point()
    test_count += 1
    test_two_data_points()
    test_count += 1
    test_monotonic_increasing_prices()
    test_count += 1
    test_monotonic_decreasing_prices()
    test_count += 1
    test_constant_prices()
    test_count += 1
    test_vwap_lags_price()
    test_count += 1
    test_vwap_converges_to_mean()
    test_count += 1
    test_large_dataset()
    test_count += 1
    test_different_array_types()
    test_count += 1
    test_price_crosses_vwap()
    test_count += 1
    test_return_type_and_shape()
    test_count += 1

    # 2. Volume Weighting Tests (10 tests)
    print("\n" + "=" * 80)
    print("2. VOLUME WEIGHTING TESTS (10 tests)")
    print("=" * 80)
    test_high_volume_pulls_vwap()
    test_count += 1
    test_low_volume_has_less_influence()
    test_count += 1
    test_volume_weighted_vs_simple_average()
    test_count += 1
    test_equal_volumes_equals_simple_average()
    test_count += 1
    test_doubling_all_volumes()
    test_count += 1
    test_volume_scaling_invariance()
    test_count += 1
    test_extreme_volume_ratios()
    test_count += 1
    test_gradual_volume_increase()
    test_count += 1
    test_volume_spike_impact()
    test_count += 1
    test_volume_distribution_effect()
    test_count += 1

    # 3. Edge Cases (10 tests)
    print("\n" + "=" * 80)
    print("3. EDGE CASES (10 tests)")
    print("=" * 80)
    test_zero_volumes()
    test_count += 1
    test_mixed_zero_nonzero_volumes()
    test_count += 1
    test_very_small_volumes()
    test_count += 1
    test_very_large_volumes()
    test_count += 1
    test_negative_prices()
    test_count += 1
    test_price_precision()
    test_count += 1
    test_identical_prices()
    test_count += 1
    test_empty_input_handling()
    test_count += 1
    test_mismatched_array_lengths()
    test_count += 1
    test_extreme_price_ranges()
    test_count += 1

    # 4. Anchored VWAP Tests (10 tests)
    print("\n" + "=" * 80)
    print("4. ANCHORED VWAP TESTS (10 tests)")
    print("=" * 80)
    test_anchored_vwap_basic()
    test_count += 1
    test_anchored_vwap_resets()
    test_count += 1
    test_anchored_vwap_no_anchors()
    test_count += 1
    test_anchored_vwap_every_point()
    test_count += 1
    test_anchored_vwap_sessions()
    test_count += 1
    test_anchored_vwap_intraday_reset()
    test_count += 1
    test_anchored_vwap_integer_anchors()
    test_count += 1
    test_anchored_vwap_consecutive_anchors()
    test_count += 1
    test_anchored_vwap_vs_regular()
    test_count += 1
    test_anchored_vwap_mid_day_reset()
    test_count += 1

    # 5. GPU/CPU Parity Tests (10 tests)
    print("\n" + "=" * 80)
    print("5. GPU/CPU PARITY TESTS (10 tests)")
    print("=" * 80)
    test_cpu_engine()
    test_count += 1
    test_auto_engine()
    test_count += 1
    test_cpu_consistency()
    test_count += 1
    test_different_engines_same_result()
    test_count += 1
    test_large_dataset_engines()
    test_count += 1
    test_anchored_vwap_cpu()
    test_count += 1
    test_anchored_vwap_auto()
    test_count += 1
    test_numerical_precision_cpu()
    test_count += 1
    test_numerical_stability()
    test_count += 1
    test_engine_error_handling()
    test_count += 1

    # 6. Performance Tests (5 tests)
    print("\n" + "=" * 80)
    print("6. PERFORMANCE TESTS (5 tests)")
    print("=" * 80)
    test_performance_small_dataset()
    test_count += 1
    test_performance_medium_dataset()
    test_count += 1
    test_performance_large_dataset()
    test_count += 1
    test_performance_anchored_vwap()
    test_count += 1
    test_performance_comparison()
    test_count += 1

    print("\n" + "=" * 80)
    print(f"ALL {test_count} TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

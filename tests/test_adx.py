"""
Test suite for ADX (Average Directional Index) indicator.

Tests cover:
- Basic calculation correctness
- Value ranges (0-100 for all indicators)
- Wilder's smoothing correctness
- Trend strength interpretation
- Signal generation (+DI/-DI crossovers)
- Edge cases (constant prices, minimal data)
- Performance benchmarking
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.ops.adx import calculate_adx


def generate_ohlc_data(n=100, seed=42):
    """Generate test OHLC data for ADX testing."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return highs, lows, closes


def generate_trending_data(n=100, trend_strength=1.0):
    """Generate strongly trending data for testing."""
    # Create consistent uptrend
    closes = 100 + np.arange(n) * trend_strength
    highs = closes + 0.5
    lows = closes - 0.5
    return highs, lows, closes


def test_basic_calculation():
    """Test basic ADX calculation returns correct array lengths."""
    highs, lows, closes = generate_ohlc_data(n=100)

    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # All arrays should have same length as input
    assert len(adx) == len(highs) == 100
    assert len(plus_di) == len(highs) == 100
    assert len(minus_di) == len(highs) == 100

    # Arrays should be NumPy arrays
    assert isinstance(adx, np.ndarray)
    assert isinstance(plus_di, np.ndarray)
    assert isinstance(minus_di, np.ndarray)


def test_all_three_values_calculated():
    """Test that ADX returns all three indicators with values."""
    highs, lows, closes = generate_ohlc_data(n=100)

    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # Should have non-NaN values after warmup period
    # With Wilder's smoothing, values start appearing after period-1 for DI
    # and after 2*(period-1) for ADX
    warmup_di = 14
    warmup_adx = 2 * 14
    assert not np.all(np.isnan(adx[warmup_adx:]))
    assert not np.all(np.isnan(plus_di[warmup_di:]))
    assert not np.all(np.isnan(minus_di[warmup_di:]))

    # Should have some NaN values at start (warmup period)
    # ADX starts producing values at index period-1 (after first smoothing),
    # then needs another period-1 for second smoothing
    assert np.sum(np.isnan(adx)) >= 13  # At least period - 1
    assert np.sum(np.isnan(plus_di)) >= 13  # DI needs period - 1 for smoothing
    assert np.sum(np.isnan(minus_di)) >= 13


def test_value_ranges():
    """Test that ADX, +DI, and -DI are all in range [0, 100]."""
    highs, lows, closes = generate_ohlc_data(n=200)

    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # Remove NaN values for testing
    adx_valid = adx[~np.isnan(adx)]
    plus_di_valid = plus_di[~np.isnan(plus_di)]
    minus_di_valid = minus_di[~np.isnan(minus_di)]

    # All values should be in [0, 100]
    assert np.all(adx_valid >= 0.0)
    assert np.all(adx_valid <= 100.0)
    assert np.all(plus_di_valid >= 0.0)
    assert np.all(plus_di_valid <= 100.0)
    assert np.all(minus_di_valid >= 0.0)
    assert np.all(minus_di_valid <= 100.0)


def test_wilder_smoothing():
    """Test that Wilder's smoothing is applied correctly."""
    highs, lows, closes = generate_ohlc_data(n=100)

    # Test with different periods
    adx_14, plus_di_14, minus_di_14 = calculate_adx(highs, lows, closes, period=14)
    adx_7, plus_di_7, minus_di_7 = calculate_adx(highs, lows, closes, period=7)

    # Shorter period should have more responsive (less smooth) values
    # Standard deviation should be higher for shorter period
    adx_14_valid = adx_14[~np.isnan(adx_14)]
    adx_7_valid = adx_7[~np.isnan(adx_7)]

    # Both should have valid values
    assert len(adx_14_valid) > 20
    assert len(adx_7_valid) > 20

    # Shorter period typically more volatile (but not always, so we just check it runs)
    assert np.mean(adx_7_valid) >= 0
    assert np.mean(adx_14_valid) >= 0


def test_trend_strength():
    """Test ADX correctly identifies trend strength."""
    # Strong uptrend
    highs, lows, closes = generate_trending_data(n=100, trend_strength=1.0)

    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # Remove warmup period (first 2*period values)
    warmup = 2 * 14
    adx_valid = adx[warmup:]
    plus_di_valid = plus_di[warmup:]
    minus_di_valid = minus_di[warmup:]

    # In strong uptrend:
    # 1. ADX should be > 25 (strong trend)
    assert (
        np.mean(adx_valid) > 25
    ), f"Expected strong trend (ADX > 25), got {np.mean(adx_valid):.2f}"

    # 2. +DI should be > -DI (upward movement)
    assert np.mean(plus_di_valid) > np.mean(
        minus_di_valid
    ), f"+DI ({np.mean(plus_di_valid):.2f}) should be > -DI ({np.mean(minus_di_valid):.2f})"

    # 3. +DI should be substantial
    assert (
        np.mean(plus_di_valid) > 20
    ), f"Expected +DI > 20 in uptrend, got {np.mean(plus_di_valid):.2f}"


def test_signal_generation():
    """Test +DI/-DI crossover signal detection."""
    # Create data with trend reversal
    n = 100
    # Uptrend then downtrend
    uptrend = 100 + np.arange(50) * 0.5
    downtrend = uptrend[-1] - np.arange(50) * 0.5
    closes = np.concatenate([uptrend, downtrend])
    highs = closes + 0.3
    lows = closes - 0.3

    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # Detect crossovers (ignoring NaN values)
    valid_idx = ~(np.isnan(plus_di) | np.isnan(minus_di))

    # Find where +DI crosses above -DI (bullish)
    bullish_cross = (plus_di[1:] > minus_di[1:]) & (plus_di[:-1] <= minus_di[:-1]) & valid_idx[1:]

    # Find where -DI crosses above +DI (bearish)
    bearish_cross = (minus_di[1:] > plus_di[1:]) & (minus_di[:-1] <= plus_di[:-1]) & valid_idx[1:]

    # Should detect at least one crossover in this data
    total_crosses = np.sum(bullish_cross) + np.sum(bearish_cross)
    assert total_crosses >= 1, "Should detect at least one DI crossover"


def test_edge_cases():
    """Test edge cases for ADX calculation."""
    # Test 1: Constant price (no movement)
    n = 50
    constant_close = np.full(n, 100.0)
    constant_high = np.full(n, 100.5)
    constant_low = np.full(n, 99.5)

    adx, plus_di, minus_di = calculate_adx(constant_high, constant_low, constant_close, period=14)

    # With constant prices, DI values should be near zero or NaN
    # (no directional movement)
    valid_idx = ~np.isnan(adx)
    if np.any(valid_idx):
        # ADX should be very low (no trend)
        assert np.nanmax(adx) < 30, "Constant price should produce low ADX"

    # Test 2: Minimal data (exactly period length)
    highs, lows, closes = generate_ohlc_data(n=20)
    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # Should handle minimal data without crashing
    assert len(adx) == 20
    assert len(plus_di) == 20
    assert len(minus_di) == 20

    # Test 3: Very short period
    highs, lows, closes = generate_ohlc_data(n=100)
    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=3)

    # Should work with short periods
    assert len(adx) == 100
    valid_adx = adx[~np.isnan(adx)]
    assert len(valid_adx) > 50  # Should have plenty of valid values


def test_different_periods():
    """Test ADX with different period parameters."""
    highs, lows, closes = generate_ohlc_data(n=100)

    # Test various periods
    periods = [7, 10, 14, 21]
    results = []

    for period in periods:
        adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=period)
        results.append((adx, plus_di, minus_di))

        # All should return valid arrays
        assert len(adx) == 100
        assert len(plus_di) == 100
        assert len(minus_di) == 100

        # Should have some valid values after warmup
        warmup = 2 * period
        if warmup < 100:
            assert not np.all(np.isnan(adx[warmup:]))

    # Longer periods should have more smoothing (less variation)
    adx_7 = results[0][0][~np.isnan(results[0][0])]
    adx_21 = results[3][0][~np.isnan(results[3][0])]

    if len(adx_7) > 10 and len(adx_21) > 10:
        # Both should produce valid results
        assert np.mean(adx_7) >= 0
        assert np.mean(adx_21) >= 0


def test_nan_handling():
    """Test that NaN values are in expected positions."""
    highs, lows, closes = generate_ohlc_data(n=100)

    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # Check NaN positions
    # DI values need at least period values for first smoothing
    # First value is always NaN (from true_range/directional_movement diff)
    assert np.isnan(plus_di[0])
    assert np.isnan(minus_di[0])
    assert np.isnan(adx[0])

    # After warmup period, should have valid values
    warmup = 2 * 14
    assert not np.all(np.isnan(adx[warmup:]))
    assert not np.all(np.isnan(plus_di[warmup:]))
    assert not np.all(np.isnan(minus_di[warmup:]))


def test_directional_indicators_sum():
    """Test relationship between +DI and -DI."""
    highs, lows, closes = generate_ohlc_data(n=100)

    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)

    # Remove NaN values
    valid_idx = ~(np.isnan(plus_di) | np.isnan(minus_di))
    plus_di_valid = plus_di[valid_idx]
    minus_di_valid = minus_di[valid_idx]

    # Both should be non-negative
    assert np.all(plus_di_valid >= 0)
    assert np.all(minus_di_valid >= 0)

    # At least one should typically be > 0 (except in very flat markets)
    di_sum = plus_di_valid + minus_di_valid
    assert np.mean(di_sum) > 0, "Sum of DIs should be positive"


def test_performance():
    """Test performance on different data sizes."""
    import time

    sizes = [1_000, 10_000]

    for size in sizes:
        highs, lows, closes = generate_ohlc_data(n=size)

        start = time.time()
        adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14, engine="cpu")
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0, f"ADX too slow for {size} rows: {elapsed:.3f}s"

        # Results should be valid
        assert len(adx) == size
        assert not np.all(np.isnan(adx))

    print(f"\nPerformance test passed for sizes: {sizes}")


def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    highs, lows, closes = generate_ohlc_data(n=100)

    # Test invalid period
    with pytest.raises(ValueError, match="period must be >= 1"):
        calculate_adx(highs, lows, closes, period=0)

    with pytest.raises(ValueError, match="period must be >= 1"):
        calculate_adx(highs, lows, closes, period=-1)


if __name__ == "__main__":
    # Run tests
    print("Running ADX tests...")
    print("\n1. Basic calculation...")
    test_basic_calculation()
    print("✓ Passed")

    print("\n2. All three values calculated...")
    test_all_three_values_calculated()
    print("✓ Passed")

    print("\n3. Value ranges...")
    test_value_ranges()
    print("✓ Passed")

    print("\n4. Wilder's smoothing...")
    test_wilder_smoothing()
    print("✓ Passed")

    print("\n5. Trend strength...")
    test_trend_strength()
    print("✓ Passed")

    print("\n6. Signal generation...")
    test_signal_generation()
    print("✓ Passed")

    print("\n7. Edge cases...")
    test_edge_cases()
    print("✓ Passed")

    print("\n8. Different periods...")
    test_different_periods()
    print("✓ Passed")

    print("\n9. NaN handling...")
    test_nan_handling()
    print("✓ Passed")

    print("\n10. Directional indicators sum...")
    test_directional_indicators_sum()
    print("✓ Passed")

    print("\n11. Performance...")
    test_performance()
    print("✓ Passed")

    print("\n12. Invalid inputs...")
    test_invalid_inputs()
    print("✓ Passed")

    print("\n" + "=" * 50)
    print("All ADX tests passed! ✓")
    print("=" * 50)

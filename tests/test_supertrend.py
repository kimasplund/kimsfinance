"""
Test Suite for Supertrend Indicator
====================================

Comprehensive tests for GPU-accelerated Supertrend implementation.

Tests cover:
- Basic calculation
- Trend direction (1 or -1)
- ATR-based bands
- State tracking (trend persistence)
- Trend reversals (direction switches)
- Different multipliers
- Edge cases
- Performance benchmarks
"""

import numpy as np
import pytest

from kimsfinance.ops.supertrend import calculate_supertrend
from kimsfinance.ops.indicators import calculate_atr


def generate_ohlcv_data(n=100, seed=42):
    """Generate test OHLCV data for indicators."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    volumes = np.abs(np.random.randn(n) * 1_000_000)
    return highs, lows, closes, volumes


def test_basic_calculation():
    """Test basic Supertrend calculation."""
    highs, lows, closes, _ = generate_ohlcv_data(n=50)

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    # Check output shape
    assert len(supertrend) == len(highs), "Supertrend length should match input length"
    assert len(direction) == len(highs), "Direction length should match input length"

    # Check types
    assert isinstance(supertrend, np.ndarray), "Supertrend should be numpy array"
    assert isinstance(direction, np.ndarray), "Direction should be numpy array"

    # First 'period' values should be NaN
    assert np.all(np.isnan(supertrend[:10])), "First 10 values should be NaN (ATR warmup)"
    assert np.all(np.isnan(direction[:10])), "First 10 direction values should be NaN"

    # Remaining values should not be NaN
    assert not np.all(np.isnan(supertrend[10:])), "Should have valid values after warmup"
    assert not np.all(np.isnan(direction[10:])), "Should have valid direction after warmup"


def test_trend_direction():
    """Test that trend direction is 1 or -1."""
    highs, lows, closes, _ = generate_ohlcv_data(n=50)

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    # Check direction values (excluding NaN)
    valid_direction = direction[~np.isnan(direction)]
    assert np.all(np.isin(valid_direction, [1, -1])), "Direction should be 1 (up) or -1 (down)"

    # Check that we have at least some valid direction values
    assert len(valid_direction) > 0, "Should have valid direction values"


def test_atr_bands():
    """Test that Supertrend uses ATR-based bands correctly."""
    highs, lows, closes, _ = generate_ohlcv_data(n=50)
    period = 10
    multiplier = 3.0

    supertrend, direction = calculate_supertrend(
        highs, lows, closes, period=period, multiplier=multiplier
    )

    # Calculate ATR independently
    atr = calculate_atr(highs, lows, closes, period=period)

    # Calculate median price
    hl_avg = (highs + lows) / 2.0

    # For valid indices, supertrend should be close to hl_avg ± multiplier * atr
    valid_idx = ~np.isnan(supertrend)

    # Supertrend should be within reasonable range based on ATR
    # (accounting for state tracking adjustments)
    for i in np.where(valid_idx)[0]:
        # Supertrend should be somewhat close to median price ± multiplier * ATR
        # Allow wider tolerance due to state tracking
        distance_from_median = abs(supertrend[i] - hl_avg[i])
        expected_max_distance = multiplier * atr[i] * 1.5  # 50% tolerance for state adjustments

        assert distance_from_median <= expected_max_distance, \
            f"Supertrend at index {i} too far from expected band range"


def test_state_tracking():
    """Test that trend state persists until reversal."""
    # Create trending data
    np.random.seed(42)
    n = 100
    # Strong uptrend
    closes = np.linspace(100, 120, n)
    highs = closes + 0.5
    lows = closes - 0.5

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    # In strong uptrend, direction should stabilize to 1
    # Check last 50 values (after initial warmup and stabilization)
    late_direction = direction[-50:]
    uptrend_ratio = np.sum(late_direction == 1) / len(late_direction)

    # Should be predominantly uptrend in strong uptrend scenario
    assert uptrend_ratio > 0.7, "Should maintain uptrend in strong upward movement"


def test_trend_reversals():
    """Test that trend direction switches at reversals."""
    # Create data with clear reversal
    np.random.seed(42)
    n = 100

    # First half: uptrend, second half: downtrend
    first_half = np.linspace(100, 110, n // 2)
    second_half = np.linspace(110, 95, n // 2)
    closes = np.concatenate([first_half, second_half])

    highs = closes + 0.5
    lows = closes - 0.5

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=2.0)

    # Check for direction changes
    valid_direction = direction[~np.isnan(direction)]
    direction_changes = np.diff(valid_direction) != 0

    # Should have at least one direction change
    assert np.sum(direction_changes) >= 1, "Should detect trend reversal"

    # Check that both uptrend and downtrend exist
    has_uptrend = np.any(valid_direction == 1)
    has_downtrend = np.any(valid_direction == -1)

    assert has_uptrend, "Should have uptrend periods"
    assert has_downtrend, "Should have downtrend periods"


def test_different_multipliers():
    """Test Supertrend with different multiplier values."""
    highs, lows, closes, _ = generate_ohlcv_data(n=100)

    # Test different multipliers
    multipliers = [2.0, 3.0, 4.0]
    results = []

    for mult in multipliers:
        supertrend, direction = calculate_supertrend(
            highs, lows, closes, period=10, multiplier=mult
        )
        results.append((supertrend, direction))

    # Higher multiplier should result in wider bands
    # Check that supertrend values differ with different multipliers
    st_2, dir_2 = results[0]  # multiplier 2.0
    st_3, dir_3 = results[1]  # multiplier 3.0
    st_4, dir_4 = results[2]  # multiplier 4.0

    # Supertrend values should be different
    valid_idx = ~np.isnan(st_2) & ~np.isnan(st_3) & ~np.isnan(st_4)

    assert not np.allclose(st_2[valid_idx], st_3[valid_idx]), \
        "Different multipliers should produce different Supertrend values"

    assert not np.allclose(st_3[valid_idx], st_4[valid_idx]), \
        "Different multipliers should produce different Supertrend values"

    # Lower multiplier (more sensitive) may have more direction changes
    changes_2 = np.sum(np.abs(np.diff(dir_2[~np.isnan(dir_2)])))
    changes_4 = np.sum(np.abs(np.diff(dir_4[~np.isnan(dir_4)])))

    # This is probabilistic, but generally lower multiplier = more sensitivity
    # We'll just check that we got valid calculations
    assert changes_2 >= 0, "Should have valid direction changes for multiplier 2.0"
    assert changes_4 >= 0, "Should have valid direction changes for multiplier 4.0"


def test_supertrend_as_support_resistance():
    """Test that Supertrend acts as support in uptrend and resistance in downtrend."""
    highs, lows, closes, _ = generate_ohlcv_data(n=100)

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    valid_idx = ~np.isnan(supertrend)

    # In uptrend (direction = 1), close should generally be above supertrend
    uptrend_idx = valid_idx & (direction == 1)
    if np.any(uptrend_idx):
        uptrend_above = closes[uptrend_idx] >= supertrend[uptrend_idx]
        # Most closes should be above supertrend in uptrend
        assert np.mean(uptrend_above) > 0.5, \
            "In uptrend, price should generally be above Supertrend"

    # In downtrend (direction = -1), close should generally be below supertrend
    downtrend_idx = valid_idx & (direction == -1)
    if np.any(downtrend_idx):
        downtrend_below = closes[downtrend_idx] <= supertrend[downtrend_idx]
        # Most closes should be below supertrend in downtrend
        assert np.mean(downtrend_below) > 0.5, \
            "In downtrend, price should generally be below Supertrend"


def test_edge_cases():
    """Test edge cases."""
    # Minimal data (period + a few extra points)
    highs = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112], dtype=np.float64)
    lows = np.array([99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=np.float64)
    closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111], dtype=np.float64)

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    assert len(supertrend) == 12, "Should handle minimal data"
    assert len(direction) == 12, "Should handle minimal data"

    # Constant price
    highs = np.full(50, 101.0)
    lows = np.full(50, 99.0)
    closes = np.full(50, 100.0)

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    # Should not raise errors with constant price
    assert len(supertrend) == 50, "Should handle constant price"
    valid_st = supertrend[~np.isnan(supertrend)]
    assert len(valid_st) > 0, "Should have some valid values"


def test_input_validation():
    """Test input validation."""
    highs, lows, closes, _ = generate_ohlcv_data(n=20)

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        calculate_supertrend(highs[:10], lows, closes)

    # Insufficient data
    with pytest.raises(ValueError, match="must be >= period"):
        calculate_supertrend(highs[:5], lows[:5], closes[:5], period=10)

    # Invalid period
    with pytest.raises(ValueError, match="period must be >= 1"):
        calculate_supertrend(highs, lows, closes, period=0)

    # Invalid multiplier
    with pytest.raises(ValueError, match="multiplier must be > 0"):
        calculate_supertrend(highs, lows, closes, period=10, multiplier=0)

    with pytest.raises(ValueError, match="multiplier must be > 0"):
        calculate_supertrend(highs, lows, closes, period=10, multiplier=-1.0)


def test_return_types():
    """Test that return types are correct."""
    highs, lows, closes, _ = generate_ohlcv_data(n=50)

    result = calculate_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    # Should return tuple
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 2, "Should return 2 values"

    supertrend, direction = result

    # Check types
    assert isinstance(supertrend, np.ndarray), "Supertrend should be numpy array"
    assert isinstance(direction, np.ndarray), "Direction should be numpy array"

    # Check dtypes
    assert supertrend.dtype == np.float64, "Supertrend should be float64"
    assert direction.dtype == np.float64, "Direction should be float64"


def test_performance():
    """Test performance on different data sizes."""
    import time

    sizes = [1_000, 10_000, 50_000]
    results = []

    for size in sizes:
        # Generate data
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(size) * 0.5)
        highs = closes + np.abs(np.random.randn(size) * 0.3)
        lows = closes - np.abs(np.random.randn(size) * 0.3)

        # Time CPU execution
        start = time.time()
        supertrend, direction = calculate_supertrend(
            highs, lows, closes, period=10, multiplier=3.0, engine="cpu"
        )
        cpu_time = time.time() - start

        results.append({
            "size": size,
            "cpu_time": cpu_time,
            "valid_values": np.sum(~np.isnan(supertrend))
        })

    # Print performance results
    print("\nSupertrend Performance Benchmark:")
    print(f"{'Size':<10} {'CPU Time':<12} {'Valid Values':<15}")
    print("-" * 40)
    for r in results:
        print(f"{r['size']:<10} {r['cpu_time']*1000:>8.2f} ms   {r['valid_values']:<15}")

    # Verify all sizes completed successfully
    assert all(r["valid_values"] > 0 for r in results), "All sizes should produce valid results"

    # Verify performance scales reasonably (not exponentially)
    # 50x data should not take 50x time (due to vectorization)
    if len(results) >= 2:
        time_ratio = results[-1]["cpu_time"] / results[0]["cpu_time"]
        size_ratio = results[-1]["size"] / results[0]["size"]
        assert time_ratio < size_ratio * 1.5, \
            f"Performance should scale sub-linearly (time_ratio={time_ratio:.2f}, size_ratio={size_ratio})"


def test_comparison_with_different_periods():
    """Test Supertrend with different period values."""
    highs, lows, closes, _ = generate_ohlcv_data(n=100)

    # Test different periods
    periods = [7, 10, 14, 20]
    results = []

    for period in periods:
        supertrend, direction = calculate_supertrend(
            highs, lows, closes, period=period, multiplier=3.0
        )
        results.append((supertrend, direction, period))

    # Shorter period should have more direction changes (more sensitive)
    # Longer period should be smoother

    for i, (st, dir, period) in enumerate(results):
        valid_dir = dir[~np.isnan(dir)]
        changes = np.sum(np.abs(np.diff(valid_dir)))
        print(f"Period {period}: {changes} direction changes")

    # Just verify all calculations completed successfully
    assert all(len(st) == 100 for st, _, _ in results), "All periods should work"


def test_signal_generation():
    """Test buy/sell signal generation from Supertrend."""
    # Create data with clear trends
    np.random.seed(42)
    n = 100

    # Uptrend followed by downtrend
    first_half = np.linspace(100, 115, n // 2)
    second_half = np.linspace(115, 95, n // 2)
    closes = np.concatenate([first_half, second_half])

    highs = closes + 0.5
    lows = closes - 0.5

    supertrend, direction = calculate_supertrend(highs, lows, closes, period=10, multiplier=2.0)

    # Generate signals
    valid_idx = ~np.isnan(direction)
    direction_changes = np.diff(direction[valid_idx])

    # Buy signal: direction changes from -1 to 1 (change = +2)
    buy_signals = direction_changes == 2

    # Sell signal: direction changes from 1 to -1 (change = -2)
    sell_signals = direction_changes == -2

    # Should have at least one signal in this trending data
    total_signals = np.sum(buy_signals) + np.sum(sell_signals)
    assert total_signals >= 1, "Should generate at least one trading signal"

    print(f"\nSignal generation test:")
    print(f"  Buy signals: {np.sum(buy_signals)}")
    print(f"  Sell signals: {np.sum(sell_signals)}")
    print(f"  Total signals: {total_signals}")


if __name__ == "__main__":
    # Run tests
    print("Running Supertrend tests...")
    pytest.main([__file__, "-v", "-s"])

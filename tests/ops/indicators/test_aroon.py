"""
Test suite for Aroon indicator.

Tests cover:
- Basic calculation correctness
- Value ranges (0-100)
- Trend identification
- Crossover signals
- Edge cases
- GPU/CPU parity
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.ops.indicators import calculate_aroon
from kimsfinance.ops.indicators.aroon import CUPY_AVAILABLE


def generate_ohlc_data(n=100, seed=42):
    """Generate test OHLC data for Aroon testing."""
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return highs, lows, closes


def generate_uptrend_data(n=100):
    """Generate data with consistent uptrend."""
    closes = 100 + np.arange(n) * 0.5
    highs = closes + 0.3
    lows = closes - 0.3
    return highs, lows, closes


def generate_downtrend_data(n=100):
    """Generate data with consistent downtrend."""
    closes = 150 - np.arange(n) * 0.5
    highs = closes + 0.3
    lows = closes - 0.3
    return highs, lows, closes


def test_basic_calculation():
    """Test basic Aroon calculation returns correct array lengths."""
    highs, lows, closes = generate_ohlc_data(n=100)

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Arrays should have same length as input
    assert len(aroon_up) == len(highs) == 100
    assert len(aroon_down) == len(highs) == 100

    # Arrays should be NumPy arrays
    assert isinstance(aroon_up, np.ndarray)
    assert isinstance(aroon_down, np.ndarray)


def test_both_values_calculated():
    """Test that Aroon returns both Up and Down with values."""
    highs, lows, closes = generate_ohlc_data(n=100)

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Should have non-NaN values after warmup period
    warmup = 25
    assert not np.all(np.isnan(aroon_up[warmup:]))
    assert not np.all(np.isnan(aroon_down[warmup:]))

    # Should have NaN values at start (warmup period)
    assert np.all(np.isnan(aroon_up[: warmup - 1]))
    assert np.all(np.isnan(aroon_down[: warmup - 1]))


def test_value_ranges():
    """Test that Aroon Up and Down are in range [0, 100]."""
    highs, lows, closes = generate_ohlc_data(n=200)

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Remove NaN values for testing
    aroon_up_valid = aroon_up[~np.isnan(aroon_up)]
    aroon_down_valid = aroon_down[~np.isnan(aroon_down)]

    # All values should be in [0, 100]
    assert np.all(aroon_up_valid >= 0.0)
    assert np.all(aroon_up_valid <= 100.0)
    assert np.all(aroon_down_valid >= 0.0)
    assert np.all(aroon_down_valid <= 100.0)


def test_uptrend_detection():
    """Test Aroon correctly identifies uptrend."""
    highs, lows, closes = generate_uptrend_data(n=100)

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Remove warmup period
    warmup = 25
    aroon_up_valid = aroon_up[warmup:]
    aroon_down_valid = aroon_down[warmup:]

    # In strong uptrend:
    # 1. Aroon Up should be high (near 100)
    assert (
        np.mean(aroon_up_valid) > 70
    ), f"Expected Aroon Up > 70 in uptrend, got {np.mean(aroon_up_valid):.2f}"

    # 2. Aroon Up should be > Aroon Down
    assert np.mean(aroon_up_valid) > np.mean(
        aroon_down_valid
    ), f"Aroon Up ({np.mean(aroon_up_valid):.2f}) should be > Aroon Down ({np.mean(aroon_down_valid):.2f})"

    # 3. Most recent value should be 100 (just made new high)
    assert (
        aroon_up[-1] == 100.0
    ), f"Expected Aroon Up = 100 at end of uptrend, got {aroon_up[-1]:.2f}"


def test_downtrend_detection():
    """Test Aroon correctly identifies downtrend."""
    highs, lows, closes = generate_downtrend_data(n=100)

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Remove warmup period
    warmup = 25
    aroon_up_valid = aroon_up[warmup:]
    aroon_down_valid = aroon_down[warmup:]

    # In strong downtrend:
    # 1. Aroon Down should be high (near 100)
    assert (
        np.mean(aroon_down_valid) > 70
    ), f"Expected Aroon Down > 70 in downtrend, got {np.mean(aroon_down_valid):.2f}"

    # 2. Aroon Down should be > Aroon Up
    assert np.mean(aroon_down_valid) > np.mean(
        aroon_up_valid
    ), f"Aroon Down ({np.mean(aroon_down_valid):.2f}) should be > Aroon Up ({np.mean(aroon_up_valid):.2f})"

    # 3. Most recent value should be 100 (just made new low)
    assert (
        aroon_down[-1] == 100.0
    ), f"Expected Aroon Down = 100 at end of downtrend, got {aroon_down[-1]:.2f}"


def test_crossover_signals():
    """Test Aroon crossover signal detection."""
    # Create data with trend reversal
    n = 100
    # Uptrend then downtrend
    uptrend = 100 + np.arange(50) * 0.5
    downtrend = uptrend[-1] - np.arange(50) * 0.5
    closes = np.concatenate([uptrend, downtrend])
    highs = closes + 0.3
    lows = closes - 0.3

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Detect crossovers (ignoring NaN values)
    valid_idx = ~(np.isnan(aroon_up) | np.isnan(aroon_down))

    # Find where Aroon Up crosses above Aroon Down (bullish)
    bullish_cross = (
        (aroon_up[1:] > aroon_down[1:]) & (aroon_up[:-1] <= aroon_down[:-1]) & valid_idx[1:]
    )

    # Find where Aroon Down crosses above Aroon Up (bearish)
    bearish_cross = (
        (aroon_down[1:] > aroon_up[1:]) & (aroon_down[:-1] <= aroon_up[:-1]) & valid_idx[1:]
    )

    # Should detect at least one crossover in this data
    total_crosses = np.sum(bullish_cross) + np.sum(bearish_cross)
    assert total_crosses >= 1, "Should detect at least one Aroon crossover"


def test_edge_cases():
    """Test edge cases for Aroon calculation."""
    # Test 1: Constant price (no movement)
    n = 50
    constant_close = np.full(n, 100.0)
    constant_high = np.full(n, 100.5)
    constant_low = np.full(n, 99.5)

    aroon_up, aroon_down = calculate_aroon(constant_high, constant_low, period=25, engine="cpu")

    # With constant prices, both should be 100 (high/low occurred at every point)
    valid_idx = ~np.isnan(aroon_up)
    if np.any(valid_idx):
        # All values should be 100 (every point is the "most recent" high/low)
        assert np.allclose(aroon_up[valid_idx], 100.0), "Constant price should give Aroon Up = 100"
        assert np.allclose(
            aroon_down[valid_idx], 100.0
        ), "Constant price should give Aroon Down = 100"

    # Test 2: Minimal data (exactly period length)
    highs, lows, closes = generate_ohlc_data(n=25)
    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Should handle minimal data without crashing
    assert len(aroon_up) == 25
    assert len(aroon_down) == 25

    # Test 3: Very short period
    highs, lows, closes = generate_ohlc_data(n=100)
    aroon_up, aroon_down = calculate_aroon(highs, lows, period=5, engine="cpu")

    # Should work with short periods
    assert len(aroon_up) == 100
    valid_aroon = aroon_up[~np.isnan(aroon_up)]
    assert len(valid_aroon) > 80  # Should have plenty of valid values


def test_different_periods():
    """Test Aroon with different period parameters."""
    highs, lows, closes = generate_ohlc_data(n=100)

    # Test various periods
    periods = [10, 14, 25, 50]
    results = []

    for period in periods:
        aroon_up, aroon_down = calculate_aroon(highs, lows, period=period, engine="cpu")
        results.append((aroon_up, aroon_down))

        # All should return valid arrays
        assert len(aroon_up) == 100
        assert len(aroon_down) == 100

        # Should have some valid values after warmup
        warmup = period
        if warmup < 100:
            assert not np.all(np.isnan(aroon_up[warmup:]))
            assert not np.all(np.isnan(aroon_down[warmup:]))

    # Longer periods should have fewer non-NaN values at start
    aroon_up_10 = results[0][0]
    aroon_up_50 = results[3][0]

    nan_count_10 = np.sum(np.isnan(aroon_up_10))
    nan_count_50 = np.sum(np.isnan(aroon_up_50))

    assert nan_count_50 > nan_count_10, "Longer period should have more NaN values"


def test_nan_handling():
    """Test that NaN values are in expected positions."""
    highs, lows, closes = generate_ohlc_data(n=100)

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

    # Check NaN positions
    # First (period-1) values should be NaN
    assert np.all(np.isnan(aroon_up[:24]))
    assert np.all(np.isnan(aroon_down[:24]))

    # After warmup period, should have valid values
    assert not np.all(np.isnan(aroon_up[25:]))
    assert not np.all(np.isnan(aroon_down[25:]))


def test_known_values():
    """Test against known Aroon values."""
    # Create simple test case with known values
    # Uptrend: each new bar makes a higher high
    highs = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    lows = np.array([99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    period = 5

    aroon_up, aroon_down = calculate_aroon(highs, lows, period=period, engine="cpu")

    # At index 4 (first valid value), the highest high is at index 4 (most recent)
    # periods_since_high = 0
    # Aroon Up = ((5 - 0) / 5) * 100 = 100
    assert aroon_up[4] == 100.0, f"Expected 100, got {aroon_up[4]}"

    # At index 4, the lowest low is at index 0 (4 periods ago)
    # periods_since_low = 4
    # Aroon Down = ((5 - 4) / 5) * 100 = 20
    assert aroon_down[4] == 20.0, f"Expected 20, got {aroon_down[4]}"

    # At index 10, highest high is at index 10 (most recent)
    # Aroon Up = 100
    assert aroon_up[10] == 100.0, f"Expected 100, got {aroon_up[10]}"

    # At index 10, lowest low is at index 6 (within 5-period window: 6,7,8,9,10)
    # periods_since_low = 4
    # Aroon Down = ((5 - 4) / 5) * 100 = 20
    assert aroon_down[10] == 20.0, f"Expected 20, got {aroon_down[10]}"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
def test_gpu_cpu_match():
    """Test GPU and CPU implementations produce identical results."""
    highs, lows, closes = generate_ohlc_data(n=200)

    cpu_up, cpu_down = calculate_aroon(highs, lows, period=25, engine="cpu")
    gpu_up, gpu_down = calculate_aroon(highs, lows, period=25, engine="gpu")

    # Should match within floating point tolerance
    np.testing.assert_allclose(cpu_up, gpu_up, rtol=1e-10)
    np.testing.assert_allclose(cpu_down, gpu_down, rtol=1e-10)


def test_invalid_period():
    """Test that invalid period raises ValueError."""
    highs, lows, closes = generate_ohlc_data(n=100)

    with pytest.raises(ValueError, match="period must be >= 1"):
        calculate_aroon(highs, lows, period=0, engine="cpu")

    with pytest.raises(ValueError, match="period must be >= 1"):
        calculate_aroon(highs, lows, period=-1, engine="cpu")


def test_insufficient_data():
    """Test that insufficient data raises ValueError."""
    short_highs = np.array([100, 101, 102])
    short_lows = np.array([99, 100, 101])

    with pytest.raises(ValueError, match="Insufficient data"):
        calculate_aroon(short_highs, short_lows, period=25, engine="cpu")


def test_mismatched_lengths():
    """Test that mismatched input lengths raise ValueError."""
    highs = np.array([100, 101, 102, 103, 104])
    lows = np.array([99, 100, 101])  # Different length

    with pytest.raises(ValueError, match="must have same length"):
        calculate_aroon(highs, lows, period=5, engine="cpu")


def test_performance():
    """Test performance on different data sizes."""
    import time

    sizes = [1_000, 10_000]

    for size in sizes:
        highs, lows, closes = generate_ohlc_data(n=size)

        start = time.time()
        aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0, f"Aroon too slow for {size} rows: {elapsed:.3f}s"

        # Results should be valid
        assert len(aroon_up) == size
        assert not np.all(np.isnan(aroon_up))

    print(f"\nPerformance test passed for sizes: {sizes}")


if __name__ == "__main__":
    # Run tests
    print("Running Aroon tests...")
    print("\n1. Basic calculation...")
    test_basic_calculation()
    print("✓ Passed")

    print("\n2. Both values calculated...")
    test_both_values_calculated()
    print("✓ Passed")

    print("\n3. Value ranges...")
    test_value_ranges()
    print("✓ Passed")

    print("\n4. Uptrend detection...")
    test_uptrend_detection()
    print("✓ Passed")

    print("\n5. Downtrend detection...")
    test_downtrend_detection()
    print("✓ Passed")

    print("\n6. Crossover signals...")
    test_crossover_signals()
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

    print("\n10. Known values...")
    test_known_values()
    print("✓ Passed")

    print("\n11. GPU/CPU match...")
    test_gpu_cpu_match()
    print("✓ Passed")

    print("\n12. Invalid period...")
    test_invalid_period()
    print("✓ Passed")

    print("\n13. Insufficient data...")
    test_insufficient_data()
    print("✓ Passed")

    print("\n14. Mismatched lengths...")
    test_mismatched_lengths()
    print("✓ Passed")

    print("\n15. Performance...")
    test_performance()
    print("✓ Passed")

    print("\n" + "=" * 50)
    print("All Aroon tests passed! ✓")
    print("=" * 50)

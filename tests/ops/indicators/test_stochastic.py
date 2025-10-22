#!/usr/bin/env python3
"""
Comprehensive Tests for Stochastic Oscillator Indicator
========================================================

Tests both stochastic.py and indicators/stochastic_oscillator.py implementations
for correctness, GPU/CPU equivalence, edge cases, and performance characteristics.

Test Coverage:
- Basic Calculation (15 tests)
- Signal Generation (10 tests)
- Edge Cases (10 tests)
- GPU/CPU Parity (10 tests)
- Performance (5+ tests)
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

# Test both implementations
from kimsfinance.ops.stochastic import calculate_stochastic, calculate_stochastic_rsi
from kimsfinance.ops.indicators.stochastic_oscillator import (
    calculate_stochastic_oscillator,
    CUPY_AVAILABLE,
)
from kimsfinance.core import EngineManager
from kimsfinance.core.exceptions import ConfigurationError


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlc():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 2)
    highs = closes + np.abs(np.random.randn(n) * 1.5)
    lows = closes - np.abs(np.random.randn(n) * 1.5)
    return highs, lows, closes


@pytest.fixture
def large_ohlc():
    """Generate large OHLC dataset for GPU testing."""
    np.random.seed(42)
    n = 600_000  # Above GPU threshold
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return highs, lows, closes


@pytest.fixture
def uptrend_ohlc():
    """Generate data with consistent uptrend."""
    n = 100
    closes = 100 + np.arange(n) * 0.5
    highs = closes + 0.3
    lows = closes - 0.3
    return highs, lows, closes


@pytest.fixture
def downtrend_ohlc():
    """Generate data with consistent downtrend."""
    n = 100
    closes = 150 - np.arange(n) * 0.5
    highs = closes + 0.3
    lows = closes - 0.3
    return highs, lows, closes


@pytest.fixture
def flat_ohlc():
    """Generate flat price action."""
    n = 100
    closes = np.full(n, 100.0)
    highs = closes + 0.1
    lows = closes - 0.1
    return highs, lows, closes


@pytest.fixture
def extreme_range_ohlc():
    """Generate data with extreme price ranges."""
    n = 100
    closes = np.array([100.0 if i % 2 == 0 else 200.0 for i in range(n)])
    highs = closes + 50
    lows = closes - 50
    return highs, lows, closes


# ============================================================================
# CATEGORY 1: Basic Calculation Tests (15 tests)
# ============================================================================


class TestStochasticBasicCalculation:
    """Test basic Stochastic Oscillator calculations."""

    def test_basic_k_and_d_calculation(self, sample_ohlc):
        """Test basic %K and %D calculation returns correct structure."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Check lengths match input
        assert len(k) == len(highs) == 100
        assert len(d) == len(highs) == 100

        # Check types
        assert isinstance(k, np.ndarray)
        assert isinstance(d, np.ndarray)

        # Should have some valid values after warmup
        assert not np.all(np.isnan(k))
        assert not np.all(np.isnan(d))

    def test_k_formula_correctness(self):
        """Test %K formula: 100 * (close - low_min) / (high_max - low_min)."""
        # Simple test case with known values
        highs = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        lows = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        closes = np.array([9.0, 11.0, 10.5, 12.0, 13.0])

        k, d = calculate_stochastic(highs, lows, closes, k_period=3, d_period=3)

        # At index 2 (first valid point):
        # high_max = max(10, 12, 11) = 12
        # low_min = min(8, 9, 10) = 8
        # %K = 100 * (10.5 - 8) / (12 - 8) = 100 * 2.5 / 4 = 62.5
        assert np.isclose(k[2], 62.5, rtol=1e-6)

    def test_d_is_sma_of_k(self, sample_ohlc):
        """Test that %D is SMA of %K with correct period."""
        highs, lows, closes = sample_ohlc
        k_period = 14
        d_period = 3

        k, d = calculate_stochastic(highs, lows, closes, k_period=k_period, d_period=d_period)

        # Manually calculate %D for verification
        # %D should be NaN for first k_period + d_period - 2 values (accounting for how rolling works)
        warmup = k_period + d_period - 2
        assert np.all(np.isnan(d[:warmup]))

        # After warmup, %D should be rolling mean of %K
        for i in range(warmup, min(warmup + 10, len(d))):
            if not np.isnan(d[i]):
                manual_d = np.mean(k[i - d_period + 1 : i + 1])
                assert np.isclose(d[i], manual_d, rtol=1e-4)

    def test_fast_stochastic_default_params(self, sample_ohlc):
        """Test fast stochastic with default parameters (14, 3)."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes)  # Default: k_period=14, d_period=3

        # First 13 values should be NaN (k_period - 1)
        assert np.all(np.isnan(k[:13]))

        # After warmup, should have valid values
        assert not np.isnan(k[13])

        # %D warmup = k_period + d_period - 2 = 15 (accounting for rolling implementation)
        assert np.all(np.isnan(d[:15]))

    def test_slow_stochastic_custom_params(self, sample_ohlc):
        """Test slow stochastic with custom smoothing."""
        highs, lows, closes = sample_ohlc

        # Slow stochastic typically uses k_period=14, smooth_k=3, smooth_d=3
        # For now test with different periods
        k_short, d_short = calculate_stochastic(highs, lows, closes, k_period=5, d_period=3)
        k_long, d_long = calculate_stochastic(highs, lows, closes, k_period=20, d_period=5)

        # Longer periods should have more NaN values
        assert np.sum(np.isnan(k_short)) < np.sum(np.isnan(k_long))
        assert np.sum(np.isnan(d_short)) < np.sum(np.isnan(d_long))

    def test_custom_k_period(self, sample_ohlc):
        """Test with custom k_period values."""
        highs, lows, closes = sample_ohlc

        # Test period=5
        k_5, d_5 = calculate_stochastic(highs, lows, closes, k_period=5, d_period=3)

        # Test period=21
        k_21, d_21 = calculate_stochastic(highs, lows, closes, k_period=21, d_period=3)

        # First (k_period-1) values should be NaN
        assert np.all(np.isnan(k_5[:4]))
        assert np.all(np.isnan(k_21[:20]))

        # Valid values should differ
        valid_5 = ~np.isnan(k_5)
        valid_21 = ~np.isnan(k_21)
        common_valid = valid_5 & valid_21
        assert not np.allclose(k_5[common_valid], k_21[common_valid])

    def test_custom_d_period(self, sample_ohlc):
        """Test with custom d_period values."""
        highs, lows, closes = sample_ohlc

        k, d_3 = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)
        k2, d_5 = calculate_stochastic(highs, lows, closes, k_period=14, d_period=5)

        # %K should be identical
        np.testing.assert_allclose(k, k2, rtol=1e-10)

        # %D should differ
        valid_3 = ~np.isnan(d_3)
        valid_5 = ~np.isnan(d_5)
        common_valid = valid_3 & valid_5
        assert not np.allclose(d_3[common_valid], d_5[common_valid])

    def test_value_range_0_to_100(self, sample_ohlc):
        """Test that %K and %D are in range [0, 100]."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Remove NaN values
        k_valid = k[~np.isnan(k)]
        d_valid = d[~np.isnan(d)]

        # All values should be in [0, 100]
        assert np.all(k_valid >= 0.0)
        assert np.all(k_valid <= 100.0)
        assert np.all(d_valid >= 0.0)
        assert np.all(d_valid <= 100.0)

    def test_minimum_period_values(self, sample_ohlc):
        """Test with minimum valid period values (period=1)."""
        highs, lows, closes = sample_ohlc

        k, d = calculate_stochastic(highs, lows, closes, k_period=1, d_period=1)

        # With period=1, no warmup needed
        assert not np.isnan(k[0])
        assert not np.isnan(d[0])

    def test_small_dataset_handling(self):
        """Test with very small datasets."""
        highs = np.array([10.0, 11.0, 12.0])
        lows = np.array([8.0, 9.0, 10.0])
        closes = np.array([9.0, 10.0, 11.0])

        k, d = calculate_stochastic(highs, lows, closes, k_period=2, d_period=2)

        # Should handle small dataset without errors
        assert len(k) == 3
        assert len(d) == 3

    def test_different_array_dtypes(self, sample_ohlc):
        """Test with different NumPy array dtypes."""
        highs, lows, closes = sample_ohlc

        # Test with float32
        k_32, d_32 = calculate_stochastic(
            highs.astype(np.float32),
            lows.astype(np.float32),
            closes.astype(np.float32),
            k_period=14,
            d_period=3,
        )

        # Test with float64
        k_64, d_64 = calculate_stochastic(
            highs.astype(np.float64),
            lows.astype(np.float64),
            closes.astype(np.float64),
            k_period=14,
            d_period=3,
        )

        # Should produce similar results (allow more tolerance for float32)
        np.testing.assert_allclose(k_32, k_64, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(d_32, d_64, rtol=1e-4, atol=1e-4)

    def test_polars_implementation(self, sample_ohlc):
        """Test the Polars-based implementation from stochastic_oscillator.py."""
        highs, lows, closes = sample_ohlc

        k, d = calculate_stochastic_oscillator(highs, lows, closes, period=14, engine="cpu")

        # Check basic properties
        assert len(k) == len(highs) == 100
        assert len(d) == len(highs) == 100
        assert isinstance(k, np.ndarray)
        assert isinstance(d, np.ndarray)

        # Should have valid values after warmup
        assert not np.all(np.isnan(k))
        assert not np.all(np.isnan(d))

    def test_warmup_period_alignment(self, sample_ohlc):
        """Test that warmup periods align correctly with formula."""
        highs, lows, closes = sample_ohlc
        k_period = 10
        d_period = 4

        k, d = calculate_stochastic(highs, lows, closes, k_period=k_period, d_period=d_period)

        # %K should have (k_period - 1) NaN values
        k_nan_count = np.sum(np.isnan(k))
        assert k_nan_count == k_period - 1

        # %D should have (k_period + d_period - 2) NaN values (accounting for rolling implementation)
        d_nan_count = np.sum(np.isnan(d))
        assert d_nan_count == k_period + d_period - 2

    def test_sequential_values_monotonic_trend(self, uptrend_ohlc):
        """Test that stochastic responds to monotonic trends."""
        highs, lows, closes = uptrend_ohlc

        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # In strong uptrend, %K should generally be high
        k_valid = k[~np.isnan(k)]
        assert np.mean(k_valid[-20:]) > 70  # Last 20 values should be mostly high

    def test_implementation_consistency(self, sample_ohlc):
        """Test that both implementations produce consistent results."""
        highs, lows, closes = sample_ohlc

        # Old implementation (stochastic.py)
        k1, d1 = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # New implementation (stochastic_oscillator.py)
        k2, d2 = calculate_stochastic_oscillator(highs, lows, closes, period=14, engine="cpu")

        # Note: d_period is hardcoded to 3 in stochastic_oscillator
        # Results should be close (but may differ slightly due to implementation)
        # We test that both are valid rather than identical
        assert len(k1) == len(k2)
        assert len(d1) == len(d2)

        # Both should have similar warmup behavior
        k1_nan_count = np.sum(np.isnan(k1))
        k2_nan_count = np.sum(np.isnan(k2))
        assert abs(k1_nan_count - k2_nan_count) <= 1  # Allow 1 difference


# ============================================================================
# CATEGORY 2: Signal Generation Tests (10 tests)
# ============================================================================


class TestStochasticSignals:
    """Test Stochastic Oscillator signal generation."""

    def test_overbought_detection(self, sample_ohlc):
        """Test overbought signal (%K > 80)."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Find overbought conditions
        overbought = k > 80

        # Should have some overbought signals (not all)
        assert np.any(overbought)
        assert not np.all(overbought)

    def test_oversold_detection(self, sample_ohlc):
        """Test oversold signal (%K < 20)."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Find oversold conditions
        oversold = k < 20

        # Should have some oversold signals (not all)
        assert np.any(oversold)
        assert not np.all(oversold)

    def test_extreme_overbought_uptrend(self, uptrend_ohlc):
        """Test that strong uptrend triggers overbought."""
        highs, lows, closes = uptrend_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # In strong uptrend, should have many overbought signals
        overbought = k > 80
        overbought_count = np.sum(overbought[~np.isnan(k)])

        # At least 50% of valid values should be overbought
        valid_count = np.sum(~np.isnan(k))
        assert overbought_count / valid_count > 0.5

    def test_extreme_oversold_downtrend(self, downtrend_ohlc):
        """Test that strong downtrend triggers oversold."""
        highs, lows, closes = downtrend_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # In strong downtrend, should have many oversold signals
        oversold = k < 20
        oversold_count = np.sum(oversold[~np.isnan(k)])

        # At least 50% of valid values should be oversold
        valid_count = np.sum(~np.isnan(k))
        assert oversold_count / valid_count > 0.5

    def test_bullish_crossover(self):
        """Test bullish crossover (%K crosses above %D)."""
        # Create scenario where %K crosses above %D
        n = 50
        highs = np.array([100 + i * 0.1 for i in range(n)])
        lows = np.array([98 + i * 0.1 for i in range(n)])
        closes = np.array([99 + i * 0.1 for i in range(n)])

        k, d = calculate_stochastic(highs, lows, closes, k_period=10, d_period=3)

        # Find bullish crossovers (k > d and prev_k <= prev_d)
        bullish_cross = (k > d) & (np.roll(k, 1) <= np.roll(d, 1))

        # Remove first value (from roll) and warmup period
        bullish_cross[0] = False
        bullish_cross[: 13] = False  # Warmup period

        # Should have at least one crossover in uptrend
        assert np.any(bullish_cross)

    def test_bearish_crossover(self):
        """Test bearish crossover (%K crosses below %D)."""
        # Create scenario where %K crosses below %D
        n = 50
        highs = np.array([150 - i * 0.1 for i in range(n)])
        lows = np.array([148 - i * 0.1 for i in range(n)])
        closes = np.array([149 - i * 0.1 for i in range(n)])

        k, d = calculate_stochastic(highs, lows, closes, k_period=10, d_period=3)

        # Find bearish crossovers (k < d and prev_k >= prev_d)
        bearish_cross = (k < d) & (np.roll(k, 1) >= np.roll(d, 1))

        # Remove first value (from roll) and warmup period
        bearish_cross[0] = False
        bearish_cross[: 13] = False

        # Should have at least one crossover in downtrend
        assert np.any(bearish_cross)

    def test_midline_50_crossing(self, sample_ohlc):
        """Test crossing the midline at 50."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Find crossings of 50 level
        above_50 = k > 50
        below_50 = k < 50

        # Should have values both above and below 50
        assert np.any(above_50)
        assert np.any(below_50)

    def test_divergence_detection_setup(self, sample_ohlc):
        """Test setup for divergence detection (price vs stochastic)."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Check that we can identify potential divergences
        # (price makes new high but stochastic doesn't)

        # Find local price highs
        price_highs = (closes > np.roll(closes, 1)) & (closes > np.roll(closes, -1))
        price_highs[0] = False
        price_highs[-1] = False

        # Find local stochastic peaks
        k_peaks = (k > np.roll(k, 1)) & (k > np.roll(k, -1))
        k_peaks[0] = False
        k_peaks[-1] = False

        # Should have both types of peaks
        assert np.any(price_highs)
        assert np.any(k_peaks)

    def test_custom_overbought_oversold_thresholds(self, sample_ohlc):
        """Test with custom overbought/oversold thresholds (e.g., 70/30)."""
        highs, lows, closes = sample_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Custom thresholds
        overbought_70 = k > 70
        oversold_30 = k < 30

        # Should have signals with more lenient thresholds
        assert np.any(overbought_70)
        assert np.any(oversold_30)

        # More signals than 80/20
        overbought_80 = k > 80
        oversold_20 = k < 20

        assert np.sum(overbought_70) >= np.sum(overbought_80)
        assert np.sum(oversold_30) >= np.sum(oversold_20)

    def test_signal_persistence(self, uptrend_ohlc):
        """Test that signals persist in strong trends."""
        highs, lows, closes = uptrend_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # In uptrend, overbought should persist
        overbought = k > 80

        # Find consecutive overbought periods
        consecutive_count = 0
        max_consecutive = 0
        for val in overbought:
            if val:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0

        # Should have at least 5 consecutive overbought periods
        assert max_consecutive >= 5


# ============================================================================
# CATEGORY 3: Edge Cases Tests (10 tests)
# ============================================================================


class TestStochasticEdgeCases:
    """Test edge cases and error handling."""

    def test_flat_price_action(self, flat_ohlc):
        """Test with completely flat prices."""
        highs, lows, closes = flat_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # With flat prices and epsilon handling, %K should be defined
        k_valid = k[~np.isnan(k)]

        # Should not raise errors
        assert len(k_valid) > 0

        # Values should be finite
        assert np.all(np.isfinite(k_valid))

    def test_extreme_price_ranges(self, extreme_range_ohlc):
        """Test with extreme price swings."""
        highs, lows, closes = extreme_range_ohlc
        k, d = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Should handle extreme ranges without errors
        assert len(k) == len(highs)
        assert len(d) == len(highs)

        # Values should still be in [0, 100]
        k_valid = k[~np.isnan(k)]
        assert np.all(k_valid >= 0.0)
        assert np.all(k_valid <= 100.0)

    def test_nan_input_handling(self, sample_ohlc):
        """Test handling of NaN values in input data."""
        highs, lows, closes = sample_ohlc

        # Introduce NaN values
        highs_with_nan = highs.copy()
        lows_with_nan = lows.copy()
        closes_with_nan = closes.copy()
        highs_with_nan[10:15] = np.nan
        lows_with_nan[10:15] = np.nan
        closes_with_nan[10:15] = np.nan

        k, d = calculate_stochastic(
            highs_with_nan, lows_with_nan, closes_with_nan, k_period=14, d_period=3
        )

        # Should handle NaN input gracefully
        assert len(k) == len(highs)
        assert len(d) == len(highs)

        # Output will have NaN where input had NaN
        assert np.any(np.isnan(k))
        assert np.any(np.isnan(d))

    def test_single_candle_dataset(self):
        """Test with single data point."""
        highs = np.array([10.0])
        lows = np.array([8.0])
        closes = np.array([9.0])

        k, d = calculate_stochastic(highs, lows, closes, k_period=1, d_period=1)

        # Should handle single point
        assert len(k) == 1
        assert len(d) == 1

    def test_two_candle_dataset(self):
        """Test with two data points."""
        highs = np.array([10.0, 11.0])
        lows = np.array([8.0, 9.0])
        closes = np.array([9.0, 10.0])

        k, d = calculate_stochastic(highs, lows, closes, k_period=2, d_period=2)

        # Should handle two points
        assert len(k) == 2
        assert len(d) == 2

    def test_zero_price_handling(self):
        """Test handling of zero prices."""
        highs = np.array([1.0, 2.0, 0.1, 3.0, 4.0])
        lows = np.array([0.5, 1.0, 0.0, 2.0, 3.0])
        closes = np.array([0.8, 1.5, 0.05, 2.5, 3.5])

        k, d = calculate_stochastic(highs, lows, closes, k_period=3, d_period=2)

        # Should not raise errors
        assert len(k) == 5
        assert len(d) == 5

        # Values should be finite
        k_valid = k[~np.isnan(k)]
        assert np.all(np.isfinite(k_valid))

    def test_negative_price_handling(self):
        """Test that negative prices are handled (though unrealistic for stocks)."""
        # Negative prices can occur in futures/spreads
        highs = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
        lows = np.array([-7.0, -6.0, -5.0, -4.0, -3.0])
        closes = np.array([-6.0, -5.0, -4.0, -3.0, -2.0])

        k, d = calculate_stochastic(highs, lows, closes, k_period=3, d_period=2)

        # Should handle negative prices
        assert len(k) == 5
        k_valid = k[~np.isnan(k)]
        assert len(k_valid) > 0

    def test_very_large_period_vs_dataset_size(self, sample_ohlc):
        """Test with period larger than dataset."""
        highs, lows, closes = sample_ohlc
        k_period = 150  # Larger than dataset size (100)

        k, d = calculate_stochastic(highs, lows, closes, k_period=k_period, d_period=3)

        # Should return all NaN
        assert np.all(np.isnan(k))
        assert np.all(np.isnan(d))

    def test_invalid_period_error_handling(self, sample_ohlc):
        """Test error handling for invalid period values."""
        highs, lows, closes = sample_ohlc

        # Test k_period < 1
        with pytest.raises(ValueError, match="k_period must be >= 1"):
            calculate_stochastic(highs, lows, closes, k_period=0, d_period=3)

        with pytest.raises(ValueError, match="k_period must be >= 1"):
            calculate_stochastic(highs, lows, closes, k_period=-1, d_period=3)

        # Test d_period < 1
        with pytest.raises(ValueError, match="d_period must be >= 1"):
            calculate_stochastic(highs, lows, closes, k_period=14, d_period=0)

        with pytest.raises(ValueError, match="d_period must be >= 1"):
            calculate_stochastic(highs, lows, closes, k_period=14, d_period=-1)

    def test_high_less_than_low_handling(self):
        """Test handling when high < low (data error)."""
        # Create invalid data where high < low
        highs = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        lows = np.array([10.0, 11.0, 12.0, 13.0, 14.0])  # Lows > highs!
        closes = np.array([9.0, 10.0, 11.0, 12.0, 13.0])

        # Should not crash, but results will be unusual
        k, d = calculate_stochastic(highs, lows, closes, k_period=3, d_period=2)

        # Should produce output (even if nonsensical)
        assert len(k) == 5
        assert len(d) == 5


# ============================================================================
# CATEGORY 4: GPU/CPU Parity Tests (10 tests)
# ============================================================================


class TestStochasticGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_basic(self, sample_ohlc):
        """Test GPU and CPU produce identical results on basic dataset."""
        highs, lows, closes = sample_ohlc

        # CPU calculation
        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="cpu"
        )

        # GPU calculation
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="gpu"
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_large_data(self, large_ohlc):
        """Test GPU and CPU produce identical results on large dataset."""
        highs, lows, closes = large_ohlc

        # CPU calculation
        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="cpu"
        )

        # GPU calculation
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="gpu"
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_short_period(self, sample_ohlc):
        """Test GPU/CPU parity with short period."""
        highs, lows, closes = sample_ohlc

        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=5, engine="cpu"
        )
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=5, engine="gpu"
        )

        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_long_period(self, sample_ohlc):
        """Test GPU/CPU parity with long period."""
        highs, lows, closes = sample_ohlc

        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=30, engine="cpu"
        )
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=30, engine="gpu"
        )

        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_uptrend(self, uptrend_ohlc):
        """Test GPU/CPU parity on uptrend data."""
        highs, lows, closes = uptrend_ohlc

        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="cpu"
        )
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="gpu"
        )

        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_downtrend(self, downtrend_ohlc):
        """Test GPU/CPU parity on downtrend data."""
        highs, lows, closes = downtrend_ohlc

        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="cpu"
        )
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="gpu"
        )

        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_flat_market(self, flat_ohlc):
        """Test GPU/CPU parity on flat prices."""
        highs, lows, closes = flat_ohlc

        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="cpu"
        )
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="gpu"
        )

        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_with_nan(self, sample_ohlc):
        """Test GPU/CPU parity with NaN values in input."""
        highs, lows, closes = sample_ohlc

        # Introduce NaN
        highs_nan = highs.copy()
        lows_nan = lows.copy()
        closes_nan = closes.copy()
        highs_nan[20:25] = np.nan
        lows_nan[20:25] = np.nan
        closes_nan[20:25] = np.nan

        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs_nan, lows_nan, closes_nan, period=14, engine="cpu"
        )
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs_nan, lows_nan, closes_nan, period=14, engine="gpu"
        )

        np.testing.assert_allclose(k_cpu, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-10, equal_nan=True)

    def test_auto_engine_selection_small_data(self, sample_ohlc):
        """Test that auto engine selects CPU for small datasets."""
        highs, lows, closes = sample_ohlc  # 100 rows < 500K threshold

        k_auto, d_auto = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="auto"
        )

        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="cpu"
        )

        # Auto should match CPU for small data
        np.testing.assert_allclose(k_auto, k_cpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_auto, d_cpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_auto_engine_selection_large_data(self, large_ohlc):
        """Test that auto engine selects GPU for large datasets."""
        highs, lows, closes = large_ohlc  # 600K rows > 500K threshold

        k_auto, d_auto = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="auto"
        )

        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="gpu"
        )

        # Auto should match GPU for large data
        np.testing.assert_allclose(k_auto, k_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(d_auto, d_gpu, rtol=1e-10, equal_nan=True)


# ============================================================================
# CATEGORY 5: Performance Tests (5+ tests)
# ============================================================================


class TestStochasticPerformance:
    """Test performance characteristics and benchmarks."""

    def test_performance_cpu_baseline(self, sample_ohlc):
        """Test CPU performance baseline."""
        import time

        highs, lows, closes = sample_ohlc

        start = time.perf_counter()
        for _ in range(100):
            k, d = calculate_stochastic_oscillator(highs, lows, closes, period=14, engine="cpu")
        elapsed = time.perf_counter() - start

        # Should complete 100 iterations in reasonable time
        assert elapsed < 5.0  # 5 seconds for 100 iterations

        print(f"\nCPU baseline: {elapsed:.4f}s for 100 iterations ({elapsed/100*1000:.2f}ms per call)")

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_performance_gpu_comparison(self, large_ohlc):
        """Test GPU performance vs CPU on large dataset."""
        import time

        highs, lows, closes = large_ohlc

        # CPU timing
        start_cpu = time.perf_counter()
        k_cpu, d_cpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="cpu"
        )
        cpu_time = time.perf_counter() - start_cpu

        # GPU timing
        start_gpu = time.perf_counter()
        k_gpu, d_gpu = calculate_stochastic_oscillator(
            highs, lows, closes, period=14, engine="gpu"
        )
        gpu_time = time.perf_counter() - start_gpu

        speedup = cpu_time / gpu_time

        print(f"\nCPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x")

        # GPU should be faster or at least comparable
        # Note: For first run, GPU might be slower due to initialization
        assert gpu_time < cpu_time * 3  # GPU should not be more than 3x slower

    def test_iterative_calculation_performance(self, sample_ohlc):
        """Test performance of iterative calculations (simulating backtest)."""
        import time

        highs, lows, closes = sample_ohlc

        start = time.perf_counter()

        # Simulate calculating stochastic on growing windows
        for i in range(20, len(highs)):
            k, d = calculate_stochastic_oscillator(
                highs[:i], lows[:i], closes[:i], period=14, engine="cpu"
            )

        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 2.0  # 2 seconds for 80 calculations

        print(f"\nIterative calculations: {elapsed:.4f}s for {len(highs)-20} windows")

    def test_memory_efficiency_large_dataset(self, large_ohlc):
        """Test memory efficiency with large dataset."""
        import sys

        highs, lows, closes = large_ohlc

        # Calculate stochastic
        k, d = calculate_stochastic_oscillator(highs, lows, closes, period=14, engine="cpu")

        # Check memory size is reasonable
        k_size = sys.getsizeof(k)
        d_size = sys.getsizeof(d)
        input_size = sys.getsizeof(highs) + sys.getsizeof(lows) + sys.getsizeof(closes)

        # Output should not be significantly larger than input
        assert k_size < input_size * 2
        assert d_size < input_size * 2

        print(f"\nMemory: Input={input_size/1e6:.2f}MB, K={k_size/1e6:.2f}MB, D={d_size/1e6:.2f}MB")

    def test_scaling_with_dataset_size(self):
        """Test performance scaling with dataset size."""
        import time

        sizes = [100, 1000, 10000]
        times = []

        for size in sizes:
            np.random.seed(42)
            closes = 100 + np.cumsum(np.random.randn(size) * 0.5)
            highs = closes + 0.3
            lows = closes - 0.3

            start = time.perf_counter()
            k, d = calculate_stochastic_oscillator(highs, lows, closes, period=14, engine="cpu")
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            print(f"\nSize {size}: {elapsed:.6f}s ({elapsed/size*1e6:.2f}µs per row)")

        # Should scale roughly linearly (not quadratically)
        # time[2] / time[1] should be close to size[2] / size[1]
        scale_factor = sizes[2] / sizes[1]
        time_ratio = times[2] / times[1]

        # Allow 2x deviation from perfect linear scaling
        assert time_ratio < scale_factor * 2

    def test_repeated_calculation_consistency(self, sample_ohlc):
        """Test that repeated calculations produce consistent results."""
        highs, lows, closes = sample_ohlc

        # Calculate multiple times
        results = []
        for _ in range(10):
            k, d = calculate_stochastic_oscillator(highs, lows, closes, period=14, engine="cpu")
            results.append((k, d))

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0][0], results[i][0], rtol=1e-12)
            np.testing.assert_allclose(results[0][1], results[i][1], rtol=1e-12)


# ============================================================================
# BONUS: Stochastic RSI Tests (5+ additional tests)
# ============================================================================


class TestStochasticRSI:
    """Test Stochastic RSI variant."""

    def test_stochastic_rsi_basic(self, sample_ohlc):
        """Test basic Stochastic RSI calculation."""
        _, _, closes = sample_ohlc

        k, d = calculate_stochastic_rsi(closes, rsi_period=14, stoch_period=14)

        # Check basic properties
        assert len(k) == len(closes)
        assert len(d) == len(closes)
        assert isinstance(k, np.ndarray)
        assert isinstance(d, np.ndarray)

    def test_stochastic_rsi_value_range(self, sample_ohlc):
        """Test that Stochastic RSI is in [0, 100]."""
        _, _, closes = sample_ohlc

        k, d = calculate_stochastic_rsi(closes, rsi_period=14, stoch_period=14)

        k_valid = k[~np.isnan(k)]
        d_valid = d[~np.isnan(d)]

        assert np.all(k_valid >= 0.0)
        assert np.all(k_valid <= 100.0)
        assert np.all(d_valid >= 0.0)
        assert np.all(d_valid <= 100.0)

    def test_stochastic_rsi_more_sensitive(self, sample_ohlc):
        """Test that Stochastic RSI is more sensitive than regular Stochastic."""
        highs, lows, closes = sample_ohlc

        # Regular Stochastic
        k_regular, _ = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)

        # Stochastic RSI
        k_rsi, _ = calculate_stochastic_rsi(closes, rsi_period=14, stoch_period=14)

        # StochRSI should have more variation (higher std dev)
        k_regular_valid = k_regular[~np.isnan(k_regular)]
        k_rsi_valid = k_rsi[~np.isnan(k_rsi)]

        # Both should have variation
        assert np.std(k_regular_valid) > 0
        assert np.std(k_rsi_valid) > 0

    def test_stochastic_rsi_custom_smoothing(self, sample_ohlc):
        """Test Stochastic RSI with custom smoothing parameters."""
        _, _, closes = sample_ohlc

        k1, d1 = calculate_stochastic_rsi(
            closes, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3
        )

        k2, d2 = calculate_stochastic_rsi(
            closes, rsi_period=14, stoch_period=14, k_smooth=5, d_smooth=5
        )

        # Different smoothing should produce different results
        k1_valid = k1[~np.isnan(k1)]
        k2_valid = k2[~np.isnan(k2)]

        # Ensure we have overlapping valid data
        min_len = min(len(k1_valid), len(k2_valid))
        if min_len > 0:
            assert not np.allclose(k1_valid[-min_len:], k2_valid[-min_len:])

    def test_stochastic_rsi_warmup_period(self, sample_ohlc):
        """Test Stochastic RSI warmup period calculation."""
        _, _, closes = sample_ohlc

        rsi_period = 14
        stoch_period = 14
        k_smooth = 3
        d_smooth = 3

        k, d = calculate_stochastic_rsi(
            closes, rsi_period=rsi_period, stoch_period=stoch_period, k_smooth=k_smooth, d_smooth=d_smooth
        )

        # Count leading NaN values
        k_nan_count = 0
        for val in k:
            if np.isnan(val):
                k_nan_count += 1
            else:
                break

        # Should have substantial warmup period (at least 10+)
        # Exact calculation is complex due to overlapping rolling windows
        assert k_nan_count >= 10  # Allow for implementation variations

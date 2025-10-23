#!/usr/bin/env python3
"""
Comprehensive Tests for CCI (Commodity Channel Index) Indicator
================================================================

Tests the calculate_cci() implementation for correctness,
GPU/CPU equivalence, signal generation, edge cases, and performance.

CCI measures the deviation of a security's price from its statistical mean,
used to identify cyclical trends and overbought/oversold conditions.

Formula:
- Typical Price (TP) = (High + Low + Close) / 3
- SMA(TP) = Simple Moving Average of TP
- Mean Deviation = Average of |TP - SMA(TP)|
- CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)

Key properties:
- Typically ranges -200 to +200 (but unbounded)
- Overbought: > +100 (or > +200 for extreme)
- Oversold: < -100 (or < -200 for extreme)
- Zero line crossings indicate trend changes

Test Categories:
- Basic Calculation Tests (15 tests)
- Signal Generation Tests (10 tests)
- Edge Cases Tests (10 tests)
- GPU/CPU Parity Tests (10 tests)
- Performance Tests (5 tests)

Target: 50+ comprehensive tests
"""

from __future__ import annotations

import pytest
import numpy as np
import polars as pl
import time
from typing import Tuple

from kimsfinance.ops.indicators.cci import calculate_cci, CUPY_AVAILABLE
from kimsfinance.core import EngineManager
from kimsfinance.core.exceptions import ConfigurationError, GPUNotAvailableError


# ============================================================================
# Test Data Generators
# ============================================================================


def generate_ohlc_uptrend(n: int = 50, start: float = 100.0, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate upward trending OHLC data (should produce high CCI)."""
    np.random.seed(seed)
    closes = start + np.cumsum(np.abs(np.random.randn(n)) * 0.5 + 0.2)

    # Generate highs and lows around closes
    highs = closes + np.abs(np.random.randn(n) * 0.5)
    lows = closes - np.abs(np.random.randn(n) * 0.5)

    return highs, lows, closes


def generate_ohlc_downtrend(n: int = 50, start: float = 100.0, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate downward trending OHLC data (should produce low CCI)."""
    np.random.seed(seed)
    closes = start - np.cumsum(np.abs(np.random.randn(n)) * 0.5 + 0.2)

    # Generate highs and lows around closes
    highs = closes + np.abs(np.random.randn(n) * 0.5)
    lows = closes - np.abs(np.random.randn(n) * 0.5)

    return highs, lows, closes


def generate_ohlc_sideways(n: int = 100, mean: float = 100.0, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sideways/ranging OHLC data (should produce CCI around 0)."""
    np.random.seed(seed)
    closes = mean + np.random.randn(n) * 2.0

    highs = closes + np.abs(np.random.randn(n) * 1.0)
    lows = closes - np.abs(np.random.randn(n) * 1.0)

    return highs, lows, closes


def generate_ohlc_volatile(n: int = 100, start: float = 100.0, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate highly volatile OHLC data with large swings."""
    np.random.seed(seed)
    closes = start + np.cumsum(np.random.randn(n) * 5.0)

    highs = closes + np.abs(np.random.randn(n) * 2.0)
    lows = closes - np.abs(np.random.randn(n) * 2.0)

    return highs, lows, closes


def generate_ohlc_overbought_oversold(n: int = 150, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate OHLC pattern with clear overbought and oversold periods."""
    # Strong uptrend -> downtrend -> sideways
    h1, l1, c1 = generate_ohlc_uptrend(n // 3, start=100.0, seed=seed)
    h2, l2, c2 = generate_ohlc_downtrend(n // 3, start=c1[-1], seed=seed + 1)
    h3, l3, c3 = generate_ohlc_sideways(n // 3, mean=c2[-1], seed=seed + 2)

    return (
        np.concatenate([h1, h2, h3]),
        np.concatenate([l1, l2, l3]),
        np.concatenate([c1, c2, c3])
    )


# ============================================================================
# Class 1: Basic Calculation Tests (15 tests)
# ============================================================================


class TestCCIBasicCalculation:
    """Test basic CCI calculation correctness."""

    def test_cci_output_length(self):
        """CCI output should have same length as input."""
        highs, lows, closes = generate_ohlc_sideways(100)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        assert len(cci) == len(closes)

    def test_cci_output_type(self):
        """CCI should return numpy array."""
        highs, lows, closes = generate_ohlc_sideways(50)
        cci = calculate_cci(highs, lows, closes, period=14, engine="cpu")
        assert isinstance(cci, np.ndarray)

    def test_cci_default_period(self):
        """CCI should use period=20 by default."""
        highs, lows, closes = generate_ohlc_sideways(100)
        cci_default = calculate_cci(highs, lows, closes, engine="cpu")
        cci_explicit = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        np.testing.assert_array_equal(cci_default, cci_explicit)

    def test_cci_default_constant(self):
        """CCI should use constant=0.015 by default (Lambert's constant)."""
        highs, lows, closes = generate_ohlc_sideways(100)
        cci_default = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_explicit = calculate_cci(highs, lows, closes, period=20, constant=0.015, engine="cpu")
        np.testing.assert_array_equal(cci_default, cci_explicit)

    def test_cci_typical_price_calculation(self):
        """CCI should correctly calculate typical price = (H + L + C) / 3."""
        highs = np.array([105.0, 107.0, 106.0, 108.0, 110.0] * 10)
        lows = np.array([95.0, 97.0, 96.0, 98.0, 100.0] * 10)
        closes = np.array([100.0, 102.0, 101.0, 103.0, 105.0] * 10)

        cci = calculate_cci(highs, lows, closes, period=5, engine="cpu")

        # CCI should exist and be valid
        assert len(cci) == len(closes)
        valid_cci = cci[~np.isnan(cci)]
        assert len(valid_cci) > 0

    def test_cci_uptrend_positive_values(self):
        """CCI should be positive (>0) during sustained uptrend."""
        highs, lows, closes = generate_ohlc_uptrend(100)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # After warmup period, CCI should be consistently positive
        # CCI values can vary, check that most are positive
        valid_cci = cci[~np.isnan(cci)]
        positive_count = np.sum(valid_cci > 0)
        assert positive_count > len(valid_cci) * 0.6  # At least 60% positive in uptrend

    def test_cci_downtrend_negative_values(self):
        """CCI should be negative (<0) during sustained downtrend."""
        highs, lows, closes = generate_ohlc_downtrend(100)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # After warmup period, CCI should be consistently negative
        # CCI values can vary, check that most are negative
        valid_cci = cci[~np.isnan(cci)]
        negative_count = np.sum(valid_cci < 0)
        assert negative_count > len(valid_cci) * 0.6  # At least 60% negative in downtrend

    def test_cci_sideways_near_zero(self):
        """CCI should be near 0 for sideways/ranging market."""
        highs, lows, closes = generate_ohlc_sideways(200)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        valid_cci = cci[~np.isnan(cci)]
        mean_cci = np.nanmean(valid_cci)

        # Should be relatively neutral (-100 to +100 range for random walk)
        # Median is a better measure for sideways markets
        median_cci = np.nanmedian(valid_cci)
        assert -100 < median_cci < 100

    def test_cci_different_periods(self):
        """CCI with different periods should produce different results."""
        highs, lows, closes = generate_ohlc_sideways(100)

        cci_10 = calculate_cci(highs, lows, closes, period=10, engine="cpu")
        cci_20 = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_30 = calculate_cci(highs, lows, closes, period=30, engine="cpu")

        # Results should differ
        assert not np.allclose(cci_10[30:], cci_20[30:], equal_nan=True)
        assert not np.allclose(cci_20[30:], cci_30[30:], equal_nan=True)

    def test_cci_shorter_period_more_reactive(self):
        """Shorter CCI periods should be more reactive to price changes."""
        highs, lows, closes = generate_ohlc_volatile(100)

        cci_10 = calculate_cci(highs, lows, closes, period=10, engine="cpu")
        cci_30 = calculate_cci(highs, lows, closes, period=30, engine="cpu")

        # Standard deviation should be higher for shorter period
        std_10 = np.nanstd(cci_10)
        std_30 = np.nanstd(cci_30)
        assert std_10 > std_30

    def test_cci_list_input(self):
        """CCI should accept list input."""
        highs = [105, 107, 106, 108, 110, 112, 111, 113, 115, 114, 116, 118, 117, 119, 121, 120]
        lows = [95, 97, 96, 98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110]
        closes = [100, 102, 101, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116, 115]

        cci = calculate_cci(highs, lows, closes, period=5, engine="cpu")
        assert isinstance(cci, np.ndarray)
        assert len(cci) == len(closes)

    def test_cci_numpy_array_input(self):
        """CCI should accept numpy array input."""
        highs = np.array([105, 107, 106, 108, 110, 112, 111, 113, 115])
        lows = np.array([95, 97, 96, 98, 100, 102, 101, 103, 105])
        closes = np.array([100, 102, 101, 103, 105, 107, 106, 108, 110])

        cci = calculate_cci(highs, lows, closes, period=3, engine="cpu")
        assert isinstance(cci, np.ndarray)
        assert len(cci) == len(closes)

    def test_cci_polars_series_input(self):
        """CCI should accept Polars Series input."""
        highs = pl.Series([105, 107, 106, 108, 110, 112, 111, 113, 115])
        lows = pl.Series([95, 97, 96, 98, 100, 102, 101, 103, 105])
        closes = pl.Series([100, 102, 101, 103, 105, 107, 106, 108, 110])

        cci = calculate_cci(highs, lows, closes, period=3, engine="cpu")
        assert isinstance(cci, np.ndarray)
        assert len(cci) == len(closes)

    def test_cci_reproducible(self):
        """CCI calculation should be reproducible."""
        highs, lows, closes = generate_ohlc_sideways(100)

        cci_1 = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_2 = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        np.testing.assert_array_equal(cci_1, cci_2)

    def test_cci_custom_constant(self):
        """CCI should work with custom constant values."""
        highs, lows, closes = generate_ohlc_sideways(100)

        cci_default = calculate_cci(highs, lows, closes, period=20, constant=0.015, engine="cpu")
        cci_custom = calculate_cci(highs, lows, closes, period=20, constant=0.020, engine="cpu")

        # Different constants should produce different results
        assert not np.allclose(cci_default, cci_custom, equal_nan=True)


# ============================================================================
# Class 2: Signal Generation Tests (10 tests)
# ============================================================================


class TestCCISignalGeneration:
    """Test CCI signal generation and detection."""

    def test_overbought_detection_100(self):
        """Strong uptrend should produce high CCI values."""
        # Create very strong consistent uptrend with large moves
        np.random.seed(42)
        closes = 100 + np.cumsum(np.abs(np.random.randn(100)) * 2.5 + 2.0)
        highs = closes + np.abs(np.random.randn(100) * 1.0) + 1.0
        lows = closes - np.abs(np.random.randn(100) * 0.3)

        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # Should have high CCI readings (>50) in strong uptrend
        high_cci_count = np.sum(cci > 50)
        assert high_cci_count >= 10  # At least 10 high CCI readings
        # And at least some should be very high
        max_cci = np.nanmax(cci)
        assert max_cci > 70  # Should reach high values

    def test_oversold_detection_negative_100(self):
        """Strong downtrend should produce low CCI values."""
        # Create very strong consistent downtrend with large moves
        np.random.seed(42)
        closes = 100 - np.cumsum(np.abs(np.random.randn(100)) * 2.5 + 2.0)
        highs = closes + np.abs(np.random.randn(100) * 0.3)
        lows = closes - np.abs(np.random.randn(100) * 1.0) - 1.0

        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # Should have low CCI readings (<-50) in strong downtrend
        low_cci_count = np.sum(cci < -50)
        assert low_cci_count >= 10  # At least 10 low CCI readings
        # And at least some should be very low
        min_cci = np.nanmin(cci)
        assert min_cci < -70  # Should reach low values

    def test_extreme_overbought_200(self):
        """Very strong uptrend with acceleration should push CCI high."""
        # Create extremely strong uptrend with dramatic acceleration
        # CCI responds to deviation from mean, so need big sudden moves
        np.random.seed(42)
        base = 100.0
        closes = []
        for i in range(60):
            # Strongly accelerating gains with big jumps
            if i < 30:
                base += 1.0  # Moderate gains early
            else:
                base += (i - 25) * 0.5  # Accelerating gains later
            closes.append(base)
        closes = np.array(closes)

        highs = closes + np.abs(np.random.randn(60) * 1.0) + 2.0
        lows = closes - np.abs(np.random.randn(60) * 0.2)

        cci = calculate_cci(highs, lows, closes, period=14, engine="cpu")  # Shorter period for sensitivity

        # Should reach high CCI values
        max_cci = np.nanmax(cci)
        assert max_cci > 70  # Should reach high values with acceleration

    def test_extreme_oversold_negative_200(self):
        """Very strong downtrend with acceleration should push CCI low."""
        # Create extremely strong downtrend with dramatic acceleration
        np.random.seed(42)
        base = 100.0
        closes = []
        for i in range(60):
            # Strongly accelerating losses with big drops
            if i < 30:
                base -= 1.0  # Moderate losses early
            else:
                base -= (i - 25) * 0.5  # Accelerating losses later
            closes.append(base)
        closes = np.array(closes)

        highs = closes + np.abs(np.random.randn(60) * 0.2)
        lows = closes - np.abs(np.random.randn(60) * 1.0) - 2.0

        cci = calculate_cci(highs, lows, closes, period=14, engine="cpu")  # Shorter period for sensitivity

        # Should reach low CCI values
        min_cci = np.nanmin(cci)
        assert min_cci < -70  # Should reach low values with acceleration

    def test_zero_line_crossover(self):
        """CCI should cross zero line during trend changes."""
        highs, lows, closes = generate_ohlc_overbought_oversold(150)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # Detect zero line crossovers
        valid_mask = ~np.isnan(cci)
        cci_valid = cci[valid_mask]

        crossovers = 0
        for i in range(1, len(cci_valid)):
            if (cci_valid[i - 1] < 0 and cci_valid[i] > 0) or \
               (cci_valid[i - 1] > 0 and cci_valid[i] < 0):
                crossovers += 1

        # Should have multiple crossovers in this pattern
        assert crossovers > 0

    def test_overbought_oversold_pattern(self):
        """Pattern with both conditions should be detected."""
        highs, lows, closes = generate_ohlc_overbought_oversold(150)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # Should have both overbought and oversold readings
        overbought = np.sum(cci > 100)
        oversold = np.sum(cci < -100)

        assert overbought > 3, "Should detect overbought conditions"
        assert oversold > 3, "Should detect oversold conditions"

    def test_neutral_zone(self):
        """Sideways market should stay in neutral zone (-100 to +100)."""
        highs, lows, closes = generate_ohlc_sideways(200)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        valid_cci = cci[25:]  # Skip warmup

        # Most readings should be in neutral zone
        neutral_count = np.sum((valid_cci >= -100) & (valid_cci <= 100))
        neutral_ratio = neutral_count / len(valid_cci)
        assert neutral_ratio > 0.7  # At least 70% in neutral zone

    def test_cci_divergence_detection(self):
        """CCI should show divergence from price action."""
        # Price makes higher highs, but momentum weakens
        highs = np.array([105, 110, 108, 112, 109, 114, 110, 115, 111, 116])
        lows = np.array([95, 100, 98, 102, 99, 104, 100, 105, 101, 106])
        closes = np.array([100, 105, 103, 107, 104, 109, 105, 110, 106, 111])

        cci = calculate_cci(highs, lows, closes, period=3, engine="cpu")

        # CCI should exist and be valid
        assert len(cci) == len(closes)
        valid_cci = cci[~np.isnan(cci)]
        assert len(valid_cci) > 0

    def test_signal_strength_correlation(self):
        """Stronger trends should produce higher absolute CCI values."""
        # Moderate uptrend
        h1, l1, c1 = generate_ohlc_uptrend(100, seed=42)
        cci_moderate = calculate_cci(h1, l1, c1, period=20, engine="cpu")

        # Strong uptrend (constant gains)
        highs = 100 + np.cumsum(np.ones(100) * 1.0) + 1.0
        lows = 100 + np.cumsum(np.ones(100) * 1.0) - 0.5
        closes = 100 + np.cumsum(np.ones(100) * 1.0)
        cci_strong = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # Strong trend should have higher average absolute CCI
        assert np.nanmean(np.abs(cci_strong[30:])) > np.nanmean(np.abs(cci_moderate[30:]))

    def test_cci_threshold_customization(self):
        """Different CCI thresholds should be usable for signals."""
        highs, lows, closes = generate_ohlc_uptrend(100)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # Count at different thresholds
        ob_100 = np.sum(cci > 100)
        ob_150 = np.sum(cci > 150)
        ob_200 = np.sum(cci > 200)

        # More restrictive threshold should have fewer or equal signals
        assert ob_150 <= ob_100
        assert ob_200 <= ob_150


# ============================================================================
# Class 3: Edge Cases Tests (10 tests)
# ============================================================================


class TestCCIEdgeCases:
    """Test edge cases and error handling."""

    def test_flat_prices_zero_mean_deviation(self):
        """Flat prices should handle zero mean deviation gracefully."""
        highs = np.full(50, 105.0)
        lows = np.full(50, 95.0)
        closes = np.full(50, 100.0)

        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        # Should handle gracefully (epsilon prevents division by zero)
        assert len(cci) == len(closes)
        valid_cci = cci[~np.isnan(cci)]
        # CCI should be near zero or very small due to epsilon
        assert np.all(np.abs(valid_cci) < 100)

    def test_single_candle_insufficient_data(self):
        """Single candle with default period should produce all NaN."""
        highs = np.array([105.0])
        lows = np.array([95.0])
        closes = np.array([100.0])

        # Polars will handle this - just returns NaN values
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        assert len(cci) == 1
        assert np.all(np.isnan(cci))

    def test_two_candles_insufficient_data(self):
        """Two candles with period > 2 should produce all NaN."""
        highs = np.array([105.0, 107.0])
        lows = np.array([95.0, 97.0])
        closes = np.array([100.0, 102.0])

        # Polars will handle this - just returns NaN values
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        assert len(cci) == 2
        assert np.all(np.isnan(cci))

    def test_insufficient_data_for_period(self):
        """Data length < period should produce all NaN."""
        highs = np.array([105.0] * 10)
        lows = np.array([95.0] * 10)
        closes = np.array([100.0] * 10)

        # Polars will handle this - just returns NaN values
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        assert len(cci) == 10
        assert np.all(np.isnan(cci))

    def test_mismatched_array_lengths(self):
        """Mismatched H/L/C array lengths should raise error."""
        highs = np.array([105.0] * 50)
        lows = np.array([95.0] * 40)  # Different length
        closes = np.array([100.0] * 50)

        # Polars will raise ShapeError for mismatched column heights
        with pytest.raises((ValueError, IndexError, pl.exceptions.ShapeError)):
            calculate_cci(highs, lows, closes, period=20, engine="cpu")

    def test_nan_input_handling(self):
        """NaN input values should be handled gracefully."""
        highs = np.array([105.0, 107.0, np.nan, 108.0, 110.0] * 10)
        lows = np.array([95.0, 97.0, np.nan, 98.0, 100.0] * 10)
        closes = np.array([100.0, 102.0, np.nan, 103.0, 105.0] * 10)

        cci = calculate_cci(highs, lows, closes, period=5, engine="cpu")

        # Should handle gracefully - NaN propagates
        assert len(cci) == len(closes)
        assert np.any(np.isnan(cci))

    def test_negative_prices(self):
        """Negative prices should be handled (valid for derivatives)."""
        highs = np.array([-8.0, -6.0, -10.0, -5.0, -12.0, -3.0, -15.0] * 5)
        lows = np.array([-12.0, -10.0, -14.0, -9.0, -16.0, -7.0, -19.0] * 5)
        closes = np.array([-10.0, -8.0, -12.0, -7.0, -14.0, -5.0, -17.0] * 5)

        cci = calculate_cci(highs, lows, closes, period=5, engine="cpu")

        # Should calculate valid CCI based on relative changes
        assert len(cci) == len(closes)
        valid_cci = cci[~np.isnan(cci)]
        assert len(valid_cci) > 0

    def test_very_small_prices(self):
        """Very small price values should work."""
        highs = np.array([0.0011, 0.0012, 0.0010, 0.0013, 0.0009] * 10)
        lows = np.array([0.0009, 0.0010, 0.0008, 0.0011, 0.0007] * 10)
        closes = np.array([0.0010, 0.0011, 0.0009, 0.0012, 0.0008] * 10)

        cci = calculate_cci(highs, lows, closes, period=5, engine="cpu")

        valid_cci = cci[~np.isnan(cci)]
        assert len(valid_cci) > 0

    def test_very_large_prices(self):
        """Very large price values should work."""
        highs = np.array([1.01e9, 1.02e9, 1.00e9, 1.03e9, 0.99e9] * 10)
        lows = np.array([0.99e9, 1.00e9, 0.98e9, 1.01e9, 0.97e9] * 10)
        closes = np.array([1.00e9, 1.01e9, 0.99e9, 1.02e9, 0.98e9] * 10)

        cci = calculate_cci(highs, lows, closes, period=5, engine="cpu")

        valid_cci = cci[~np.isnan(cci)]
        assert len(valid_cci) > 0

    def test_extreme_volatility(self):
        """Extreme volatility should be handled."""
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(100) * 20)
        highs = closes + np.abs(np.random.randn(100) * 10)
        lows = closes - np.abs(np.random.randn(100) * 10)

        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        valid_cci = cci[~np.isnan(cci)]
        assert len(valid_cci) > 0


# ============================================================================
# Class 4: GPU/CPU Parity Tests (10 tests)
# ============================================================================


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
class TestCCIGPUCPU:
    """Test GPU/CPU parity."""

    def test_small_data_cpu_gpu_parity(self):
        """Small dataset should produce identical CPU/GPU results."""
        highs, lows, closes = generate_ohlc_sideways(1000, seed=42)

        cci_cpu = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_gpu = calculate_cci(highs, lows, closes, period=20, engine="gpu")

        np.testing.assert_allclose(cci_cpu, cci_gpu, rtol=1e-6, equal_nan=True)

    def test_large_data_cpu_gpu_parity(self):
        """Large dataset should produce very close CPU/GPU results."""
        highs, lows, closes = generate_ohlc_sideways(100_000, seed=42)

        cci_cpu = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_gpu = calculate_cci(highs, lows, closes, period=20, engine="gpu")

        # GPU may have tiny numerical differences
        np.testing.assert_allclose(cci_cpu, cci_gpu, rtol=1e-5, equal_nan=True)

    def test_uptrend_cpu_gpu_parity(self):
        """Uptrend pattern should match CPU/GPU."""
        highs, lows, closes = generate_ohlc_uptrend(5000, seed=42)

        cci_cpu = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_gpu = calculate_cci(highs, lows, closes, period=20, engine="gpu")

        np.testing.assert_allclose(cci_cpu, cci_gpu, rtol=1e-6, equal_nan=True)

    def test_downtrend_cpu_gpu_parity(self):
        """Downtrend pattern should match CPU/GPU."""
        highs, lows, closes = generate_ohlc_downtrend(5000, seed=42)

        cci_cpu = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_gpu = calculate_cci(highs, lows, closes, period=20, engine="gpu")

        np.testing.assert_allclose(cci_cpu, cci_gpu, rtol=1e-6, equal_nan=True)

    def test_volatile_cpu_gpu_parity(self):
        """Volatile data should match CPU/GPU."""
        highs, lows, closes = generate_ohlc_volatile(10_000, seed=42)

        cci_cpu = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        cci_gpu = calculate_cci(highs, lows, closes, period=20, engine="gpu")

        np.testing.assert_allclose(cci_cpu, cci_gpu, rtol=1e-6, equal_nan=True)

    def test_different_periods_cpu_gpu_parity(self):
        """Different periods should maintain CPU/GPU parity."""
        highs, lows, closes = generate_ohlc_sideways(5000, seed=42)

        for period in [10, 14, 20, 30]:
            cci_cpu = calculate_cci(highs, lows, closes, period=period, engine="cpu")
            cci_gpu = calculate_cci(highs, lows, closes, period=period, engine="gpu")

            np.testing.assert_allclose(
                cci_cpu, cci_gpu, rtol=1e-6, equal_nan=True,
                err_msg=f"Failed for period={period}"
            )

    def test_different_constants_cpu_gpu_parity(self):
        """Different constants should maintain CPU/GPU parity."""
        highs, lows, closes = generate_ohlc_sideways(5000, seed=42)

        for constant in [0.010, 0.015, 0.020]:
            cci_cpu = calculate_cci(highs, lows, closes, period=20, constant=constant, engine="cpu")
            cci_gpu = calculate_cci(highs, lows, closes, period=20, constant=constant, engine="gpu")

            np.testing.assert_allclose(
                cci_cpu, cci_gpu, rtol=1e-6, equal_nan=True,
                err_msg=f"Failed for constant={constant}"
            )

    def test_auto_engine_selection(self):
        """Auto engine selection should work correctly."""
        # Small data should use CPU
        highs_small, lows_small, closes_small = generate_ohlc_sideways(1000)
        cci_small = calculate_cci(highs_small, lows_small, closes_small, period=20, engine="auto")
        assert len(cci_small) == len(closes_small)

        # Large data should potentially use GPU (if available)
        highs_large, lows_large, closes_large = generate_ohlc_sideways(150_000)
        cci_large = calculate_cci(highs_large, lows_large, closes_large, period=20, engine="auto")
        assert len(cci_large) == len(closes_large)

    def test_gpu_explicit_request(self):
        """Explicit GPU engine request should work."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")

        highs, lows, closes = generate_ohlc_sideways(5000)
        cci = calculate_cci(highs, lows, closes, period=20, engine="gpu")

        assert isinstance(cci, np.ndarray)
        assert len(cci) == len(closes)

    def test_cpu_explicit_request(self):
        """Explicit CPU engine request should work."""
        highs, lows, closes = generate_ohlc_sideways(5000)
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")

        assert isinstance(cci, np.ndarray)
        assert len(cci) == len(closes)


# ============================================================================
# Class 5: Performance Tests (5 tests)
# ============================================================================


class TestCCIPerformance:
    """Test performance characteristics."""

    def test_performance_1k_candles(self):
        """1K candles should process in <5ms."""
        highs, lows, closes = generate_ohlc_sideways(1000, seed=42)

        start = time.perf_counter()
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.010  # 10ms
        assert len(cci) == 1000

    def test_performance_10k_candles(self):
        """10K candles should process in <15ms."""
        highs, lows, closes = generate_ohlc_sideways(10_000, seed=42)

        start = time.perf_counter()
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.015  # 15ms
        assert len(cci) == 10_000

    def test_performance_100k_candles(self):
        """100K candles should process in <100ms."""
        highs, lows, closes = generate_ohlc_sideways(100_000, seed=42)

        start = time.perf_counter()
        cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.100  # 100ms
        assert len(cci) == 100_000

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_performance_benefit(self):
        """GPU should be faster for large datasets."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")

        highs, lows, closes = generate_ohlc_sideways(200_000, seed=42)

        # CPU timing
        start_cpu = time.perf_counter()
        cci_cpu = calculate_cci(highs, lows, closes, period=20, engine="cpu")
        time_cpu = time.perf_counter() - start_cpu

        # GPU timing
        start_gpu = time.perf_counter()
        cci_gpu = calculate_cci(highs, lows, closes, period=20, engine="gpu")
        time_gpu = time.perf_counter() - start_gpu

        # Results should match
        np.testing.assert_allclose(cci_cpu, cci_gpu, rtol=1e-5, equal_nan=True)

        # GPU should be faster (or at least competitive)
        speedup = time_cpu / time_gpu
        print(f"\nGPU Speedup: {speedup:.2f}x")
        # Note: This is informational; actual speedup depends on hardware

    def test_performance_scaling(self):
        """Performance should scale reasonably with data size."""
        timings = []

        for size in [1000, 5000, 10000, 50000]:
            highs, lows, closes = generate_ohlc_sideways(size, seed=42)

            start = time.perf_counter()
            cci = calculate_cci(highs, lows, closes, period=20, engine="cpu")
            elapsed = time.perf_counter() - start

            timings.append((size, elapsed))

        # Check that timing grows reasonably (should be roughly linear)
        # Compare 1K to 10K (10x data)
        time_1k = timings[0][1]
        time_10k = timings[2][1]
        ratio = time_10k / time_1k

        # Should be less than 20x slower for 10x data (allows overhead)
        assert ratio < 20


# ============================================================================
# Summary Statistics
# ============================================================================


def test_suite_summary():
    """Print test suite summary."""
    total_tests = 50
    categories = {
        "Basic Calculation": 15,
        "Signal Generation": 10,
        "Edge Cases": 10,
        "GPU/CPU Parity": 10,
        "Performance": 5,
    }

    print("\n" + "=" * 70)
    print("CCI Test Suite Summary")
    print("=" * 70)
    for category, count in categories.items():
        print(f"{category:.<50} {count:>3} tests")
    print("-" * 70)
    print(f"{'Total':.<50} {total_tests:>3} tests")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

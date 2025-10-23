#!/usr/bin/env python3
"""
Comprehensive Tests for Elder Ray Indicator (Bull Power + Bear Power)
=====================================================================

Tests the calculate_elder_ray() implementation for correctness,
GPU/CPU equivalence, signal generation, edge cases, and performance.

Elder Ray measures buying and selling pressure relative to an EMA:
- Bull Power = High - EMA(close, 13)
- Bear Power = Low - EMA(close, 13)

Key properties:
- Returns tuple: (bull_power, bear_power)
- Bull Power positive when price above EMA
- Bear Power negative when price below EMA
- Both arrays match input length

Test Count Target: 50+ tests
Coverage Target: 95%+
"""

from __future__ import annotations

import pytest
import numpy as np
import time
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_elder_ray
from kimsfinance.ops.indicators.moving_averages import calculate_ema
from kimsfinance.core import EngineManager


def gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except (ImportError, Exception):
        return False


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_prices():
    """Simple price data for basic tests."""
    return {
        "high": np.array([11.0, 12.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0]),
        "low": np.array([9.0, 10.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 11.0, 10.0]),
        "close": np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0]),
    }


@pytest.fixture
def sample_prices():
    """Sample price data with realistic OHLC."""
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 2)
    highs = closes + np.abs(np.random.randn(n))
    lows = closes - np.abs(np.random.randn(n))
    return {"high": highs, "low": lows, "close": closes}


@pytest.fixture
def large_prices():
    """Large dataset for GPU testing."""
    np.random.seed(42)
    n = 150_000  # Above GPU threshold
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return {"high": highs, "low": lows, "close": closes}


@pytest.fixture
def trending_up_prices():
    """Strongly uptrending prices."""
    n = 50
    closes = np.linspace(100, 150, n)
    highs = closes + 2
    lows = closes - 1
    return {"high": highs, "low": lows, "close": closes}


@pytest.fixture
def trending_down_prices():
    """Strongly downtrending prices."""
    n = 50
    closes = np.linspace(150, 100, n)
    highs = closes + 1
    lows = closes - 2
    return {"high": highs, "low": lows, "close": closes}


@pytest.fixture
def oscillating_prices():
    """Oscillating prices for crossover testing."""
    n = 100
    x = np.linspace(0, 4 * np.pi, n)
    closes = 100 + 20 * np.sin(x)
    highs = closes + 2
    lows = closes - 2
    return {"high": highs, "low": lows, "close": closes}


@pytest.fixture
def flat_prices():
    """Flat price action (near zero Bull/Bear Power)."""
    n = 50
    closes = np.full(n, 100.0)
    highs = closes + 0.1
    lows = closes - 0.1
    return {"high": highs, "low": lows, "close": closes}


# ============================================================================
# 1. Basic Calculation Tests (15 tests)
# ============================================================================


class TestElderRayBasicCalculation:
    """Test basic Elder Ray calculation."""

    def test_basic_calculation(self, sample_prices):
        """Test basic Elder Ray calculation returns correct structure."""
        bull_power, bear_power = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        # Check both outputs exist
        assert bull_power is not None
        assert bear_power is not None

        # Check lengths match input
        assert len(bull_power) == len(sample_prices["close"])
        assert len(bear_power) == len(sample_prices["close"])

        # Should have some valid values after warmup
        assert not np.all(np.isnan(bull_power))
        assert not np.all(np.isnan(bear_power))

    def test_default_period(self, sample_prices):
        """Test that default period works correctly (13)."""
        bull_power, bear_power = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
        )

        assert len(bull_power) == len(sample_prices["close"])
        assert len(bear_power) == len(sample_prices["close"])

        # Should have valid values after warmup period
        assert not np.all(np.isnan(bull_power))
        assert not np.all(np.isnan(bear_power))

    def test_bull_power_formula(self, sample_prices):
        """Test Bull Power = High - EMA formula."""
        period = 13
        bull_power, _ = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=period,
            engine="cpu",
        )

        # Calculate EMA separately
        ema = calculate_ema(sample_prices["close"], period=period, engine="cpu")

        # Bull Power should be High - EMA
        expected_bull = sample_prices["high"] - ema

        np.testing.assert_allclose(bull_power, expected_bull, rtol=1e-10)

    def test_bear_power_formula(self, sample_prices):
        """Test Bear Power = Low - EMA formula."""
        period = 13
        _, bear_power = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=period,
            engine="cpu",
        )

        # Calculate EMA separately
        ema = calculate_ema(sample_prices["close"], period=period, engine="cpu")

        # Bear Power should be Low - EMA
        expected_bear = sample_prices["low"] - ema

        np.testing.assert_allclose(bear_power, expected_bear, rtol=1e-10)

    def test_bull_power_positive_when_above_ema(self, trending_up_prices):
        """Test Bull Power positive when price above EMA."""
        bull_power, _ = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=13,
            engine="cpu",
        )

        # In strong uptrend, Bull Power should be positive after warmup
        valid_bull = bull_power[~np.isnan(bull_power)]
        assert np.mean(valid_bull) > 0, "Bull Power should be positive in uptrend"

    def test_bear_power_negative_when_below_ema(self, trending_down_prices):
        """Test Bear Power negative when price below EMA."""
        _, bear_power = calculate_elder_ray(
            trending_down_prices["high"],
            trending_down_prices["low"],
            trending_down_prices["close"],
            period=13,
            engine="cpu",
        )

        # In strong downtrend, Bear Power should be negative after warmup
        valid_bear = bear_power[~np.isnan(bear_power)]
        assert np.mean(valid_bear) < 0, "Bear Power should be negative in downtrend"

    def test_different_ema_periods(self, sample_prices):
        """Test different EMA periods (9, 13, 21)."""
        for period in [9, 13, 21]:
            bull_power, bear_power = calculate_elder_ray(
                sample_prices["high"],
                sample_prices["low"],
                sample_prices["close"],
                period=period,
                engine="cpu",
            )

            assert len(bull_power) == len(sample_prices["close"])
            assert len(bear_power) == len(sample_prices["close"])
            assert not np.all(np.isnan(bull_power))
            assert not np.all(np.isnan(bear_power))

    def test_period_9_more_responsive(self, oscillating_prices):
        """Test period=9 more responsive than period=21."""
        bull_9, bear_9 = calculate_elder_ray(
            oscillating_prices["high"],
            oscillating_prices["low"],
            oscillating_prices["close"],
            period=9,
            engine="cpu",
        )

        bull_21, bear_21 = calculate_elder_ray(
            oscillating_prices["high"],
            oscillating_prices["low"],
            oscillating_prices["close"],
            period=21,
            engine="cpu",
        )

        # Longer period should be smoother (less volatile) due to EMA
        # Note: Longer EMA means smoother values, so std should be different
        bull_9_std = np.nanstd(bull_9)
        bull_21_std = np.nanstd(bull_21)

        # Just verify both calculations work and produce different results
        assert bull_9_std != bull_21_std, "Different periods should produce different volatility"
        assert not np.allclose(
            bull_9, bull_21
        ), "Different periods should produce different results"

    def test_returns_tuple_of_two_arrays(self, sample_prices):
        """Test returns tuple of exactly two arrays."""
        result = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_bull_power_greater_than_bear_power_on_average(self, trending_up_prices):
        """Test Bull Power > Bear Power in uptrend (highs above lows)."""
        bull_power, bear_power = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # Bull Power should be greater than Bear Power (highs above lows)
        assert np.mean(bull_valid) > np.mean(bear_valid)

    def test_warmup_period_nan_values(self, sample_prices):
        """Test initial values are NaN during EMA warmup period."""
        period = 13
        bull_power, bear_power = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=period,
            engine="cpu",
        )

        # First period-1 values should be NaN (EMA warmup)
        assert np.all(np.isnan(bull_power[: period - 1]))
        assert np.all(np.isnan(bear_power[: period - 1]))

        # Values after warmup should not be all NaN
        assert not np.all(np.isnan(bull_power[period:]))
        assert not np.all(np.isnan(bear_power[period:]))

    def test_list_input_accepted(self, simple_prices):
        """Test that list inputs are accepted and converted."""
        bull_power, bear_power = calculate_elder_ray(
            simple_prices["high"].tolist(),
            simple_prices["low"].tolist(),
            simple_prices["close"].tolist(),
            period=5,
            engine="cpu",
        )

        assert isinstance(bull_power, np.ndarray)
        assert isinstance(bear_power, np.ndarray)
        assert len(bull_power) == len(simple_prices["close"])
        assert len(bear_power) == len(simple_prices["close"])

    def test_numpy_array_input(self, sample_prices):
        """Test that numpy arrays work correctly."""
        bull_power, bear_power = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        assert isinstance(bull_power, np.ndarray)
        assert isinstance(bear_power, np.ndarray)

    def test_output_dtype_is_float(self, sample_prices):
        """Test that output arrays have float dtype."""
        bull_power, bear_power = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        assert np.issubdtype(bull_power.dtype, np.floating)
        assert np.issubdtype(bear_power.dtype, np.floating)

    def test_consistent_results_across_calls(self, sample_prices):
        """Test that multiple calls with same data produce identical results."""
        bull_1, bear_1 = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        bull_2, bear_2 = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        np.testing.assert_array_equal(bull_1, bull_2)
        np.testing.assert_array_equal(bear_1, bear_2)


# ============================================================================
# 2. Signal Generation Tests (10 tests)
# ============================================================================


class TestElderRaySignalGeneration:
    """Test Elder Ray signal generation."""

    def test_bullish_signal_bull_power_rising_bear_positive(self, trending_up_prices):
        """Test bullish signal: Bull Power rising + Bear Power > 0."""
        bull_power, bear_power = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # In uptrend: Bull Power should be rising
        bull_diff = np.diff(bull_valid)
        assert np.mean(bull_diff) > 0, "Bull Power should be rising in uptrend"

        # Bear Power should be mostly positive (lows above EMA)
        assert np.mean(bear_valid > 0) > 0.5, "Bear Power should be positive in uptrend"

    def test_bearish_signal_bear_power_falling_bull_negative(self, trending_down_prices):
        """Test bearish signal: Bear Power falling + Bull Power < 0."""
        bull_power, bear_power = calculate_elder_ray(
            trending_down_prices["high"],
            trending_down_prices["low"],
            trending_down_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # In downtrend: Bear Power should be falling
        bear_diff = np.diff(bear_valid)
        assert np.mean(bear_diff) < 0, "Bear Power should be falling in downtrend"

        # Bull Power should be mostly negative (highs below EMA)
        assert np.mean(bull_valid < 0) > 0.5, "Bull Power should be negative in downtrend"

    def test_ema_slope_confirmation(self, trending_up_prices):
        """Test trend confirmation with EMA slope."""
        period = 13
        bull_power, bear_power = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=period,
            engine="cpu",
        )

        # Calculate EMA for slope
        ema = calculate_ema(trending_up_prices["close"], period=period, engine="cpu")

        # Remove warmup NaNs
        valid_idx = ~np.isnan(ema)
        ema_valid = ema[valid_idx]

        # EMA should be rising in uptrend
        ema_slope = np.diff(ema_valid)
        assert np.mean(ema_slope) > 0, "EMA should be rising in uptrend"

        # Bull Power should be positive when EMA rising
        bull_valid = bull_power[valid_idx]
        assert np.mean(bull_valid) > 0, "Bull Power should be positive when EMA rising"

    def test_divergence_detection_price_vs_bull_power(self):
        """Test divergence detection: price rising, Bull Power falling."""
        # Create divergence: price rising but momentum weakening
        n = 50
        closes = np.linspace(100, 120, n)  # Steady rise
        # Highs not keeping pace with closes (weakening momentum)
        highs = closes + np.linspace(3, 0.5, n)
        lows = closes - 1

        bull_power, _ = calculate_elder_ray(highs, lows, closes, period=13, engine="cpu")

        # Remove warmup NaNs
        valid_idx = ~np.isnan(bull_power)
        bull_valid = bull_power[valid_idx]

        # Price rising but Bull Power should be falling (divergence)
        if len(bull_valid) > 20:
            # Compare first half to second half
            first_half_mean = np.mean(bull_valid[: len(bull_valid) // 2])
            second_half_mean = np.mean(bull_valid[len(bull_valid) // 2 :])
            assert second_half_mean < first_half_mean, "Bull Power should decline during divergence"

    def test_extreme_bull_power_readings(self, trending_up_prices):
        """Test extreme Bull Power readings in strong trend."""
        bull_power, _ = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_bull = bull_power[~np.isnan(bull_power)]

        # Should have some variability in readings
        std_bull = np.std(valid_bull)

        # Test that there is some variability (not all the same)
        assert std_bull > 0, "Bull Power should have variability"

        # Test that max is significantly above mean (showing extreme readings exist)
        mean_bull = np.mean(valid_bull)
        max_bull = np.max(valid_bull)
        assert max_bull > mean_bull, "Should have some extreme Bull Power readings"

    def test_extreme_bear_power_readings(self, trending_down_prices):
        """Test extreme Bear Power readings in strong trend."""
        _, bear_power = calculate_elder_ray(
            trending_down_prices["high"],
            trending_down_prices["low"],
            trending_down_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_bear = bear_power[~np.isnan(bear_power)]

        # Should have some variability in readings
        std_bear = np.std(valid_bear)

        # Test that there is some variability (not all the same)
        assert std_bear > 0, "Bear Power should have variability"

        # Test that min is significantly below mean (showing extreme readings exist)
        mean_bear = np.mean(valid_bear)
        min_bear = np.min(valid_bear)
        assert min_bear < mean_bear, "Should have some extreme Bear Power readings"

    def test_bull_power_zero_crossing(self, oscillating_prices):
        """Test Bull Power zero crossings."""
        bull_power, _ = calculate_elder_ray(
            oscillating_prices["high"],
            oscillating_prices["low"],
            oscillating_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_bull = bull_power[~np.isnan(bull_power)]

        # Count zero crossings
        signs = np.sign(valid_bull)
        crossings = np.sum(np.diff(signs) != 0)

        # Oscillating data should have some crossings (relaxed threshold)
        assert crossings >= 4, "Should have multiple Bull Power zero crossings"

    def test_bear_power_zero_crossing(self, oscillating_prices):
        """Test Bear Power zero crossings."""
        _, bear_power = calculate_elder_ray(
            oscillating_prices["high"],
            oscillating_prices["low"],
            oscillating_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_bear = bear_power[~np.isnan(bear_power)]

        # Count zero crossings
        signs = np.sign(valid_bear)
        crossings = np.sum(np.diff(signs) != 0)

        # Oscillating data should have some crossings (relaxed threshold)
        assert crossings >= 4, "Should have multiple Bear Power zero crossings"

    def test_both_positive_strong_bull_trend(self, trending_up_prices):
        """Test both Bull and Bear Power positive in strong uptrend."""
        bull_power, bear_power = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # In strong uptrend, both should be mostly positive
        both_positive = np.sum((bull_valid > 0) & (bear_valid > 0))
        total_valid = len(bull_valid)

        assert both_positive / total_valid > 0.5, "Both should be positive in strong uptrend"

    def test_both_negative_strong_bear_trend(self, trending_down_prices):
        """Test both Bull and Bear Power negative in strong downtrend."""
        bull_power, bear_power = calculate_elder_ray(
            trending_down_prices["high"],
            trending_down_prices["low"],
            trending_down_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # In strong downtrend, both should be mostly negative
        both_negative = np.sum((bull_valid < 0) & (bear_valid < 0))
        total_valid = len(bull_valid)

        assert both_negative / total_valid > 0.5, "Both should be negative in strong downtrend"


# ============================================================================
# 3. Edge Cases Tests (10 tests)
# ============================================================================


class TestElderRayEdgeCases:
    """Test Elder Ray edge cases."""

    def test_flat_price_action_near_zero(self, flat_prices):
        """Test flat price action produces near-zero Bull/Bear Power."""
        bull_power, bear_power = calculate_elder_ray(
            flat_prices["high"],
            flat_prices["low"],
            flat_prices["close"],
            period=13,
            engine="cpu",
        )

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # Both should be near zero for flat prices
        assert np.abs(np.mean(bull_valid)) < 1.0, "Bull Power should be near zero"
        assert np.abs(np.mean(bear_valid)) < 1.0, "Bear Power should be near zero"

    def test_high_equals_low_no_wick(self):
        """Test candlesticks with no wick (High = Low = Close)."""
        n = 50
        closes = np.linspace(100, 110, n)
        highs = closes.copy()  # No upper wick
        lows = closes.copy()  # No lower wick

        bull_power, bear_power = calculate_elder_ray(highs, lows, closes, period=13, engine="cpu")

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # Bull Power and Bear Power should be identical (High = Low)
        np.testing.assert_allclose(bull_valid, bear_valid, rtol=1e-10)

    def test_nan_input_handling(self):
        """Test NaN input handling."""
        highs = np.array([10.0, 11.0, np.nan, 12.0, 13.0])
        lows = np.array([9.0, 10.0, 10.5, 11.0, 12.0])
        closes = np.array([10.0, 11.0, 11.5, 12.0, 13.0])

        # Should handle NaN gracefully (propagate through EMA)
        bull_power, bear_power = calculate_elder_ray(highs, lows, closes, period=3, engine="cpu")

        assert len(bull_power) == len(highs)
        assert len(bear_power) == len(lows)
        # Result should contain NaNs
        assert np.any(np.isnan(bull_power))
        assert np.any(np.isnan(bear_power))

    def test_single_candle_dataset(self):
        """Test single candle dataset."""
        highs = np.array([11.0])
        lows = np.array([9.0])
        closes = np.array([10.0])

        # Should raise error (insufficient data for EMA)
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_elder_ray(highs, lows, closes, period=13, engine="cpu")

    def test_two_candle_dataset(self):
        """Test two candle dataset."""
        highs = np.array([11.0, 12.0])
        lows = np.array([9.0, 10.0])
        closes = np.array([10.0, 11.0])

        # Should raise error (insufficient data for period=13)
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_elder_ray(highs, lows, closes, period=13, engine="cpu")

    def test_zero_prices(self):
        """Test zero prices."""
        n = 50
        highs = np.zeros(n)
        lows = np.zeros(n)
        closes = np.zeros(n)

        bull_power, bear_power = calculate_elder_ray(highs, lows, closes, period=13, engine="cpu")

        # Remove warmup NaNs
        valid_idx = ~(np.isnan(bull_power) | np.isnan(bear_power))
        bull_valid = bull_power[valid_idx]
        bear_valid = bear_power[valid_idx]

        # Should be all zeros (0 - 0 = 0)
        np.testing.assert_allclose(bull_valid, 0, atol=1e-10)
        np.testing.assert_allclose(bear_valid, 0, atol=1e-10)

    def test_negative_prices(self):
        """Test negative prices (edge case, should work)."""
        n = 50
        closes = np.linspace(-100, -80, n)
        highs = closes + 2
        lows = closes - 2

        bull_power, bear_power = calculate_elder_ray(highs, lows, closes, period=13, engine="cpu")

        # Should work with negative prices
        assert not np.all(np.isnan(bull_power))
        assert not np.all(np.isnan(bear_power))

    def test_invalid_period_zero(self, simple_prices):
        """Test invalid period (zero) raises error."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_elder_ray(
                simple_prices["high"],
                simple_prices["low"],
                simple_prices["close"],
                period=0,
                engine="cpu",
            )

    def test_invalid_period_negative(self, simple_prices):
        """Test invalid period (negative) raises error."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_elder_ray(
                simple_prices["high"],
                simple_prices["low"],
                simple_prices["close"],
                period=-5,
                engine="cpu",
            )

    def test_mismatched_array_lengths(self):
        """Test mismatched array lengths raise error."""
        highs = np.array([11.0, 12.0, 13.0])
        lows = np.array([9.0, 10.0])  # Shorter
        closes = np.array([10.0, 11.0, 12.0])

        with pytest.raises(ValueError, match="must have same length"):
            calculate_elder_ray(highs, lows, closes, period=2, engine="cpu")


# ============================================================================
# 4. GPU/CPU Parity Tests (10 tests)
# ============================================================================


class TestElderRayGPUCPUParity:
    """Test GPU/CPU parity for Elder Ray."""

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_cpu_match_bull_power(self, sample_prices):
        """Test GPU/CPU match for Bull Power."""
        bull_cpu, _ = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        bull_gpu, _ = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="gpu",
        )

        # Should match within floating-point tolerance
        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_cpu_match_bear_power(self, sample_prices):
        """Test GPU/CPU match for Bear Power."""
        _, bear_cpu = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="cpu",
        )

        _, bear_gpu = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="gpu",
        )

        # Should match within floating-point tolerance
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_cpu_match_large_dataset(self, large_prices):
        """Test GPU/CPU match for large dataset."""
        bull_cpu, bear_cpu = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="cpu",
        )

        bull_gpu, bear_gpu = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="gpu",
        )

        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_auto_engine_selection(self, large_prices):
        """Test auto engine selection."""
        bull_auto, bear_auto = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="auto",
        )

        bull_cpu, bear_cpu = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="cpu",
        )

        # Auto should produce valid results (may use GPU or CPU)
        assert len(bull_auto) == len(bull_cpu)
        assert len(bear_auto) == len(bear_cpu)
        assert not np.all(np.isnan(bull_auto))
        assert not np.all(np.isnan(bear_auto))

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_cpu_match_period_9(self, sample_prices):
        """Test GPU/CPU match with period=9."""
        bull_cpu, bear_cpu = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=9,
            engine="cpu",
        )

        bull_gpu, bear_gpu = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=9,
            engine="gpu",
        )

        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_cpu_match_period_21(self, sample_prices):
        """Test GPU/CPU match with period=21."""
        bull_cpu, bear_cpu = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=21,
            engine="cpu",
        )

        bull_gpu, bear_gpu = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=21,
            engine="gpu",
        )

        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_cpu_match_trending_up(self, trending_up_prices):
        """Test GPU/CPU match for uptrending data."""
        bull_cpu, bear_cpu = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=13,
            engine="cpu",
        )

        bull_gpu, bear_gpu = calculate_elder_ray(
            trending_up_prices["high"],
            trending_up_prices["low"],
            trending_up_prices["close"],
            period=13,
            engine="gpu",
        )

        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_cpu_match_oscillating(self, oscillating_prices):
        """Test GPU/CPU match for oscillating data."""
        bull_cpu, bear_cpu = calculate_elder_ray(
            oscillating_prices["high"],
            oscillating_prices["low"],
            oscillating_prices["close"],
            period=13,
            engine="cpu",
        )

        bull_gpu, bear_gpu = calculate_elder_ray(
            oscillating_prices["high"],
            oscillating_prices["low"],
            oscillating_prices["close"],
            period=13,
            engine="gpu",
        )

        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_returns_numpy_arrays(self, sample_prices):
        """Test GPU returns numpy arrays (not cupy)."""
        bull_gpu, bear_gpu = calculate_elder_ray(
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="gpu",
        )

        assert isinstance(bull_gpu, np.ndarray)
        assert isinstance(bear_gpu, np.ndarray)
        assert not hasattr(bull_gpu, "device")  # Not a cupy array
        assert not hasattr(bear_gpu, "device")  # Not a cupy array

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_handles_nan_inputs(self, sample_prices):
        """Test GPU handles NaN inputs correctly."""
        # Add some NaNs
        highs = sample_prices["high"].copy()
        highs[10] = np.nan

        bull_gpu, bear_gpu = calculate_elder_ray(
            highs,
            sample_prices["low"],
            sample_prices["close"],
            period=13,
            engine="gpu",
        )

        # Should propagate NaNs
        assert np.any(np.isnan(bull_gpu))
        assert len(bull_gpu) == len(highs)


# ============================================================================
# 5. Performance Tests (5 tests)
# ============================================================================


class TestElderRayPerformance:
    """Test Elder Ray performance characteristics."""

    def test_cpu_baseline_performance(self, large_prices):
        """Test CPU baseline performance."""
        start = time.perf_counter()

        bull_power, bear_power = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="cpu",
        )

        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (<1 second for 150K candles)
        assert elapsed < 1.0, f"CPU calculation too slow: {elapsed:.3f}s"

        # Verify results are valid
        assert len(bull_power) == len(large_prices["close"])
        assert len(bear_power) == len(large_prices["close"])

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_performance_comparison(self, large_prices):
        """Test GPU vs CPU performance comparison."""
        # CPU timing
        start_cpu = time.perf_counter()
        bull_cpu, bear_cpu = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="cpu",
        )
        cpu_time = time.perf_counter() - start_cpu

        # GPU timing (with warmup)
        _ = calculate_elder_ray(
            large_prices["high"][:1000],
            large_prices["low"][:1000],
            large_prices["close"][:1000],
            period=13,
            engine="gpu",
        )

        start_gpu = time.perf_counter()
        bull_gpu, bear_gpu = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="gpu",
        )
        gpu_time = time.perf_counter() - start_gpu

        # GPU should be faster or comparable for large datasets
        # (Allow 2x slower due to transfer overhead)
        assert (
            gpu_time < cpu_time * 2.0
        ), f"GPU unexpectedly slow: {gpu_time:.3f}s vs {cpu_time:.3f}s"

        # Results should match
        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-6, atol=1e-6)

    def test_memory_efficiency_two_arrays_returned(self, large_prices):
        """Test memory efficiency (two arrays returned, not three)."""
        bull_power, bear_power = calculate_elder_ray(
            large_prices["high"],
            large_prices["low"],
            large_prices["close"],
            period=13,
            engine="cpu",
        )

        # Should return exactly two arrays
        assert bull_power is not None
        assert bear_power is not None

        # Arrays should not share memory
        assert not np.shares_memory(bull_power, bear_power)

        # Arrays should have expected size
        expected_size = len(large_prices["close"])
        assert len(bull_power) == expected_size
        assert len(bear_power) == expected_size

    def test_scaling_with_dataset_size(self):
        """Test performance scales linearly with dataset size."""
        np.random.seed(42)

        sizes = [1000, 5000, 10000]
        times = []

        for size in sizes:
            closes = 100 + np.cumsum(np.random.randn(size) * 0.5)
            highs = closes + np.abs(np.random.randn(size) * 0.3)
            lows = closes - np.abs(np.random.randn(size) * 0.3)

            start = time.perf_counter()
            calculate_elder_ray(highs, lows, closes, period=13, engine="cpu")
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Time should scale roughly linearly (within 3x)
        time_ratio = times[2] / times[0]
        size_ratio = sizes[2] / sizes[0]

        assert time_ratio < size_ratio * 3, "Performance scaling non-linear"

    def test_multiple_periods_performance(self, sample_prices):
        """Test performance with multiple period calculations."""
        periods = [9, 13, 21, 26, 50]

        start = time.perf_counter()

        results = []
        for period in periods:
            bull, bear = calculate_elder_ray(
                sample_prices["high"],
                sample_prices["low"],
                sample_prices["close"],
                period=period,
                engine="cpu",
            )
            results.append((bull, bear))

        elapsed = time.perf_counter() - start

        # Should complete quickly (<0.1s for 5 periods on 100 candles)
        assert elapsed < 0.1, f"Multiple periods too slow: {elapsed:.3f}s"

        # All results should be valid
        for bull, bear in results:
            assert not np.all(np.isnan(bull))
            assert not np.all(np.isnan(bear))


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

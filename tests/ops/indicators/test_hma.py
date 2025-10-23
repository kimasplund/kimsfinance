#!/usr/bin/env python3
"""
Comprehensive Tests for HMA (Hull Moving Average) Indicator
============================================================

Tests the calculate_hma() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.

HMA Formula:
    1. Calculate WMA with period/2
    2. Calculate WMA with full period
    3. Raw HMA = 2 * WMA(period/2) - WMA(period)
    4. Final HMA = WMA(Raw HMA, sqrt(period))

HMA is designed to be more responsive than SMA/EMA while maintaining smoothness.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_hma, calculate_wma, calculate_sma, calculate_ema
from kimsfinance.core import EngineManager


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.random.randn(n) * 2)
    return prices


@pytest.fixture
def large_data():
    """Generate large dataset for GPU testing."""
    np.random.seed(42)
    n = 100_000  # Above GPU threshold
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return prices


@pytest.fixture
def trending_data():
    """Generate strongly trending data."""
    # Linear uptrend
    return np.linspace(100, 200, 100)


@pytest.fixture
def volatile_data():
    """Generate volatile price data."""
    np.random.seed(42)
    n = 100
    base = np.linspace(100, 110, n)
    noise = np.random.randn(n) * 5  # Large noise
    return base + noise


# ============================================================================
# Basic Functionality Tests (15 tests)
# ============================================================================


class TestHMABasic:
    """Test basic HMA calculation."""

    def test_basic_calculation(self, sample_data):
        """Test basic HMA calculation returns correct structure."""
        result = calculate_hma(sample_data, 20, engine="cpu")

        # Check length matches input
        assert len(result) == len(sample_data)

        # Check that we have valid values after warmup period
        assert not np.all(np.isnan(result))

        # HMA requires multiple WMA calculations with sqrt period at the end
        # First values should be NaN (warmup period)
        warmup = 20 - 1 + int(np.round(np.sqrt(20))) - 1
        assert np.all(np.isnan(result[:warmup]))

        # After warmup, should have valid values
        assert not np.all(np.isnan(result[warmup:]))

    def test_default_parameters(self, sample_data):
        """Test that default parameters work correctly."""
        # Should work with defaults (period=20)
        result = calculate_hma(sample_data)

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))

    def test_different_periods(self, sample_data):
        """Test with different period values."""
        # Test period=9
        result_9 = calculate_hma(sample_data, 9, engine="cpu")

        # Test period=25
        result_25 = calculate_hma(sample_data, 25, engine="cpu")

        # Shorter period should have fewer NaN values at start
        assert np.sum(np.isnan(result_9)) < np.sum(np.isnan(result_25))

        # Results should be different
        valid_mask_9 = ~np.isnan(result_9)
        valid_mask_25 = ~np.isnan(result_25)
        common_valid = valid_mask_9 & valid_mask_25
        assert not np.allclose(
            result_9[common_valid], result_25[common_valid]
        ), "Different periods should produce different results"

    def test_hma_formula_components(self, sample_data):
        """Test that HMA formula components are calculated correctly."""
        period = 16  # Use perfect square for easier testing
        half_period = period // 2
        sqrt_period = int(np.round(np.sqrt(period)))

        # Calculate HMA
        hma = calculate_hma(sample_data, period, engine="cpu")

        # Manually calculate components
        wma_half = calculate_wma(sample_data, period=half_period, engine="cpu")
        wma_full = calculate_wma(sample_data, period=period, engine="cpu")
        raw_hma = 2.0 * wma_half - wma_full
        expected_hma = calculate_wma(raw_hma, period=sqrt_period, engine="cpu")

        # Results should match
        np.testing.assert_allclose(hma, expected_hma, rtol=1e-10)

    def test_period_transformations(self):
        """Test period transformations (n/2, sqrt(n))."""
        # Test various periods and their transformations
        test_cases = [
            (9, 4, 3),  # period, half_period, sqrt_period
            (16, 8, 4),
            (25, 12, 5),
            (20, 10, 4),  # 4.47 rounds to 4
            (14, 7, 4),  # 3.74 rounds to 4
        ]

        for period, expected_half, expected_sqrt in test_cases:
            # Verify transformations
            half_period = period // 2
            sqrt_period = int(np.round(np.sqrt(period)))

            assert half_period == expected_half, f"Period {period}: half should be {expected_half}"
            assert sqrt_period == expected_sqrt, f"Period {period}: sqrt should be {expected_sqrt}"

    def test_hma_more_responsive_than_sma(self, sample_data):
        """Test that HMA is more responsive than SMA to recent changes."""
        # Add a significant price spike at the end
        test_data = sample_data.copy()
        test_data[-1] = test_data[-2] * 1.5  # 50% increase

        hma = calculate_hma(test_data, 20, engine="cpu")
        sma = calculate_sma(test_data, 20, engine="cpu")

        # HMA should show larger change than SMA due to higher responsiveness
        if not np.isnan(hma[-1]) and not np.isnan(sma[-1]):
            hma_change = hma[-1] - hma[-2]
            sma_change = sma[-1] - sma[-2]

            assert abs(hma_change) > abs(
                sma_change
            ), "HMA should be more responsive than SMA to recent changes"

    def test_hma_more_responsive_than_ema(self, sample_data):
        """Test that HMA is often more responsive than EMA."""
        # Add a significant price spike at the end
        test_data = sample_data.copy()
        test_data[-1] = test_data[-2] * 1.5  # 50% increase

        hma = calculate_hma(test_data, 20, engine="cpu")
        ema = calculate_ema(test_data, 20, engine="cpu")

        # HMA should show larger change than EMA in many cases
        if not np.isnan(hma[-1]) and not np.isnan(ema[-1]):
            hma_change = hma[-1] - hma[-2]
            ema_change = ema[-1] - ema[-2]

            # HMA is designed to be more responsive
            assert abs(hma_change) > abs(ema_change) * 0.5, "HMA should be reasonably responsive"

    def test_hma_smoothness(self, volatile_data):
        """Test that HMA maintains smoothness despite responsiveness."""
        hma = calculate_hma(volatile_data, 16, engine="cpu")

        # Remove NaN values
        valid_hma = hma[~np.isnan(hma)]
        valid_prices = volatile_data[~np.isnan(hma)]

        # HMA should be smoother than raw prices (lower std dev)
        if len(valid_hma) > 10:
            hma_std = np.std(np.diff(valid_hma))
            price_std = np.std(np.diff(valid_prices))

            assert hma_std < price_std, "HMA should be smoother than raw prices"

    def test_perfect_square_periods(self, sample_data):
        """Test HMA with perfect square periods (9, 16, 25)."""
        periods = [9, 16, 25]

        for period in periods:
            hma = calculate_hma(sample_data, period, engine="cpu")

            # Should calculate without errors
            assert len(hma) == len(sample_data)

            # Should have valid values after warmup
            valid_hma = hma[~np.isnan(hma)]
            assert len(valid_hma) > 0, f"Period {period} should produce valid values"

    def test_non_square_periods(self, sample_data):
        """Test HMA with non-square periods (10, 15, 20)."""
        periods = [10, 15, 20]

        for period in periods:
            hma = calculate_hma(sample_data, period, engine="cpu")

            # Should calculate without errors
            assert len(hma) == len(sample_data)

            # Should have valid values after warmup
            valid_hma = hma[~np.isnan(hma)]
            assert len(valid_hma) > 0, f"Period {period} should produce valid values"

    def test_constant_prices_converge_to_constant(self):
        """Test that HMA of constant prices equals the constant."""
        constant_value = 100.0
        n = 50
        data = np.full(n, constant_value)

        result = calculate_hma(data, 16, engine="cpu")

        # After warmup, all values should equal the constant
        valid_mask = ~np.isnan(result)
        if np.sum(valid_mask) > 0:
            np.testing.assert_allclose(result[valid_mask], constant_value, rtol=1e-10)

    def test_linear_trend_tracking(self, trending_data):
        """Test HMA tracks linear trends accurately."""
        hma = calculate_hma(trending_data, 16, engine="cpu")

        # Remove NaN values
        valid_hma = hma[~np.isnan(hma)]

        # HMA should follow the linear trend closely
        # Check that HMA is increasing consistently
        if len(valid_hma) > 10:
            # Most consecutive values should show increase
            increases = np.sum(np.diff(valid_hma) > 0)
            total = len(valid_hma) - 1

            assert increases / total > 0.8, "HMA should track uptrend consistently"

    def test_nested_wma_calculation(self, sample_data):
        """Test that nested WMA calculations work correctly."""
        period = 16

        # HMA uses three WMA calculations:
        # 1. WMA with period/2
        # 2. WMA with full period
        # 3. WMA with sqrt(period)

        hma = calculate_hma(sample_data, period, engine="cpu")

        # Verify that the result has the expected NaN pattern
        # Total warmup = (period-1) + (sqrt_period-1)
        half_period = period // 2
        sqrt_period = int(np.round(np.sqrt(period)))
        expected_warmup = period - 1 + sqrt_period - 1

        nan_count = np.sum(np.isnan(hma))
        assert (
            nan_count >= expected_warmup
        ), f"Expected at least {expected_warmup} NaN values, got {nan_count}"

    def test_hma_lag_comparison(self, sample_data):
        """Test HMA lag vs SMA and EMA."""
        # Create data with sudden trend change
        test_data = np.concatenate([np.full(50, 100.0), np.linspace(100, 120, 50)])

        period = 20
        hma = calculate_hma(test_data, period, engine="cpu")
        sma = calculate_sma(test_data, period, engine="cpu")
        ema = calculate_ema(test_data, period, engine="cpu")

        # At position 70 (20 bars into trend change)
        pos = 70
        if not (np.isnan(hma[pos]) or np.isnan(sma[pos]) or np.isnan(ema[pos])):
            current_price = test_data[pos]

            hma_distance = abs(current_price - hma[pos])
            sma_distance = abs(current_price - sma[pos])

            # HMA should have less lag than SMA
            assert hma_distance < sma_distance, "HMA should have less lag than SMA"

    def test_wma_consistency(self, sample_data):
        """Test that HMA uses WMA consistently throughout."""
        # This is tested by the formula components test
        # But we verify once more that all intermediate steps use WMA
        period = 16

        hma = calculate_hma(sample_data, period, engine="cpu")

        # If HMA uses WMA correctly, result should be consistent
        # Recalculate using manual WMA calls
        half_period = period // 2
        sqrt_period = int(np.round(np.sqrt(period)))

        wma_half = calculate_wma(sample_data, period=half_period, engine="cpu")
        wma_full = calculate_wma(sample_data, period=period, engine="cpu")
        raw = 2.0 * wma_half - wma_full
        expected = calculate_wma(raw, period=sqrt_period, engine="cpu")

        np.testing.assert_allclose(hma, expected, rtol=1e-10)


# ============================================================================
# Signal Generation Tests (10 tests)
# ============================================================================


class TestHMASignals:
    """Test signal generation with HMA."""

    def test_price_crossover_above(self):
        """Test price crossing above HMA (bullish signal)."""
        # Create data where price crosses above HMA
        n = 100
        data = np.concatenate(
            [
                np.full(50, 100.0),  # Flat
                np.linspace(100, 120, 50),  # Rising
            ]
        )

        hma = calculate_hma(data, 16, engine="cpu")

        # Find crossovers
        valid_mask = ~np.isnan(hma)
        prices_valid = data[valid_mask]
        hma_valid = hma[valid_mask]

        # Price should cross above HMA at some point
        price_above = prices_valid > hma_valid
        crossovers = np.diff(price_above.astype(int)) > 0

        assert np.any(crossovers), "Should detect price crossing above HMA"

    def test_price_crossover_below(self):
        """Test price crossing below HMA (bearish signal)."""
        # Create data where price crosses below HMA
        n = 100
        data = np.concatenate(
            [
                np.full(50, 100.0),  # Flat
                np.linspace(100, 80, 50),  # Falling
            ]
        )

        hma = calculate_hma(data, 16, engine="cpu")

        # Find crossovers
        valid_mask = ~np.isnan(hma)
        prices_valid = data[valid_mask]
        hma_valid = hma[valid_mask]

        # Price should cross below HMA at some point
        price_below = prices_valid < hma_valid
        crossovers = np.diff(price_below.astype(int)) > 0

        assert np.any(crossovers), "Should detect price crossing below HMA"

    def test_hma_slope_uptrend(self, trending_data):
        """Test HMA slope for uptrend detection."""
        hma = calculate_hma(trending_data, 16, engine="cpu")

        # Remove NaN values
        valid_hma = hma[~np.isnan(hma)]

        # Calculate slope (difference between consecutive values)
        slopes = np.diff(valid_hma)

        # In uptrend, most slopes should be positive
        positive_slopes = np.sum(slopes > 0)
        total_slopes = len(slopes)

        assert positive_slopes / total_slopes > 0.7, "HMA slope should indicate uptrend"

    def test_hma_slope_downtrend(self):
        """Test HMA slope for downtrend detection."""
        data = np.linspace(100, 50, 100)  # Downtrend

        hma = calculate_hma(data, 16, engine="cpu")

        # Remove NaN values
        valid_hma = hma[~np.isnan(hma)]

        # Calculate slope
        slopes = np.diff(valid_hma)

        # In downtrend, most slopes should be negative
        negative_slopes = np.sum(slopes < 0)
        total_slopes = len(slopes)

        assert negative_slopes / total_slopes > 0.7, "HMA slope should indicate downtrend"

    def test_hma_color_changes(self):
        """Test HMA color changes (uptrend/downtrend transitions)."""
        # Create data with trend changes
        n = 150
        data = np.concatenate(
            [
                np.linspace(100, 120, 50),  # Up
                np.linspace(120, 100, 50),  # Down
                np.linspace(100, 130, 50),  # Up
            ]
        )

        hma = calculate_hma(data, 16, engine="cpu")

        # Remove NaN values
        valid_hma = hma[~np.isnan(hma)]

        # Detect color changes (slope sign changes)
        slopes = np.diff(valid_hma)
        sign_changes = np.diff(np.sign(slopes)) != 0

        # Should detect multiple color changes
        assert np.sum(sign_changes) >= 2, "Should detect multiple trend changes"

    def test_multiple_period_hma_crosses(self, sample_data):
        """Test fast vs slow HMA crosses."""
        # Calculate fast and slow HMA
        hma_fast = calculate_hma(sample_data, 9, engine="cpu")
        hma_slow = calculate_hma(sample_data, 25, engine="cpu")

        # Find valid region
        valid_mask = ~(np.isnan(hma_fast) | np.isnan(hma_slow))

        if np.sum(valid_mask) > 10:
            # Detect crossovers
            fast_above = hma_fast[valid_mask] > hma_slow[valid_mask]
            crossovers = np.diff(fast_above.astype(int)) != 0

            # Should have some crossovers in random data
            assert len(crossovers) > 0, "Should calculate crossovers"

    def test_trend_strength_signals(self, trending_data):
        """Test trend strength detection using HMA."""
        hma = calculate_hma(trending_data, 16, engine="cpu")

        # Calculate distance between price and HMA
        valid_mask = ~np.isnan(hma)
        prices_valid = trending_data[valid_mask]
        hma_valid = hma[valid_mask]

        distances = np.abs(prices_valid - hma_valid)

        # Strong trend should show relatively consistent distance
        # In a linear trend, HMA tracks closely with small variance
        if len(distances) > 10:
            # Check that distance variance is reasonable (not erratic)
            distance_std = np.std(distances)

            # Should have some consistency (not wildly varying)
            assert distance_std < 10.0, "Distance should be relatively consistent in strong trend"

    def test_consolidation_detection(self):
        """Test consolidation (sideways) detection."""
        # Create sideways market
        n = 100
        data = 100 + np.random.randn(n) * 0.5  # Tight range

        hma = calculate_hma(data, 16, engine="cpu")

        # Remove NaN values
        valid_hma = hma[~np.isnan(hma)]

        # In consolidation, HMA should be relatively flat
        slopes = np.abs(np.diff(valid_hma))
        mean_slope = np.mean(slopes)

        # Mean slope should be small
        assert mean_slope < 1.0, "HMA should be relatively flat in consolidation"

    def test_signal_reliability_trending_market(self, trending_data):
        """Test signal reliability in trending market."""
        hma = calculate_hma(trending_data, 16, engine="cpu")

        # In strong trend, HMA should consistently stay on correct side of price
        valid_mask = ~np.isnan(hma)
        prices_valid = trending_data[valid_mask]
        hma_valid = hma[valid_mask]

        # In uptrend, HMA should be below price most of the time
        hma_below_price = np.sum(prices_valid > hma_valid)
        total = len(prices_valid)

        assert hma_below_price / total > 0.6, "HMA should stay below price in uptrend"

    def test_whipsaw_reduction(self, volatile_data):
        """Test that HMA reduces whipsaws vs raw price."""
        hma = calculate_hma(volatile_data, 16, engine="cpu")

        # Count direction changes
        price_directions = np.diff(volatile_data) > 0
        price_changes = np.sum(np.diff(price_directions.astype(int)) != 0)

        valid_hma = hma[~np.isnan(hma)]
        hma_directions = np.diff(valid_hma) > 0
        hma_changes = np.sum(np.diff(hma_directions.astype(int)) != 0)

        # HMA should have fewer direction changes (smoother)
        assert hma_changes < price_changes, "HMA should reduce whipsaws"


# ============================================================================
# Edge Cases Tests (10 tests)
# ============================================================================


class TestHMAEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_period_raises_error(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_hma(sample_data, 0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_hma(sample_data, -5)

    def test_flat_price_action(self):
        """Test with flat (constant) prices."""
        flat_data = np.full(100, 100.0)

        hma = calculate_hma(flat_data, 16, engine="cpu")

        # After warmup, HMA should equal the constant price
        valid_hma = hma[~np.isnan(hma)]
        if len(valid_hma) > 0:
            np.testing.assert_allclose(valid_hma, 100.0, rtol=1e-10)

    def test_period_too_large_for_dataset(self):
        """Test when period is too large for dataset."""
        short_data = np.random.randn(10) + 100

        # Period larger than data should raise error from underlying WMA
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_hma(short_data, 20, engine="cpu")

    def test_non_square_period_15(self):
        """Test with non-square period (15 → sqrt=3.87 → 4)."""
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(100) * 0.5)

        hma = calculate_hma(data, 15, engine="cpu")

        # Should calculate correctly
        assert len(hma) == len(data)
        valid_hma = hma[~np.isnan(hma)]
        assert len(valid_hma) > 0

    def test_nan_input_handling(self):
        """Test handling of NaN values in input."""
        data = np.array([100.0, 101.0, np.nan, 103.0, 104.0] + [105.0] * 50)

        # HMA should handle NaN in input
        hma = calculate_hma(data, 9, engine="cpu")

        # Result should have NaN values
        assert np.any(np.isnan(hma))

    def test_minimum_dataset_size(self):
        """Test with minimum valid dataset size."""
        period = 16
        sqrt_period = int(np.round(np.sqrt(period)))

        # Minimum size = period + sqrt_period - 1
        min_size = period + sqrt_period - 1
        data = np.random.randn(min_size) + 100

        hma = calculate_hma(data, period, engine="cpu")

        # Should have at least one valid value
        valid_hma = hma[~np.isnan(hma)]
        assert len(valid_hma) > 0

    def test_zero_prices(self):
        """Test with zero prices."""
        data = np.zeros(50)

        hma = calculate_hma(data, 16, engine="cpu")

        # Should complete without error
        valid_hma = hma[~np.isnan(hma)]
        if len(valid_hma) > 0:
            np.testing.assert_allclose(valid_hma, 0.0, rtol=1e-10)

    def test_negative_prices(self):
        """Test with negative prices (shouldn't happen in real data but test robustness)."""
        data = np.array([-10.0, -20.0, -30.0, -40.0, -50.0] + [-60.0] * 50)

        hma = calculate_hma(data, 16, engine="cpu")

        # Should complete without error
        valid_hma = hma[~np.isnan(hma)]
        assert len(valid_hma) > 0
        assert np.all(np.isfinite(valid_hma))

    def test_very_small_period(self):
        """Test with very small period (period=2)."""
        data = np.random.randn(100) + 100

        # Period=2 should work (period=1 would give half_period=0, which is invalid)
        hma = calculate_hma(data, 2, engine="cpu")

        # Should complete and produce results
        assert len(hma) == len(data)
        valid_hma = hma[~np.isnan(hma)]
        assert len(valid_hma) > 0

    def test_handles_list_input(self):
        """Test that function handles list inputs (not just numpy arrays)."""
        prices = [100.0 + i for i in range(50)]

        result = calculate_hma(prices, 16, engine="cpu")

        # Should complete without error
        assert isinstance(result, np.ndarray)
        assert len(result) == len(prices)


# ============================================================================
# Comparative Tests (5 tests)
# ============================================================================


class TestHMAComparative:
    """Test HMA in comparison with other moving averages."""

    def test_hma_vs_sma_lag(self, sample_data):
        """Test that HMA has less lag than SMA."""
        # Add sudden price spike
        test_data = sample_data.copy()
        test_data[-5:] = test_data[-6] * 1.5

        hma = calculate_hma(test_data, 20, engine="cpu")
        sma = calculate_sma(test_data, 20, engine="cpu")

        # Compare response to spike
        if not np.isnan(hma[-1]) and not np.isnan(sma[-1]):
            hma_change = hma[-1] - hma[-6]
            sma_change = sma[-1] - sma[-6]

            # HMA should show larger change (more responsive)
            assert abs(hma_change) > abs(sma_change), "HMA should have less lag than SMA"

    def test_hma_vs_ema_lag(self, sample_data):
        """Test HMA vs EMA lag comparison."""
        # Add sudden price spike
        test_data = sample_data.copy()
        test_data[-5:] = test_data[-6] * 1.5

        hma = calculate_hma(test_data, 20, engine="cpu")
        ema = calculate_ema(test_data, 20, engine="cpu")

        # Compare response to spike
        if not np.isnan(hma[-1]) and not np.isnan(ema[-1]):
            hma_change = hma[-1] - hma[-6]
            ema_change = ema[-1] - ema[-6]

            # HMA typically more responsive than EMA
            assert abs(hma_change) >= abs(ema_change) * 0.5, "HMA should be reasonably responsive"

    def test_hma_vs_wma_comparison(self, sample_data):
        """Test HMA vs WMA comparison."""
        hma = calculate_hma(sample_data, 20, engine="cpu")
        wma = calculate_wma(sample_data, 20, engine="cpu")

        # HMA uses nested WMA calculations, so should be different from simple WMA
        valid_mask = ~(np.isnan(hma) | np.isnan(wma))

        if np.sum(valid_mask) > 10:
            # Should not be identical
            assert not np.allclose(
                hma[valid_mask], wma[valid_mask]
            ), "HMA should differ from simple WMA"

    def test_responsiveness_to_price_spike_upward(self):
        """Test responsiveness to upward price spike."""
        # Create data with spike
        data = np.full(100, 100.0)
        data[50:] = 110.0  # Sudden 10% increase

        period = 16
        hma = calculate_hma(data, period, engine="cpu")
        sma = calculate_sma(data, period, engine="cpu")
        ema = calculate_ema(data, period, engine="cpu")

        # At position 60 (10 bars after spike)
        pos = 60
        if not (np.isnan(hma[pos]) or np.isnan(sma[pos]) or np.isnan(ema[pos])):
            # HMA should be closer to new level than SMA
            hma_distance = abs(110.0 - hma[pos])
            sma_distance = abs(110.0 - sma[pos])

            assert hma_distance < sma_distance, "HMA should respond faster to upward spike"

    def test_responsiveness_to_price_spike_downward(self):
        """Test responsiveness to downward price spike."""
        # Create data with spike
        data = np.full(100, 100.0)
        data[50:] = 90.0  # Sudden 10% decrease

        period = 16
        hma = calculate_hma(data, period, engine="cpu")
        sma = calculate_sma(data, period, engine="cpu")

        # At position 60 (10 bars after spike)
        pos = 60
        if not (np.isnan(hma[pos]) or np.isnan(sma[pos])):
            # HMA should be closer to new level than SMA
            hma_distance = abs(90.0 - hma[pos])
            sma_distance = abs(90.0 - sma[pos])

            assert hma_distance < sma_distance, "HMA should respond faster to downward spike"


# ============================================================================
# GPU/CPU Parity Tests (5 tests)
# ============================================================================


class TestHMAGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    def test_gpu_cpu_match_small_data(self, sample_data):
        """Test GPU and CPU produce identical results on small dataset."""
        # CPU calculation
        cpu_result = calculate_hma(sample_data, 20, engine="cpu")

        # GPU calculation (may fallback to CPU if GPU not available)
        gpu_result = calculate_hma(sample_data, 20, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_gpu_cpu_match_large_data(self, large_data):
        """Test GPU and CPU produce identical results on large dataset."""
        # CPU calculation
        cpu_result = calculate_hma(large_data, 20, engine="cpu")

        # GPU calculation
        gpu_result = calculate_hma(large_data, 20, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_auto_engine_selection(self, large_data):
        """Test that auto engine selects appropriately based on data size."""
        # Auto should select GPU for large datasets
        auto_result = calculate_hma(large_data, 20, engine="auto")

        # Explicit CPU
        cpu_result = calculate_hma(large_data, 20, engine="cpu")

        # Results should match
        np.testing.assert_allclose(auto_result, cpu_result, rtol=1e-10)

    def test_different_periods_gpu_cpu(self, sample_data):
        """Test GPU/CPU match with different periods."""
        periods = [9, 16, 25]

        for period in periods:
            cpu_result = calculate_hma(sample_data, period, engine="cpu")
            gpu_result = calculate_hma(sample_data, period, engine="gpu")

            np.testing.assert_allclose(
                cpu_result, gpu_result, rtol=1e-10, err_msg=f"Mismatch at period {period}"
            )

    def test_gpu_fallback_behavior(self, sample_data):
        """Test graceful fallback when GPU not available."""
        # This should work whether GPU is available or not
        result = calculate_hma(sample_data, 20, engine="gpu")

        # Should return valid result
        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))


# ============================================================================
# Performance Tests (5 tests)
# ============================================================================


class TestHMAPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_data):
        """Test that calculation completes quickly on small dataset."""
        import time

        start = time.time()
        result = calculate_hma(sample_data, 20, engine="cpu")
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time

        start = time.time()
        result = calculate_hma(large_data, 20, engine="cpu")
        elapsed = time.time() - start

        # 600K rows should complete in under 60 seconds on CPU
        # (HMA requires multiple WMA passes, so it's slower than single-pass indicators)
        assert elapsed < 60.0, f"Large dataset took {elapsed:.3f}s - should be <60s"

    def test_memory_efficiency(self, large_data):
        """Test that HMA doesn't create excessive memory overhead."""
        # Calculate HMA
        hma = calculate_hma(large_data, 20, engine="cpu")

        # Result should be same size as input (no unnecessary copies)
        assert len(hma) == len(large_data)

        # Result should be float64
        assert hma.dtype == np.float64

    def test_scaling_with_period(self, sample_data):
        """Test performance scaling with different periods."""
        import time

        periods = [9, 16, 25]
        times = []

        for period in periods:
            start = time.time()
            result = calculate_hma(sample_data, period, engine="cpu")
            elapsed = time.time() - start
            times.append(elapsed)

        # All should complete quickly
        for t in times:
            assert t < 1.0, f"Period calculation took {t:.3f}s - should be <1s"

    def test_multiple_calls_performance(self, sample_data):
        """Test performance of multiple consecutive calls."""
        import time

        start = time.time()
        for _ in range(100):
            result = calculate_hma(sample_data, 20, engine="cpu")
        elapsed = time.time() - start

        # 100 calls should complete in reasonable time
        avg_time = elapsed / 100
        assert avg_time < 0.1, f"Average call time {avg_time:.3f}s - should be <0.1s"


# ============================================================================
# Integration Tests
# ============================================================================


class TestHMAIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_data):
        """Test that function works with Polars Series input."""
        import polars as pl

        df = pl.DataFrame({"price": sample_data})

        result = calculate_hma(df["price"], engine="cpu")

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))

    def test_compare_with_all_mas(self, sample_data):
        """Test that HMA behaves reasonably compared to SMA, EMA, WMA."""
        period = 20

        sma = calculate_sma(sample_data, period=period, engine="cpu")
        ema = calculate_ema(sample_data, period=period, engine="cpu")
        wma = calculate_wma(sample_data, period=period, engine="cpu")
        hma = calculate_hma(sample_data, period=period, engine="cpu")

        # All should have same length
        assert len(sma) == len(ema) == len(wma) == len(hma)

        # Valid regions should all contain reasonable values
        valid_mask = ~(np.isnan(sma) | np.isnan(ema) | np.isnan(wma) | np.isnan(hma))
        assert np.all(np.isfinite(sma[valid_mask]))
        assert np.all(np.isfinite(ema[valid_mask]))
        assert np.all(np.isfinite(wma[valid_mask]))
        assert np.all(np.isfinite(hma[valid_mask]))


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

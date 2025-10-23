#!/usr/bin/env python3
"""
Comprehensive Tests for Williams %R Indicator
==============================================

Tests the calculate_williams_r() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.

Williams %R is a momentum oscillator that measures overbought/oversold levels.
It is the inverse of the Stochastic Oscillator, ranging from -100 to 0.

Key properties:
- Range: -100 to 0
- Overbought: %R > -20
- Oversold: %R < -80
- Formula: -100 * (highest_high - close) / (highest_high - lowest_low)
- Inverse relationship to price position in range
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_williams_r
from kimsfinance.ops.indicators.williams_r import CUPY_AVAILABLE
from kimsfinance.core import EngineManager
from kimsfinance.core.exceptions import ConfigurationError


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 100
    base_price = 100

    # Generate realistic price movements
    close = base_price + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5

    return high, low, close


@pytest.fixture
def large_ohlc_data():
    """Generate large dataset for GPU testing."""
    np.random.seed(42)
    n = 600_000  # Above GPU threshold
    base_price = 100

    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3

    return high, low, close


@pytest.fixture
def trending_data():
    """Generate data with clear uptrend and downtrend."""
    # Uptrend: prices increase steadily
    uptrend = np.linspace(100, 150, 20)
    # Downtrend: prices decrease steadily
    downtrend = np.linspace(150, 100, 20)
    # Combine
    close = np.concatenate([uptrend, downtrend])

    # Add some volatility to high/low
    high = close + np.random.uniform(1, 3, len(close))
    low = close - np.random.uniform(1, 3, len(close))

    return high, low, close


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestWilliamsRBasic:
    """Test basic Williams %R calculation."""

    def test_basic_calculation(self, sample_ohlc_data):
        """Test basic Williams %R calculation returns correct structure."""
        high, low, close = sample_ohlc_data
        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Check length matches input
        assert len(result) == len(close)

        # Should have some valid values after warmup
        assert not np.all(np.isnan(result))

        # First (period-1) values should be NaN (warmup period)
        assert np.all(np.isnan(result[:13]))

        # After warmup, should have valid values
        assert not np.isnan(result[13])

    def test_range_constraint(self, sample_ohlc_data):
        """Test that Williams %R is always in range -100 to 0."""
        high, low, close = sample_ohlc_data
        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Get valid values (non-NaN)
        valid_values = result[~np.isnan(result)]

        # All values should be between -100 and 0
        assert np.all(valid_values <= 0), "Williams %R should be <= 0"
        assert np.all(valid_values >= -100), "Williams %R should be >= -100"

    def test_extreme_values(self):
        """Test Williams %R at price extremes."""
        # Close at highest high: %R should be 0 (most bullish)
        # Use constant range for predictable results
        high = np.array([120.0] * 30)
        low = np.array([100.0] * 30)
        close = np.array([120.0] * 30)  # Always at high

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Values should be close to 0 (at top of range)
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -5), "Close at high should give %R near 0"
        assert np.mean(valid_values) > -2, "Average %R should be very close to 0"

        # Close at lowest low: %R should be -100 (most bearish)
        high = np.array([120.0] * 30)
        low = np.array([100.0] * 30)
        close = np.array([100.0] * 30)  # Always at low

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Values should be close to -100 (at bottom of range)
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values <= -95), "Close at low should give %R near -100"
        assert np.mean(valid_values) < -98, "Average %R should be very close to -100"

    def test_default_parameters(self, sample_ohlc_data):
        """Test that default parameters work correctly."""
        high, low, close = sample_ohlc_data

        # Should work with defaults (period=14)
        result = calculate_williams_r(high, low, close)

        assert len(result) == len(close)
        assert not np.all(np.isnan(result))

    def test_different_periods(self, sample_ohlc_data):
        """Test with different period values."""
        high, low, close = sample_ohlc_data

        # Test period=5
        result_5 = calculate_williams_r(high, low, close, period=5, engine="cpu")

        # Test period=20
        result_20 = calculate_williams_r(high, low, close, period=20, engine="cpu")

        # Shorter period should have fewer NaN values at start
        assert np.sum(np.isnan(result_5)) < np.sum(np.isnan(result_20))

        # Results should be different
        valid_mask_5 = ~np.isnan(result_5)
        valid_mask_20 = ~np.isnan(result_20)
        common_valid = valid_mask_5 & valid_mask_20
        assert not np.allclose(
            result_5[common_valid], result_20[common_valid]
        ), "Different periods should produce different results"


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================


class TestWilliamsRAlgorithm:
    """Test algorithm correctness against known values and properties."""

    def test_known_values_simple_case(self):
        """Test against hand-calculated values."""
        # Create simple test data with known Williams %R values
        high = np.array([110.0, 115.0, 120.0, 125.0, 130.0])
        low = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        close = np.array([105.0, 110.0, 115.0, 120.0, 125.0])  # Mid-range

        result = calculate_williams_r(high, low, close, period=3, engine="cpu")

        # First 2 values should be NaN (warmup for period=3)
        assert np.all(np.isnan(result[:2]))

        # Index 2: highest_high=120, lowest_low=100, close=115
        # %R = -100 * (120 - 115) / (120 - 100) = -100 * 5 / 20 = -25.0
        assert np.isclose(result[2], -25.0, rtol=1e-6)

        # Index 3: highest_high=125, lowest_low=105, close=120
        # %R = -100 * (125 - 120) / (125 - 105) = -100 * 5 / 20 = -25.0
        assert np.isclose(result[3], -25.0, rtol=1e-6)

        # Index 4: highest_high=130, lowest_low=110, close=125
        # %R = -100 * (130 - 125) / (130 - 110) = -100 * 5 / 20 = -25.0
        assert np.isclose(result[4], -25.0, rtol=1e-6)

    def test_williams_r_formula_consistency(self):
        """Test Williams %R formula is applied consistently."""
        high = np.array([115, 120, 125, 130, 135, 140])
        low = np.array([105, 110, 115, 120, 125, 130])
        close = np.array([110, 115, 120, 125, 130, 135])
        period = 3

        result = calculate_williams_r(high, low, close, period=period, engine="cpu")

        # Manually calculate expected Williams %R values
        for i in range(period - 1, len(close)):
            window_start = max(0, i - period + 1)
            highest_high = np.max(high[window_start : i + 1])
            lowest_low = np.min(low[window_start : i + 1])
            current_close = close[i]

            expected = -100.0 * (highest_high - current_close) / (highest_high - lowest_low)

            assert np.isclose(
                result[i], expected, rtol=1e-6
            ), f"Williams %R mismatch at index {i}: got {result[i]}, expected {expected}"

    def test_inverse_relationship_to_price_position(self):
        """Test that Williams %R shows inverse relationship to price position."""
        # Price near top of range should give %R near 0
        high = np.array([100] * 20)
        low = np.array([80] * 20)
        close = np.array([98] * 20)  # Near high

        result_top = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Price near bottom of range should give %R near -100
        close_bottom = np.array([82] * 20)  # Near low
        result_bottom = calculate_williams_r(high, low, close_bottom, period=14, engine="cpu")

        # %R when price is near top should be closer to 0
        valid_top = result_top[~np.isnan(result_top)]
        valid_bottom = result_bottom[~np.isnan(result_bottom)]

        assert np.mean(valid_top) > np.mean(
            valid_bottom
        ), "Price near top should give higher %R (closer to 0)"

    def test_comparison_with_stochastic(self):
        """Test relationship between Williams %R and Stochastic %K."""
        from kimsfinance.ops.indicators import calculate_stochastic_oscillator

        high = np.array([115, 120, 125, 130, 135, 140, 135, 130, 125, 120, 115] * 3)
        low = np.array([105, 110, 115, 120, 125, 130, 125, 120, 115, 110, 105] * 3)
        close = np.array([110, 115, 120, 125, 130, 135, 130, 125, 120, 115, 110] * 3)

        williams_r = calculate_williams_r(high, low, close, period=14, engine="cpu")
        stoch_k, _ = calculate_stochastic_oscillator(high, low, close, period=14, engine="cpu")

        # Williams %R = Stochastic %K - 100
        # Therefore: Stochastic %K = Williams %R + 100
        valid_mask = ~np.isnan(williams_r) & ~np.isnan(stoch_k)

        expected_stoch = williams_r[valid_mask] + 100
        assert np.allclose(
            stoch_k[valid_mask], expected_stoch, rtol=1e-5
        ), "Williams %R should be inverse of Stochastic %K"


# ============================================================================
# Signal Generation Tests
# ============================================================================


class TestWilliamsRSignals:
    """Test signal generation properties."""

    def test_overbought_condition(self):
        """Test overbought condition detection (%R > -20)."""
        # Create data where price is consistently near top of range
        high = np.array([100] * 30)
        low = np.array([80] * 30)
        close = np.array([97] * 30)  # Very close to high

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Most values should indicate overbought
        valid_values = result[~np.isnan(result)]
        overbought_count = np.sum(valid_values > -20)

        assert (
            overbought_count > len(valid_values) * 0.8
        ), "Price near high should show overbought condition"

    def test_oversold_condition(self):
        """Test oversold condition detection (%R < -80)."""
        # Create data where price is consistently near bottom of range
        high = np.array([100] * 30)
        low = np.array([80] * 30)
        close = np.array([83] * 30)  # Very close to low

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Most values should indicate oversold
        valid_values = result[~np.isnan(result)]
        oversold_count = np.sum(valid_values < -80)

        assert (
            oversold_count > len(valid_values) * 0.8
        ), "Price near low should show oversold condition"

    def test_momentum_shifts(self, trending_data):
        """Test that Williams %R responds to momentum shifts."""
        high, low, close = trending_data

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # During uptrend (first half), %R should generally be higher (closer to 0)
        uptrend_values = result[13:20]  # After warmup, during uptrend

        # During downtrend (second half), %R should generally be lower (closer to -100)
        downtrend_values = result[-7:]  # During downtrend

        assert np.mean(uptrend_values) > np.mean(
            downtrend_values
        ), "Uptrend should have higher %R than downtrend"

    def test_volatility_response(self):
        """Test Williams %R response to volatility changes."""
        # Low volatility period
        high_low = np.array([102, 104, 103, 105, 104] * 5)
        low_low = np.array([98, 96, 97, 95, 96] * 5)
        close_low = np.array([100, 100, 100, 100, 100] * 5)

        result_low = calculate_williams_r(high_low, low_low, close_low, period=14, engine="cpu")

        # High volatility period
        high_high = np.array([120, 130, 125, 135, 128] * 5)
        low_high = np.array([80, 70, 75, 65, 72] * 5)
        close_high = np.array([100, 100, 100, 100, 100] * 5)

        result_high = calculate_williams_r(high_high, low_high, close_high, period=14, engine="cpu")

        # Higher volatility should lead to more varied %R values
        valid_low = result_low[~np.isnan(result_low)]
        valid_high = result_high[~np.isnan(result_high)]

        # Both should be around -50 (middle of range) but high volatility might vary more
        assert len(valid_low) > 0 and len(valid_high) > 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestWilliamsREdgeCases:
    """Test edge cases and error handling."""

    def test_flat_prices(self):
        """Test with constant prices (no range)."""
        # All prices are identical
        high = np.full(50, 100.0)
        low = np.full(50, 100.0)
        close = np.full(50, 100.0)

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Should handle gracefully (epsilon prevents division by zero)
        # Result should be defined (not NaN) after warmup
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0, "Should produce values even with flat prices"

    def test_extreme_price_ranges(self):
        """Test with extreme price ranges."""
        # Very wide range
        high = np.array([1000.0] * 30)
        low = np.array([10.0] * 30)
        close = np.array([500.0] * 30)  # Middle

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Should calculate correctly even with extreme ranges
        valid_values = result[~np.isnan(result)]

        # Close is at middle, so %R should be around -50
        assert np.all(valid_values >= -100) and np.all(valid_values <= 0)
        assert np.allclose(valid_values, -50.505050, atol=5), "Middle price should give %R near -50"

    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        high = np.array([110.0, 115.0, np.nan, 125.0, 130.0] * 5)
        low = np.array([100.0, 105.0, 110.0, np.nan, 120.0] * 5)
        close = np.array([105.0, 110.0, 115.0, 120.0, 125.0] * 5)

        # Should not raise an error
        result = calculate_williams_r(high, low, close, period=5, engine="cpu")

        # Result should be same length
        assert len(result) == len(close)

    def test_insufficient_data(self):
        """Test with insufficient data for the period."""
        # Only 10 data points, but period is 14
        high = np.random.uniform(100, 110, 10)
        low = np.random.uniform(90, 100, 10)
        close = np.random.uniform(95, 105, 10)

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Should return all NaN (not enough data)
        assert np.all(np.isnan(result))

    def test_minimal_data_size(self):
        """Test with minimal valid data size."""
        # Exactly period rows (minimum for valid calculation)
        period = 14
        n = 14
        np.random.seed(42)
        high = 100 + np.random.uniform(5, 10, n)
        low = 100 - np.random.uniform(5, 10, n)
        close = 100 + np.random.uniform(-3, 3, n)

        result = calculate_williams_r(high, low, close, period=period, engine="cpu")

        # Should complete without error
        assert len(result) == n
        # Should have exactly one valid value (at index period-1)
        valid_count = np.sum(~np.isnan(result))
        assert valid_count == 1

    def test_handles_list_input(self):
        """Test that function handles list inputs (not just numpy arrays)."""
        high = [110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
        low = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170]
        close = [105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175]

        result = calculate_williams_r(high, low, close, period=5, engine="cpu")

        # Should complete without error
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_zero_range_handling(self):
        """Test handling when high equals low."""
        # High equals low (zero range)
        high = np.array([100.0, 105.0, 100.0, 100.0, 110.0] * 5)
        low = np.array([100.0, 105.0, 100.0, 100.0, 110.0] * 5)  # Same as high
        close = np.array([100.0, 105.0, 100.0, 100.0, 110.0] * 5)

        result = calculate_williams_r(high, low, close, period=5, engine="cpu")

        # Should not raise an error (epsilon prevents division by zero)
        # Result should be defined
        assert len(result) == len(close)


# ============================================================================
# GPU/CPU Equivalence Tests
# ============================================================================


class TestWilliamsRGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_small_data(self, sample_ohlc_data):
        """Test GPU and CPU produce identical results on small dataset."""
        high, low, close = sample_ohlc_data

        # CPU calculation
        result_cpu = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # GPU calculation
        result_gpu = calculate_williams_r(high, low, close, period=14, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-10)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_large_data(self, large_ohlc_data):
        """Test GPU and CPU produce identical results on large dataset."""
        high, low, close = large_ohlc_data

        # CPU calculation
        result_cpu = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # GPU calculation
        result_gpu = calculate_williams_r(high, low, close, period=14, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-10)

    def test_auto_engine_selection(self, large_ohlc_data):
        """Test that auto engine selects appropriately based on data size."""
        high, low, close = large_ohlc_data

        # Auto should select GPU for large datasets
        result_auto = calculate_williams_r(high, low, close, period=14, engine="auto")

        # Explicit CPU
        result_cpu = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Results should match
        np.testing.assert_allclose(result_auto, result_cpu, rtol=1e-10)


# ============================================================================
# Type and API Tests
# ============================================================================


class TestWilliamsRAPI:
    """Test API correctness and return types."""

    def test_return_type_is_ndarray(self, sample_ohlc_data):
        """Test that function returns numpy array."""
        high, low, close = sample_ohlc_data
        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_result_length_matches_input(self, sample_ohlc_data):
        """Test that result length matches input length."""
        high, low, close = sample_ohlc_data
        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        assert len(result) == len(close)

    def test_invalid_engine_raises_error(self, sample_ohlc_data):
        """Test that invalid engine parameter raises error."""
        high, low, close = sample_ohlc_data

        with pytest.raises(ConfigurationError, match="Invalid engine"):
            calculate_williams_r(high, low, close, period=14, engine="invalid")

    def test_mismatched_input_lengths(self):
        """Test that mismatched input lengths are handled."""
        high = np.array([110, 115, 120])
        low = np.array([100, 105])  # Different length
        close = np.array([105, 110, 115])

        # Should raise an error or handle gracefully
        with pytest.raises(Exception):
            calculate_williams_r(high, low, close, period=2, engine="cpu")

    def test_invalid_period_values(self, sample_ohlc_data):
        """Test that invalid period values produce expected behavior."""
        high, low, close = sample_ohlc_data

        # Period of 0 should produce all NaN or handle gracefully
        # (Polars rolling_max/min with window_size=0 may not raise error)
        try:
            result = calculate_williams_r(high, low, close, period=0, engine="cpu")
            # If it doesn't raise, check that it handles gracefully
            assert len(result) == len(close)
        except Exception:
            # If it does raise, that's also acceptable
            pass

        # Negative period should be handled
        try:
            result = calculate_williams_r(high, low, close, period=-5, engine="cpu")
            assert len(result) == len(close)
        except Exception:
            # Exception is acceptable for negative period
            pass


# ============================================================================
# Performance Characteristics Tests
# ============================================================================


class TestWilliamsRPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_ohlc_data):
        """Test that calculation completes quickly on small dataset."""
        import time

        high, low, close = sample_ohlc_data

        start = time.time()
        result = calculate_williams_r(high, low, close, period=14, engine="cpu")
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_ohlc_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time

        high, low, close = large_ohlc_data

        start = time.time()
        result = calculate_williams_r(high, low, close, period=14, engine="cpu")
        elapsed = time.time() - start

        # 600K rows should complete in under 5 seconds on CPU
        assert elapsed < 5.0, f"Large dataset took {elapsed:.3f}s - should be <5s"

    def test_rolling_operations_efficiency(self, sample_ohlc_data):
        """Test that rolling operations are efficient."""
        import time

        high, low, close = sample_ohlc_data

        # Multiple periods should scale linearly
        periods = [5, 10, 20]
        times = []

        for period in periods:
            start = time.time()
            calculate_williams_r(high, low, close, period=period, engine="cpu")
            elapsed = time.time() - start
            times.append(elapsed)

        # All should complete quickly
        assert all(t < 1.0 for t in times), "All periods should complete quickly"


# ============================================================================
# Integration Tests
# ============================================================================


class TestWilliamsRIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_ohlc_data):
        """Test that function works with Polars Series input."""
        import polars as pl

        high, low, close = sample_ohlc_data

        df = pl.DataFrame({"high": high, "low": low, "close": close})

        result = calculate_williams_r(df["high"], df["low"], df["close"], period=14, engine="cpu")

        assert len(result) == len(close)
        assert not np.all(np.isnan(result))

    def test_consistent_with_other_momentum_indicators(self, sample_ohlc_data):
        """Test that Williams %R behaves consistently with other momentum indicators."""
        from kimsfinance.ops.indicators import calculate_rsi

        high, low, close = sample_ohlc_data

        # Calculate Williams %R and RSI
        williams = calculate_williams_r(high, low, close, period=14, engine="cpu")
        rsi = calculate_rsi(close, period=14, engine="cpu")

        # Both should have same structure (same length)
        assert len(williams) == len(rsi)

        # Williams %R should have NaN values at start (warmup period)
        assert np.sum(np.isnan(williams)) > 0

        # Both should respond to price changes (not constant)
        valid_williams = williams[~np.isnan(williams)]
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.std(valid_williams) > 0
        assert np.std(valid_rsi) > 0


# ============================================================================
# Statistical Properties Tests
# ============================================================================


class TestWilliamsRStatisticalProperties:
    """Test statistical properties of Williams %R indicator."""

    def test_williams_r_symmetry(self):
        """Test Williams %R response to symmetric price movements."""
        # Create symmetrical price pattern
        high = np.array([110, 115, 120, 125, 130, 125, 120, 115, 110] * 3)
        low = np.array([100, 105, 110, 115, 120, 115, 110, 105, 100] * 3)
        close = np.array([105, 110, 115, 120, 125, 120, 115, 110, 105] * 3)

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Should produce valid results
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # Should be in valid range
        assert np.all(valid_values >= -100) and np.all(valid_values <= 0)

    def test_williams_r_sensitivity_to_period(self):
        """Test that Williams %R sensitivity changes with period."""
        high = np.array([110, 115, 120, 125, 130, 135, 140, 145, 150] * 5)
        low = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140] * 5)
        close = np.array([105, 110, 115, 120, 125, 130, 135, 140, 145] * 5)

        # Short period should be more responsive
        result_short = calculate_williams_r(high, low, close, period=5, engine="cpu")

        # Long period should be smoother
        result_long = calculate_williams_r(high, low, close, period=20, engine="cpu")

        # Get common valid indices
        valid_mask = ~np.isnan(result_short) & ~np.isnan(result_long)

        # Short period should have higher variance (more responsive)
        assert np.std(result_short[valid_mask]) >= np.std(
            result_long[valid_mask]
        ), "Shorter period should have higher variance"

    def test_williams_r_mean_reversion(self):
        """Test Williams %R mean reversion properties."""
        # Create oscillating price pattern
        n_cycles = 10
        cycle_length = 20
        close = np.tile(
            np.concatenate(
                [np.linspace(100, 120, cycle_length // 2), np.linspace(120, 100, cycle_length // 2)]
            ),
            n_cycles,
        )

        high = close + 2
        low = close - 2

        result = calculate_williams_r(high, low, close, period=14, engine="cpu")

        # Williams %R should oscillate (not trend)
        valid_values = result[~np.isnan(result)]

        # Mean should be somewhere in the middle range
        mean_wr = np.mean(valid_values)
        assert (
            -70 < mean_wr < -30
        ), f"Mean Williams %R should be near middle of range, got {mean_wr}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

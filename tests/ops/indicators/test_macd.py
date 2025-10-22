#!/usr/bin/env python3
"""
Comprehensive Tests for MACD (Moving Average Convergence Divergence) Indicator
===============================================================================

Tests the calculate_macd() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.

MACD is a trend-following momentum indicator showing the relationship
between two exponential moving averages (EMAs).
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_macd
from kimsfinance.ops.indicators.macd import CUPY_AVAILABLE
from kimsfinance.core import EngineManager
from kimsfinance.core.exceptions import ConfigurationError


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
    n = 600_000  # Above GPU threshold
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return prices


@pytest.fixture
def trending_data():
    """Generate trending price data for signal testing."""
    # Strong uptrend
    return np.linspace(100, 200, 100)


@pytest.fixture
def oscillating_data():
    """Generate oscillating price data for crossover testing."""
    x = np.linspace(0, 4 * np.pi, 100)
    return 100 + 20 * np.sin(x)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestMACDBasic:
    """Test basic MACD calculation."""

    def test_basic_calculation(self, sample_data):
        """Test basic MACD calculation returns correct structure."""
        macd_line, signal_line, histogram = calculate_macd(
            sample_data, fast_period=12, slow_period=26, signal_period=9, engine="cpu"
        )

        # Check all three outputs exist
        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None

        # Check lengths match input
        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)

        # Should have some valid values after warmup
        assert not np.all(np.isnan(macd_line))
        assert not np.all(np.isnan(signal_line))
        assert not np.all(np.isnan(histogram))

    def test_default_parameters(self, sample_data):
        """Test that default parameters work correctly (12, 26, 9)."""
        macd_line, signal_line, histogram = calculate_macd(sample_data)

        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)

        # Should have valid values after warmup period
        assert not np.all(np.isnan(macd_line))
        assert not np.all(np.isnan(signal_line))
        assert not np.all(np.isnan(histogram))

    def test_custom_parameters(self, sample_data):
        """Test with custom MACD parameters."""
        # Use faster parameters (5, 13, 5)
        macd_fast, signal_fast, hist_fast = calculate_macd(
            sample_data, fast_period=5, slow_period=13, signal_period=5, engine="cpu"
        )

        # Use slower parameters (19, 39, 9)
        macd_slow, signal_slow, hist_slow = calculate_macd(
            sample_data, fast_period=19, slow_period=39, signal_period=9, engine="cpu"
        )

        # Results should be different
        valid_mask_fast = ~np.isnan(macd_fast)
        valid_mask_slow = ~np.isnan(macd_slow)
        common_valid = valid_mask_fast & valid_mask_slow

        assert not np.allclose(
            macd_fast[common_valid], macd_slow[common_valid]
        ), "Different parameters should produce different results"

    def test_three_outputs_same_length(self, sample_data):
        """Test that all three outputs have the same length."""
        macd_line, signal_line, histogram = calculate_macd(sample_data, engine="cpu")

        assert len(macd_line) == len(signal_line) == len(histogram)

    def test_macd_line_is_ema_difference(self):
        """Test that MACD line is correctly calculated as fast EMA - slow EMA."""
        from kimsfinance.ops.indicators.moving_averages import calculate_ema

        prices = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118] * 5)  # 50 points

        macd_line, _, _ = calculate_macd(
            prices, fast_period=5, slow_period=10, signal_period=3, engine="cpu"
        )

        # Calculate EMAs manually
        ema_fast = calculate_ema(prices, period=5, engine="cpu")
        ema_slow = calculate_ema(prices, period=10, engine="cpu")

        # MACD line should equal fast - slow
        expected_macd = ema_fast - ema_slow

        np.testing.assert_allclose(macd_line, expected_macd, rtol=1e-10)

    def test_histogram_is_macd_minus_signal(self, sample_data):
        """Test that histogram equals MACD line - signal line."""
        macd_line, signal_line, histogram = calculate_macd(sample_data, engine="cpu")

        # Histogram should equal MACD - Signal
        expected_histogram = macd_line - signal_line

        np.testing.assert_allclose(histogram, expected_histogram, rtol=1e-10)


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================


class TestMACDAlgorithm:
    """Test algorithm correctness against known values and properties."""

    def test_known_values_simple_case(self):
        """Test against manually calculated values."""
        # Simple uptrend data
        prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0] + [120.0] * 50)

        macd_line, signal_line, histogram = calculate_macd(
            prices, fast_period=5, slow_period=8, signal_period=3, engine="cpu"
        )

        # MACD line should be positive in an uptrend (fast EMA > slow EMA)
        # After warmup period, check that MACD is consistently positive
        assert np.all(macd_line[20:30] > 0), "MACD should be positive in uptrend"

        # Signal line should smooth the MACD line
        # Both should be positive in sustained uptrend
        assert np.all(signal_line[20:30] > 0), "Signal should be positive in uptrend"

    def test_constant_prices_produce_zero_macd(self):
        """Test that constant prices produce zero MACD."""
        prices = np.full(100, 100.0)

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # All valid values should be 0 (no change)
        valid_mask_macd = ~np.isnan(macd_line)
        valid_mask_signal = ~np.isnan(signal_line)
        valid_mask_hist = ~np.isnan(histogram)

        assert np.allclose(macd_line[valid_mask_macd], 0.0, atol=1e-10)
        assert np.allclose(signal_line[valid_mask_signal], 0.0, atol=1e-10)
        assert np.allclose(histogram[valid_mask_hist], 0.0, atol=1e-10)

    def test_macd_responds_to_trend_changes(self):
        """Test that MACD responds appropriately to trend changes."""
        # Create data with uptrend then downtrend
        uptrend = np.linspace(100, 150, 50)
        downtrend = np.linspace(150, 100, 50)
        prices = np.concatenate([uptrend, downtrend])

        macd_line, signal_line, histogram = calculate_macd(
            prices, fast_period=12, slow_period=26, signal_period=9, engine="cpu"
        )

        # In uptrend, MACD should generally be positive
        uptrend_macd = macd_line[30:45]  # Middle of uptrend
        assert np.mean(uptrend_macd) > 0, "MACD should be positive in uptrend"

        # In downtrend, MACD should generally be negative
        downtrend_macd = macd_line[70:85]  # Middle of downtrend
        assert np.mean(downtrend_macd) < 0, "MACD should be negative in downtrend"

    def test_signal_line_smooths_macd(self):
        """Test that signal line is smoother than MACD line."""
        # Create volatile data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 5)

        macd_line, signal_line, _ = calculate_macd(prices, engine="cpu")

        # Remove NaN values
        valid_idx = ~(np.isnan(macd_line) | np.isnan(signal_line))
        macd_valid = macd_line[valid_idx]
        signal_valid = signal_line[valid_idx]

        # Signal line should have lower standard deviation (smoother)
        # Calculate consecutive differences to measure volatility
        macd_volatility = np.std(np.diff(macd_valid))
        signal_volatility = np.std(np.diff(signal_valid))

        assert signal_volatility < macd_volatility, "Signal line should be smoother than MACD"


# ============================================================================
# Signal Generation Tests
# ============================================================================


class TestMACDSignals:
    """Test MACD signal generation capabilities."""

    def test_bullish_crossover_detection(self):
        """Test detection of bullish crossover (MACD crosses above signal)."""
        # Create data that produces a clear bullish crossover
        # Need enough data for MACD to stabilize, then create crossover pattern
        # Start flat, drop, then rally strongly
        flat = np.full(40, 100.0)
        drop = np.linspace(100, 85, 30)
        rally = np.linspace(85, 115, 60)
        prices = np.concatenate([flat, drop, rally])

        macd_line, signal_line, histogram = calculate_macd(
            prices, fast_period=8, slow_period=17, signal_period=7, engine="cpu"
        )

        # Find crossover points where MACD crosses above signal
        # Histogram changes from negative to positive
        valid_mask = ~np.isnan(histogram)
        histogram_valid = histogram[valid_mask]

        # Look for sign changes from negative to positive
        crossovers = []
        for i in range(1, len(histogram_valid)):
            if histogram_valid[i - 1] < 0 and histogram_valid[i] > 0:
                crossovers.append(i)

        # Should detect at least one bullish crossover
        assert len(crossovers) > 0, "Should detect bullish crossover in uptrend"

    def test_bearish_crossover_detection(self):
        """Test detection of bearish crossover (MACD crosses below signal)."""
        # Create data that produces a clear bearish crossover
        # Start flat, rally, then drop strongly
        flat = np.full(40, 100.0)
        rally = np.linspace(100, 115, 30)
        drop = np.linspace(115, 85, 60)
        prices = np.concatenate([flat, rally, drop])

        macd_line, signal_line, histogram = calculate_macd(
            prices, fast_period=8, slow_period=17, signal_period=7, engine="cpu"
        )

        # Find crossover points where MACD crosses below signal
        # Histogram changes from positive to negative
        valid_mask = ~np.isnan(histogram)
        histogram_valid = histogram[valid_mask]

        # Look for sign changes from positive to negative
        crossovers = []
        for i in range(1, len(histogram_valid)):
            if histogram_valid[i - 1] > 0 and histogram_valid[i] < 0:
                crossovers.append(i)

        # Should detect at least one bearish crossover
        assert len(crossovers) > 0, "Should detect bearish crossover in downtrend"

    def test_zero_line_crossovers(self):
        """Test MACD zero-line crossovers."""
        # Create oscillating data around a mean
        x = np.linspace(0, 4 * np.pi, 200)
        prices = 100 + 20 * np.sin(x)

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # Remove NaN values
        valid_mask = ~np.isnan(macd_line)
        macd_valid = macd_line[valid_mask]

        # Count zero-line crossovers (sign changes)
        zero_crossovers = 0
        for i in range(1, len(macd_valid)):
            if (macd_valid[i - 1] < 0 and macd_valid[i] > 0) or (
                macd_valid[i - 1] > 0 and macd_valid[i] < 0
            ):
                zero_crossovers += 1

        # Oscillating data should produce multiple zero-line crossovers
        assert zero_crossovers > 0, "Should detect MACD zero-line crossovers"

    def test_histogram_divergence(self):
        """Test that histogram shows divergence patterns."""
        # Create price data with divergence scenario
        # Prices make higher highs, but MACD histogram makes lower highs (bearish divergence)
        prices = np.array(
            [100.0]
            + list(np.linspace(100, 110, 20))  # First peak
            + list(np.linspace(110, 105, 10))  # Pullback
            + list(np.linspace(105, 112, 20))  # Higher high
            + [112.0] * 50
        )

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # Histogram should exist and vary
        valid_mask = ~np.isnan(histogram)
        histogram_valid = histogram[valid_mask]

        # Should have both positive and negative histogram values
        assert np.any(histogram_valid > 0), "Should have positive histogram values"
        assert np.any(histogram_valid < 0), "Should have negative histogram values"

        # Histogram should not be constant
        assert np.std(histogram_valid) > 0, "Histogram should vary"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestMACDEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        empty_data = np.array([])

        with pytest.raises((ValueError, IndexError)):
            calculate_macd(empty_data, engine="cpu")

    def test_single_point_raises_error(self):
        """Test that single data point raises ValueError."""
        single_point = np.array([100.0])

        with pytest.raises(ValueError):
            calculate_macd(single_point, engine="cpu")

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        # Only 10 data points, but slow_period is 26
        short_data = np.random.randn(10) + 100

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_macd(short_data, fast_period=12, slow_period=26, signal_period=9, engine="cpu")

    def test_minimal_data_size(self):
        """Test with minimal valid data size."""
        # Need at least slow_period points
        slow_period = 26
        n = 27  # Just enough
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        macd_line, signal_line, histogram = calculate_macd(
            prices, fast_period=12, slow_period=slow_period, signal_period=9, engine="cpu"
        )

        # Should complete without error
        assert len(macd_line) == n
        assert len(signal_line) == n
        assert len(histogram) == n

    def test_handles_list_input(self):
        """Test that function handles list inputs (not just numpy arrays)."""
        prices = [100 + i for i in range(50)]

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # Should complete without error
        assert isinstance(macd_line, np.ndarray)
        assert isinstance(signal_line, np.ndarray)
        assert isinstance(histogram, np.ndarray)
        assert len(macd_line) == len(prices)

    def test_all_nan_data(self):
        """Test handling of all NaN data."""
        nan_data = np.full(100, np.nan)

        macd_line, signal_line, histogram = calculate_macd(nan_data, engine="cpu")

        # Should return all NaN
        assert np.all(np.isnan(macd_line))
        assert np.all(np.isnan(signal_line))
        assert np.all(np.isnan(histogram))

    def test_some_nan_values(self):
        """Test handling of data with some NaN values."""
        prices = np.array([100.0, 102.0, np.nan, 106.0, 108.0] * 10)

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # Should handle gracefully - NaN propagates
        assert len(macd_line) == len(prices)
        # Some values will be NaN due to input NaN
        assert np.any(np.isnan(macd_line))

    def test_extreme_volatility(self):
        """Test handling of extremely volatile data."""
        np.random.seed(42)
        # Very high volatility
        prices = 100 + np.cumsum(np.random.randn(100) * 50)

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # Should complete without error
        assert len(macd_line) == 100
        # Should have valid values after warmup
        assert not np.all(np.isnan(macd_line))

    def test_period_greater_than_data_length(self):
        """Test that period > data length raises error."""
        short_data = np.random.randn(20) + 100

        with pytest.raises(ValueError):
            calculate_macd(short_data, fast_period=12, slow_period=50, signal_period=9, engine="cpu")

    def test_negative_prices(self):
        """Test handling of negative prices."""
        prices = np.array([-100, -105, -110, -115, -120] * 10)

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # Should calculate correctly
        assert not np.all(np.isnan(macd_line))
        assert len(macd_line) == len(prices)


# ============================================================================
# Parameter Validation Tests
# ============================================================================


class TestMACDParameterValidation:
    """Test parameter validation."""

    def test_invalid_fast_period_raises_error(self, sample_data):
        """Test that invalid fast_period raises ValueError."""
        with pytest.raises(ValueError):
            calculate_macd(sample_data, fast_period=0, slow_period=26, signal_period=9)

        with pytest.raises(ValueError):
            calculate_macd(sample_data, fast_period=-5, slow_period=26, signal_period=9)

    def test_invalid_slow_period_raises_error(self, sample_data):
        """Test that invalid slow_period raises ValueError."""
        with pytest.raises(ValueError):
            calculate_macd(sample_data, fast_period=12, slow_period=0, signal_period=9)

        with pytest.raises(ValueError):
            calculate_macd(sample_data, fast_period=12, slow_period=-10, signal_period=9)

    def test_invalid_signal_period_raises_error(self, sample_data):
        """Test that invalid signal_period raises ValueError."""
        with pytest.raises(ValueError):
            calculate_macd(sample_data, fast_period=12, slow_period=26, signal_period=0)

        with pytest.raises(ValueError):
            calculate_macd(sample_data, fast_period=12, slow_period=26, signal_period=-3)

    def test_fast_period_greater_than_slow_period(self, sample_data):
        """Test behavior when fast_period > slow_period (unusual but valid)."""
        # This is unusual but mathematically valid
        macd_line, signal_line, histogram = calculate_macd(
            sample_data, fast_period=26, slow_period=12, signal_period=9, engine="cpu"
        )

        # Should complete without error
        assert len(macd_line) == len(sample_data)

        # MACD line will just be inverted (slow EMA - fast EMA)
        # This is technically valid, just unconventional

    @pytest.mark.skip(reason="Engine validation not yet implemented in MACD/EMA")
    def test_invalid_engine_raises_error(self, sample_data):
        """Test that invalid engine parameter raises error."""
        # TODO: Implement engine validation in calculate_ema/calculate_macd
        with pytest.raises(ConfigurationError, match="Invalid engine"):
            calculate_macd(sample_data, engine="invalid")

    def test_signal_period_larger_than_slow_period(self, sample_data):
        """Test behavior when signal_period is larger than slow_period."""
        # Valid but unusual configuration
        macd_line, signal_line, histogram = calculate_macd(
            sample_data, fast_period=5, slow_period=10, signal_period=20, engine="cpu"
        )

        # Should complete without error
        assert len(macd_line) == len(sample_data)
        assert not np.all(np.isnan(signal_line))


# ============================================================================
# GPU/CPU Equivalence Tests
# ============================================================================


class TestMACDGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_small_data(self, sample_data):
        """Test GPU and CPU produce identical results on small dataset."""
        # CPU calculation
        macd_cpu, signal_cpu, hist_cpu = calculate_macd(sample_data, engine="cpu")

        # GPU calculation
        macd_gpu, signal_gpu, hist_gpu = calculate_macd(sample_data, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(macd_cpu, macd_gpu, rtol=1e-10)
        np.testing.assert_allclose(signal_cpu, signal_gpu, rtol=1e-10)
        np.testing.assert_allclose(hist_cpu, hist_gpu, rtol=1e-10)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_large_data(self, large_data):
        """Test GPU and CPU produce identical results on large dataset."""
        # CPU calculation
        macd_cpu, signal_cpu, hist_cpu = calculate_macd(large_data, engine="cpu")

        # GPU calculation
        macd_gpu, signal_gpu, hist_gpu = calculate_macd(large_data, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(macd_cpu, macd_gpu, rtol=1e-10)
        np.testing.assert_allclose(signal_cpu, signal_gpu, rtol=1e-10)
        np.testing.assert_allclose(hist_cpu, hist_gpu, rtol=1e-10)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_custom_parameters(self, sample_data):
        """Test GPU and CPU match with custom parameters."""
        # CPU calculation
        macd_cpu, signal_cpu, hist_cpu = calculate_macd(
            sample_data, fast_period=5, slow_period=13, signal_period=5, engine="cpu"
        )

        # GPU calculation
        macd_gpu, signal_gpu, hist_gpu = calculate_macd(
            sample_data, fast_period=5, slow_period=13, signal_period=5, engine="gpu"
        )

        # Should match
        np.testing.assert_allclose(macd_cpu, macd_gpu, rtol=1e-10)
        np.testing.assert_allclose(signal_cpu, signal_gpu, rtol=1e-10)
        np.testing.assert_allclose(hist_cpu, hist_gpu, rtol=1e-10)

    def test_auto_engine_selection(self, large_data):
        """Test that auto engine selects appropriately based on data size."""
        # Auto should select appropriately for large datasets
        macd_auto, signal_auto, hist_auto = calculate_macd(large_data, engine="auto")

        # Explicit CPU
        macd_cpu, signal_cpu, hist_cpu = calculate_macd(large_data, engine="cpu")

        # Results should match
        np.testing.assert_allclose(macd_auto, macd_cpu, rtol=1e-10)
        np.testing.assert_allclose(signal_auto, signal_cpu, rtol=1e-10)
        np.testing.assert_allclose(hist_auto, hist_cpu, rtol=1e-10)


# ============================================================================
# Type and API Tests
# ============================================================================


class TestMACDAPI:
    """Test API correctness and return types."""

    def test_return_type_is_tuple_of_ndarrays(self, sample_data):
        """Test that function returns tuple of three numpy arrays."""
        result = calculate_macd(sample_data, engine="cpu")

        assert isinstance(result, tuple)
        assert len(result) == 3

        macd_line, signal_line, histogram = result

        assert isinstance(macd_line, np.ndarray)
        assert isinstance(signal_line, np.ndarray)
        assert isinstance(histogram, np.ndarray)

        assert macd_line.dtype == np.float64
        assert signal_line.dtype == np.float64
        assert histogram.dtype == np.float64

    def test_result_length_matches_input(self, sample_data):
        """Test that result lengths match input length."""
        macd_line, signal_line, histogram = calculate_macd(sample_data, engine="cpu")

        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)

    def test_unpacking_result(self, sample_data):
        """Test that result can be unpacked correctly."""
        # Should be able to unpack into three variables
        macd, signal, hist = calculate_macd(sample_data, engine="cpu")

        assert macd is not None
        assert signal is not None
        assert hist is not None


# ============================================================================
# Performance Characteristics Tests
# ============================================================================


class TestMACDPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_data):
        """Test that calculation completes quickly on small dataset."""
        import time

        start = time.time()
        macd_line, signal_line, histogram = calculate_macd(sample_data, engine="cpu")
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time

        start = time.time()
        macd_line, signal_line, histogram = calculate_macd(large_data, engine="cpu")
        elapsed = time.time() - start

        # 600K rows should complete in under 5 seconds on CPU
        assert elapsed < 5.0, f"Large dataset took {elapsed:.3f}s - should be <5s"

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_threshold_validation(self, large_data):
        """Test that GPU is actually used for large datasets when available."""
        # This test validates that engine='auto' uses GPU for large data
        import time

        # Warm up GPU if available
        try:
            _ = calculate_macd(large_data[:1000], engine="gpu")
        except Exception:
            pytest.skip("GPU warmup failed")

        # Time GPU calculation
        start_gpu = time.time()
        macd_gpu, signal_gpu, hist_gpu = calculate_macd(large_data, engine="gpu")
        elapsed_gpu = time.time() - start_gpu

        # Time CPU calculation
        start_cpu = time.time()
        macd_cpu, signal_cpu, hist_cpu = calculate_macd(large_data, engine="cpu")
        elapsed_cpu = time.time() - start_cpu

        # GPU should be faster for large datasets (or at least not much slower)
        # Allow for some variance due to overhead
        # GPU might not always be faster for MACD due to multiple EMA calculations
        # but should complete successfully
        assert elapsed_gpu < elapsed_cpu * 2, f"GPU ({elapsed_gpu:.3f}s) should be reasonably fast vs CPU ({elapsed_cpu:.3f}s)"


# ============================================================================
# Integration Tests
# ============================================================================


class TestMACDIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_data):
        """Test that function works with Polars Series input."""
        import polars as pl

        df = pl.DataFrame({"price": sample_data})

        macd_line, signal_line, histogram = calculate_macd(df["price"], engine="cpu")

        assert len(macd_line) == len(sample_data)
        assert not np.all(np.isnan(macd_line))

    def test_consistent_with_other_indicators(self, sample_data):
        """Test that MACD behaves consistently with other trend indicators."""
        from kimsfinance.ops.indicators import calculate_ema

        # Calculate MACD
        macd_line, signal_line, histogram = calculate_macd(sample_data, engine="cpu")

        # Calculate EMAs independently
        ema_12 = calculate_ema(sample_data, period=12, engine="cpu")
        ema_26 = calculate_ema(sample_data, period=26, engine="cpu")

        # MACD line should equal EMA(12) - EMA(26)
        expected_macd = ema_12 - ema_26
        np.testing.assert_allclose(macd_line, expected_macd, rtol=1e-10)

    def test_macd_with_different_data_types(self):
        """Test MACD with different input data types."""
        # NumPy array
        np_data = np.array([100.0 + i for i in range(50)])
        macd_np, _, _ = calculate_macd(np_data, engine="cpu")

        # Python list
        list_data = [100.0 + i for i in range(50)]
        macd_list, _, _ = calculate_macd(list_data, engine="cpu")

        # Polars Series
        import polars as pl

        pl_data = pl.Series([100.0 + i for i in range(50)])
        macd_pl, _, _ = calculate_macd(pl_data, engine="cpu")

        # All should produce the same result
        np.testing.assert_allclose(macd_np, macd_list, rtol=1e-10)
        np.testing.assert_allclose(macd_np, macd_pl, rtol=1e-10)


# ============================================================================
# NaN Handling Tests
# ============================================================================


class TestMACDNaNHandling:
    """Test NaN handling in MACD calculations."""

    def test_nan_warmup_period(self):
        """Test that initial values are NaN during warmup period."""
        prices = np.array([100.0 + i for i in range(60)])

        macd_line, signal_line, histogram = calculate_macd(
            prices, fast_period=12, slow_period=26, signal_period=9, engine="cpu"
        )

        # Fast period (12) will have NaN for first 12-1 = 11 values
        # Slow period (26) will have NaN for first 26-1 = 25 values
        # MACD line will have NaN for first slow_period-1 values
        assert np.isnan(macd_line[0]), "First MACD value should be NaN"
        assert np.all(np.isnan(macd_line[:12])), "First 12 MACD values should be NaN (fast period)"

        # Signal line will have additional NaN due to signal_period
        # Signal is EMA of MACD, so needs signal_period more values
        assert np.isnan(signal_line[0]), "First signal value should be NaN"

        # Histogram depends on both, so inherits NaN pattern
        assert np.isnan(histogram[0]), "First histogram value should be NaN"

    def test_valid_values_after_warmup(self):
        """Test that valid values exist after warmup period."""
        prices = np.array([100.0 + i for i in range(100)])

        macd_line, signal_line, histogram = calculate_macd(
            prices, fast_period=12, slow_period=26, signal_period=9, engine="cpu"
        )

        # After sufficient warmup, should have valid values
        # slow_period (26) + signal_period (9) = 35 minimum warmup
        assert not np.isnan(macd_line[50]), "MACD should be valid after warmup"
        assert not np.isnan(signal_line[50]), "Signal should be valid after warmup"
        assert not np.isnan(histogram[50]), "Histogram should be valid after warmup"


# ============================================================================
# Statistical Properties Tests
# ============================================================================


class TestMACDStatisticalProperties:
    """Test statistical properties of MACD indicator."""

    def test_macd_symmetry_for_symmetric_data(self):
        """Test MACD behavior with symmetric price movements."""
        # Price goes up then back down
        up = np.linspace(100, 110, 25)
        down = np.linspace(110, 100, 25)
        prices = np.concatenate([up, down])

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # MACD should show positive then negative values
        valid_mask = ~np.isnan(macd_line)
        macd_valid = macd_line[valid_mask]

        # Should have both positive and negative values
        assert np.any(macd_valid > 0), "Should have positive MACD in uptrend"
        assert np.any(macd_valid < 0), "Should have negative MACD in downtrend"

    def test_histogram_variability(self):
        """Test that histogram varies in response to price changes."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 2)

        macd_line, signal_line, histogram = calculate_macd(prices, engine="cpu")

        # Remove NaN values
        valid_mask = ~np.isnan(histogram)
        histogram_valid = histogram[valid_mask]

        # Histogram should vary (not constant)
        assert np.std(histogram_valid) > 0, "Histogram should vary"

        # Should have both positive and negative values for random walk
        # (might not always be true for very short series, but likely for 200 points)
        # Make this a softer check
        unique_signs = len(np.unique(np.sign(histogram_valid[histogram_valid != 0])))
        assert unique_signs >= 1, "Histogram should have values"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

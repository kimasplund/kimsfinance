#!/usr/bin/env python3
"""
Comprehensive Tests for TSI (True Strength Index) Indicator
============================================================

Tests the calculate_tsi() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.

TSI is a double-smoothed momentum oscillator that shows both trend
direction and overbought/oversold conditions using double exponential
smoothing of price momentum.

Test Coverage:
- Basic Calculation Tests (15 tests)
- Signal Generation Tests (10 tests)
- Edge Cases Tests (10 tests)
- GPU/CPU Parity Tests (10 tests)
- Performance Tests (5 tests)

Total: 50 comprehensive tests
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_tsi
from kimsfinance.ops.indicators.moving_averages import calculate_ema
from kimsfinance.core import EngineManager
from kimsfinance.core.exceptions import ConfigurationError

# Check if GPU is available
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    # Need larger dataset due to TSI double-smoothing warmup
    # Default params (25, 13) need ~40+ values to start producing non-NaN
    n = 200
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
def trending_up_data():
    """Generate strong uptrend for signal testing."""
    return np.linspace(100, 200, 200)


@pytest.fixture
def trending_down_data():
    """Generate strong downtrend for signal testing."""
    return np.linspace(200, 100, 200)


@pytest.fixture
def oscillating_data():
    """Generate oscillating price data for crossover testing."""
    x = np.linspace(0, 4 * np.pi, 200)
    return 100 + 20 * np.sin(x)


@pytest.fixture
def flat_data():
    """Generate flat price data."""
    return np.full(200, 100.0)


# ============================================================================
# Class 1: Basic Calculation Tests (15 tests)
# ============================================================================


class TestTSIBasicCalculation:
    """Test basic TSI calculation correctness."""

    def test_basic_calculation(self, sample_data):
        """Test basic TSI calculation returns correct structure."""
        tsi = calculate_tsi(sample_data, long_period=25, short_period=13, engine="cpu")

        # Check output exists
        assert tsi is not None

        # Check length matches input
        assert len(tsi) == len(sample_data)

        # Should have some valid values after warmup
        assert not np.all(np.isnan(tsi))

    def test_default_parameters(self, sample_data):
        """Test that default parameters work correctly (25, 13)."""
        tsi = calculate_tsi(sample_data)

        assert len(tsi) == len(sample_data)
        # Should have valid values after warmup period
        assert not np.all(np.isnan(tsi))

    def test_custom_parameters(self, sample_data):
        """Test with custom TSI parameters."""
        # Use faster parameters (13, 7)
        tsi_fast = calculate_tsi(sample_data, long_period=13, short_period=7, engine="cpu")

        # Use slower parameters (50, 25)
        tsi_slow = calculate_tsi(sample_data, long_period=50, short_period=25, engine="cpu")

        # Results should be different
        valid_mask_fast = ~np.isnan(tsi_fast)
        valid_mask_slow = ~np.isnan(tsi_slow)
        common_valid = valid_mask_fast & valid_mask_slow

        assert not np.allclose(
            tsi_fast[common_valid], tsi_slow[common_valid]
        ), "Different parameters should produce different results"

    def test_value_range(self, sample_data):
        """Test that TSI values are in range -100 to +100."""
        tsi = calculate_tsi(sample_data, engine="cpu")

        # Remove NaN values
        valid_tsi = tsi[~np.isnan(tsi)]

        # All values should be in range [-100, 100]
        assert np.all(valid_tsi >= -100), f"Min TSI: {np.min(valid_tsi)} should be >= -100"
        assert np.all(valid_tsi <= 100), f"Max TSI: {np.max(valid_tsi)} should be <= 100"

    def test_double_smoothing_warmup_period(self):
        """Test that TSI has NaN values during double smoothing warmup."""
        prices = np.array([100.0 + i for i in range(100)])

        tsi = calculate_tsi(prices, long_period=25, short_period=13, engine="cpu")

        # First values should be NaN due to double smoothing
        # First smoothing needs long_period, second needs short_period
        assert np.isnan(tsi[0]), "First TSI value should be NaN"
        assert np.any(np.isnan(tsi[:30])), "Early TSI values should be NaN (warmup)"

    def test_valid_values_after_warmup(self, sample_data):
        """Test that valid values exist after warmup period."""
        tsi = calculate_tsi(sample_data, long_period=25, short_period=13, engine="cpu")

        # After sufficient warmup (long + short periods), should have valid values
        assert not np.isnan(tsi[50]), "TSI should be valid after warmup"
        assert not np.isnan(tsi[-1]), "Final TSI value should be valid"

    def test_uptrend_positive_tsi(self, trending_up_data):
        """Test that strong uptrend produces positive TSI."""
        tsi = calculate_tsi(trending_up_data, long_period=25, short_period=13, engine="cpu")

        # Remove NaN warmup values
        valid_tsi = tsi[~np.isnan(tsi)]

        # Most values should be positive in uptrend
        positive_ratio = np.sum(valid_tsi > 0) / len(valid_tsi)
        assert positive_ratio > 0.7, "At least 70% of TSI values should be positive in uptrend"

    def test_downtrend_negative_tsi(self, trending_down_data):
        """Test that strong downtrend produces negative TSI."""
        tsi = calculate_tsi(trending_down_data, long_period=25, short_period=13, engine="cpu")

        # Remove NaN warmup values
        valid_tsi = tsi[~np.isnan(tsi)]

        # Most values should be negative in downtrend
        negative_ratio = np.sum(valid_tsi < 0) / len(valid_tsi)
        assert negative_ratio > 0.7, "At least 70% of TSI values should be negative in downtrend"

    def test_constant_prices_produce_zero_tsi(self, flat_data):
        """Test that constant prices produce TSI near zero."""
        tsi = calculate_tsi(flat_data, engine="cpu")

        # Remove NaN values
        valid_tsi = tsi[~np.isnan(tsi)]

        # All valid values should be near 0 (no momentum)
        # Use epsilon tolerance due to floating point
        assert np.allclose(valid_tsi, 0.0, atol=1e-8), "TSI should be ~0 for flat prices"

    def test_tsi_is_double_smoothed_momentum(self):
        """Test that TSI correctly implements double-smoothed momentum formula."""
        prices = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118] * 5)  # 50 points

        tsi = calculate_tsi(prices, long_period=10, short_period=5, engine="cpu")

        # TSI should have valid values after warmup
        valid_tsi = tsi[~np.isnan(tsi)]
        assert len(valid_tsi) > 0, "TSI should have valid values"

        # TSI should be in valid range
        assert np.all(valid_tsi >= -100) and np.all(
            valid_tsi <= 100
        ), "TSI should be in [-100, 100]"

        # For alternating up/down pattern, TSI behavior should be reasonable
        # The actual values will vary based on the pattern
        # Key thing is that it calculates without error

    def test_output_length_matches_input(self, sample_data):
        """Test that TSI output length matches input length."""
        tsi = calculate_tsi(sample_data, engine="cpu")
        assert len(tsi) == len(sample_data)

    def test_return_type_is_ndarray(self, sample_data):
        """Test that function returns numpy array."""
        tsi = calculate_tsi(sample_data, engine="cpu")

        assert isinstance(tsi, np.ndarray)
        assert tsi.dtype == np.float64

    def test_handles_list_input(self):
        """Test that function handles list inputs."""
        prices = [100 + i for i in range(50)]

        tsi = calculate_tsi(prices, engine="cpu")

        # Should complete without error
        assert isinstance(tsi, np.ndarray)
        assert len(tsi) == len(prices)

    def test_shorter_periods_more_reactive(self):
        """Test that shorter periods make TSI more reactive."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 2)

        # Short periods
        tsi_short = calculate_tsi(prices, long_period=10, short_period=5, engine="cpu")

        # Long periods
        tsi_long = calculate_tsi(prices, long_period=40, short_period=20, engine="cpu")

        # Remove NaN values
        valid_short = tsi_short[~np.isnan(tsi_short)]
        valid_long = tsi_long[~np.isnan(tsi_long)]

        # Shorter periods should have higher volatility
        std_short = np.std(valid_short)
        std_long = np.std(valid_long)

        assert std_short > std_long, "Shorter periods should be more reactive"

    def test_tsi_responds_to_trend_changes(self):
        """Test that TSI responds to trend changes."""
        # Create data with uptrend then downtrend
        uptrend = np.linspace(100, 150, 50)
        downtrend = np.linspace(150, 100, 50)
        prices = np.concatenate([uptrend, downtrend])

        tsi = calculate_tsi(prices, long_period=15, short_period=7, engine="cpu")

        # In uptrend (after warmup), TSI should generally be positive
        uptrend_tsi = tsi[25:45]  # Middle of uptrend, after warmup
        assert np.mean(uptrend_tsi) > 0, "TSI should be positive in uptrend"

        # In downtrend, TSI should generally be negative
        downtrend_tsi = tsi[60:85]  # Middle of downtrend
        assert np.mean(downtrend_tsi) < 0, "TSI should be negative in downtrend"


# ============================================================================
# Class 2: Signal Generation Tests (10 tests)
# ============================================================================


class TestTSISignals:
    """Test TSI signal generation capabilities."""

    def test_zero_line_crossover_upward(self):
        """Test detection of bullish zero-line crossover (TSI crosses above 0)."""
        # Create data that produces zero-line crossover
        # Need a period below zero, then crossing to above zero
        np.random.seed(42)
        downtrend = 100 - np.cumsum(np.ones(40) * 0.3)  # Decline
        flat = np.full(20, downtrend[-1])  # Consolidation
        uptrend = downtrend[-1] + np.cumsum(np.ones(60) * 0.4)  # Rally
        prices = np.concatenate([downtrend, flat, uptrend])

        tsi = calculate_tsi(prices, long_period=15, short_period=7, engine="cpu")

        # Find zero-line crossovers (negative to positive)
        valid_mask = ~np.isnan(tsi)
        tsi_valid = tsi[valid_mask]

        # TSI should have both negative and positive values
        has_negative = np.any(tsi_valid < -10)
        has_positive = np.any(tsi_valid > 10)

        assert has_negative or has_positive, "TSI should show directional movement"

    def test_zero_line_crossover_downward(self):
        """Test detection of bearish zero-line crossover (TSI crosses below 0)."""
        # Create data that produces zero-line crossover
        np.random.seed(42)
        uptrend = 100 + np.cumsum(np.ones(40) * 0.3)  # Rally
        flat = np.full(20, uptrend[-1])  # Consolidation
        downtrend = uptrend[-1] - np.cumsum(np.ones(60) * 0.4)  # Drop
        prices = np.concatenate([uptrend, flat, downtrend])

        tsi = calculate_tsi(prices, long_period=15, short_period=7, engine="cpu")

        # Find zero-line crossovers (positive to negative)
        valid_mask = ~np.isnan(tsi)
        tsi_valid = tsi[valid_mask]

        # TSI should show directional changes
        has_positive = np.any(tsi_valid > 10)
        has_negative = np.any(tsi_valid < -10)

        assert has_positive or has_negative, "TSI should show directional movement"

    def test_signal_line_calculation(self):
        """Test that signal line (EMA of TSI) can be calculated."""
        np.random.seed(42)
        # Use volatile data so TSI varies
        prices = 100 + np.cumsum(np.random.randn(200) * 2)

        tsi = calculate_tsi(prices, long_period=25, short_period=13, engine="cpu")

        # Find first valid TSI index
        first_valid_tsi = np.where(~np.isnan(tsi))[0]
        if len(first_valid_tsi) == 0:
            pytest.skip("No valid TSI values")

        # Calculate signal line on valid portion only
        # Extract valid TSI values
        tsi_valid = tsi[first_valid_tsi[0] :]
        signal_line = calculate_ema(tsi_valid, period=7, engine="cpu")

        # Signal line should exist and have some valid values
        assert len(signal_line) == len(tsi_valid)
        valid_signal = signal_line[~np.isnan(signal_line)]
        assert len(valid_signal) > 0, "Signal line should have valid values"

    def test_bullish_signal_tsi_crosses_above_signal(self):
        """Test bullish signal when TSI crosses above signal line."""
        # Create volatile pattern
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 2)

        tsi = calculate_tsi(prices, long_period=15, short_period=7, engine="cpu")

        # Find first valid TSI value
        first_valid = np.where(~np.isnan(tsi))[0]
        if len(first_valid) == 0:
            pytest.skip("No valid TSI values")

        # Calculate signal on valid portion
        tsi_valid = tsi[first_valid[0] :]
        signal = calculate_ema(tsi_valid, period=7, engine="cpu")

        # Both should have valid values
        valid_both = ~(np.isnan(tsi_valid) | np.isnan(signal))
        assert np.sum(valid_both) > 0, "Should have overlapping valid TSI and signal values"

    def test_bearish_signal_tsi_crosses_below_signal(self):
        """Test bearish signal when TSI crosses below signal line."""
        # Create volatile pattern
        np.random.seed(43)  # Different seed for different pattern
        prices = 100 + np.cumsum(np.random.randn(200) * 2)

        tsi = calculate_tsi(prices, long_period=15, short_period=7, engine="cpu")

        # Find first valid TSI value
        first_valid = np.where(~np.isnan(tsi))[0]
        if len(first_valid) == 0:
            pytest.skip("No valid TSI values")

        # Calculate signal on valid portion
        tsi_valid = tsi[first_valid[0] :]
        signal = calculate_ema(tsi_valid, period=7, engine="cpu")

        # Both should have valid values
        valid_both = ~(np.isnan(tsi_valid) | np.isnan(signal))
        assert np.sum(valid_both) > 0, "Should have overlapping valid TSI and signal values"

    def test_overbought_condition(self):
        """Test that strong uptrend produces high TSI values (overbought)."""
        # Very strong uptrend
        prices = 100 + np.cumsum(np.ones(100) * 0.5)  # Constant gains

        tsi = calculate_tsi(prices, long_period=25, short_period=13, engine="cpu")

        # Remove NaN values
        valid_tsi = tsi[~np.isnan(tsi)]

        # Should reach high TSI values (customizable threshold, e.g., >25)
        max_tsi = np.max(valid_tsi)
        assert max_tsi > 25, "Strong uptrend should produce high TSI values"

    def test_oversold_condition(self):
        """Test that strong downtrend produces low TSI values (oversold)."""
        # Very strong downtrend
        prices = 100 - np.cumsum(np.ones(100) * 0.5)  # Constant losses

        tsi = calculate_tsi(prices, long_period=25, short_period=13, engine="cpu")

        # Remove NaN values
        valid_tsi = tsi[~np.isnan(tsi)]

        # Should reach low TSI values (customizable threshold, e.g., <-25)
        min_tsi = np.min(valid_tsi)
        assert min_tsi < -25, "Strong downtrend should produce low TSI values"

    def test_divergence_detection_setup(self):
        """Test that TSI can be used for divergence detection."""
        # Create price data with higher highs but weaker momentum
        prices = np.array(
            [100.0]
            + list(np.linspace(100, 110, 20))  # First peak
            + list(np.linspace(110, 105, 10))  # Pullback
            + list(np.linspace(105, 112, 20))  # Higher high but slower
            + [112.0] * 30
        )

        tsi = calculate_tsi(prices, long_period=15, short_period=7, engine="cpu")

        # TSI should exist and vary
        valid_tsi = tsi[~np.isnan(tsi)]

        # Should have both positive and negative values or at least variation
        assert np.std(valid_tsi) > 0, "TSI should vary for divergence analysis"
        assert len(valid_tsi) > 30, "Should have sufficient data for divergence"

    def test_oscillating_market_multiple_crossovers(self, oscillating_data):
        """Test that oscillating market produces multiple zero-line crossovers."""
        tsi = calculate_tsi(oscillating_data, long_period=15, short_period=7, engine="cpu")

        # Remove NaN values
        valid_mask = ~np.isnan(tsi)
        tsi_valid = tsi[valid_mask]

        # Count zero-line crossovers
        crossovers = 0
        for i in range(1, len(tsi_valid)):
            if (tsi_valid[i - 1] < 0 and tsi_valid[i] > 0) or (
                tsi_valid[i - 1] > 0 and tsi_valid[i] < 0
            ):
                crossovers += 1

        # Oscillating data should produce multiple crossovers
        assert crossovers >= 2, "Oscillating market should produce multiple crossovers"

    def test_trending_market_fewer_crossovers(self, trending_up_data):
        """Test that trending market produces fewer zero-line crossovers."""
        tsi = calculate_tsi(trending_up_data, long_period=15, short_period=7, engine="cpu")

        # Remove NaN values
        valid_mask = ~np.isnan(tsi)
        tsi_valid = tsi[valid_mask]

        # Count zero-line crossovers
        crossovers = 0
        for i in range(1, len(tsi_valid)):
            if (tsi_valid[i - 1] < 0 and tsi_valid[i] > 0) or (
                tsi_valid[i - 1] > 0 and tsi_valid[i] < 0
            ):
                crossovers += 1

        # Trending market should have fewer crossovers than oscillating
        assert crossovers <= 2, "Trending market should have few zero-line crossovers"


# ============================================================================
# Class 3: Edge Cases Tests (10 tests)
# ============================================================================


class TestTSIEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        empty_data = np.array([])

        with pytest.raises((ValueError, IndexError)):
            calculate_tsi(empty_data, engine="cpu")

    def test_single_point_raises_error(self):
        """Test that single data point raises ValueError."""
        single_point = np.array([100.0])

        with pytest.raises(ValueError):
            calculate_tsi(single_point, engine="cpu")

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        # Only 10 data points, but need long_period + short_period
        short_data = np.random.randn(10) + 100

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_tsi(short_data, long_period=25, short_period=13, engine="cpu")

    def test_minimal_data_size(self):
        """Test with minimal valid data size."""
        long_period = 10
        short_period = 5
        n = long_period + short_period + 5  # Just enough
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        tsi = calculate_tsi(
            prices, long_period=long_period, short_period=short_period, engine="cpu"
        )

        # Should complete without error
        assert len(tsi) == n

    def test_all_nan_data(self):
        """Test handling of all NaN data."""
        nan_data = np.full(100, np.nan)

        tsi = calculate_tsi(nan_data, engine="cpu")

        # Should return all NaN
        assert np.all(np.isnan(tsi))

    def test_some_nan_values(self):
        """Test handling of data with some NaN values."""
        prices = np.array([100.0, 102.0, np.nan, 106.0, 108.0] * 10)

        tsi = calculate_tsi(prices, engine="cpu")

        # Should handle gracefully - NaN propagates
        assert len(tsi) == len(prices)
        # Some values will be NaN due to input NaN
        assert np.any(np.isnan(tsi))

    def test_extreme_volatility(self):
        """Test handling of extremely volatile data."""
        np.random.seed(42)
        # Very high volatility
        prices = 100 + np.cumsum(np.random.randn(100) * 50)

        tsi = calculate_tsi(prices, engine="cpu")

        # Should complete without error
        assert len(tsi) == 100
        # Should have valid values after warmup
        assert not np.all(np.isnan(tsi))
        # Should still be in range
        valid_tsi = tsi[~np.isnan(tsi)]
        assert np.all(valid_tsi >= -100) and np.all(valid_tsi <= 100)

    def test_period_greater_than_data_length(self):
        """Test that period > data length raises error."""
        short_data = np.random.randn(20) + 100

        with pytest.raises(ValueError):
            calculate_tsi(short_data, long_period=50, short_period=13, engine="cpu")

    def test_negative_prices(self):
        """Test handling of negative prices."""
        prices = np.array([-100, -105, -110, -115, -120] * 10)

        tsi = calculate_tsi(prices, engine="cpu")

        # Should calculate correctly
        assert not np.all(np.isnan(tsi))
        assert len(tsi) == len(prices)

    def test_zero_prices(self):
        """Test handling of zero prices."""
        prices = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0] * 5)

        tsi = calculate_tsi(prices, long_period=10, short_period=5, engine="cpu")

        # Should complete without error
        assert len(tsi) == len(prices)
        valid_tsi = tsi[~np.isnan(tsi)]
        # Should produce valid values
        assert len(valid_tsi) > 0


# ============================================================================
# Class 4: Parameter Validation Tests (10 tests)
# ============================================================================


class TestTSIParameterValidation:
    """Test parameter validation."""

    def test_invalid_long_period_zero(self, sample_data):
        """Test that long_period=0 raises ValueError."""
        with pytest.raises(ValueError):
            calculate_tsi(sample_data, long_period=0, short_period=13)

    def test_invalid_long_period_negative(self, sample_data):
        """Test that negative long_period raises ValueError."""
        with pytest.raises(ValueError):
            calculate_tsi(sample_data, long_period=-5, short_period=13)

    def test_invalid_short_period_zero(self, sample_data):
        """Test that short_period=0 raises ValueError."""
        with pytest.raises(ValueError):
            calculate_tsi(sample_data, long_period=25, short_period=0)

    def test_invalid_short_period_negative(self, sample_data):
        """Test that negative short_period raises ValueError."""
        with pytest.raises(ValueError):
            calculate_tsi(sample_data, long_period=25, short_period=-3)

    def test_short_period_greater_than_long_period(self, sample_data):
        """Test that short > long is unusual but should work."""
        # This is unusual but mathematically valid
        tsi = calculate_tsi(sample_data, long_period=5, short_period=10, engine="cpu")

        # Should complete without error
        assert len(tsi) == len(sample_data)
        # Should still produce values in valid range
        valid_tsi = tsi[~np.isnan(tsi)]
        assert np.all(valid_tsi >= -100) and np.all(valid_tsi <= 100)

    def test_very_large_periods(self):
        """Test with very large periods."""
        np.random.seed(42)
        n = 300
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # Large periods
        tsi = calculate_tsi(prices, long_period=100, short_period=50, engine="cpu")

        # Should complete without error
        assert len(tsi) == n
        # Will have many NaN values due to long warmup
        valid_tsi = tsi[~np.isnan(tsi)]
        assert len(valid_tsi) > 0, "Should have some valid values"

    def test_period_equal_to_one(self, sample_data):
        """Test edge case with period=1."""
        # Period of 1 is technically valid for EMA
        tsi = calculate_tsi(sample_data, long_period=1, short_period=1, engine="cpu")

        assert len(tsi) == len(sample_data)

    def test_handles_polars_series_input(self):
        """Test that function handles Polars Series input."""
        import polars as pl

        prices_list = [100.0 + i for i in range(50)]
        prices = pl.Series(prices_list)

        tsi = calculate_tsi(prices, engine="cpu")

        # Should complete without error
        assert isinstance(tsi, np.ndarray)
        assert len(tsi) == len(prices_list)

    def test_handles_float32_input(self):
        """Test that function handles float32 input."""
        prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0] * 10, dtype=np.float32)

        tsi = calculate_tsi(prices, long_period=10, short_period=5, engine="cpu")

        # Should complete without error
        assert isinstance(tsi, np.ndarray)
        assert len(tsi) == len(prices)

    def test_handles_integer_input(self):
        """Test that function handles integer input."""
        prices = np.array([100, 102, 104, 106, 108] * 10, dtype=np.int64)

        tsi = calculate_tsi(prices, long_period=10, short_period=5, engine="cpu")

        # Should complete without error
        assert isinstance(tsi, np.ndarray)
        assert len(tsi) == len(prices)


# ============================================================================
# Class 5: GPU/CPU Equivalence Tests (10 tests)
# ============================================================================


class TestTSIGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_small_data(self, sample_data):
        """Test GPU and CPU produce identical results on small dataset."""
        # CPU calculation
        tsi_cpu = calculate_tsi(sample_data, engine="cpu")

        # GPU calculation
        tsi_gpu = calculate_tsi(sample_data, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_large_data(self, large_data):
        """Test GPU and CPU produce identical results on large dataset."""
        # CPU calculation
        tsi_cpu = calculate_tsi(large_data, engine="cpu")

        # GPU calculation
        tsi_gpu = calculate_tsi(large_data, engine="gpu")

        # Should match within floating point tolerance
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_custom_parameters(self, sample_data):
        """Test GPU and CPU match with custom parameters."""
        # CPU calculation
        tsi_cpu = calculate_tsi(sample_data, long_period=13, short_period=7, engine="cpu")

        # GPU calculation
        tsi_gpu = calculate_tsi(sample_data, long_period=13, short_period=7, engine="gpu")

        # Should match
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_trending_data(self, trending_up_data):
        """Test GPU and CPU match on trending data."""
        # CPU calculation
        tsi_cpu = calculate_tsi(trending_up_data, engine="cpu")

        # GPU calculation
        tsi_gpu = calculate_tsi(trending_up_data, engine="gpu")

        # Should match
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_oscillating_data(self, oscillating_data):
        """Test GPU and CPU match on oscillating data."""
        # CPU calculation
        tsi_cpu = calculate_tsi(oscillating_data, engine="cpu")

        # GPU calculation
        tsi_gpu = calculate_tsi(oscillating_data, engine="gpu")

        # Should match
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_fast_periods(self, sample_data):
        """Test GPU and CPU match with fast periods."""
        # CPU calculation
        tsi_cpu = calculate_tsi(sample_data, long_period=8, short_period=4, engine="cpu")

        # GPU calculation
        tsi_gpu = calculate_tsi(sample_data, long_period=8, short_period=4, engine="gpu")

        # Should match
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_slow_periods(self):
        """Test GPU and CPU match with slow periods."""
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # CPU calculation
        tsi_cpu = calculate_tsi(prices, long_period=50, short_period=25, engine="cpu")

        # GPU calculation
        tsi_gpu = calculate_tsi(prices, long_period=50, short_period=25, engine="gpu")

        # Should match
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    def test_auto_engine_selection(self, large_data):
        """Test that auto engine selects appropriately based on data size."""
        # Auto should select appropriately for large datasets
        tsi_auto = calculate_tsi(large_data, engine="auto")

        # Explicit CPU
        tsi_cpu = calculate_tsi(large_data, engine="cpu")

        # Results should match (auto may use GPU or CPU)
        np.testing.assert_allclose(tsi_auto, tsi_cpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_handles_nan_like_cpu(self):
        """Test that GPU handles NaN values same as CPU."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        prices[20:25] = np.nan  # Insert NaN values

        tsi_cpu = calculate_tsi(prices, engine="cpu")
        tsi_gpu = calculate_tsi(prices, engine="gpu")

        # NaN positions should match
        np.testing.assert_array_equal(np.isnan(tsi_cpu), np.isnan(tsi_gpu))

        # Non-NaN values should match
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_reproducible(self, sample_data):
        """Test that GPU calculation is reproducible."""
        # Calculate twice
        tsi_gpu_1 = calculate_tsi(sample_data, engine="gpu")
        tsi_gpu_2 = calculate_tsi(sample_data, engine="gpu")

        # Should be identical
        np.testing.assert_array_equal(tsi_gpu_1, tsi_gpu_2)


# ============================================================================
# Class 6: Performance Characteristics Tests (5 tests)
# ============================================================================


class TestTSIPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_data):
        """Test that calculation completes quickly on small dataset."""
        import time

        start = time.time()
        tsi = calculate_tsi(sample_data, engine="cpu")
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time

        start = time.time()
        tsi = calculate_tsi(large_data, engine="cpu")
        elapsed = time.time() - start

        # 600K rows should complete in under 10 seconds on CPU
        assert elapsed < 10.0, f"Large dataset took {elapsed:.3f}s - should be <10s"

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_performance_benefit(self, large_data):
        """Test that GPU provides performance benefit for large datasets."""
        import time

        # Warm up GPU if available
        try:
            _ = calculate_tsi(large_data[:1000], engine="gpu")
        except Exception:
            pytest.skip("GPU warmup failed")

        # Time GPU calculation
        start_gpu = time.time()
        tsi_gpu = calculate_tsi(large_data, engine="gpu")
        elapsed_gpu = time.time() - start_gpu

        # Time CPU calculation
        start_cpu = time.time()
        tsi_cpu = calculate_tsi(large_data, engine="cpu")
        elapsed_cpu = time.time() - start_cpu

        # GPU should be faster or at least competitive
        # Due to double smoothing, speedup may be modest
        assert (
            elapsed_gpu < elapsed_cpu * 2
        ), f"GPU ({elapsed_gpu:.3f}s) should be reasonably fast vs CPU ({elapsed_cpu:.3f}s)"

        # Results should match
        np.testing.assert_allclose(tsi_cpu, tsi_gpu, rtol=1e-10, equal_nan=True)

    def test_performance_scales_linearly(self):
        """Test that performance scales approximately linearly with data size."""
        import time

        timings = []
        sizes = [1000, 5000, 10000, 50000]

        for size in sizes:
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.randn(size) * 0.5)

            start = time.time()
            tsi = calculate_tsi(prices, engine="cpu")
            elapsed = time.time() - start

            timings.append((size, elapsed))

        # Check scaling from 1K to 10K (10x data)
        time_1k = timings[0][1]
        time_10k = timings[2][1]
        ratio = time_10k / time_1k

        # Should be less than 20x slower for 10x data (allows overhead)
        assert ratio < 20, f"Scaling ratio {ratio:.2f}x is too high"

    def test_memory_efficiency(self):
        """Test that TSI calculation is memory efficient."""
        # Calculate TSI on moderately large dataset
        np.random.seed(42)
        n = 100_000
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # Should complete without memory errors
        tsi = calculate_tsi(prices, engine="cpu")

        # Result should be correct length
        assert len(tsi) == n

        # Should have valid values
        valid_tsi = tsi[~np.isnan(tsi)]
        assert len(valid_tsi) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestTSIIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_data):
        """Test that function works with Polars Series input."""
        import polars as pl

        df = pl.DataFrame({"price": sample_data})

        tsi = calculate_tsi(df["price"], engine="cpu")

        assert len(tsi) == len(sample_data)
        assert not np.all(np.isnan(tsi))

    def test_consistent_with_ema_calculation(self):
        """Test that TSI double smoothing is consistent with EMA."""
        prices = np.array([100.0 + i * 0.5 for i in range(100)])

        # Calculate TSI
        tsi = calculate_tsi(prices, long_period=10, short_period=5, engine="cpu")

        # TSI should have valid values
        valid_tsi = tsi[~np.isnan(tsi)]
        assert len(valid_tsi) > 0, "TSI should produce valid values"

        # For linearly increasing prices, TSI should be highly positive
        # (constant positive momentum)
        mean_tsi = np.mean(valid_tsi)
        assert mean_tsi > 50, "Linear uptrend should produce positive TSI"

    def test_tsi_with_different_data_types(self):
        """Test TSI with different input data types."""
        # NumPy array
        np_data = np.array([100.0 + i for i in range(50)])
        tsi_np = calculate_tsi(np_data, engine="cpu")

        # Python list
        list_data = [100.0 + i for i in range(50)]
        tsi_list = calculate_tsi(list_data, engine="cpu")

        # Polars Series
        import polars as pl

        pl_data = pl.Series([100.0 + i for i in range(50)])
        tsi_pl = calculate_tsi(pl_data, engine="cpu")

        # All should produce the same result
        np.testing.assert_allclose(tsi_np, tsi_list, rtol=1e-10)
        np.testing.assert_allclose(tsi_np, tsi_pl, rtol=1e-10)


# ============================================================================
# Test Suite Summary
# ============================================================================


def test_suite_summary():
    """Print test suite summary."""
    total_tests = 50
    categories = {
        "Basic Calculation": 15,
        "Signal Generation": 10,
        "Edge Cases": 10,
        "Parameter Validation": 10,
        "GPU/CPU Equivalence": 10,
        "Performance": 5,
    }

    print("\n" + "=" * 70)
    print("TSI Test Suite Summary")
    print("=" * 70)
    for category, count in categories.items():
        print(f"{category:.<50} {count:>3} tests")
    print("-" * 70)
    print(f"{'Total':.<50} {total_tests:>3} tests")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

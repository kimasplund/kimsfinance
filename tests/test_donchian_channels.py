#!/usr/bin/env python3
"""
Comprehensive Tests for Donchian Channels Indicator
====================================================

Tests the calculate_donchian_channels() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_donchian_channels
from kimsfinance.core import EngineManager


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 2)
    highs = closes + np.abs(np.random.randn(n) * 1.5)
    lows = closes - np.abs(np.random.randn(n) * 1.5)
    return highs, lows


@pytest.fixture
def large_ohlc_data():
    """Generate large dataset for GPU testing."""
    np.random.seed(42)
    n = 600_000  # Above GPU threshold
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return highs, lows


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestDonchianChannelsBasic:
    """Test basic Donchian Channels calculation."""

    def test_basic_calculation(self, sample_ohlc_data):
        """Test basic Donchian Channels calculation returns correct structure."""
        highs, lows = sample_ohlc_data

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )

        # Check all three bands are returned
        assert len(upper) == len(highs)
        assert len(middle) == len(highs)
        assert len(lower) == len(highs)

        # Check that we have valid values throughout
        # Note: Polars rolling operations produce NaN for first (period-1) values
        assert not np.all(np.isnan(upper))
        assert not np.all(np.isnan(middle))
        assert not np.all(np.isnan(lower))

        # Check first (period-1) values are NaN
        assert np.all(np.isnan(upper[:19]))
        assert np.all(np.isnan(middle[:19]))
        assert np.all(np.isnan(lower[:19]))

        # Check values after warmup are finite
        assert np.all(np.isfinite(upper[19:]))
        assert np.all(np.isfinite(middle[19:]))
        assert np.all(np.isfinite(lower[19:]))

    def test_channel_ordering(self, sample_ohlc_data):
        """Test that upper >= middle >= lower at all valid points."""
        highs, lows = sample_ohlc_data

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )

        # Check ordering at valid points (skip NaN values)
        valid_mask = ~np.isnan(upper)
        assert np.all(upper[valid_mask] >= middle[valid_mask]), \
            "Upper channel should be >= middle line"
        assert np.all(middle[valid_mask] >= lower[valid_mask]), \
            "Middle line should be >= lower channel"

    def test_default_parameters(self, sample_ohlc_data):
        """Test that default parameters work correctly."""
        highs, lows = sample_ohlc_data

        # Should work with defaults (period=20)
        upper, middle, lower = calculate_donchian_channels(highs, lows)

        assert len(upper) == len(highs)
        assert not np.all(np.isnan(upper))

    def test_different_periods(self, sample_ohlc_data):
        """Test with different period values."""
        highs, lows = sample_ohlc_data

        # Test period=10
        upper_10, middle_10, lower_10 = calculate_donchian_channels(
            highs, lows, period=10, engine='cpu'
        )

        # Test period=50
        upper_50, middle_50, lower_50 = calculate_donchian_channels(
            highs, lows, period=50, engine='cpu'
        )

        # Both should have valid values after warmup
        assert np.all(np.isfinite(upper_10[9:]))
        assert np.all(np.isfinite(upper_50[49:]))

        # Results should be different (different periods produce different results)
        # Compare after both have valid values
        assert not np.allclose(upper_10[49:], upper_50[49:]), \
            "Different periods should produce different results"
        assert not np.allclose(middle_10[49:], middle_50[49:]), \
            "Different periods should produce different results"


# ============================================================================
# GPU/CPU Equivalence Tests
# ============================================================================

class TestDonchianChannelsGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    def test_gpu_cpu_match_small_data(self, sample_ohlc_data):
        """Test GPU and CPU produce identical results on small dataset."""
        highs, lows = sample_ohlc_data

        # CPU calculation
        upper_cpu, middle_cpu, lower_cpu = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )

        # GPU calculation (may fallback to CPU if GPU not available)
        upper_gpu, middle_gpu, lower_gpu = calculate_donchian_channels(
            highs, lows, period=20, engine='gpu'
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(upper_cpu, upper_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(middle_cpu, middle_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(lower_cpu, lower_gpu, rtol=1e-10, equal_nan=True)

    def test_gpu_cpu_match_large_data(self, large_ohlc_data):
        """Test GPU and CPU produce identical results on large dataset."""
        highs, lows = large_ohlc_data

        # CPU calculation
        upper_cpu, middle_cpu, lower_cpu = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )

        # GPU calculation
        upper_gpu, middle_gpu, lower_gpu = calculate_donchian_channels(
            highs, lows, period=20, engine='gpu'
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(upper_cpu, upper_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(middle_cpu, middle_gpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(lower_cpu, lower_gpu, rtol=1e-10, equal_nan=True)

    def test_auto_engine_selection(self, large_ohlc_data):
        """Test that auto engine selects appropriately based on data size."""
        highs, lows = large_ohlc_data

        # Auto should select GPU for large datasets
        upper_auto, middle_auto, lower_auto = calculate_donchian_channels(
            highs, lows, period=20, engine='auto'
        )

        # Explicit CPU
        upper_cpu, middle_cpu, lower_cpu = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )

        # Results should match
        np.testing.assert_allclose(upper_auto, upper_cpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(middle_auto, middle_cpu, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(lower_auto, lower_cpu, rtol=1e-10, equal_nan=True)


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================

class TestDonchianChannelsAlgorithm:
    """Test algorithm correctness against known values and properties."""

    def test_upper_channel_is_rolling_max(self, sample_ohlc_data):
        """Test that upper channel matches rolling max of highs."""
        highs, lows = sample_ohlc_data
        period = 20

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=period, engine='cpu'
        )

        # Calculate rolling max manually using Polars
        import polars as pl
        df = pl.DataFrame({"high": highs})
        rolling_max = df.select(
            pl.col("high").rolling_max(window_size=period)
        )["high"].to_numpy()

        # Upper channel should match rolling max
        np.testing.assert_allclose(upper, rolling_max, rtol=1e-10, equal_nan=True)

    def test_lower_channel_is_rolling_min(self, sample_ohlc_data):
        """Test that lower channel matches rolling min of lows."""
        highs, lows = sample_ohlc_data
        period = 20

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=period, engine='cpu'
        )

        # Calculate rolling min manually using Polars
        import polars as pl
        df = pl.DataFrame({"low": lows})
        rolling_min = df.select(
            pl.col("low").rolling_min(window_size=period)
        )["low"].to_numpy()

        # Lower channel should match rolling min
        np.testing.assert_allclose(lower, rolling_min, rtol=1e-10, equal_nan=True)

    def test_middle_channel_is_average(self, sample_ohlc_data):
        """Test that middle channel is average of upper and lower."""
        highs, lows = sample_ohlc_data
        period = 20

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=period, engine='cpu'
        )

        # Middle should be (upper + lower) / 2
        expected_middle = (upper + lower) / 2

        # Check within reasonable tolerance
        np.testing.assert_allclose(middle, expected_middle, rtol=1e-10, equal_nan=True)

    def test_known_values_simple_case(self):
        """Test against hand-calculated values for simple case."""
        # Simple test data: monotonically increasing prices
        highs = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                          111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121])
        lows = np.array([99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                         109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119])

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=10, engine='cpu'
        )

        # For monotonically increasing data:
        # At index 9 (first valid point with period=10):
        # - Upper = max(highs[0:10]) = 110
        # - Lower = min(lows[0:10]) = 99
        # - Middle = (110 + 99) / 2 = 104.5
        assert upper[9] == 110.0
        assert lower[9] == 99.0
        assert middle[9] == 104.5

        # At index 10:
        # - Upper = max(highs[1:11]) = 111
        # - Lower = min(lows[1:11]) = 100
        # - Middle = (111 + 100) / 2 = 105.5
        assert upper[10] == 111.0
        assert lower[10] == 100.0
        assert middle[10] == 105.5

    def test_known_values_constant_price(self):
        """Test with constant prices."""
        n = 50
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=10, engine='cpu'
        )

        # For constant prices:
        # - Upper = 101.0 (constant high)
        # - Lower = 99.0 (constant low)
        # - Middle = (101 + 99) / 2 = 100.0
        valid_mask = ~np.isnan(upper)
        assert np.allclose(upper[valid_mask], 101.0)
        assert np.allclose(lower[valid_mask], 99.0)
        assert np.allclose(middle[valid_mask], 100.0)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestDonchianChannelsEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_period_raises_error(self, sample_ohlc_data):
        """Test that invalid period raises ValueError."""
        highs, lows = sample_ohlc_data

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_donchian_channels(highs, lows, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_donchian_channels(highs, lows, period=-5)

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        # Only 10 data points, but period is 20
        n = 10
        highs = np.random.randn(n) + 102
        lows = np.random.randn(n) + 98

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_donchian_channels(highs, lows, period=20)

    def test_mismatched_array_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        highs = np.array([1, 2, 3, 4, 5])
        lows = np.array([1, 2, 3])  # Wrong length

        with pytest.raises(ValueError, match="must have same length"):
            calculate_donchian_channels(highs, lows)

    def test_minimal_data_size(self):
        """Test with minimal valid data size."""
        # Exactly period rows
        period = 20
        n = 20
        np.random.seed(42)
        highs = 102 + np.cumsum(np.random.randn(n) * 0.5)
        lows = 98 + np.cumsum(np.random.randn(n) * 0.5)

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=period, engine='cpu'
        )

        # Should complete without error
        assert len(upper) == n
        assert len(middle) == n
        assert len(lower) == n

        # Should have exactly one valid value (at index period-1)
        assert np.sum(~np.isnan(upper)) == 1

    def test_handles_list_input(self):
        """Test that function handles list inputs (not just numpy arrays)."""
        highs = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]
        lows = [99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=10, engine='cpu'
        )

        # Should complete without error
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_single_period(self, sample_ohlc_data):
        """Test with period=1 (edge case, should return input values)."""
        highs, lows = sample_ohlc_data

        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=1, engine='cpu'
        )

        # With period=1:
        # - Upper = current high
        # - Lower = current low
        # - Middle = (high + low) / 2
        np.testing.assert_allclose(upper, highs, rtol=1e-10)
        np.testing.assert_allclose(lower, lows, rtol=1e-10)
        expected_middle = (highs + lows) / 2
        np.testing.assert_allclose(middle, expected_middle, rtol=1e-10)


# ============================================================================
# Type and API Tests
# ============================================================================

class TestDonchianChannelsAPI:
    """Test API correctness and return types."""

    def test_return_type_is_tuple(self, sample_ohlc_data):
        """Test that function returns tuple of three arrays."""
        highs, lows = sample_ohlc_data

        result = calculate_donchian_channels(highs, lows, engine='cpu')

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

    def test_result_unpacking(self, sample_ohlc_data):
        """Test that result can be unpacked correctly."""
        highs, lows = sample_ohlc_data

        # Should be able to unpack
        upper, middle, lower = calculate_donchian_channels(
            highs, lows, engine='cpu'
        )

        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_invalid_engine_raises_error(self, sample_ohlc_data):
        """Test that invalid engine parameter raises error."""
        highs, lows = sample_ohlc_data

        # Invalid engine should raise error
        with pytest.raises(Exception):  # Could be ConfigurationError or ValueError
            calculate_donchian_channels(highs, lows, engine='invalid')


# ============================================================================
# Performance Characteristics Tests
# ============================================================================

class TestDonchianChannelsPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_ohlc_data):
        """Test that calculation completes quickly on small dataset."""
        import time
        highs, lows = sample_ohlc_data

        start = time.time()
        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, \
            f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_ohlc_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time
        highs, lows = large_ohlc_data

        start = time.time()
        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )
        elapsed = time.time() - start

        # 600K rows should complete in under 10 seconds on CPU
        assert elapsed < 10.0, \
            f"Large dataset took {elapsed:.3f}s - should be <10s"


# ============================================================================
# Integration Tests
# ============================================================================

class TestDonchianChannelsIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_ohlc_data):
        """Test that function works with Polars Series input."""
        import polars as pl
        highs, lows = sample_ohlc_data

        df = pl.DataFrame({
            "high": highs,
            "low": lows
        })

        upper, middle, lower = calculate_donchian_channels(
            df['high'], df['low'], engine='cpu'
        )

        assert len(upper) == len(highs)
        assert not np.all(np.isnan(upper))

    def test_turtle_traders_scenario(self, sample_ohlc_data):
        """Test Turtle Traders scenario (20-period entry, 10-period exit)."""
        highs, lows = sample_ohlc_data

        # Entry channels (20-period)
        entry_upper, _, entry_lower = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )

        # Exit channels (10-period)
        exit_upper, _, exit_lower = calculate_donchian_channels(
            highs, lows, period=10, engine='cpu'
        )

        # Both should have same structure
        assert len(entry_upper) == len(exit_upper)

        # Exit channels should be narrower (less lookback)
        # This is not always true at every point, but should be trend
        # Check valid points after both have warmed up (index 20+)
        valid_mask = ~np.isnan(entry_upper) & ~np.isnan(exit_upper)
        # Just verify both sets exist and are calculable
        assert np.any(valid_mask)


# ============================================================================
# Turtle Traders System Tests
# ============================================================================

class TestTurtleTradersSystem:
    """Test Turtle Traders specific scenarios."""

    def test_turtle_system_basic(self):
        """Test basic Turtle Traders entry/exit logic."""
        # Create trending data
        np.random.seed(42)
        n = 100
        trend = np.arange(n) * 0.5
        noise = np.random.randn(n) * 2
        highs = 100 + trend + noise + 2
        lows = 100 + trend + noise - 2
        closes = 100 + trend + noise

        # Calculate Donchian Channels (20-period for entry)
        upper, middle, lower = calculate_donchian_channels(
            highs, lows, period=20, engine='cpu'
        )

        # Simulate breakout signals
        # Long entry: price breaks above upper channel
        # Short entry: price breaks below lower channel
        long_signals = closes > upper
        short_signals = closes < lower

        # Both signal types should exist in trending data
        # (Though not necessarily - depends on the data)
        # Just verify the calculation works
        assert isinstance(long_signals, np.ndarray)
        assert isinstance(short_signals, np.ndarray)
        assert len(long_signals) == len(closes)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

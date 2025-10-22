#!/usr/bin/env python3
"""
Comprehensive Tests for Keltner Channels Indicator
===================================================

Tests the calculate_keltner_channels() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_keltner_channels, calculate_atr
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
def sample_ohlc_data():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 2)
    highs = closes + np.abs(np.random.randn(n) * 1.5)
    lows = closes - np.abs(np.random.randn(n) * 1.5)
    return highs, lows, closes


@pytest.fixture
def large_ohlc_data():
    """Generate large dataset for GPU testing."""
    np.random.seed(42)
    n = 600_000  # Above GPU threshold
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    return highs, lows, closes


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestKeltnerChannelsBasic:
    """Test basic Keltner Channels calculation."""

    def test_basic_calculation(self, sample_ohlc_data):
        """Test basic Keltner Channels calculation returns correct structure."""
        highs, lows, closes = sample_ohlc_data

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=20, multiplier=2.0, engine="cpu"
        )

        # Check all three bands are returned
        assert len(upper) == len(closes)
        assert len(middle) == len(closes)
        assert len(lower) == len(closes)

        # Check that we have valid values throughout
        # Note: Polars EMA doesn't produce NaN values - it calculates from the start
        assert not np.all(np.isnan(upper))
        assert not np.all(np.isnan(middle))
        assert not np.all(np.isnan(lower))

        # Check values are finite
        assert np.all(np.isfinite(upper))
        assert np.all(np.isfinite(middle))
        assert np.all(np.isfinite(lower))

    def test_channel_ordering(self, sample_ohlc_data):
        """Test that upper > middle > lower at all valid points."""
        highs, lows, closes = sample_ohlc_data

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=20, multiplier=2.0, engine="cpu"
        )

        # Check ordering at valid points (skip NaN values)
        valid_mask = ~np.isnan(upper)
        assert np.all(
            upper[valid_mask] >= middle[valid_mask]
        ), "Upper channel should be >= middle line"
        assert np.all(
            middle[valid_mask] >= lower[valid_mask]
        ), "Middle line should be >= lower channel"

    def test_default_parameters(self, sample_ohlc_data):
        """Test that default parameters work correctly."""
        highs, lows, closes = sample_ohlc_data

        # Should work with defaults (period=20, multiplier=2.0)
        upper, middle, lower = calculate_keltner_channels(highs, lows, closes)

        assert len(upper) == len(closes)
        assert not np.all(np.isnan(upper))

    def test_different_periods(self, sample_ohlc_data):
        """Test with different period values."""
        highs, lows, closes = sample_ohlc_data

        # Test period=10
        upper_10, middle_10, lower_10 = calculate_keltner_channels(
            highs, lows, closes, period=10, engine="cpu"
        )

        # Test period=50
        upper_50, middle_50, lower_50 = calculate_keltner_channels(
            highs, lows, closes, period=50, engine="cpu"
        )

        # Both should have all finite values (Polars EMA doesn't produce NaN)
        assert np.all(np.isfinite(upper_10))
        assert np.all(np.isfinite(upper_50))

        # Results should be different (different periods produce different smoothing)
        assert not np.allclose(
            upper_10, upper_50
        ), "Different periods should produce different results"
        assert not np.allclose(
            middle_10, middle_50
        ), "Different periods should produce different results"

    def test_different_multipliers(self, sample_ohlc_data):
        """Test with different multiplier values."""
        highs, lows, closes = sample_ohlc_data

        # Test multiplier=1.0
        upper_1, middle_1, lower_1 = calculate_keltner_channels(
            highs, lows, closes, period=20, multiplier=1.0, engine="cpu"
        )

        # Test multiplier=3.0
        upper_3, middle_3, lower_3 = calculate_keltner_channels(
            highs, lows, closes, period=20, multiplier=3.0, engine="cpu"
        )

        # Middle line should be the same (it's just EMA)
        np.testing.assert_allclose(middle_1, middle_3, rtol=1e-10)

        # Larger multiplier should produce wider bands
        valid_mask = ~np.isnan(upper_1)
        width_1 = upper_1[valid_mask] - lower_1[valid_mask]
        width_3 = upper_3[valid_mask] - lower_3[valid_mask]
        assert np.all(width_3 > width_1), "Larger multiplier should produce wider channels"


# ============================================================================
# GPU/CPU Equivalence Tests
# ============================================================================


@pytest.mark.skipif(not gpu_available(), reason="GPU not available")
class TestKeltnerChannelsGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    def test_gpu_cpu_match_small_data(self, sample_ohlc_data):
        """Test GPU and CPU produce identical results on small dataset."""
        highs, lows, closes = sample_ohlc_data

        # CPU calculation
        upper_cpu, middle_cpu, lower_cpu = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="cpu"
        )

        # GPU calculation (may fallback to CPU if GPU not available)
        upper_gpu, middle_gpu, lower_gpu = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="gpu"
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(upper_cpu, upper_gpu, rtol=1e-10)
        np.testing.assert_allclose(middle_cpu, middle_gpu, rtol=1e-10)
        np.testing.assert_allclose(lower_cpu, lower_gpu, rtol=1e-10)

    def test_gpu_cpu_match_large_data(self, large_ohlc_data):
        """Test GPU and CPU produce identical results on large dataset."""
        highs, lows, closes = large_ohlc_data

        # CPU calculation
        upper_cpu, middle_cpu, lower_cpu = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="cpu"
        )

        # GPU calculation
        upper_gpu, middle_gpu, lower_gpu = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="gpu"
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(upper_cpu, upper_gpu, rtol=1e-10)
        np.testing.assert_allclose(middle_cpu, middle_gpu, rtol=1e-10)
        np.testing.assert_allclose(lower_cpu, lower_gpu, rtol=1e-10)

    def test_auto_engine_selection(self, large_ohlc_data):
        """Test that auto engine selects appropriately based on data size."""
        highs, lows, closes = large_ohlc_data

        # Auto should select GPU for large datasets
        upper_auto, middle_auto, lower_auto = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="auto"
        )

        # Explicit CPU
        upper_cpu, middle_cpu, lower_cpu = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="cpu"
        )

        # Results should match
        np.testing.assert_allclose(upper_auto, upper_cpu, rtol=1e-10)
        np.testing.assert_allclose(middle_auto, middle_cpu, rtol=1e-10)
        np.testing.assert_allclose(lower_auto, lower_cpu, rtol=1e-10)


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================


class TestKeltnerChannelsAlgorithm:
    """Test algorithm correctness against known values and properties."""

    def test_middle_line_is_ema(self, sample_ohlc_data):
        """Test that middle line matches EMA of closes."""
        highs, lows, closes = sample_ohlc_data
        period = 20

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=period, engine="cpu"
        )

        # Calculate EMA manually using Polars
        import polars as pl

        df = pl.DataFrame({"close": closes})
        ema = df.select(pl.col("close").ewm_mean(span=period, adjust=False))["close"].to_numpy()

        # Middle line should match EMA
        np.testing.assert_allclose(middle, ema, rtol=1e-10)

    def test_channel_width_uses_atr(self, sample_ohlc_data):
        """Test that channel width is based on ATR * multiplier."""
        highs, lows, closes = sample_ohlc_data
        period = 20
        multiplier = 2.0

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=period, multiplier=multiplier, engine="cpu"
        )

        # Calculate ATR separately
        atr = calculate_atr(highs, lows, closes, period=period, engine="cpu")

        # Upper channel should be middle + (multiplier * atr)
        expected_upper = middle + (multiplier * atr)
        expected_lower = middle - (multiplier * atr)

        # Check within reasonable tolerance (accounting for EMA differences)
        valid_mask = ~np.isnan(upper) & ~np.isnan(atr)
        np.testing.assert_allclose(upper[valid_mask], expected_upper[valid_mask], rtol=1e-8)
        np.testing.assert_allclose(lower[valid_mask], expected_lower[valid_mask], rtol=1e-8)

    def test_known_values_simple_case(self):
        """Test against hand-calculated values for simple case."""
        # Simple test data: constant price with no volatility
        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=10, multiplier=2.0, engine="cpu"
        )

        # For constant prices, EMA should converge to the constant value
        # After warmup, middle should be close to 100
        assert np.abs(middle[-1] - 100.0) < 0.1, "Middle line should converge to constant price"

        # Bands should be symmetric around middle
        valid_mask = ~np.isnan(upper)
        upper_distance = upper[valid_mask] - middle[valid_mask]
        lower_distance = middle[valid_mask] - lower[valid_mask]
        np.testing.assert_allclose(upper_distance, lower_distance, rtol=1e-6)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestKeltnerChannelsEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_period_raises_error(self, sample_ohlc_data):
        """Test that invalid period raises ValueError."""
        highs, lows, closes = sample_ohlc_data

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_keltner_channels(highs, lows, closes, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_keltner_channels(highs, lows, closes, period=-5)

    def test_invalid_multiplier_raises_error(self, sample_ohlc_data):
        """Test that negative multiplier raises ValueError."""
        highs, lows, closes = sample_ohlc_data

        with pytest.raises(ValueError, match="multiplier must be >= 0"):
            calculate_keltner_channels(highs, lows, closes, multiplier=-1.0)

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        # Only 10 data points, but period is 20
        n = 10
        closes = np.random.randn(n) + 100
        highs = closes + 1
        lows = closes - 1

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_keltner_channels(highs, lows, closes, period=20)

    def test_mismatched_array_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        highs = np.array([1, 2, 3, 4, 5])
        lows = np.array([1, 2, 3])  # Wrong length
        closes = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="must have same length"):
            calculate_keltner_channels(highs, lows, closes)

    def test_minimal_data_size(self):
        """Test with minimal valid data size."""
        # Exactly period rows
        period = 20
        n = 20
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + 1
        lows = closes - 1

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=period, engine="cpu"
        )

        # Should complete without error
        assert len(upper) == n
        assert len(middle) == n
        assert len(lower) == n

    def test_zero_multiplier(self, sample_ohlc_data):
        """Test with multiplier=0 (bands collapse to middle line)."""
        highs, lows, closes = sample_ohlc_data

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=20, multiplier=0.0, engine="cpu"
        )

        # With multiplier=0, all three lines should be equal
        valid_mask = ~np.isnan(upper)
        np.testing.assert_allclose(upper[valid_mask], middle[valid_mask], rtol=1e-10)
        np.testing.assert_allclose(lower[valid_mask], middle[valid_mask], rtol=1e-10)

    def test_handles_list_input(self):
        """Test that function handles list inputs (not just numpy arrays)."""
        highs = [
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
        ]
        lows = [
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
        ]
        closes = [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
        ]

        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=10, engine="cpu"
        )

        # Should complete without error
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)


# ============================================================================
# Type and API Tests
# ============================================================================


class TestKeltnerChannelsAPI:
    """Test API correctness and return types."""

    def test_return_type_is_tuple(self, sample_ohlc_data):
        """Test that function returns tuple of three arrays."""
        highs, lows, closes = sample_ohlc_data

        result = calculate_keltner_channels(highs, lows, closes, engine="cpu")

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

    def test_result_unpacking(self, sample_ohlc_data):
        """Test that result can be unpacked correctly."""
        highs, lows, closes = sample_ohlc_data

        # Should be able to unpack
        upper, middle, lower = calculate_keltner_channels(highs, lows, closes, engine="cpu")

        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_invalid_engine_raises_error(self, sample_ohlc_data):
        """Test that invalid engine parameter raises error."""
        highs, lows, closes = sample_ohlc_data

        # Invalid engine should raise ConfigurationError (from EngineManager)
        # This depends on the actual implementation in EngineManager
        # If it raises a different error type, this test should be adjusted
        with pytest.raises(Exception):  # Could be ConfigurationError or ValueError
            calculate_keltner_channels(highs, lows, closes, engine="invalid")


# ============================================================================
# Performance Characteristics Tests
# ============================================================================


class TestKeltnerChannelsPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_ohlc_data):
        """Test that calculation completes quickly on small dataset."""
        import time

        highs, lows, closes = sample_ohlc_data

        start = time.time()
        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="cpu"
        )
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_ohlc_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time

        highs, lows, closes = large_ohlc_data

        start = time.time()
        upper, middle, lower = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="cpu"
        )
        elapsed = time.time() - start

        # 600K rows should complete in under 10 seconds on CPU
        assert elapsed < 10.0, f"Large dataset took {elapsed:.3f}s - should be <10s"


# ============================================================================
# Integration Tests
# ============================================================================


class TestKeltnerChannelsIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_ohlc_data):
        """Test that function works with Polars Series input."""
        import polars as pl

        highs, lows, closes = sample_ohlc_data

        df = pl.DataFrame({"high": highs, "low": lows, "close": closes})

        upper, middle, lower = calculate_keltner_channels(
            df["high"], df["low"], df["close"], engine="cpu"
        )

        assert len(upper) == len(closes)
        assert not np.all(np.isnan(upper))

    def test_consistent_with_bollinger_bands_behavior(self, sample_ohlc_data):
        """Test that Keltner behaves similarly to Bollinger (structure-wise)."""
        from kimsfinance.ops.indicators import calculate_bollinger_bands

        highs, lows, closes = sample_ohlc_data

        # Calculate Keltner Channels
        kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
            highs, lows, closes, period=20, engine="cpu"
        )

        # Calculate Bollinger Bands (similar channel indicator)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes, period=20, engine="cpu")

        # Both should have same structure (3 bands, same length)
        assert len(kc_upper) == len(bb_upper)
        assert len(kc_middle) == len(bb_middle)
        assert len(kc_lower) == len(bb_lower)

        # Both should return tuples of 3 arrays
        assert isinstance(kc_upper, np.ndarray)
        assert isinstance(bb_upper, np.ndarray)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

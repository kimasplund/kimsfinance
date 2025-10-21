#!/usr/bin/env python3
"""
Comprehensive Tests for WMA (Weighted Moving Average) Indicator
================================================================

Tests the calculate_wma() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_wma
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
    n = 600_000  # Above GPU threshold
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return prices


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestWMABasic:
    """Test basic WMA calculation."""

    def test_basic_calculation(self, sample_data):
        """Test basic WMA calculation returns correct structure."""
        result = calculate_wma(sample_data, 20, engine='cpu')

        # Check length matches input
        assert len(result) == len(sample_data)

        # Check that we have valid values after warmup period
        assert not np.all(np.isnan(result))

        # First (period-1) values should be NaN (warmup period)
        assert np.all(np.isnan(result[:19]))

        # After warmup, should have valid values
        assert not np.isnan(result[19])

    def test_default_parameters(self, sample_data):
        """Test that default parameters work correctly."""
        # Should work with defaults (period=20)
        result = calculate_wma(sample_data)

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))

    def test_different_periods(self, sample_data):
        """Test with different period values."""
        # Test period=10
        result_10 = calculate_wma(sample_data, 10, engine='cpu')

        # Test period=50
        result_50 = calculate_wma(sample_data, 50, engine='cpu')

        # Shorter period should have fewer NaN values at start
        assert np.sum(np.isnan(result_10)) < np.sum(np.isnan(result_50))

        # Results should be different
        valid_mask_10 = ~np.isnan(result_10)
        valid_mask_50 = ~np.isnan(result_50)
        common_valid = valid_mask_10 & valid_mask_50
        assert not np.allclose(result_10[common_valid], result_50[common_valid]), \
            "Different periods should produce different results"

    def test_wma_more_responsive_than_sma(self, sample_data):
        """Test that WMA is more responsive than SMA to recent changes."""
        from kimsfinance.ops.indicators import calculate_sma

        # Add a significant price spike at the end
        test_data = sample_data.copy()
        test_data[-1] = test_data[-2] * 1.5  # 50% increase

        wma = calculate_wma(test_data, 20, engine='cpu')
        sma = calculate_sma(test_data, 20, engine='cpu')

        # WMA should show larger change than SMA due to higher weight on recent prices
        wma_change = wma[-1] - wma[-2]
        sma_change = sma[-1] - sma[-2]

        assert abs(wma_change) > abs(sma_change), \
            "WMA should be more responsive than SMA to recent changes"


# ============================================================================
# GPU/CPU Equivalence Tests
# ============================================================================

class TestWMAGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    def test_gpu_cpu_match_small_data(self, sample_data):
        """Test GPU and CPU produce identical results on small dataset."""
        # CPU calculation
        cpu_result = calculate_wma(sample_data, 20, engine='cpu')

        # GPU calculation (may fallback to CPU if GPU not available)
        gpu_result = calculate_wma(sample_data, 20, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_gpu_cpu_match_large_data(self, large_data):
        """Test GPU and CPU produce identical results on large dataset."""
        # CPU calculation
        cpu_result = calculate_wma(large_data, 20, engine='cpu')

        # GPU calculation
        gpu_result = calculate_wma(large_data, 20, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_auto_engine_selection(self, large_data):
        """Test that auto engine selects appropriately based on data size."""
        # Auto should select GPU for large datasets
        auto_result = calculate_wma(large_data, 20, engine='auto')

        # Explicit CPU
        cpu_result = calculate_wma(large_data, 20, engine='cpu')

        # Results should match
        np.testing.assert_allclose(auto_result, cpu_result, rtol=1e-10)


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================

class TestWMAAlgorithm:
    """Test algorithm correctness against known values and properties."""

    def test_known_values_simple_case(self):
        """Test against hand-calculated values for simple case."""
        # Simple test data: [1, 2, 3, 4, 5]
        # WMA(5) = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (1+2+3+4+5)
        #        = (1 + 4 + 9 + 16 + 25) / 15
        #        = 55 / 15
        #        = 3.6666...

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_wma(data, 5, engine='cpu')

        expected = np.array([np.nan, np.nan, np.nan, np.nan, 3.666666666666667])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_known_values_longer_sequence(self):
        """Test with longer sequence and multiple WMA values."""
        # Data: [10, 20, 30, 40, 50, 60]
        # WMA(3) for position 2: (10*1 + 20*2 + 30*3) / 6 = 140/6 = 23.333...
        # WMA(3) for position 3: (20*1 + 30*2 + 40*3) / 6 = 200/6 = 33.333...
        # WMA(3) for position 4: (30*1 + 40*2 + 50*3) / 6 = 260/6 = 43.333...
        # WMA(3) for position 5: (40*1 + 50*2 + 60*3) / 6 = 320/6 = 53.333...

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result = calculate_wma(data, 3, engine='cpu')

        expected = np.array([
            np.nan,
            np.nan,
            23.333333333333332,
            33.333333333333336,
            43.333333333333336,
            53.333333333333336
        ])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_weights_linearly_increase(self):
        """Test that weights increase linearly (1, 2, 3, ..., N)."""
        # For constant prices, WMA should equal the price
        # But we can verify the weighting scheme by checking a linear sequence

        # Linear increasing data [1, 2, 3, ..., 10]
        data = np.arange(1, 11, dtype=np.float64)
        result = calculate_wma(data, 5, engine='cpu')

        # For position 4 (index 4): data = [1,2,3,4,5]
        # WMA = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15 = 55/15 = 3.666...
        expected_pos4 = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15.0

        np.testing.assert_allclose(result[4], expected_pos4, rtol=1e-10)

    def test_constant_prices_converge_to_constant(self):
        """Test that WMA of constant prices equals the constant."""
        # For constant prices, WMA should equal the constant value
        constant_value = 100.0
        n = 50
        data = np.full(n, constant_value)

        result = calculate_wma(data, 10, engine='cpu')

        # After warmup, all values should equal the constant
        valid_mask = ~np.isnan(result)
        np.testing.assert_allclose(result[valid_mask], constant_value, rtol=1e-10)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestWMAEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_period_raises_error(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_wma(sample_data, 0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_wma(sample_data, -5)

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        # Only 10 data points, but period is 20
        short_data = np.random.randn(10) + 100

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_wma(short_data, 20)

    def test_minimal_data_size(self):
        """Test with minimal valid data size."""
        # Exactly period rows
        period = 20
        n = 20
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = calculate_wma(data, period, engine='cpu')

        # Should complete without error
        assert len(result) == n
        # Should have exactly one non-NaN value (at position period-1)
        assert np.sum(~np.isnan(result)) == 1
        assert not np.isnan(result[period - 1])

    def test_handles_list_input(self):
        """Test that function handles list inputs (not just numpy arrays)."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]

        result = calculate_wma(prices, 10, engine='cpu')

        # Should complete without error
        assert isinstance(result, np.ndarray)
        assert len(result) == len(prices)

    def test_single_period(self, sample_data):
        """Test with period=1 (should equal input data)."""
        result = calculate_wma(sample_data, 1, engine='cpu')

        # With period=1, WMA should equal the input (weight is just 1)
        # WMA = Price[i] * 1 / 1 = Price[i]
        np.testing.assert_allclose(result, sample_data, rtol=1e-10)


# ============================================================================
# Type and API Tests
# ============================================================================

class TestWMAAPI:
    """Test API correctness and return types."""

    def test_return_type_is_array(self, sample_data):
        """Test that function returns numpy array."""
        result = calculate_wma(sample_data, engine='cpu')

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_invalid_engine_raises_error(self, sample_data):
        """Test that invalid engine parameter raises error."""
        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_wma(sample_data, 20, engine='invalid')


# ============================================================================
# Performance Characteristics Tests
# ============================================================================

class TestWMAPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_data):
        """Test that calculation completes quickly on small dataset."""
        import time

        start = time.time()
        result = calculate_wma(sample_data, 20, engine='cpu')
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, \
            f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time

        start = time.time()
        result = calculate_wma(large_data, 20, engine='cpu')
        elapsed = time.time() - start

        # 600K rows should complete in under 15 seconds on CPU
        assert elapsed < 15.0, \
            f"Large dataset took {elapsed:.3f}s - should be <15s"


# ============================================================================
# Integration Tests
# ============================================================================

class TestWMAIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_data):
        """Test that function works with Polars Series input."""
        import polars as pl

        df = pl.DataFrame({"price": sample_data})

        result = calculate_wma(df['price'], engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))

    def test_compare_with_sma_and_ema(self, sample_data):
        """Test that WMA behaves reasonably compared to SMA and EMA."""
        from kimsfinance.ops.indicators import calculate_sma, calculate_ema

        period = 20

        sma = calculate_sma(sample_data, period=period, engine='cpu')
        ema = calculate_ema(sample_data, period=period, engine='cpu')
        wma = calculate_wma(sample_data, period=period, engine='cpu')

        # All should have same length
        assert len(sma) == len(ema) == len(wma)

        # All should have same NaN pattern at start
        # (EMA may differ slightly, but should be similar)
        sma_nans = np.sum(np.isnan(sma))
        wma_nans = np.sum(np.isnan(wma))
        assert sma_nans == wma_nans, "SMA and WMA should have same warmup period"

        # Valid regions should all contain reasonable values
        valid_mask = ~np.isnan(sma) & ~np.isnan(ema) & ~np.isnan(wma)
        assert np.all(np.isfinite(sma[valid_mask]))
        assert np.all(np.isfinite(ema[valid_mask]))
        assert np.all(np.isfinite(wma[valid_mask]))

    def test_wma_between_sma_and_ema_responsiveness(self):
        """Test that WMA is more responsive than SMA (WMA vs EMA depends on context)."""
        from kimsfinance.ops.indicators import calculate_sma, calculate_ema

        # Create data with a trend change
        n = 100
        data = np.concatenate([
            np.full(50, 100.0),  # Flat at 100
            np.linspace(100, 120, 50)  # Linear increase to 120
        ])

        period = 20
        sma = calculate_sma(data, period=period, engine='cpu')
        ema = calculate_ema(data, period=period, engine='cpu')
        wma = calculate_wma(data, period=period, engine='cpu')

        # Look at position 70 (10 periods into the trend change)
        # WMA should be more responsive than SMA (guaranteed)
        # WMA vs EMA comparison depends on period and data pattern

        pos = 70
        if not (np.isnan(sma[pos]) or np.isnan(ema[pos]) or np.isnan(wma[pos])):
            # WMA should be closer to current price than SMA
            current_price = data[pos]

            wma_distance = abs(current_price - wma[pos])
            sma_distance = abs(current_price - sma[pos])

            # WMA more responsive than SMA (this is the key property)
            assert wma_distance < sma_distance, "WMA should be more responsive than SMA"

            # All three should produce different values
            assert not np.isclose(sma[pos], wma[pos]), "SMA and WMA should differ"
            assert not np.isclose(sma[pos], ema[pos]), "SMA and EMA should differ"


# ============================================================================
# Regression Tests
# ============================================================================

class TestWMARegression:
    """Test for regression issues and edge cases discovered in production."""

    def test_no_overflow_with_large_values(self):
        """Test that large price values don't cause overflow."""
        # Test with very large prices
        large_prices = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4])
        result = calculate_wma(large_prices, period=5, engine='cpu')

        # Should complete without overflow
        assert np.isfinite(result[-1])
        assert not np.isnan(result[-1])

    def test_handles_negative_prices(self):
        """Test that negative prices are handled correctly."""
        # Financial data shouldn't have negative prices, but test robustness
        data = np.array([-10.0, -20.0, -30.0, -40.0, -50.0])
        result = calculate_wma(data, period=5, engine='cpu')

        # Should complete and produce finite result
        assert np.isfinite(result[-1])

    def test_mixed_positive_negative_prices(self):
        """Test with mixed positive and negative prices."""
        data = np.array([-5.0, 10.0, -15.0, 20.0, -25.0, 30.0])
        result = calculate_wma(data, period=3, engine='cpu')

        # Should complete and produce finite results
        valid_mask = ~np.isnan(result)
        assert np.all(np.isfinite(result[valid_mask]))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

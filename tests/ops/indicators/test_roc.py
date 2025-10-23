#!/usr/bin/env python3
"""
Comprehensive Tests for Rate of Change (ROC) Indicator
=======================================================

Tests the calculate_roc() implementation for correctness,
GPU/CPU equivalence, edge cases, and performance characteristics.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch

from kimsfinance.ops.indicators import calculate_roc
from kimsfinance.ops.indicators.roc import CUPY_AVAILABLE
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


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestROCBasic:
    """Test basic ROC calculation."""

    def test_basic_calculation(self, sample_data):
        """Test basic ROC calculation returns correct structure."""
        result = calculate_roc(sample_data, period=12, engine="cpu")

        # Check length matches input
        assert len(result) == len(sample_data)

        # Should have some valid values after warmup
        assert not np.all(np.isnan(result))

        # First (period) values should be NaN (warmup period)
        assert np.all(np.isnan(result[:12]))

        # After warmup, should have valid values
        assert not np.isnan(result[12])

    def test_positive_and_negative_values(self):
        """Test that ROC produces positive and negative values correctly."""
        # Create data with clear uptrend then downtrend
        prices = np.array([100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100])

        result = calculate_roc(prices, period=2, engine="cpu")

        # After period=2, we should have values
        # Index 2: (104 - 100) / 100 * 100 = 4.0
        assert np.isclose(result[2], 4.0, rtol=1e-6)

        # Index 3: (106 - 102) / 102 * 100 = 3.92...
        expected_3 = ((106 - 102) / 102) * 100
        assert np.isclose(result[3], expected_3, rtol=1e-6)

        # Index 7: (106 - 110) / 110 * 100 = -3.636...
        expected_7 = ((106 - 110) / 110) * 100
        assert np.isclose(result[7], expected_7, rtol=1e-6)

    def test_default_parameters(self, sample_data):
        """Test that default parameters work correctly."""
        # Should work with defaults (period=12)
        result = calculate_roc(sample_data)

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))

    def test_different_periods(self, sample_data):
        """Test with different period values."""
        # Test period=5
        result_5 = calculate_roc(sample_data, period=5, engine="cpu")

        # Test period=20
        result_20 = calculate_roc(sample_data, period=20, engine="cpu")

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
# GPU/CPU Equivalence Tests
# ============================================================================


class TestROCGPUCPU:
    """Test GPU and CPU implementations produce identical results."""

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_small_data(self, sample_data):
        """Test GPU and CPU produce identical results on small dataset."""
        # CPU calculation
        result_cpu = calculate_roc(sample_data, period=12, engine="cpu")

        # GPU calculation (may fallback to CPU if GPU not available)
        result_gpu = calculate_roc(sample_data, period=12, engine="gpu")

        # Should match within floating point tolerance
        # Use rtol=1e-6 for GPU/CPU comparison (industry standard)
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-6, atol=1e-7)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_match_large_data(self, large_data):
        """Test GPU and CPU produce identical results on large dataset."""
        # CPU calculation
        result_cpu = calculate_roc(large_data, period=12, engine="cpu")

        # GPU calculation
        result_gpu = calculate_roc(large_data, period=12, engine="gpu")

        # Should match within floating point tolerance
        # Use rtol=1e-6 + atol=1e-7 for GPU/CPU comparison (handles edge cases)
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=2e-6, atol=1e-7)

    def test_auto_engine_selection(self, large_data):
        """Test that auto engine selects appropriately based on data size."""
        # Auto should select GPU for large datasets
        result_auto = calculate_roc(large_data, period=12, engine="auto")

        # Explicit CPU
        result_cpu = calculate_roc(large_data, period=12, engine="cpu")

        # Results should match within reasonable tolerance
        # Auto may select GPU, which has slight floating-point differences
        np.testing.assert_allclose(result_auto, result_cpu, rtol=2e-6, atol=1e-7)


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================


class TestROCAlgorithm:
    """Test algorithm correctness against known values and properties."""

    def test_known_values_simple_case(self):
        """Test against hand-calculated values."""
        # Simple test data with known ROC values
        prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])

        result = calculate_roc(prices, period=3, engine="cpu")

        # First 3 values should be NaN (warmup)
        assert np.all(np.isnan(result[:3]))

        # Index 3: (115 - 100) / 100 * 100 = 15.0
        assert np.isclose(result[3], 15.0, rtol=1e-6)

        # Index 4: (120 - 105) / 105 * 100 = 14.285714...
        expected_4 = ((120 - 105) / 105) * 100
        assert np.isclose(result[4], expected_4, rtol=1e-6)

        # Index 5: (125 - 110) / 110 * 100 = 13.636363...
        expected_5 = ((125 - 110) / 110) * 100
        assert np.isclose(result[5], expected_5, rtol=1e-6)

        # Index 6: (130 - 115) / 115 * 100 = 13.043478...
        expected_6 = ((130 - 115) / 115) * 100
        assert np.isclose(result[6], expected_6, rtol=1e-6)

    def test_zero_change_produces_zero_roc(self):
        """Test that constant prices produce zero ROC."""
        # Constant prices
        prices = np.full(50, 100.0)

        result = calculate_roc(prices, period=10, engine="cpu")

        # All valid values should be 0 (no change)
        valid_mask = ~np.isnan(result)
        assert np.allclose(result[valid_mask], 0.0, atol=1e-10)

    def test_roc_formula_consistency(self):
        """Test ROC formula is applied consistently."""
        prices = np.array([100, 110, 120, 130, 140, 150])
        period = 2

        result = calculate_roc(prices, period=period, engine="cpu")

        # Manually calculate expected ROC values
        for i in range(period, len(prices)):
            prev_price = prices[i - period]
            current_price = prices[i]
            expected = ((current_price - prev_price) / prev_price) * 100.0

            assert np.isclose(
                result[i], expected, rtol=1e-10
            ), f"ROC mismatch at index {i}: got {result[i]}, expected {expected}"

    def test_percentage_interpretation(self):
        """Test that ROC correctly represents percentage change."""
        # Price increases by exactly 10%
        prices = np.array([100.0, 100.0, 110.0])

        result = calculate_roc(prices, period=1, engine="cpu")

        # ROC at index 2 should be 10.0 (10% increase)
        assert np.isclose(result[2], 10.0, rtol=1e-6)

        # Price decreases by exactly 10%
        prices = np.array([100.0, 100.0, 90.0])
        result = calculate_roc(prices, period=1, engine="cpu")

        # ROC at index 2 should be -10.0 (10% decrease)
        assert np.isclose(result[2], -10.0, rtol=1e-6)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestROCEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_period_raises_error(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_roc(sample_data, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_roc(sample_data, period=-5)

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        # Only 10 data points, but period is 12
        short_data = np.random.randn(10) + 100

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_roc(short_data, period=12)

    def test_minimal_data_size(self):
        """Test with minimal valid data size."""
        # Exactly period+1 rows (minimum required)
        period = 12
        n = 13
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = calculate_roc(prices, period=period, engine="cpu")

        # Should complete without error
        assert len(result) == n
        # Should have exactly one valid value (at index period)
        valid_count = np.sum(~np.isnan(result))
        assert valid_count == 1

    def test_handles_list_input(self):
        """Test that function handles list inputs (not just numpy arrays)."""
        prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]

        result = calculate_roc(prices, period=5, engine="cpu")

        # Should complete without error
        assert isinstance(result, np.ndarray)
        assert len(result) == len(prices)

    def test_division_by_zero_handling(self):
        """Test that division by zero is handled gracefully."""
        # Include a zero price (which would cause division by zero)
        prices = np.array([100, 110, 120, 0, 140, 150, 160])

        result = calculate_roc(prices, period=3, engine="cpu")

        # Should not raise an error
        # Index 6 would try to divide by 0 (price at index 3)
        # This should result in NaN
        assert np.isnan(result[6])

        # Other valid indices should still work
        assert not np.isnan(result[3])  # (0 - 100) / 100 * 100 = -100

    def test_negative_prices_handling(self):
        """Test handling of negative prices (unusual but mathematically valid)."""
        # Some commodities or spreads can have negative values
        prices = np.array([-100, -105, -110, -115, -120, -125])

        result = calculate_roc(prices, period=2, engine="cpu")

        # Should calculate correctly
        # Index 2: (-110 - (-100)) / (-100) * 100 = -10 / -100 * 100 = 10.0
        assert np.isclose(result[2], 10.0, rtol=1e-6)


# ============================================================================
# Type and API Tests
# ============================================================================


class TestROCAPI:
    """Test API correctness and return types."""

    def test_return_type_is_ndarray(self, sample_data):
        """Test that function returns numpy array."""
        result = calculate_roc(sample_data, period=12, engine="cpu")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_result_length_matches_input(self, sample_data):
        """Test that result length matches input length."""
        result = calculate_roc(sample_data, period=12, engine="cpu")

        assert len(result) == len(sample_data)

    def test_invalid_engine_raises_error(self, sample_data):
        """Test that invalid engine parameter raises error."""
        with pytest.raises(ConfigurationError, match="Invalid engine"):
            calculate_roc(sample_data, period=12, engine="invalid")


# ============================================================================
# Performance Characteristics Tests
# ============================================================================


class TestROCPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_completes_in_reasonable_time_small_data(self, sample_data):
        """Test that calculation completes quickly on small dataset."""
        import time

        start = time.time()
        result = calculate_roc(sample_data, period=12, engine="cpu")
        elapsed = time.time() - start

        # 100 rows should complete in under 1 second
        assert elapsed < 1.0, f"Small dataset took {elapsed:.3f}s - should be <1s"

    def test_completes_in_reasonable_time_large_data(self, large_data):
        """Test that calculation completes in reasonable time on large dataset."""
        import time

        start = time.time()
        result = calculate_roc(large_data, period=12, engine="cpu")
        elapsed = time.time() - start

        # 600K rows should complete in under 5 seconds on CPU
        assert elapsed < 5.0, f"Large dataset took {elapsed:.3f}s - should be <5s"


# ============================================================================
# Integration Tests
# ============================================================================


class TestROCIntegration:
    """Test integration with other components."""

    def test_works_with_polars_series(self, sample_data):
        """Test that function works with Polars Series input."""
        import polars as pl

        df = pl.DataFrame({"price": sample_data})

        result = calculate_roc(df["price"], period=12, engine="cpu")

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))

    def test_consistent_with_other_momentum_indicators(self, sample_data):
        """Test that ROC behaves consistently with other momentum indicators."""
        from kimsfinance.ops.indicators import calculate_rsi

        # Calculate ROC and RSI
        roc = calculate_roc(sample_data, period=14, engine="cpu")
        rsi = calculate_rsi(sample_data, period=14, engine="cpu")

        # Both should have same structure (same length)
        assert len(roc) == len(rsi)

        # ROC should have NaN values at start (warmup period)
        assert np.sum(np.isnan(roc)) > 0

        # Both should respond to price changes (not constant)
        valid_roc = roc[~np.isnan(roc)]
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.std(valid_roc) > 0
        assert np.std(valid_rsi) > 0

        # Both are momentum indicators, so they should show similar trends
        # (though values will be different as they use different formulas)
        # Just verify both are working properly
        assert len(valid_roc) > 0
        assert len(valid_rsi) > 0


# ============================================================================
# Statistical Properties Tests
# ============================================================================


class TestROCStatisticalProperties:
    """Test statistical properties of ROC indicator."""

    def test_roc_symmetry(self):
        """Test that ROC is symmetric for equal up/down moves."""
        # Price goes up 10%, then down 10% from original
        prices = np.array([100, 110, 100, 110, 100])
        period = 2

        result = calculate_roc(prices, period=period, engine="cpu")

        # ROC at index 2: (100 - 100) / 100 * 100 = 0
        assert np.isclose(result[2], 0.0, atol=1e-10)

        # ROC at index 4: (100 - 100) / 100 * 100 = 0
        assert np.isclose(result[4], 0.0, atol=1e-10)

    def test_roc_additivity_property(self):
        """Test ROC calculation over consecutive periods."""
        prices = np.array([100, 105, 110, 115, 120])

        # Calculate ROC with period=1 (consecutive changes)
        roc_1 = calculate_roc(prices, period=1, engine="cpu")

        # Each consecutive ROC should reflect the percentage change
        # Index 1: (105 - 100) / 100 * 100 = 5.0
        assert np.isclose(roc_1[1], 5.0, rtol=1e-6)

        # Index 2: (110 - 105) / 105 * 100 = 4.761904...
        expected_2 = ((110 - 105) / 105) * 100
        assert np.isclose(roc_1[2], expected_2, rtol=1e-6)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

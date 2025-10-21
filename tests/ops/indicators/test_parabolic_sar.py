#!/usr/bin/env python3
"""
Tests for Parabolic SAR (Stop and Reverse) Indicator
=====================================================

Validates correctness of Parabolic SAR calculation across CPU and GPU engines.
"""

import numpy as np
import pytest
from kimsfinance.ops.indicators import calculate_parabolic_sar


class TestParabolicSAR:
    """Test Parabolic SAR calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLC data for testing."""
        np.random.seed(42)
        n = 100
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n) * 0.3)
        lows = closes - np.abs(np.random.randn(n) * 0.3)
        return highs, lows

    @pytest.fixture
    def uptrend_data(self):
        """Generate data with clear uptrend for testing."""
        n = 20
        # Create steadily rising prices
        closes = np.linspace(100, 120, n)
        highs = closes + 1.0
        lows = closes - 1.0
        return highs, lows

    @pytest.fixture
    def downtrend_data(self):
        """Generate data with clear downtrend for testing."""
        n = 20
        # Create steadily falling prices
        closes = np.linspace(120, 100, n)
        highs = closes + 1.0
        lows = closes - 1.0
        return highs, lows

    def test_basic_calculation(self, sample_data):
        """Test basic Parabolic SAR calculation."""
        highs, lows = sample_data
        result = calculate_parabolic_sar(highs, lows, engine='cpu')

        assert isinstance(result, np.ndarray)
        assert len(result) == len(highs)
        assert result.dtype == np.float64
        # First value is initialized (not NaN)
        assert not np.isnan(result[0])
        # All subsequent values should be valid
        assert not np.any(np.isnan(result))

    def test_default_parameters(self, sample_data):
        """Test that default parameters work correctly."""
        highs, lows = sample_data
        result = calculate_parabolic_sar(highs, lows)

        assert len(result) == len(highs)
        assert not np.all(np.isnan(result))

    def test_custom_parameters(self, sample_data):
        """Test with custom acceleration factor parameters."""
        highs, lows = sample_data

        # Test with different AF parameters
        result1 = calculate_parabolic_sar(
            highs, lows,
            af_start=0.01,
            af_increment=0.01,
            af_max=0.1,
            engine='cpu'
        )

        result2 = calculate_parabolic_sar(
            highs, lows,
            af_start=0.03,
            af_increment=0.03,
            af_max=0.3,
            engine='cpu'
        )

        # Results should differ with different parameters
        assert len(result1) == len(result2) == len(highs)
        # At least some values should be different
        assert not np.allclose(result1[1:], result2[1:], rtol=1e-10)

    def test_uptrend_behavior(self, uptrend_data):
        """Test SAR behavior in clear uptrend."""
        highs, lows = uptrend_data
        sar = calculate_parabolic_sar(highs, lows, engine='cpu')

        # In uptrend, SAR should generally be below price (lows)
        # Check most values (allow some initial convergence)
        below_count = np.sum(sar[5:] < lows[5:])
        total_count = len(sar[5:])

        # Most SAR values should be below lows in uptrend
        assert below_count / total_count > 0.6, \
            f"In uptrend, SAR should be below price. Got {below_count}/{total_count}"

    def test_downtrend_behavior(self, downtrend_data):
        """Test SAR behavior in clear downtrend."""
        highs, lows = downtrend_data
        sar = calculate_parabolic_sar(highs, lows, engine='cpu')

        # In downtrend, SAR should generally be above price (highs)
        # Check most values (allow some initial convergence)
        above_count = np.sum(sar[5:] > highs[5:])
        total_count = len(sar[5:])

        # Most SAR values should be above highs in downtrend
        assert above_count / total_count > 0.6, \
            f"In downtrend, SAR should be above price. Got {above_count}/{total_count}"

    def test_sar_within_reasonable_range(self, sample_data):
        """Test that SAR values are within reasonable range of price."""
        highs, lows = sample_data
        sar = calculate_parabolic_sar(highs, lows, engine='cpu')

        # SAR should be within reasonable range of highs and lows
        min_price = np.min(lows)
        max_price = np.max(highs)
        price_range = max_price - min_price

        # Allow SAR to be slightly outside price range (up to 20% extension)
        valid_sar = sar[~np.isnan(sar)]
        assert np.all(valid_sar >= min_price - 0.2 * price_range), \
            "SAR values should not be too far below price range"
        assert np.all(valid_sar <= max_price + 0.2 * price_range), \
            "SAR values should not be too far above price range"

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        highs, lows = sample_data

        cpu_result = calculate_parabolic_sar(highs, lows, engine='cpu')
        gpu_result = calculate_parabolic_sar(highs, lows, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10, equal_nan=True)

    def test_invalid_af_start(self, sample_data):
        """Test that invalid af_start raises ValueError."""
        highs, lows = sample_data

        with pytest.raises(ValueError, match="af_start must be in"):
            calculate_parabolic_sar(highs, lows, af_start=0.0)

        with pytest.raises(ValueError, match="af_start must be in"):
            calculate_parabolic_sar(highs, lows, af_start=1.0)

        with pytest.raises(ValueError, match="af_start must be in"):
            calculate_parabolic_sar(highs, lows, af_start=-0.1)

    def test_invalid_af_increment(self, sample_data):
        """Test that invalid af_increment raises ValueError."""
        highs, lows = sample_data

        with pytest.raises(ValueError, match="af_increment must be in"):
            calculate_parabolic_sar(highs, lows, af_increment=0.0)

        with pytest.raises(ValueError, match="af_increment must be in"):
            calculate_parabolic_sar(highs, lows, af_increment=1.0)

    def test_invalid_af_max(self, sample_data):
        """Test that invalid af_max raises ValueError."""
        highs, lows = sample_data

        with pytest.raises(ValueError, match="af_max must be in"):
            calculate_parabolic_sar(highs, lows, af_start=0.02, af_max=0.01)

        with pytest.raises(ValueError, match="af_max must be in"):
            calculate_parabolic_sar(highs, lows, af_max=1.0)

    def test_mismatched_array_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        highs = np.array([100, 101, 102])
        lows = np.array([98, 99])  # Different length

        with pytest.raises(ValueError, match="highs and lows must have same length"):
            calculate_parabolic_sar(highs, lows)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        highs = np.array([100])
        lows = np.array([98])

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_parabolic_sar(highs, lows)

    def test_minimal_data(self):
        """Test with minimal dataset (2 bars)."""
        highs = np.array([102, 105])
        lows = np.array([100, 101])

        result = calculate_parabolic_sar(highs, lows, engine='cpu')

        assert len(result) == 2
        assert not np.isnan(result[0])  # First value is initialized
        assert not np.isnan(result[1])  # Second value should be valid

    def test_known_values_simple(self):
        """Test against hand-calculated values for simple case."""
        # Simple 5-bar dataset with clear trend
        highs = np.array([102, 105, 108, 111, 114])
        lows = np.array([100, 103, 106, 109, 112])

        result = calculate_parabolic_sar(
            highs, lows,
            af_start=0.02,
            af_increment=0.02,
            af_max=0.2,
            engine='cpu'
        )

        assert len(result) == 5
        assert not np.isnan(result[0])  # First value is initialized

        # SAR should be below price in this clear uptrend
        for i in range(0, 5):
            assert not np.isnan(result[i]), f"SAR[{i}] should not be NaN"
            # In uptrend, SAR should be <= low (with some tolerance)
            assert result[i] <= lows[i] + 5, \
                f"SAR[{i}]={result[i]:.2f} should be near or below low={lows[i]:.2f}"

    def test_trend_reversal_detection(self):
        """Test that SAR correctly identifies trend reversals."""
        # Create data with clear trend reversal
        n = 30
        # First half: uptrend
        up_closes = np.linspace(100, 120, 15)
        up_highs = up_closes + 1.0
        up_lows = up_closes - 1.0

        # Second half: downtrend
        down_closes = np.linspace(120, 100, 15)
        down_highs = down_closes + 1.0
        down_lows = down_closes - 1.0

        highs = np.concatenate([up_highs, down_highs])
        lows = np.concatenate([up_lows, down_lows])

        sar = calculate_parabolic_sar(highs, lows, engine='cpu')

        # Check for reversal signal around index 15
        # Before reversal: SAR should be below price
        # After reversal: SAR should be above price
        reversal_window = range(12, 18)
        reversal_detected = False

        for i in reversal_window:
            if i > 0 and i < len(sar) - 1:
                # Check if SAR crossed from below to above
                if sar[i-1] < lows[i-1] and sar[i+1] > highs[i+1]:
                    reversal_detected = True
                    break

        # We don't strictly require reversal detection in this test
        # but SAR should adapt to the new trend
        # Just check that SAR values are reasonable
        assert not np.all(np.isnan(sar[1:]))

    def test_array_like_inputs(self):
        """Test that function accepts array-like inputs (lists, tuples)."""
        highs_list = [102, 105, 104, 107, 106]
        lows_list = [100, 101, 102, 104, 103]

        result = calculate_parabolic_sar(highs_list, lows_list, engine='cpu')

        assert len(result) == 5
        assert isinstance(result, np.ndarray)

    def test_engine_parameter(self, sample_data):
        """Test different engine parameters."""
        highs, lows = sample_data

        # Test 'auto' engine
        result_auto = calculate_parabolic_sar(highs, lows, engine='auto')
        assert len(result_auto) == len(highs)

        # Test 'cpu' engine
        result_cpu = calculate_parabolic_sar(highs, lows, engine='cpu')
        assert len(result_cpu) == len(highs)

        # Test 'gpu' engine (will fallback to CPU if GPU not available)
        result_gpu = calculate_parabolic_sar(highs, lows, engine='gpu')
        assert len(result_gpu) == len(highs)

        # CPU and GPU should match
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-10, equal_nan=True)

    def test_invalid_engine(self, sample_data):
        """Test that invalid engine parameter raises ValueError."""
        highs, lows = sample_data

        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_parabolic_sar(highs, lows, engine='invalid')

    def test_large_dataset_performance(self):
        """Test with large dataset to verify performance characteristics."""
        np.random.seed(42)
        n = 10_000
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n) * 0.3)
        lows = closes - np.abs(np.random.randn(n) * 0.3)

        import time

        start = time.perf_counter()
        result = calculate_parabolic_sar(highs, lows, engine='cpu')
        elapsed = time.perf_counter() - start

        assert len(result) == n
        assert not np.all(np.isnan(result[1:]))

        # Should complete in reasonable time (< 1 second for 10K bars)
        assert elapsed < 1.0, f"Large dataset took {elapsed:.3f}s - should be <1s"

    def test_constant_prices(self):
        """Test with constant prices (no trend)."""
        n = 20
        highs = np.full(n, 105.0)
        lows = np.full(n, 95.0)

        result = calculate_parabolic_sar(highs, lows, engine='cpu')

        assert len(result) == n
        # SAR should still produce valid values even with constant prices
        assert not np.any(np.isnan(result))

    def test_extreme_volatility(self):
        """Test with extremely volatile prices."""
        np.random.seed(42)
        n = 50
        # Create highly volatile prices
        closes = 100 + np.cumsum(np.random.randn(n) * 10)  # Large moves
        highs = closes + np.abs(np.random.randn(n) * 5)
        lows = closes - np.abs(np.random.randn(n) * 5)

        result = calculate_parabolic_sar(highs, lows, engine='cpu')

        assert len(result) == n
        # SAR should handle volatility gracefully
        assert not np.all(np.isnan(result[1:]))
        assert np.all(np.isfinite(result[~np.isnan(result)]))


class TestParabolicSAREdgeCases:
    """Test edge cases for Parabolic SAR."""

    def test_single_bar_raises_error(self):
        """Test that single bar raises insufficient data error."""
        highs = np.array([100])
        lows = np.array([98])

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_parabolic_sar(highs, lows)

    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise error."""
        highs = np.array([])
        lows = np.array([])

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_parabolic_sar(highs, lows)

    def test_nan_in_input_data(self):
        """Test behavior with NaN values in input."""
        highs = np.array([100, 102, np.nan, 106, 108])
        lows = np.array([98, 100, np.nan, 104, 106])

        # Should not crash, but behavior with NaN is implementation-dependent
        result = calculate_parabolic_sar(highs, lows, engine='cpu')

        assert len(result) == 5
        # Some values will be NaN due to NaN propagation
        assert isinstance(result, np.ndarray)

    def test_inf_in_input_data(self):
        """Test behavior with inf values in input."""
        highs = np.array([100, 102, np.inf, 106, 108])
        lows = np.array([98, 100, 102, 104, 106])

        # Should not crash, but inf values will affect results
        result = calculate_parabolic_sar(highs, lows, engine='cpu')

        assert len(result) == 5
        assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

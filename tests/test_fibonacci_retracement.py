"""Tests for Fibonacci Retracement indicator."""

import numpy as np
import pytest
from kimsfinance.ops import calculate_fibonacci_retracement


class TestFibonacciRetracement:
    """Test Fibonacci Retracement calculation."""

    def test_basic_calculation(self):
        """Test basic Fibonacci retracement calculation."""
        high = 150.0
        low = 100.0

        result = calculate_fibonacci_retracement(high, low)

        # Verify result is a dictionary
        assert isinstance(result, dict)

        # Verify all expected keys are present
        expected_keys = {'0.0%', '23.6%', '38.2%', '50.0%', '61.8%', '100.0%'}
        assert set(result.keys()) == expected_keys

        # Verify all values are floats
        for value in result.values():
            assert isinstance(value, float)

    def test_known_values(self):
        """Test against hand-calculated Fibonacci levels."""
        high = 150.0
        low = 100.0

        result = calculate_fibonacci_retracement(high, low)

        # Hand-calculated expected values
        # Range = 150 - 100 = 50
        # Formula: level = high - (range * ratio)
        expected = {
            '0.0%': 150.0,      # 150 - (50 * 0.0)
            '23.6%': 138.2,     # 150 - (50 * 0.236)
            '38.2%': 130.9,     # 150 - (50 * 0.382)
            '50.0%': 125.0,     # 150 - (50 * 0.500)
            '61.8%': 119.1,     # 150 - (50 * 0.618)
            '100.0%': 100.0     # 150 - (50 * 1.0)
        }

        # Check each level with tolerance for floating point precision
        for key in expected:
            assert np.isclose(result[key], expected[key], atol=1e-10), \
                f"Level {key}: expected {expected[key]}, got {result[key]}"

    def test_known_values_precise(self):
        """Test with more precise values to verify exact calculation."""
        high = 200.0
        low = 150.0

        result = calculate_fibonacci_retracement(high, low)

        # Range = 50
        expected = {
            '0.0%': 200.0,
            '23.6%': 200.0 - (50.0 * 0.236),  # 188.2
            '38.2%': 200.0 - (50.0 * 0.382),  # 180.9
            '50.0%': 200.0 - (50.0 * 0.500),  # 175.0
            '61.8%': 200.0 - (50.0 * 0.618),  # 169.1
            '100.0%': 150.0
        }

        for key in expected:
            assert np.isclose(result[key], expected[key], rtol=1e-10), \
                f"Level {key}: expected {expected[key]}, got {result[key]}"

    def test_small_range(self):
        """Test with small price range."""
        high = 100.5
        low = 100.0

        result = calculate_fibonacci_retracement(high, low)

        # Range = 0.5
        expected = {
            '0.0%': 100.5,
            '23.6%': 100.5 - (0.5 * 0.236),
            '38.2%': 100.5 - (0.5 * 0.382),
            '50.0%': 100.5 - (0.5 * 0.500),
            '61.8%': 100.5 - (0.5 * 0.618),
            '100.0%': 100.0
        }

        for key in expected:
            assert np.isclose(result[key], expected[key], rtol=1e-10)

    def test_large_range(self):
        """Test with large price range."""
        high = 10000.0
        low = 5000.0

        result = calculate_fibonacci_retracement(high, low)

        # Range = 5000
        expected = {
            '0.0%': 10000.0,
            '23.6%': 10000.0 - (5000.0 * 0.236),
            '38.2%': 10000.0 - (5000.0 * 0.382),
            '50.0%': 10000.0 - (5000.0 * 0.500),
            '61.8%': 10000.0 - (5000.0 * 0.618),
            '100.0%': 5000.0
        }

        for key in expected:
            assert np.isclose(result[key], expected[key], rtol=1e-10)

    def test_invalid_range_equal(self):
        """Test that equal high and low raises ValueError."""
        with pytest.raises(ValueError, match="high must be > low"):
            calculate_fibonacci_retracement(100.0, 100.0)

    def test_invalid_range_inverted(self):
        """Test that inverted range (high < low) raises ValueError."""
        with pytest.raises(ValueError, match="high must be > low"):
            calculate_fibonacci_retracement(100.0, 150.0)

    def test_engine_parameter_accepted(self):
        """Test that engine parameter is accepted (for consistency with other indicators)."""
        high = 150.0
        low = 100.0

        # Should not raise any errors with engine parameter
        result_auto = calculate_fibonacci_retracement(high, low, engine='auto')
        result_cpu = calculate_fibonacci_retracement(high, low, engine='cpu')
        result_gpu = calculate_fibonacci_retracement(high, low, engine='gpu')

        # All results should be identical (engine parameter is ignored)
        for key in result_auto:
            assert result_auto[key] == result_cpu[key] == result_gpu[key]

    def test_level_ordering(self):
        """Test that levels are properly ordered from high to low."""
        high = 150.0
        low = 100.0

        result = calculate_fibonacci_retracement(high, low)

        # Verify ordering: 0% should be at high, 100% at low
        assert result['0.0%'] == high
        assert result['100.0%'] == low

        # Verify intermediate levels are between high and low
        assert low < result['61.8%'] < high
        assert low < result['50.0%'] < high
        assert low < result['38.2%'] < high
        assert low < result['23.6%'] < high

        # Verify descending order
        assert result['0.0%'] > result['23.6%'] > result['38.2%'] > \
               result['50.0%'] > result['61.8%'] > result['100.0%']

    def test_negative_prices(self):
        """Test with negative prices (theoretical edge case)."""
        high = -50.0
        low = -100.0

        result = calculate_fibonacci_retracement(high, low)

        # Range = 50
        expected = {
            '0.0%': -50.0,
            '23.6%': -50.0 - (50.0 * 0.236),
            '38.2%': -50.0 - (50.0 * 0.382),
            '50.0%': -50.0 - (50.0 * 0.500),
            '61.8%': -50.0 - (50.0 * 0.618),
            '100.0%': -100.0
        }

        for key in expected:
            assert np.isclose(result[key], expected[key], rtol=1e-10)

    def test_floating_point_precision(self):
        """Test that floating point operations maintain precision."""
        high = 123.456789
        low = 98.765432

        result = calculate_fibonacci_retracement(high, low)

        # Verify no precision loss in boundary values
        assert result['0.0%'] == high
        assert result['100.0%'] == low

        # Verify 50% level is exactly the midpoint
        expected_midpoint = (high + low) / 2
        assert np.isclose(result['50.0%'], expected_midpoint, rtol=1e-10)

    def test_very_small_range(self):
        """Test with very small price range (sub-penny)."""
        high = 100.001
        low = 100.0

        result = calculate_fibonacci_retracement(high, low)

        # Should still calculate correctly despite tiny range
        assert result['0.0%'] == high
        assert result['100.0%'] == low
        assert low < result['50.0%'] < high

    def test_realistic_trading_scenario(self):
        """Test with realistic trading scenario values."""
        # Example: Stock moves from $50 to $80, then retraces
        high = 80.0
        low = 50.0

        result = calculate_fibonacci_retracement(high, low)

        # Common trading interpretation:
        # - Strong support at 61.8% level
        # - Watch for reversal at 50% level
        # - Shallow pullback if holds 23.6%

        assert np.isclose(result['61.8%'], 61.46, atol=0.01)  # Strong support
        assert np.isclose(result['50.0%'], 65.00, atol=0.01)  # Mid-point
        assert np.isclose(result['38.2%'], 68.54, atol=0.01)  # Minor support
        assert np.isclose(result['23.6%'], 72.92, atol=0.01)  # Shallow retracement

    def test_cryptocurrency_range(self):
        """Test with cryptocurrency-style high volatility range."""
        high = 69000.0  # Bitcoin ATH example
        low = 15000.0   # Bear market low

        result = calculate_fibonacci_retracement(high, low)

        # Verify calculation works with large ranges
        assert result['0.0%'] == 69000.0
        assert result['100.0%'] == 15000.0

        # Golden ratio level should be around 35,628
        expected_618 = 69000.0 - (54000.0 * 0.618)
        assert np.isclose(result['61.8%'], expected_618, rtol=1e-10)

    def test_return_type(self):
        """Test that return type is exactly dict[str, float]."""
        result = calculate_fibonacci_retracement(150.0, 100.0)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, float)

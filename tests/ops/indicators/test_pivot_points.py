"""Tests for Pivot Points indicator."""

import numpy as np
import pytest
from kimsfinance.ops.indicators import calculate_pivot_points


class TestPivotPoints:
    """Test Pivot Points calculation."""

    def test_basic_calculation(self):
        """Test basic pivot points calculation with known values."""
        # Test case from standard pivot points formula
        high = 110.0
        low = 100.0
        close = 105.0

        result = calculate_pivot_points(high, low, close)

        # Verify all keys are present
        expected_keys = {"PP", "R1", "R2", "R3", "S1", "S2", "S3"}
        assert set(result.keys()) == expected_keys

        # Manual calculation verification
        pp = (110 + 100 + 105) / 3  # 105.0
        assert result["PP"] == pytest.approx(105.0)

        # Resistance levels
        r1 = 2 * pp - low  # 2*105 - 100 = 110
        assert result["R1"] == pytest.approx(110.0)

        r2 = pp + (high - low)  # 105 + 10 = 115
        assert result["R2"] == pytest.approx(115.0)

        r3 = high + 2 * (pp - low)  # 110 + 2*(105-100) = 120
        assert result["R3"] == pytest.approx(120.0)

        # Support levels
        s1 = 2 * pp - high  # 2*105 - 110 = 100
        assert result["S1"] == pytest.approx(100.0)

        s2 = pp - (high - low)  # 105 - 10 = 95
        assert result["S2"] == pytest.approx(95.0)

        s3 = low - 2 * (high - pp)  # 100 - 2*(110-105) = 90
        assert result["S3"] == pytest.approx(90.0)

    def test_symmetry_relationships(self):
        """Test that resistance and support levels maintain proper relationships."""
        high = 150.0
        low = 130.0
        close = 140.0

        result = calculate_pivot_points(high, low, close)

        # PP should be between S1 and R1
        assert result["S1"] < result["PP"] < result["R1"]

        # Support levels should be ordered: S3 < S2 < S1 < PP
        assert result["S3"] < result["S2"] < result["S1"] < result["PP"]

        # Resistance levels should be ordered: PP < R1 < R2 < R3
        assert result["PP"] < result["R1"] < result["R2"] < result["R3"]

    def test_with_close_equals_high(self):
        """Test pivot points when close equals high (bullish close)."""
        high = 120.0
        low = 110.0
        close = 120.0

        result = calculate_pivot_points(high, low, close)

        # Should still produce valid levels
        pp = (120 + 110 + 120) / 3  # 116.666...
        assert result["PP"] == pytest.approx(116.666667, abs=1e-5)

        # All resistance levels should be above PP
        assert result["R1"] > result["PP"]
        assert result["R2"] > result["PP"]
        assert result["R3"] > result["PP"]

    def test_with_close_equals_low(self):
        """Test pivot points when close equals low (bearish close)."""
        high = 120.0
        low = 110.0
        close = 110.0

        result = calculate_pivot_points(high, low, close)

        # Should still produce valid levels
        pp = (120 + 110 + 110) / 3  # 113.333...
        assert result["PP"] == pytest.approx(113.333333, abs=1e-5)

        # All support levels should be below PP
        assert result["S1"] < result["PP"]
        assert result["S2"] < result["PP"]
        assert result["S3"] < result["PP"]

    def test_with_very_small_range(self):
        """Test pivot points with very small price range."""
        high = 100.01
        low = 100.00
        close = 100.005

        result = calculate_pivot_points(high, low, close)

        # Should produce valid values even with tiny range
        assert all(np.isfinite(v) for v in result.values())

        # PP should be approximately the average
        expected_pp = (100.01 + 100.00 + 100.005) / 3
        assert result["PP"] == pytest.approx(expected_pp)

    def test_with_large_numbers(self):
        """Test pivot points with large price values."""
        high = 50000.0
        low = 45000.0
        close = 48000.0

        result = calculate_pivot_points(high, low, close)

        # Should handle large numbers correctly
        pp = (50000 + 45000 + 48000) / 3
        assert result["PP"] == pytest.approx(pp)

        # Verify all values are finite
        assert all(np.isfinite(v) for v in result.values())

    def test_invalid_high_less_than_low(self):
        """Test that high < low raises ValueError."""
        with pytest.raises(ValueError, match="high .* must be >= low"):
            calculate_pivot_points(high=100.0, low=110.0, close=105.0)

    def test_invalid_nan_input(self):
        """Test that NaN inputs raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            calculate_pivot_points(high=np.nan, low=100.0, close=105.0)

        with pytest.raises(ValueError, match="must be finite"):
            calculate_pivot_points(high=110.0, low=np.nan, close=105.0)

        with pytest.raises(ValueError, match="must be finite"):
            calculate_pivot_points(high=110.0, low=100.0, close=np.nan)

    def test_invalid_inf_input(self):
        """Test that infinite inputs raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            calculate_pivot_points(high=np.inf, low=100.0, close=105.0)

        with pytest.raises(ValueError, match="must be finite"):
            calculate_pivot_points(high=110.0, low=-np.inf, close=105.0)

    def test_engine_parameter_accepted(self):
        """Test that engine parameter is accepted (for consistency)."""
        # Should work with all engine types
        result_auto = calculate_pivot_points(110.0, 100.0, 105.0, engine="auto")
        result_cpu = calculate_pivot_points(110.0, 100.0, 105.0, engine="cpu")
        result_gpu = calculate_pivot_points(110.0, 100.0, 105.0, engine="gpu")

        # All should produce identical results (scalar calculation uses CPU)
        assert result_auto == result_cpu == result_gpu

    def test_real_world_example_1(self):
        """Test with realistic stock market data (SPY-like)."""
        # Example: Previous day's data
        high = 452.30
        low = 448.75
        close = 451.20

        result = calculate_pivot_points(high, low, close)

        # Verify pivot point
        expected_pp = (452.30 + 448.75 + 451.20) / 3
        assert result["PP"] == pytest.approx(expected_pp)

        # Verify proper ordering and reasonable values
        assert result["S3"] < result["S2"] < result["S1"] < result["PP"]
        assert result["PP"] < result["R1"] < result["R2"] < result["R3"]

    def test_real_world_example_2(self):
        """Test with crypto market data (BTC-like)."""
        # Example: Previous day's Bitcoin data
        high = 43500.50
        low = 41200.00
        close = 42800.75

        result = calculate_pivot_points(high, low, close)

        # Verify pivot point
        expected_pp = (43500.50 + 41200.00 + 42800.75) / 3
        assert result["PP"] == pytest.approx(expected_pp, abs=0.01)

        # Verify all levels are within reasonable range
        assert 38000 < result["S3"] < 45000
        assert 38000 < result["R3"] < 50000

    def test_formula_verification(self):
        """Verify the exact formulas are implemented correctly."""
        H = 115.0
        L = 105.0
        C = 110.0

        result = calculate_pivot_points(H, L, C)

        # PP = (H + L + C) / 3
        assert result["PP"] == pytest.approx((H + L + C) / 3)

        PP = result["PP"]

        # R1 = 2*PP - L
        assert result["R1"] == pytest.approx(2 * PP - L)

        # R2 = PP + (H - L)
        assert result["R2"] == pytest.approx(PP + (H - L))

        # R3 = H + 2*(PP - L)
        assert result["R3"] == pytest.approx(H + 2 * (PP - L))

        # S1 = 2*PP - H
        assert result["S1"] == pytest.approx(2 * PP - H)

        # S2 = PP - (H - L)
        assert result["S2"] == pytest.approx(PP - (H - L))

        # S3 = L - 2*(H - PP)
        assert result["S3"] == pytest.approx(L - 2 * (H - PP))

    def test_equal_high_low_close(self):
        """Test when high, low, and close are all equal (doji-like)."""
        high = 100.0
        low = 100.0
        close = 100.0

        result = calculate_pivot_points(high, low, close)

        # PP should equal the price
        assert result["PP"] == pytest.approx(100.0)

        # When range is 0, all levels collapse to PP or are equidistant
        assert result["R1"] == pytest.approx(100.0)
        assert result["S1"] == pytest.approx(100.0)
        assert result["R2"] == pytest.approx(100.0)
        assert result["S2"] == pytest.approx(100.0)
        # R3 and S3 still follow their formulas but should be symmetric around PP
        assert abs(result["R3"] - 100.0) == pytest.approx(abs(result["S3"] - 100.0))

    def test_return_type(self):
        """Test that return type is correct dictionary."""
        result = calculate_pivot_points(110.0, 100.0, 105.0)

        # Should be a dictionary
        assert isinstance(result, dict)

        # All values should be floats
        assert all(isinstance(v, float) for v in result.values())

        # Should have exactly 7 entries
        assert len(result) == 7

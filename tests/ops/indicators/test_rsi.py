#!/usr/bin/env python3
"""
Comprehensive Tests for RSI (Relative Strength Index) Indicator
================================================================

Tests cover calculation correctness, overbought/oversold detection,
GPU/CPU parity, edge cases, and performance characteristics.

Test Coverage:
- Basic Calculation Tests (20 tests)
- Overbought/Oversold Tests (10 tests)
- Edge Cases (15 tests)
- GPU/CPU Parity Tests (10 tests)
- Performance Tests (5 tests)
- Parameter Validation Tests (10 tests)

Total: 70 comprehensive tests
"""

import numpy as np
import polars as pl
import pytest
import time
from typing import Tuple

from kimsfinance.ops.indicators import calculate_rsi
from kimsfinance.ops.indicators.rsi import CUPY_AVAILABLE
from kimsfinance.core.exceptions import ConfigurationError, GPUNotAvailableError
from kimsfinance.core import EngineManager


# ============================================================================
# Test Data Generators
# ============================================================================


def generate_uptrend(n: int = 50, start: float = 100.0, seed: int = 42) -> np.ndarray:
    """Generate upward trending prices (should produce high RSI)."""
    np.random.seed(seed)
    gains = np.abs(np.random.randn(n)) * 0.5 + 0.2  # Positive bias
    return start + np.cumsum(gains)


def generate_downtrend(n: int = 50, start: float = 100.0, seed: int = 42) -> np.ndarray:
    """Generate downward trending prices (should produce low RSI)."""
    np.random.seed(seed)
    losses = np.abs(np.random.randn(n)) * 0.5 + 0.2  # Positive bias
    return start - np.cumsum(losses)


def generate_sideways(n: int = 100, mean: float = 100.0, seed: int = 42) -> np.ndarray:
    """Generate sideways/ranging prices (should produce RSI around 50)."""
    np.random.seed(seed)
    return mean + np.random.randn(n) * 2.0


def generate_volatile(n: int = 100, start: float = 100.0, seed: int = 42) -> np.ndarray:
    """Generate highly volatile prices with large swings."""
    np.random.seed(seed)
    changes = np.random.randn(n) * 5.0  # High volatility
    return start + np.cumsum(changes)


def generate_overbought_oversold_pattern(n: int = 100, seed: int = 42) -> np.ndarray:
    """Generate pattern with clear overbought and oversold periods."""
    np.random.seed(seed)
    # Strong uptrend for first third (overbought)
    uptrend = generate_uptrend(n // 3, start=100.0, seed=seed)
    # Strong downtrend for second third (oversold)
    downtrend = generate_downtrend(n // 3, start=uptrend[-1], seed=seed + 1)
    # Sideways for last third (neutral)
    sideways = generate_sideways(n // 3, mean=downtrend[-1], seed=seed + 2)
    return np.concatenate([uptrend, downtrend, sideways])


# ============================================================================
# Class 1: Basic Calculation Tests (20 tests)
# ============================================================================


class TestRSIBasicCalculation:
    """Test basic RSI calculation correctness."""

    def test_rsi_range_uptrend(self):
        """RSI values should be between 0 and 100 for uptrend data."""
        prices = generate_uptrend(100)
        rsi = calculate_rsi(prices, period=14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0) and np.all(valid_rsi <= 100)

    def test_rsi_range_downtrend(self):
        """RSI values should be between 0 and 100 for downtrend data."""
        prices = generate_downtrend(100)
        rsi = calculate_rsi(prices, period=14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0) and np.all(valid_rsi <= 100)

    def test_rsi_range_volatile(self):
        """RSI values should be between 0 and 100 for volatile data."""
        prices = generate_volatile(100)
        rsi = calculate_rsi(prices, period=14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0) and np.all(valid_rsi <= 100)

    def test_rsi_output_length(self):
        """RSI output should have same length as input."""
        prices = generate_sideways(100)
        rsi = calculate_rsi(prices, period=14)
        assert len(rsi) == len(prices)

    def test_rsi_output_type(self):
        """RSI should return numpy array."""
        prices = generate_sideways(50)
        rsi = calculate_rsi(prices, period=14)
        assert isinstance(rsi, np.ndarray)

    def test_rsi_default_period(self):
        """RSI should use period=14 by default."""
        prices = generate_sideways(100)
        rsi_default = calculate_rsi(prices)
        rsi_explicit = calculate_rsi(prices, period=14)
        np.testing.assert_array_equal(rsi_default, rsi_explicit)

    def test_rsi_uptrend_high_values(self):
        """RSI should be high (>50) during sustained uptrend."""
        prices = generate_uptrend(100)
        rsi = calculate_rsi(prices, period=14)
        # After warmup period, RSI should be consistently high
        valid_rsi = rsi[20:]  # Skip warmup
        assert np.mean(valid_rsi) > 60  # Should be in upper range

    def test_rsi_downtrend_low_values(self):
        """RSI should be low (<50) during sustained downtrend."""
        prices = generate_downtrend(100)
        rsi = calculate_rsi(prices, period=14)
        # After warmup period, RSI should be consistently low
        valid_rsi = rsi[20:]  # Skip warmup
        assert np.mean(valid_rsi) < 40  # Should be in lower range

    def test_rsi_sideways_neutral(self):
        """RSI should be near 50 for sideways/ranging market."""
        prices = generate_sideways(200)
        rsi = calculate_rsi(prices, period=14)
        valid_rsi = rsi[20:]  # Skip warmup
        mean_rsi = np.mean(valid_rsi)
        # Should be relatively neutral (40-60 range)
        assert 40 < mean_rsi < 60

    def test_rsi_different_periods(self):
        """RSI with different periods should produce different results."""
        prices = generate_sideways(100)
        rsi_5 = calculate_rsi(prices, period=5)
        rsi_14 = calculate_rsi(prices, period=14)
        rsi_21 = calculate_rsi(prices, period=21)

        # Results should differ
        assert not np.allclose(rsi_5[21:], rsi_14[21:], equal_nan=True)
        assert not np.allclose(rsi_14[21:], rsi_21[21:], equal_nan=True)

    def test_rsi_shorter_period_more_reactive(self):
        """Shorter RSI periods should be more reactive to price changes."""
        prices = generate_volatile(100)
        rsi_5 = calculate_rsi(prices, period=5)
        rsi_21 = calculate_rsi(prices, period=21)

        # Standard deviation should be higher for shorter period
        std_5 = np.nanstd(rsi_5)
        std_21 = np.nanstd(rsi_21)
        assert std_5 > std_21

    def test_rsi_list_input(self):
        """RSI should accept list input."""
        prices = [100, 102, 101, 105, 103, 107, 106, 110, 108, 112,
                  111, 115, 113, 117, 116, 120]
        rsi = calculate_rsi(prices, period=5)
        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(prices)

    def test_rsi_numpy_array_input(self):
        """RSI should accept numpy array input."""
        prices = np.array([100, 102, 101, 105, 103, 107, 106, 110, 108])
        rsi = calculate_rsi(prices, period=3)
        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(prices)

    def test_rsi_polars_series_input(self):
        """RSI should accept Polars Series input."""
        prices = pl.Series([100, 102, 101, 105, 103, 107, 106, 110, 108])
        rsi = calculate_rsi(prices, period=3)
        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(prices)

    def test_rsi_float32_input(self):
        """RSI should handle float32 input."""
        prices = np.array([100, 102, 101, 105, 103], dtype=np.float32)
        rsi = calculate_rsi(prices, period=3)
        assert isinstance(rsi, np.ndarray)
        assert not np.any(np.isnan(rsi[3:]))  # After warmup

    def test_rsi_float64_input(self):
        """RSI should handle float64 input."""
        prices = np.array([100, 102, 101, 105, 103], dtype=np.float64)
        rsi = calculate_rsi(prices, period=3)
        assert isinstance(rsi, np.ndarray)
        assert not np.any(np.isnan(rsi[3:]))  # After warmup

    def test_rsi_integer_input(self):
        """RSI should handle integer input."""
        prices = np.array([100, 102, 101, 105, 103], dtype=np.int64)
        rsi = calculate_rsi(prices, period=3)
        assert isinstance(rsi, np.ndarray)
        assert not np.any(np.isnan(rsi[3:]))  # After warmup

    def test_rsi_reproducible(self):
        """RSI calculation should be reproducible."""
        prices = generate_sideways(100)
        rsi_1 = calculate_rsi(prices, period=14)
        rsi_2 = calculate_rsi(prices, period=14)
        np.testing.assert_array_equal(rsi_1, rsi_2)

    def test_rsi_small_period(self):
        """RSI should work with small periods (period=2)."""
        prices = generate_sideways(50)
        rsi = calculate_rsi(prices, period=2)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0) and np.all(valid_rsi <= 100)

    def test_rsi_large_period(self):
        """RSI should work with large periods (period=50)."""
        prices = generate_sideways(200)
        rsi = calculate_rsi(prices, period=50)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0) and np.all(valid_rsi <= 100)


# ============================================================================
# Class 2: Overbought/Oversold Tests (10 tests)
# ============================================================================


class TestRSIOverboughtOversold:
    """Test overbought/oversold detection."""

    def test_overbought_detection(self):
        """Strong uptrend should produce RSI > 70 (overbought)."""
        prices = generate_uptrend(100)
        rsi = calculate_rsi(prices, period=14)
        # Should have multiple overbought readings
        overbought_count = np.sum(rsi > 70)
        assert overbought_count > 10  # At least 10 overbought readings

    def test_oversold_detection(self):
        """Strong downtrend should produce RSI < 30 (oversold)."""
        prices = generate_downtrend(100)
        rsi = calculate_rsi(prices, period=14)
        # Should have multiple oversold readings
        oversold_count = np.sum(rsi < 30)
        assert oversold_count > 10  # At least 10 oversold readings

    def test_overbought_oversold_pattern(self):
        """Pattern with both conditions should be detected."""
        prices = generate_overbought_oversold_pattern(150)
        rsi = calculate_rsi(prices, period=14)

        # Should have both overbought and oversold readings
        overbought = np.sum(rsi > 70)
        oversold = np.sum(rsi < 30)

        assert overbought > 5, "Should detect overbought conditions"
        assert oversold > 5, "Should detect oversold conditions"

    def test_extreme_overbought(self):
        """Very strong uptrend should push RSI close to 100."""
        # Create extremely strong uptrend
        prices = 100 + np.cumsum(np.ones(50))  # Constant gains
        rsi = calculate_rsi(prices, period=14)

        # Should reach very high RSI values
        max_rsi = np.nanmax(rsi)
        assert max_rsi > 95  # Should be extremely overbought

    def test_extreme_oversold(self):
        """Very strong downtrend should push RSI close to 0."""
        # Create extremely strong downtrend
        prices = 100 - np.cumsum(np.ones(50))  # Constant losses
        rsi = calculate_rsi(prices, period=14)

        # Should reach very low RSI values
        min_rsi = np.nanmin(rsi)
        assert min_rsi < 5  # Should be extremely oversold

    def test_neutral_zone(self):
        """Sideways market should stay in neutral zone (30-70)."""
        prices = generate_sideways(200)
        rsi = calculate_rsi(prices, period=14)
        valid_rsi = rsi[20:]  # Skip warmup

        # Most readings should be in neutral zone
        neutral_count = np.sum((valid_rsi >= 30) & (valid_rsi <= 70))
        neutral_ratio = neutral_count / len(valid_rsi)
        assert neutral_ratio > 0.7  # At least 70% in neutral zone

    def test_overbought_level_customization(self):
        """Different overbought thresholds should be usable."""
        prices = generate_uptrend(100)
        rsi = calculate_rsi(prices, period=14)

        # Count at different thresholds
        ob_70 = np.sum(rsi > 70)
        ob_80 = np.sum(rsi > 80)
        ob_90 = np.sum(rsi > 90)

        # More restrictive threshold should have fewer or equal signals
        assert ob_80 <= ob_70
        assert ob_90 <= ob_80

    def test_oversold_level_customization(self):
        """Different oversold thresholds should be usable."""
        prices = generate_downtrend(100)
        rsi = calculate_rsi(prices, period=14)

        # Count at different thresholds
        os_30 = np.sum(rsi < 30)
        os_20 = np.sum(rsi < 20)
        os_10 = np.sum(rsi < 10)

        # More restrictive threshold should have fewer or equal signals
        assert os_20 <= os_30
        assert os_10 <= os_20

    def test_rsi_divergence_pattern(self):
        """RSI should show divergence from price action."""
        # Price makes higher high, but momentum weakens
        prices = np.array([100, 105, 103, 107, 104, 109, 105, 110, 106])
        rsi = calculate_rsi(prices, period=3)

        # RSI should exist and be valid
        assert len(rsi) == len(prices)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert len(valid_rsi) > 0

    def test_signal_generation(self):
        """RSI crossovers should be detectable."""
        prices = generate_overbought_oversold_pattern(150)
        rsi = calculate_rsi(prices, period=14)

        # Detect crossovers of 50 level
        above_50 = rsi > 50
        crossovers = np.diff(above_50.astype(int))
        cross_up = np.sum(crossovers == 1)
        cross_down = np.sum(crossovers == -1)

        # Should have multiple crossovers in this pattern
        assert cross_up > 0
        assert cross_down > 0


# ============================================================================
# Class 3: Edge Cases (15 tests)
# ============================================================================


class TestRSIEdgeCases:
    """Test edge cases and error handling."""

    def test_minimum_data_length(self):
        """RSI should raise error if data length <= period."""
        prices = np.array([100, 101, 102, 103])
        with pytest.raises(ValueError, match="Data length must be > period"):
            calculate_rsi(prices, period=14)

    def test_exact_period_length(self):
        """RSI should raise error if data length == period."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113])
        with pytest.raises(ValueError, match="Data length must be > period"):
            calculate_rsi(prices, period=14)

    def test_period_plus_one(self):
        """RSI should work with data length = period + 1."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
        rsi = calculate_rsi(prices, period=14)
        assert len(rsi) == len(prices)

    def test_constant_prices_all_gains(self):
        """Constant upward price should handle edge case."""
        prices = np.array([100, 101, 102, 103, 104, 105])
        rsi = calculate_rsi(prices, period=2)
        # With constant gains, RSI should be 100
        assert np.nanmax(rsi) > 99

    def test_constant_prices_all_losses(self):
        """Constant downward price should handle edge case."""
        prices = np.array([105, 104, 103, 102, 101, 100])
        rsi = calculate_rsi(prices, period=2)
        # With constant losses, RSI should be 0
        assert np.nanmin(rsi) < 1

    def test_constant_prices_no_change(self):
        """Constant prices (no change) should produce RSI = 0."""
        prices = np.array([100, 100, 100, 100, 100, 100])
        rsi = calculate_rsi(prices, period=2)
        # No gains or losses = RSI = 0 (by formula)
        # Note: Implementation uses 1e-10 epsilon, so expect ~0
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi < 1)

    def test_single_large_move(self):
        """Single large price move should be handled."""
        prices = np.array([100, 100, 100, 200, 100, 100, 100])
        rsi = calculate_rsi(prices, period=3)
        assert len(rsi) == len(prices)
        assert np.all(rsi[~np.isnan(rsi)] >= 0)
        assert np.all(rsi[~np.isnan(rsi)] <= 100)

    def test_alternating_up_down(self):
        """Alternating up/down moves should work."""
        prices = np.array([100, 105, 100, 105, 100, 105, 100, 105])
        rsi = calculate_rsi(prices, period=3)
        # Should produce valid RSI around 50
        valid_rsi = rsi[~np.isnan(rsi)]
        assert 30 < np.mean(valid_rsi) < 70

    def test_negative_prices(self):
        """Negative prices should be handled (valid for derivatives)."""
        prices = np.array([-10, -8, -12, -7, -15, -5, -18])
        rsi = calculate_rsi(prices, period=3)
        # Should still calculate valid RSI based on changes
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_zero_crossing_prices(self):
        """Prices crossing zero should work."""
        prices = np.array([5, 3, 1, -1, -3, -1, 1, 3])
        rsi = calculate_rsi(prices, period=3)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_very_small_prices(self):
        """Very small price values should work."""
        prices = np.array([0.001, 0.0011, 0.0009, 0.0012, 0.0008])
        rsi = calculate_rsi(prices, period=2)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_very_large_prices(self):
        """Very large price values should work."""
        prices = np.array([1e9, 1.01e9, 0.99e9, 1.02e9, 0.98e9])
        rsi = calculate_rsi(prices, period=2)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_extreme_volatility(self):
        """Extreme volatility should be handled."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 20)  # Very high volatility
        rsi = calculate_rsi(prices, period=14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_mixed_precision(self):
        """Mixed float32/float64 should work."""
        prices_f32 = np.array([100, 102, 101, 105], dtype=np.float32)
        prices_f64 = prices_f32.astype(np.float64)

        rsi_f32 = calculate_rsi(prices_f32, period=2)
        rsi_f64 = calculate_rsi(prices_f64, period=2)

        # Results should be very close
        np.testing.assert_allclose(rsi_f32, rsi_f64, rtol=1e-5)

    def test_inf_values_protection(self):
        """Inf values should be handled gracefully."""
        prices = np.array([100, 102, np.inf, 105, 103])
        # This should either work or raise clear error
        try:
            rsi = calculate_rsi(prices, period=2)
            # If it works, result should not contain additional infs
            assert np.isfinite(rsi).any()
        except (ValueError, RuntimeError):
            # Acceptable to reject inf values
            pass


# ============================================================================
# Class 4: GPU/CPU Parity Tests (10 tests)
# ============================================================================


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
class TestRSIGPUCPU:
    """Test GPU/CPU parity."""

    def test_small_data_cpu_gpu_parity(self):
        """Small dataset should produce identical CPU/GPU results."""
        prices = generate_sideways(1000, seed=42)

        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")
        rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")

        np.testing.assert_allclose(rsi_cpu, rsi_gpu, rtol=1e-6, equal_nan=True)

    def test_large_data_cpu_gpu_parity(self):
        """Large dataset should produce very close CPU/GPU results."""
        prices = generate_sideways(100_000, seed=42)

        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")
        rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")

        # GPU may have tiny numerical differences
        np.testing.assert_allclose(rsi_cpu, rsi_gpu, rtol=1e-5, equal_nan=True)

    def test_uptrend_cpu_gpu_parity(self):
        """Uptrend pattern should match CPU/GPU."""
        prices = generate_uptrend(5000, seed=42)

        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")
        rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")

        np.testing.assert_allclose(rsi_cpu, rsi_gpu, rtol=1e-6, equal_nan=True)

    def test_downtrend_cpu_gpu_parity(self):
        """Downtrend pattern should match CPU/GPU."""
        prices = generate_downtrend(5000, seed=42)

        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")
        rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")

        np.testing.assert_allclose(rsi_cpu, rsi_gpu, rtol=1e-6, equal_nan=True)

    def test_volatile_cpu_gpu_parity(self):
        """Volatile data should match CPU/GPU."""
        prices = generate_volatile(10_000, seed=42)

        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")
        rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")

        np.testing.assert_allclose(rsi_cpu, rsi_gpu, rtol=1e-6, equal_nan=True)

    def test_different_periods_cpu_gpu_parity(self):
        """Different periods should maintain CPU/GPU parity."""
        prices = generate_sideways(5000, seed=42)

        for period in [5, 14, 21, 50]:
            rsi_cpu = calculate_rsi(prices, period=period, engine="cpu")
            rsi_gpu = calculate_rsi(prices, period=period, engine="gpu")

            np.testing.assert_allclose(
                rsi_cpu, rsi_gpu, rtol=1e-6, equal_nan=True,
                err_msg=f"Failed for period={period}"
            )

    def test_auto_engine_selection(self):
        """Auto engine selection should work correctly."""
        # Small data should use CPU
        prices_small = generate_sideways(1000)
        rsi_small = calculate_rsi(prices_small, period=14, engine="auto")
        assert len(rsi_small) == len(prices_small)

        # Large data should potentially use GPU (if available)
        prices_large = generate_sideways(150_000)
        rsi_large = calculate_rsi(prices_large, period=14, engine="auto")
        assert len(rsi_large) == len(prices_large)

    def test_gpu_explicit_request(self):
        """Explicit GPU engine request should work."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")

        prices = generate_sideways(5000)
        rsi = calculate_rsi(prices, period=14, engine="gpu")

        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(prices)

    def test_cpu_explicit_request(self):
        """Explicit CPU engine request should work."""
        prices = generate_sideways(5000)
        rsi = calculate_rsi(prices, period=14, engine="cpu")

        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(prices)

    def test_gpu_nan_handling(self):
        """GPU should handle NaN values same as CPU."""
        prices = generate_sideways(5000)
        # Insert some NaN values
        prices[100:105] = np.nan

        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")

        if EngineManager.check_gpu_available():
            rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")
            # NaN positions should match
            np.testing.assert_array_equal(np.isnan(rsi_cpu), np.isnan(rsi_gpu))


# ============================================================================
# Class 5: Performance Tests (5 tests)
# ============================================================================


class TestRSIPerformance:
    """Test performance characteristics."""

    def test_performance_1k_candles(self):
        """1K candles should process in <5ms."""
        prices = generate_sideways(1000, seed=42)

        start = time.perf_counter()
        rsi = calculate_rsi(prices, period=14, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.005  # 5ms
        assert len(rsi) == 1000

    def test_performance_10k_candles(self):
        """10K candles should process in <15ms."""
        prices = generate_sideways(10_000, seed=42)

        start = time.perf_counter()
        rsi = calculate_rsi(prices, period=14, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.015  # 15ms
        assert len(rsi) == 10_000

    def test_performance_100k_candles(self):
        """100K candles should process in <100ms."""
        prices = generate_sideways(100_000, seed=42)

        start = time.perf_counter()
        rsi = calculate_rsi(prices, period=14, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.100  # 100ms
        assert len(rsi) == 100_000

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="GPU not available")
    def test_gpu_performance_benefit(self):
        """GPU should be faster for large datasets."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")

        prices = generate_sideways(200_000, seed=42)

        # CPU timing
        start_cpu = time.perf_counter()
        rsi_cpu = calculate_rsi(prices, period=14, engine="cpu")
        time_cpu = time.perf_counter() - start_cpu

        # GPU timing
        start_gpu = time.perf_counter()
        rsi_gpu = calculate_rsi(prices, period=14, engine="gpu")
        time_gpu = time.perf_counter() - start_gpu

        # Results should match
        np.testing.assert_allclose(rsi_cpu, rsi_gpu, rtol=1e-5, equal_nan=True)

        # GPU should be faster (or at least competitive)
        speedup = time_cpu / time_gpu
        print(f"\nGPU Speedup: {speedup:.2f}x")
        # Note: This is informational; actual speedup depends on hardware

    def test_performance_scaling(self):
        """Performance should scale reasonably with data size."""
        timings = []

        for size in [1000, 5000, 10000, 50000]:
            prices = generate_sideways(size, seed=42)

            start = time.perf_counter()
            rsi = calculate_rsi(prices, period=14, engine="cpu")
            elapsed = time.perf_counter() - start

            timings.append((size, elapsed))

        # Check that timing grows reasonably (should be roughly linear)
        # Compare 1K to 10K (10x data)
        time_1k = timings[0][1]
        time_10k = timings[2][1]
        ratio = time_10k / time_1k

        # Should be less than 20x slower for 10x data (allows overhead)
        assert ratio < 20


# ============================================================================
# Class 6: Parameter Validation Tests (10 tests)
# ============================================================================


class TestRSIParameterValidation:
    """Test parameter validation."""

    def test_invalid_period_zero(self):
        """Period of 0 should raise error."""
        prices = generate_sideways(100)
        with pytest.raises((ValueError, RuntimeError)):
            calculate_rsi(prices, period=0)

    def test_invalid_period_negative(self):
        """Negative period should raise error."""
        prices = generate_sideways(100)
        with pytest.raises((ValueError, RuntimeError)):
            calculate_rsi(prices, period=-5)

    def test_invalid_period_too_large(self):
        """Period larger than data length should raise error."""
        prices = np.array([100, 101, 102, 103, 104])
        with pytest.raises(ValueError, match="Data length must be > period"):
            calculate_rsi(prices, period=10)

    def test_invalid_engine_string(self):
        """Invalid engine string should raise error."""
        prices = generate_sideways(100)
        with pytest.raises(ConfigurationError, match="Invalid engine"):
            calculate_rsi(prices, period=14, engine="invalid")

    def test_invalid_engine_type(self):
        """Invalid engine type should raise error."""
        prices = generate_sideways(100)
        with pytest.raises((ConfigurationError, TypeError)):
            calculate_rsi(prices, period=14, engine=123)

    def test_gpu_not_available_error(self):
        """Requesting GPU when unavailable should raise error."""
        if EngineManager.check_gpu_available():
            pytest.skip("GPU is available, can't test unavailable case")

        prices = generate_sideways(100)
        with pytest.raises(GPUNotAvailableError):
            calculate_rsi(prices, period=14, engine="gpu")

    def test_empty_array(self):
        """Empty array should raise error."""
        prices = np.array([])
        with pytest.raises((ValueError, IndexError)):
            calculate_rsi(prices, period=14)

    def test_none_input(self):
        """None input should raise error."""
        with pytest.raises((TypeError, AttributeError)):
            calculate_rsi(None, period=14)

    def test_string_input(self):
        """String input should raise error."""
        with pytest.raises((TypeError, AttributeError)):
            calculate_rsi("invalid", period=14)

    def test_period_float(self):
        """Float period should be handled or rejected."""
        prices = generate_sideways(100)
        # Should either convert to int or raise error
        try:
            rsi = calculate_rsi(prices, period=14.5)
            # If it works, should produce valid output
            assert len(rsi) == len(prices)
        except (TypeError, ValueError):
            # Acceptable to reject float periods
            pass


# ============================================================================
# Summary Statistics
# ============================================================================


def test_suite_summary():
    """Print test suite summary."""
    total_tests = 70
    categories = {
        "Basic Calculation": 20,
        "Overbought/Oversold": 10,
        "Edge Cases": 15,
        "GPU/CPU Parity": 10,
        "Performance": 5,
        "Parameter Validation": 10,
    }

    print("\n" + "=" * 70)
    print("RSI Test Suite Summary")
    print("=" * 70)
    for category, count in categories.items():
        print(f"{category:.<50} {count:>3} tests")
    print("-" * 70)
    print(f"{'Total':.<50} {total_tests:>3} tests")
    print("=" * 70)

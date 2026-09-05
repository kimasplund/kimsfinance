#!/usr/bin/env python3
"""
Comprehensive Tests for MFI (Money Flow Index) Indicator
=========================================================

Tests cover calculation correctness, overbought/oversold detection,
GPU/CPU parity, edge cases, and performance characteristics.

Test Coverage:
- Basic Calculation Tests (15 tests)
- Overbought/Oversold Tests (8 tests)
- Edge Cases (10 tests)
- GPU/CPU Parity Tests (8 tests)
- Performance Tests (4 tests)
- Parameter Validation Tests (8 tests)

Total: 53 comprehensive tests
"""

import numpy as np
import polars as pl
import pytest
import time
from typing import Tuple

from kimsfinance.ops.indicators import calculate_mfi
from _gpu import POLARS_GPU_AVAILABLE, requires_polars_gpu
from kimsfinance.core.exceptions import ConfigurationError, GPUNotAvailableError

# ============================================================================
# Test Data Generators
# ============================================================================


def generate_uptrend_with_volume(
    n: int = 50, start: float = 100.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate upward trending OHLC with volume (should produce high MFI)."""
    np.random.seed(seed)

    closes = start + np.cumsum(np.abs(np.random.randn(n)) * 0.5 + 0.2)
    highs = closes + np.abs(np.random.randn(n)) * 0.3
    lows = closes - np.abs(np.random.randn(n)) * 0.3
    volumes = 1000 + np.abs(np.random.randn(n)) * 200

    return highs, lows, closes, volumes


def generate_downtrend_with_volume(
    n: int = 50, start: float = 100.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate downward trending OHLC with volume (should produce low MFI)."""
    np.random.seed(seed)

    closes = start - np.cumsum(np.abs(np.random.randn(n)) * 0.5 + 0.2)
    highs = closes + np.abs(np.random.randn(n)) * 0.3
    lows = closes - np.abs(np.random.randn(n)) * 0.3
    volumes = 1000 + np.abs(np.random.randn(n)) * 200

    return highs, lows, closes, volumes


def generate_sideways_with_volume(
    n: int = 100, mean: float = 100.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sideways OHLC with volume (should produce MFI around 50)."""
    np.random.seed(seed)

    closes = mean + np.random.randn(n) * 2.0
    highs = closes + np.abs(np.random.randn(n)) * 0.5
    lows = closes - np.abs(np.random.randn(n)) * 0.5
    volumes = 1000 + np.abs(np.random.randn(n)) * 200

    return highs, lows, closes, volumes


def generate_high_volume_uptrend(
    n: int = 50, start: float = 100.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate uptrend with increasing volume (strong buying pressure)."""
    np.random.seed(seed)

    closes = start + np.cumsum(np.abs(np.random.randn(n)) * 0.5 + 0.2)
    highs = closes + np.abs(np.random.randn(n)) * 0.3
    lows = closes - np.abs(np.random.randn(n)) * 0.3
    volumes = np.linspace(1000, 2000, n) + np.abs(np.random.randn(n)) * 100

    return highs, lows, closes, volumes


# ============================================================================
# Class 1: Basic Calculation Tests (15 tests)
# ============================================================================


class TestMFIBasicCalculation:
    """Test basic MFI calculation correctness."""

    def test_mfi_range_uptrend(self):
        """MFI values should be between 0 and 100 for uptrend data."""
        high, low, close, volume = generate_uptrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        valid_mfi = mfi[~np.isnan(mfi)]
        assert np.all(valid_mfi >= 0) and np.all(valid_mfi <= 100)

    def test_mfi_range_downtrend(self):
        """MFI values should be between 0 and 100 for downtrend data."""
        high, low, close, volume = generate_downtrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        valid_mfi = mfi[~np.isnan(mfi)]
        assert np.all(valid_mfi >= 0) and np.all(valid_mfi <= 100)

    def test_mfi_output_length(self):
        """MFI output should have same length as input."""
        high, low, close, volume = generate_sideways_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        assert len(mfi) == len(close)

    def test_mfi_output_type(self):
        """MFI should return numpy array."""
        high, low, close, volume = generate_sideways_with_volume(50)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        assert isinstance(mfi, np.ndarray)

    def test_mfi_default_period(self):
        """MFI should use period=14 by default."""
        high, low, close, volume = generate_sideways_with_volume(100)
        mfi_default = calculate_mfi(high, low, close, volume)
        mfi_explicit = calculate_mfi(high, low, close, volume, period=14)
        np.testing.assert_array_equal(mfi_default, mfi_explicit)

    def test_mfi_uptrend_high_values(self):
        """MFI should be high (>50) during sustained uptrend."""
        high, low, close, volume = generate_uptrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        valid_mfi = mfi[20:]  # Skip warmup
        assert np.mean(valid_mfi) > 50

    def test_mfi_downtrend_low_values(self):
        """MFI should be low (<50) during sustained downtrend."""
        high, low, close, volume = generate_downtrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        valid_mfi = mfi[20:]  # Skip warmup
        assert np.mean(valid_mfi) < 50

    def test_mfi_sideways_neutral(self):
        """MFI should be near 50 for sideways market."""
        high, low, close, volume = generate_sideways_with_volume(200)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        valid_mfi = mfi[20:]  # Skip warmup
        mean_mfi = np.mean(valid_mfi)
        assert 35 < mean_mfi < 65

    def test_mfi_different_periods(self):
        """MFI with different periods should produce different results."""
        high, low, close, volume = generate_sideways_with_volume(100)
        mfi_5 = calculate_mfi(high, low, close, volume, period=5)
        mfi_14 = calculate_mfi(high, low, close, volume, period=14)
        mfi_21 = calculate_mfi(high, low, close, volume, period=21)

        # Results should differ
        assert not np.allclose(mfi_5[21:], mfi_14[21:], equal_nan=True)
        assert not np.allclose(mfi_14[21:], mfi_21[21:], equal_nan=True)

    def test_mfi_high_volume_impact(self):
        """High volume during uptrend should strengthen MFI signal."""
        high, low, close, volume = generate_high_volume_uptrend(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        valid_mfi = mfi[20:]
        # Strong volume with uptrend should push MFI higher
        assert np.mean(valid_mfi) > 55

    def test_mfi_numpy_array_input(self):
        """MFI should accept numpy array input."""
        high = np.array([105, 107, 106, 110, 108, 112, 111, 115])
        low = np.array([100, 102, 101, 105, 103, 107, 106, 110])
        close = np.array([103, 106, 104, 108, 106, 110, 109, 113])
        volume = np.array([1000, 1200, 900, 1500, 1100, 1300, 1000, 1400])
        mfi = calculate_mfi(high, low, close, volume, period=3)
        assert isinstance(mfi, np.ndarray)
        assert len(mfi) == len(close)

    def test_mfi_polars_series_input(self):
        """MFI should accept Polars Series input."""
        high = pl.Series([105, 107, 106, 110, 108, 112, 111, 115])
        low = pl.Series([100, 102, 101, 105, 103, 107, 106, 110])
        close = pl.Series([103, 106, 104, 108, 106, 110, 109, 113])
        volume = pl.Series([1000, 1200, 900, 1500, 1100, 1300, 1000, 1400])
        mfi = calculate_mfi(high, low, close, volume, period=3)
        assert isinstance(mfi, np.ndarray)
        assert len(mfi) == len(close)

    def test_mfi_list_input(self):
        """MFI should accept list input."""
        high = [105, 107, 106, 110, 108, 112, 111, 115]
        low = [100, 102, 101, 105, 103, 107, 106, 110]
        close = [103, 106, 104, 108, 106, 110, 109, 113]
        volume = [1000, 1200, 900, 1500, 1100, 1300, 1000, 1400]
        mfi = calculate_mfi(high, low, close, volume, period=3)
        assert isinstance(mfi, np.ndarray)
        assert len(mfi) == len(close)

    def test_mfi_reproducible(self):
        """MFI calculation should be reproducible."""
        high, low, close, volume = generate_sideways_with_volume(100)
        mfi_1 = calculate_mfi(high, low, close, volume, period=14)
        mfi_2 = calculate_mfi(high, low, close, volume, period=14)
        np.testing.assert_array_equal(mfi_1, mfi_2)

    def test_mfi_small_period(self):
        """MFI should work with small periods (period=2)."""
        high, low, close, volume = generate_sideways_with_volume(50)
        mfi = calculate_mfi(high, low, close, volume, period=2)
        valid_mfi = mfi[~np.isnan(mfi)]
        assert np.all(valid_mfi >= 0) and np.all(valid_mfi <= 100)


# ============================================================================
# Class 2: Overbought/Oversold Tests (8 tests)
# ============================================================================


class TestMFIOverboughtOversold:
    """Test overbought/oversold detection."""

    def test_overbought_detection(self):
        """Strong uptrend should produce MFI > 80 (overbought)."""
        high, low, close, volume = generate_uptrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        # Should have some overbought readings
        overbought_count = np.sum(mfi > 80)
        assert overbought_count > 0

    def test_oversold_detection(self):
        """Strong downtrend should produce MFI < 20 (oversold)."""
        high, low, close, volume = generate_downtrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        # Should have some oversold readings
        oversold_count = np.sum(mfi < 20)
        assert oversold_count > 0

    def test_extreme_overbought(self):
        """Very strong uptrend with high volume should push MFI high."""
        # Create extremely strong uptrend
        n = 50
        closes = 100 + np.cumsum(np.ones(n) * 2)
        highs = closes + 0.5
        lows = closes - 0.5
        volumes = np.ones(n) * 1000

        mfi = calculate_mfi(highs, lows, closes, volumes, period=14)
        max_mfi = np.nanmax(mfi)
        assert max_mfi > 90

    def test_extreme_oversold(self):
        """Very strong downtrend with high volume should push MFI low."""
        # Create extremely strong downtrend
        n = 50
        closes = 100 - np.cumsum(np.ones(n) * 2)
        highs = closes + 0.5
        lows = closes - 0.5
        volumes = np.ones(n) * 1000

        mfi = calculate_mfi(highs, lows, closes, volumes, period=14)
        min_mfi = np.nanmin(mfi)
        assert min_mfi < 10

    def test_neutral_zone(self):
        """Sideways market should stay in neutral zone (20-80)."""
        high, low, close, volume = generate_sideways_with_volume(200)
        mfi = calculate_mfi(high, low, close, volume, period=14)
        valid_mfi = mfi[20:]  # Skip warmup

        # Most readings should be in neutral zone
        neutral_count = np.sum((valid_mfi >= 20) & (valid_mfi <= 80))
        neutral_ratio = neutral_count / len(valid_mfi)
        assert neutral_ratio > 0.6

    def test_overbought_level_customization(self):
        """Different overbought thresholds should be usable."""
        high, low, close, volume = generate_uptrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)

        # Count at different thresholds
        ob_70 = np.sum(mfi > 70)
        ob_80 = np.sum(mfi > 80)
        ob_90 = np.sum(mfi > 90)

        # More restrictive threshold should have fewer or equal signals
        assert ob_80 <= ob_70
        assert ob_90 <= ob_80

    def test_oversold_level_customization(self):
        """Different oversold thresholds should be usable."""
        high, low, close, volume = generate_downtrend_with_volume(100)
        mfi = calculate_mfi(high, low, close, volume, period=14)

        # Count at different thresholds
        os_30 = np.sum(mfi < 30)
        os_20 = np.sum(mfi < 20)
        os_10 = np.sum(mfi < 10)

        # More restrictive threshold should have fewer or equal signals
        assert os_20 <= os_30
        assert os_10 <= os_20

    def test_signal_generation(self):
        """MFI crossovers should be detectable."""
        high, low, close, volume = generate_sideways_with_volume(150)
        mfi = calculate_mfi(high, low, close, volume, period=14)

        # Detect crossovers of 50 level
        above_50 = mfi > 50
        crossovers = np.diff(above_50.astype(int))
        cross_up = np.sum(crossovers == 1)
        cross_down = np.sum(crossovers == -1)

        # Should have some crossovers
        assert cross_up + cross_down > 0


# ============================================================================
# Class 3: Edge Cases (10 tests)
# ============================================================================


class TestMFIEdgeCases:
    """Test edge cases and error handling."""

    def test_minimum_data_length(self):
        """MFI should raise error if data length <= period."""
        high = np.array([105, 107, 106, 110])
        low = np.array([100, 102, 101, 105])
        close = np.array([103, 106, 104, 108])
        volume = np.array([1000, 1200, 900, 1500])
        with pytest.raises(ValueError, match="Data length must be > period"):
            calculate_mfi(high, low, close, volume, period=14)

    def test_mismatched_array_lengths(self):
        """MFI should raise error if arrays have different lengths."""
        high = np.array([105, 107, 106, 110, 108])
        low = np.array([100, 102, 101, 105])  # One less
        close = np.array([103, 106, 104, 108, 106])
        volume = np.array([1000, 1200, 900, 1500, 1100])
        with pytest.raises(ValueError, match="All input arrays must have the same length"):
            calculate_mfi(high, low, close, volume, period=3)

    def test_constant_prices(self):
        """Constant prices should handle edge case."""
        high = np.array([100, 100, 100, 100, 100, 100])
        low = np.array([100, 100, 100, 100, 100, 100])
        close = np.array([100, 100, 100, 100, 100, 100])
        volume = np.array([1000, 1000, 1000, 1000, 1000, 1000])
        mfi = calculate_mfi(high, low, close, volume, period=2)
        # No price changes should result in near-zero or NaN values
        valid_mfi = mfi[~np.isnan(mfi)]
        assert len(valid_mfi) >= 0  # Should not crash

    def test_zero_volume(self):
        """Zero volume should be handled."""
        high = np.array([105, 107, 106, 110, 108])
        low = np.array([100, 102, 101, 105, 103])
        close = np.array([103, 106, 104, 108, 106])
        volume = np.array([0, 0, 0, 0, 0])
        mfi = calculate_mfi(high, low, close, volume, period=2)
        # Should handle zero volume gracefully
        assert len(mfi) == len(close)

    def test_very_high_volume(self):
        """Very high volume values should work."""
        high = np.array([105, 107, 106, 110, 108])
        low = np.array([100, 102, 101, 105, 103])
        close = np.array([103, 106, 104, 108, 106])
        volume = np.array([1e9, 1.2e9, 0.9e9, 1.5e9, 1.1e9])
        mfi = calculate_mfi(high, low, close, volume, period=2)
        valid_mfi = mfi[~np.isnan(mfi)]
        assert np.all(valid_mfi >= 0) and np.all(valid_mfi <= 100)

    def test_negative_volume(self):
        """Negative volume should be handled (though unusual)."""
        high = np.array([105, 107, 106, 110, 108])
        low = np.array([100, 102, 101, 105, 103])
        close = np.array([103, 106, 104, 108, 106])
        volume = np.array([-1000, -1200, -900, -1500, -1100])
        try:
            mfi = calculate_mfi(high, low, close, volume, period=2)
            # If it works, should produce some result
            assert len(mfi) == len(close)
        except (ValueError, RuntimeError):
            # Acceptable to reject negative volume
            pass

    def test_mixed_precision(self):
        """Mixed float32/float64 should work."""
        high_f32 = np.array([105, 107, 106, 110, 108], dtype=np.float32)
        low_f32 = np.array([100, 102, 101, 105, 103], dtype=np.float32)
        close_f32 = np.array([103, 106, 104, 108, 106], dtype=np.float32)
        volume_f32 = np.array([1000, 1200, 900, 1500, 1100], dtype=np.float32)

        high_f64 = high_f32.astype(np.float64)
        low_f64 = low_f32.astype(np.float64)
        close_f64 = close_f32.astype(np.float64)
        volume_f64 = volume_f32.astype(np.float64)

        mfi_f32 = calculate_mfi(high_f32, low_f32, close_f32, volume_f32, period=2)
        mfi_f64 = calculate_mfi(high_f64, low_f64, close_f64, volume_f64, period=2)

        np.testing.assert_allclose(mfi_f32, mfi_f64, rtol=1e-5)

    def test_high_low_inversion(self):
        """High < Low should be handled or rejected."""
        high = np.array([100, 102, 101, 105, 103])  # Lower values
        low = np.array([105, 107, 106, 110, 108])  # Higher values (inverted)
        close = np.array([103, 106, 104, 108, 106])
        volume = np.array([1000, 1200, 900, 1500, 1100])

        # Should either work or raise clear error
        try:
            mfi = calculate_mfi(high, low, close, volume, period=2)
            # If it works, typical price calculation should still be valid
            assert len(mfi) == len(close)
        except (ValueError, RuntimeError):
            pass

    def test_extreme_volatility(self):
        """Extreme price volatility should be handled."""
        np.random.seed(42)
        n = 50
        closes = 100 + np.cumsum(np.random.randn(n) * 20)
        highs = closes + np.abs(np.random.randn(n)) * 10
        lows = closes - np.abs(np.random.randn(n)) * 10
        volumes = 1000 + np.abs(np.random.randn(n)) * 500

        mfi = calculate_mfi(highs, lows, closes, volumes, period=14)
        valid_mfi = mfi[~np.isnan(mfi)]
        assert np.all(valid_mfi >= 0) and np.all(valid_mfi <= 100)

    def test_single_large_volume_spike(self):
        """Single large volume spike should be handled."""
        high = np.array([105, 107, 106, 110, 108, 109, 111])
        low = np.array([100, 102, 101, 105, 103, 104, 106])
        close = np.array([103, 106, 104, 108, 106, 107, 109])
        volume = np.array([1000, 1000, 10000, 1000, 1000, 1000, 1000])  # Spike
        mfi = calculate_mfi(high, low, close, volume, period=3)
        assert len(mfi) == len(close)
        assert np.all(mfi[~np.isnan(mfi)] >= 0)
        assert np.all(mfi[~np.isnan(mfi)] <= 100)


# ============================================================================
# Class 4: GPU/CPU Parity Tests (8 tests)
# ============================================================================


@requires_polars_gpu
class TestMFIGPUCPU:
    """Test GPU/CPU parity."""

    def test_small_data_cpu_gpu_parity(self):
        """Small dataset should produce identical CPU/GPU results."""
        high, low, close, volume = generate_sideways_with_volume(1000, seed=42)

        mfi_cpu = calculate_mfi(high, low, close, volume, period=14, engine="cpu")
        mfi_gpu = calculate_mfi(high, low, close, volume, period=14, engine="gpu")

        np.testing.assert_allclose(mfi_cpu, mfi_gpu, rtol=1e-6, equal_nan=True)

    def test_large_data_cpu_gpu_parity(self):
        """Large dataset should produce very close CPU/GPU results."""
        high, low, close, volume = generate_sideways_with_volume(100_000, seed=42)

        mfi_cpu = calculate_mfi(high, low, close, volume, period=14, engine="cpu")
        mfi_gpu = calculate_mfi(high, low, close, volume, period=14, engine="gpu")

        # GPU may have tiny numerical differences
        np.testing.assert_allclose(mfi_cpu, mfi_gpu, rtol=1e-5, equal_nan=True)

    def test_uptrend_cpu_gpu_parity(self):
        """Uptrend pattern should match CPU/GPU."""
        high, low, close, volume = generate_uptrend_with_volume(5000, seed=42)

        mfi_cpu = calculate_mfi(high, low, close, volume, period=14, engine="cpu")
        mfi_gpu = calculate_mfi(high, low, close, volume, period=14, engine="gpu")

        np.testing.assert_allclose(mfi_cpu, mfi_gpu, rtol=1e-6, equal_nan=True)

    def test_downtrend_cpu_gpu_parity(self):
        """Downtrend pattern should match CPU/GPU."""
        high, low, close, volume = generate_downtrend_with_volume(5000, seed=42)

        mfi_cpu = calculate_mfi(high, low, close, volume, period=14, engine="cpu")
        mfi_gpu = calculate_mfi(high, low, close, volume, period=14, engine="gpu")

        np.testing.assert_allclose(mfi_cpu, mfi_gpu, rtol=1e-6, equal_nan=True)

    def test_different_periods_cpu_gpu_parity(self):
        """Different periods should maintain CPU/GPU parity."""
        high, low, close, volume = generate_sideways_with_volume(5000, seed=42)

        for period in [5, 14, 21]:
            mfi_cpu = calculate_mfi(high, low, close, volume, period=period, engine="cpu")
            mfi_gpu = calculate_mfi(high, low, close, volume, period=period, engine="gpu")

            np.testing.assert_allclose(
                mfi_cpu, mfi_gpu, rtol=1e-6, equal_nan=True, err_msg=f"Failed for period={period}"
            )

    def test_auto_engine_selection(self):
        """Auto engine selection should work correctly."""
        # Small data should use CPU
        high, low, close, volume = generate_sideways_with_volume(1000)
        mfi_small = calculate_mfi(high, low, close, volume, period=14, engine="auto")
        assert len(mfi_small) == len(close)

        # Large data should potentially use GPU (if available)
        high, low, close, volume = generate_sideways_with_volume(150_000)
        mfi_large = calculate_mfi(high, low, close, volume, period=14, engine="auto")
        assert len(mfi_large) == len(close)

    @requires_polars_gpu
    def test_gpu_explicit_request(self):
        """Explicit GPU engine request should work."""
        high, low, close, volume = generate_sideways_with_volume(5000)
        mfi = calculate_mfi(high, low, close, volume, period=14, engine="gpu")

        assert isinstance(mfi, np.ndarray)
        assert len(mfi) == len(close)

    def test_cpu_explicit_request(self):
        """Explicit CPU engine request should work."""
        high, low, close, volume = generate_sideways_with_volume(5000)
        mfi = calculate_mfi(high, low, close, volume, period=14, engine="cpu")

        assert isinstance(mfi, np.ndarray)
        assert len(mfi) == len(close)


# ============================================================================
# Class 5: Performance Tests (4 tests)
# ============================================================================


class TestMFIPerformance:
    """Test performance characteristics."""

    def test_performance_1k_candles(self):
        """1K candles should process in reasonable time."""
        high, low, close, volume = generate_sideways_with_volume(1000, seed=42)

        start = time.perf_counter()
        mfi = calculate_mfi(high, low, close, volume, period=14, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.050  # 50ms
        assert len(mfi) == 1000

    def test_performance_10k_candles(self):
        """10K candles should process in <20ms."""
        high, low, close, volume = generate_sideways_with_volume(10_000, seed=42)

        start = time.perf_counter()
        mfi = calculate_mfi(high, low, close, volume, period=14, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.020  # 20ms
        assert len(mfi) == 10_000

    def test_performance_100k_candles(self):
        """100K candles should process in <150ms."""
        high, low, close, volume = generate_sideways_with_volume(100_000, seed=42)

        start = time.perf_counter()
        mfi = calculate_mfi(high, low, close, volume, period=14, engine="cpu")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.150  # 150ms
        assert len(mfi) == 100_000

    def test_performance_scaling(self):
        """Performance should scale reasonably with data size."""
        timings = []

        for size in [1000, 5000, 10000, 50000]:
            high, low, close, volume = generate_sideways_with_volume(size, seed=42)

            start = time.perf_counter()
            calculate_mfi(high, low, close, volume, period=14, engine="cpu")
            elapsed = time.perf_counter() - start

            timings.append((size, elapsed))

        # Check that timing grows reasonably (should be roughly linear)
        time_1k = timings[0][1]
        time_10k = timings[2][1]
        ratio = time_10k / time_1k

        # Should be less than 20x slower for 10x data
        assert ratio < 20


# ============================================================================
# Class 6: Parameter Validation Tests (8 tests)
# ============================================================================


class TestMFIParameterValidation:
    """Test parameter validation."""

    def test_invalid_period_zero(self):
        """Period of 0 should raise error."""
        high, low, close, volume = generate_sideways_with_volume(100)
        with pytest.raises((ValueError, RuntimeError)):
            calculate_mfi(high, low, close, volume, period=0)

    def test_invalid_period_negative(self):
        """Negative period should raise error."""
        high, low, close, volume = generate_sideways_with_volume(100)
        with pytest.raises((ValueError, RuntimeError)):
            calculate_mfi(high, low, close, volume, period=-5)

    def test_invalid_period_too_large(self):
        """Period larger than data length should raise error."""
        high = np.array([105, 107, 106, 110, 108])
        low = np.array([100, 102, 101, 105, 103])
        close = np.array([103, 106, 104, 108, 106])
        volume = np.array([1000, 1200, 900, 1500, 1100])
        with pytest.raises(ValueError, match="Data length must be > period"):
            calculate_mfi(high, low, close, volume, period=10)

    def test_invalid_engine_string(self):
        """Invalid engine string should raise error."""
        high, low, close, volume = generate_sideways_with_volume(100)
        with pytest.raises(ConfigurationError, match="Invalid engine"):
            calculate_mfi(high, low, close, volume, period=14, engine="invalid")

    def test_invalid_engine_type(self):
        """Invalid engine type should raise error."""
        high, low, close, volume = generate_sideways_with_volume(100)
        with pytest.raises((ConfigurationError, TypeError)):
            calculate_mfi(high, low, close, volume, period=14, engine=123)

    def test_gpu_not_available_error(self):
        """Requesting GPU when unavailable should raise error."""
        if POLARS_GPU_AVAILABLE:
            pytest.skip("Polars GPU engine is available, can't test unavailable case")

        high, low, close, volume = generate_sideways_with_volume(100)
        with pytest.raises(GPUNotAvailableError):
            calculate_mfi(high, low, close, volume, period=14, engine="gpu")

    def test_empty_arrays(self):
        """Empty arrays should raise error."""
        high = np.array([])
        low = np.array([])
        close = np.array([])
        volume = np.array([])
        with pytest.raises((ValueError, IndexError)):
            calculate_mfi(high, low, close, volume, period=14)

    def test_none_input(self):
        """None input should raise error."""
        with pytest.raises((TypeError, AttributeError)):
            calculate_mfi(None, None, None, None, period=14)


# ============================================================================
# Summary Statistics
# ============================================================================


def test_suite_summary():
    """Print test suite summary."""
    total_tests = 53
    categories = {
        "Basic Calculation": 15,
        "Overbought/Oversold": 8,
        "Edge Cases": 10,
        "GPU/CPU Parity": 8,
        "Performance": 4,
        "Parameter Validation": 8,
    }

    print("\n" + "=" * 70)
    print("MFI Test Suite Summary")
    print("=" * 70)
    for category, count in categories.items():
        print(f"{category:.<50} {count:>3} tests")
    print("-" * 70)
    print(f"{'Total':.<50} {total_tests:>3} tests")
    print("=" * 70)

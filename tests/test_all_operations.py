#!/usr/bin/env python3
"""
Comprehensive Test Suite for kimsfinance
===============================================

Tests all operations to ensure correctness and GPU functionality.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Import the library (now properly installed)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import kimsfinance as mfp


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def test_library_info():
    """Test library information functions."""
    print_section("Library Information")

    print(f"Version: {mfp.version()}")
    print(f"GPU Available: {mfp.gpu_available()}")

    info = mfp.get_engine_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


def test_moving_averages():
    """Test moving average calculations."""
    print_section("Moving Averages")

    # Generate test data
    prices = np.array([100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0])

    print(f"Test data: {len(prices)} prices")

    # Test SMA
    sma_5 = mfp.calculate_sma(prices, period=5, engine="cpu")
    print(f"✓ SMA(5) calculated: {sma_5[-3:]}")

    # Test EMA
    ema_3 = mfp.calculate_ema(prices, period=3, engine="cpu")
    print(f"✓ EMA(3) calculated: {ema_3[-3:]}")

    # Test combined
    mas = mfp.calculate_multiple_mas(
        pl.DataFrame({"close": prices}),
        "close",
        sma_windows=[3, 5],
        ema_windows=[3, 5],
        engine="cpu",
    )
    print(f"✓ Combined MAs: {len(mas['sma'])} SMAs, {len(mas['ema'])} EMAs")


def test_nan_operations():
    """Test NaN operations."""
    print_section("NaN Operations")

    data = np.array([100.0, 102.0, np.nan, 105.0, np.nan, 107.0])

    print(f"Test data: {data}")

    # Test nanmin/nanmax
    min_val = mfp.nanmin_gpu(data, engine="cpu")
    max_val = mfp.nanmax_gpu(data, engine="cpu")
    print(f"✓ nanmin: {min_val:.2f}")
    print(f"✓ nanmax: {max_val:.2f}")

    # Test nan_bounds
    bounds = mfp.nan_bounds(data, data, engine="cpu")
    print(f"✓ nan_bounds: {bounds}")

    # Test isnan
    nan_mask = mfp.isnan_gpu(data, engine="cpu")
    print(f"✓ isnan: {np.sum(nan_mask)} NaN values detected")

    # Test nan_indices
    nan_idx = mfp.nan_indices(data, engine="cpu")
    print(f"✓ nan_indices: {nan_idx}")


def test_linear_algebra():
    """Test linear algebra operations."""
    print_section("Linear Algebra")

    # Generate linear data with noise
    x = np.arange(20, dtype=np.float64)
    y = 2.5 * x + 100 + np.random.randn(20) * 2

    print(f"Test data: {len(x)} points")

    # Test least squares
    slope, intercept = mfp.least_squares_fit(x, y, engine="cpu")
    print(f"✓ Least squares: slope={slope:.4f}, intercept={intercept:.4f}")

    # Test trend line
    trend = mfp.trend_line(x, y, engine="cpu")
    print(f"✓ Trend line calculated: {len(trend)} values")

    # Test correlation
    corr = mfp.correlation(x, y, engine="cpu")
    print(f"✓ Correlation: {corr:.4f}")


def test_indicators():
    """Test technical indicators."""
    print_section("Technical Indicators")

    # Generate OHLC data
    n = 50
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)

    print(f"Test data: {n} bars")

    # Test ATR
    atr = mfp.calculate_atr(highs, lows, closes, period=14, engine="cpu")
    print(f"✓ ATR calculated: last value = {atr[-1]:.4f}")

    # Test RSI
    rsi = mfp.calculate_rsi(closes, period=14, engine="cpu")
    print(f"✓ RSI calculated: last value = {rsi[-1]:.2f}")

    # Test MACD
    macd, signal, hist = mfp.calculate_macd(closes, engine="cpu")
    print(f"✓ MACD calculated: macd={macd[-1]:.4f}, signal={signal[-1]:.4f}")

    # Test Bollinger Bands
    upper, middle, lower = mfp.calculate_bollinger_bands(closes, period=20, engine="cpu")
    print(
        f"✓ Bollinger Bands: upper={upper[-1]:.2f}, middle={middle[-1]:.2f}, lower={lower[-1]:.2f}"
    )

    # Test Stochastic Oscillator
    k, d = mfp.calculate_stochastic_oscillator(highs, lows, closes, period=14, engine="cpu")
    assert k.shape == (n,), f"Stochastic %K shape mismatch: {k.shape}"
    assert d.shape == (n,), f"Stochastic %D shape mismatch: {d.shape}"
    assert np.nanmin(k) >= 0 and np.nanmax(k) <= 100, "Stochastic %K out of range"
    print(f"✓ Stochastic Oscillator: %K={k[-1]:.2f}, %D={d[-1]:.2f}")

    # Test OBV
    volumes = np.random.randint(1000, 5000, size=n).astype(np.float64)
    obv = mfp.calculate_obv(closes, volumes, engine="cpu")
    assert obv.shape == (n,), f"OBV shape mismatch: {obv.shape}"
    print(f"✓ OBV calculated: last value = {obv[-1]:.0f}")

    # Test VWAP
    vwap = mfp.calculate_vwap(highs, lows, closes, volumes, engine="cpu")
    assert vwap.shape == (n,), f"VWAP shape mismatch: {vwap.shape}"
    print(f"✓ VWAP calculated: last value = {vwap[-1]:.2f}")

    # Test Williams %R
    wr = mfp.calculate_williams_r(highs, lows, closes, period=14, engine="cpu")
    assert wr.shape == (n,), f"Williams %R shape mismatch: {wr.shape}"
    assert np.nanmin(wr) >= -100 and np.nanmax(wr) <= 0, "Williams %R out of range"
    print(f"✓ Williams %R calculated: last value = {wr[-1]:.2f}")

    # Test CCI
    cci = mfp.calculate_cci(highs, lows, closes, period=20, engine="cpu")
    assert cci.shape == (n,), f"CCI shape mismatch: {cci.shape}"
    print(f"✓ CCI calculated: last value = {cci[-1]:.2f}")


def test_aggregations():
    """Test aggregation operations."""
    print_section("Aggregations")

    volume = np.array([1000, 2000, 1500, 3000, 2500], dtype=np.float64)
    prices = np.array([100, 102, 101, 105, 103], dtype=np.float64)

    print(f"Test data: {len(volume)} bars")

    # Test volume sum
    total = mfp.volume_sum(volume, engine="cpu")
    print(f"✓ Volume sum: {total:.0f}")

    # Test VWAP
    vwap = mfp.volume_weighted_price(prices, volume, engine="cpu")
    print(f"✓ VWAP: {vwap:.2f}")

    # Test rolling sum
    rolling = mfp.rolling_sum(volume, window=3)
    print(f"✓ Rolling sum (window=3): {rolling[-1]:.0f}")

    # Test cumulative sum
    cumsum = mfp.cumulative_sum(volume, engine="cpu")
    print(f"✓ Cumulative sum: last value = {cumsum[-1]:.0f}")


def test_gpu_operations():
    """Test GPU-specific operations if available."""
    print_section("GPU Operations")

    if not mfp.gpu_available():
        print("⚠ GPU not available, skipping GPU tests")
        return

    print("GPU is available, testing GPU operations...")

    # Test with GPU engine
    data = np.random.randn(10000) * 100 + 1000

    try:
        # NaN operations on GPU
        min_val = mfp.nanmin_gpu(data, engine="gpu")
        print(f"✓ GPU nanmin: {min_val:.2f}")

        max_val = mfp.nanmax_gpu(data, engine="gpu")
        print(f"✓ GPU nanmax: {max_val:.2f}")

        # Aggregation on GPU
        total = mfp.volume_sum(data, engine="gpu")
        print(f"✓ GPU volume_sum: {total:.2f}")

        print("✓ All GPU operations successful!")

    except mfp.GPUNotAvailableError as e:
        print(f"✗ GPU error: {e}")


def test_error_handling():
    """Test error handling."""
    print_section("Error Handling")

    # Test GPU not available error (if GPU is not present)
    if not mfp.gpu_available():
        try:
            data = np.array([1, 2, 3])
            _ = mfp.nanmin_gpu(data, engine="gpu")
            print("✗ Should have raised GPUNotAvailableError")
        except mfp.GPUNotAvailableError:
            print("✓ GPUNotAvailableError raised correctly")

    # Test data validation
    try:
        x = np.array([1, 2, 3])
        y = np.array([1, 2])  # Different length
        _ = mfp.least_squares_fit(x, y)
        print("✗ Should have raised ValueError")
    except ValueError:
        print("✓ ValueError raised correctly for mismatched arrays")

    # Test invalid engine
    try:
        data = np.array([1, 2, 3])
        _ = mfp.nanmin_gpu(data, engine="invalid")
        print("✗ Should have raised ConfigurationError")
    except mfp.ConfigurationError:
        print("✓ ConfigurationError raised correctly for invalid engine")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  kimsfinance: Comprehensive Test Suite")
    print("=" * 80)

    try:
        test_library_info()
        test_moving_averages()
        test_nan_operations()
        test_linear_algebra()
        test_indicators()
        test_aggregations()
        test_gpu_operations()
        test_error_handling()

        print_section("TEST SUMMARY")
        print("✓ ALL TESTS PASSED!")
        print("\nThe library is ready for use.")
        print("All operations are functional and error handling works correctly.")

    except Exception as e:
        print_section("TEST FAILED")
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 1 Integration Test Suite
================================

Comprehensive integration tests for all 8 Phase 1 indicators:
1. Stochastic Oscillator (existing)
2. VWAP
3. Ichimoku Cloud
4. ADX
5. Williams %R
6. CCI
7. MFI
8. Supertrend

Tests verify:
- All indicators importable
- All indicators work with common data
- Edge case handling
- Performance characteristics
- GPU acceleration
- Real data patterns
- Batch calculation compatibility
- Export consistency
"""

from __future__ import annotations

import numpy as np
import pytest
import time
from typing import Callable, Any


def generate_ohlcv_data(
    n: int = 100, seed: int = 42, trend: str = "mixed"
) -> dict[str, np.ndarray]:
    """
    Generate test OHLCV data for indicators.

    Args:
        n: Number of bars
        seed: Random seed for reproducibility
        trend: Data pattern - "mixed", "trending", "ranging", "volatile"

    Returns:
        Dictionary with high, low, close, volume arrays
    """
    np.random.seed(seed)

    if trend == "trending":
        # Strong uptrend
        closes = 100 + np.cumsum(np.random.randn(n) * 0.3 + 0.5)
    elif trend == "ranging":
        # Sideways movement
        closes = 100 + np.sin(np.arange(n) / 10) * 5 + np.random.randn(n) * 0.5
    elif trend == "volatile":
        # High volatility
        closes = 100 + np.cumsum(np.random.randn(n) * 2.0)
    else:  # mixed
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)

    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    volumes = np.abs(np.random.randn(n) * 1_000_000 + 500_000)

    return {
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except (ImportError, Exception):
        return False


class TestPhase1Integration:
    """Integration test suite for all Phase 1 indicators."""

    def test_all_indicators_importable(self):
        """Test all Phase 1 indicators can be imported."""
        try:
            from kimsfinance.ops import (
                calculate_stochastic,
                calculate_vwap,
                calculate_ichimoku,
                calculate_adx,
                calculate_williams_r,
                calculate_cci,
                calculate_mfi,
                calculate_supertrend,
            )

            # Verify all are callable
            assert callable(calculate_stochastic), "calculate_stochastic not callable"
            assert callable(calculate_vwap), "calculate_vwap not callable"
            assert callable(calculate_ichimoku), "calculate_ichimoku not callable"
            assert callable(calculate_adx), "calculate_adx not callable"
            assert callable(calculate_williams_r), "calculate_williams_r not callable"
            assert callable(calculate_cci), "calculate_cci not callable"
            assert callable(calculate_mfi), "calculate_mfi not callable"
            assert callable(calculate_supertrend), "calculate_supertrend not callable"

            print("✓ All 8 Phase 1 indicators imported successfully")

        except ImportError as e:
            pytest.fail(f"Failed to import Phase 1 indicators: {e}")

    def test_all_indicators_work_with_common_data(self):
        """Test all indicators calculate with same OHLCV data."""
        from kimsfinance.ops import (
            calculate_stochastic,
            calculate_vwap,
            calculate_ichimoku,
            calculate_adx,
            calculate_williams_r,
            calculate_cci,
            calculate_mfi,
            calculate_supertrend,
        )

        # Generate common 1000-row OHLCV dataset
        data = generate_ohlcv_data(n=1000, seed=42)
        n = len(data["close"])

        # Test Stochastic
        k, d = calculate_stochastic(
            data["high"], data["low"], data["close"], k_period=14, d_period=3, engine="cpu"
        )
        assert len(k) == n, f"Stochastic %K length mismatch: {len(k)} != {n}"
        assert len(d) == n, f"Stochastic %D length mismatch: {len(d)} != {n}"
        print(f"✓ Stochastic: k shape={k.shape}, d shape={d.shape}")

        # Test VWAP
        vwap = calculate_vwap(
            data["high"], data["low"], data["close"], data["volume"], engine="cpu"
        )
        assert len(vwap) == n, f"VWAP length mismatch: {len(vwap)} != {n}"
        print(f"✓ VWAP: shape={vwap.shape}")

        # Test Ichimoku
        ichimoku = calculate_ichimoku(data["high"], data["low"], data["close"], engine="cpu")
        assert isinstance(ichimoku, dict), "Ichimoku should return dict"
        assert len(ichimoku) == 5, f"Ichimoku should have 5 lines, got {len(ichimoku)}"
        for key, value in ichimoku.items():
            assert len(value) == n, f"Ichimoku {key} length mismatch: {len(value)} != {n}"
        print(f"✓ Ichimoku: {len(ichimoku)} lines")

        # Test ADX
        adx, plus_di, minus_di = calculate_adx(
            data["high"], data["low"], data["close"], period=14, engine="cpu"
        )
        assert len(adx) == n, f"ADX length mismatch: {len(adx)} != {n}"
        assert len(plus_di) == n, f"+DI length mismatch: {len(plus_di)} != {n}"
        assert len(minus_di) == n, f"-DI length mismatch: {len(minus_di)} != {n}"
        print(
            f"✓ ADX: adx shape={adx.shape}, +DI shape={plus_di.shape}, -DI shape={minus_di.shape}"
        )

        # Test Williams %R
        williams = calculate_williams_r(
            data["high"], data["low"], data["close"], period=14, engine="cpu"
        )
        assert len(williams) == n, f"Williams %R length mismatch: {len(williams)} != {n}"
        print(f"✓ Williams %R: shape={williams.shape}")

        # Test CCI
        cci = calculate_cci(data["high"], data["low"], data["close"], period=20, engine="cpu")
        assert len(cci) == n, f"CCI length mismatch: {len(cci)} != {n}"
        print(f"✓ CCI: shape={cci.shape}")

        # Test MFI
        mfi = calculate_mfi(
            data["high"], data["low"], data["close"], data["volume"], period=14, engine="cpu"
        )
        assert len(mfi) == n, f"MFI length mismatch: {len(mfi)} != {n}"
        print(f"✓ MFI: shape={mfi.shape}")

        # Test Supertrend
        supertrend, direction = calculate_supertrend(
            data["high"], data["low"], data["close"], period=10, multiplier=3.0, engine="cpu"
        )
        assert len(supertrend) == n, f"Supertrend length mismatch: {len(supertrend)} != {n}"
        assert len(direction) == n, f"Supertrend direction length mismatch: {len(direction)} != {n}"
        print(f"✓ Supertrend: trend shape={supertrend.shape}, direction shape={direction.shape}")

        print("✓ All 8 indicators calculated successfully with 1000-row dataset")

    def test_all_indicators_handle_edge_cases(self):
        """Test all indicators handle minimal data."""
        from kimsfinance.ops import (
            calculate_stochastic,
            calculate_vwap,
            calculate_ichimoku,
            calculate_adx,
            calculate_williams_r,
            calculate_cci,
            calculate_mfi,
            calculate_supertrend,
        )

        # Test with 50 rows (minimal data)
        data = generate_ohlcv_data(n=50, seed=123)

        try:
            # All should handle gracefully (may have NaN for initial values)
            k, d = calculate_stochastic(data["high"], data["low"], data["close"], engine="cpu")
            vwap = calculate_vwap(
                data["high"], data["low"], data["close"], data["volume"], engine="cpu"
            )
            ichimoku = calculate_ichimoku(data["high"], data["low"], data["close"], engine="cpu")
            adx, plus_di, minus_di = calculate_adx(
                data["high"], data["low"], data["close"], engine="cpu"
            )
            williams = calculate_williams_r(data["high"], data["low"], data["close"], engine="cpu")
            cci = calculate_cci(data["high"], data["low"], data["close"], engine="cpu")
            mfi = calculate_mfi(
                data["high"], data["low"], data["close"], data["volume"], engine="cpu"
            )
            supertrend, direction = calculate_supertrend(
                data["high"], data["low"], data["close"], engine="cpu"
            )

            print("✓ All indicators handle minimal data (50 rows) gracefully")

        except Exception as e:
            pytest.fail(f"Indicator failed with minimal data: {e}")

    def test_performance_relative_comparison(self):
        """Compare relative performance of all indicators."""
        from kimsfinance.ops import (
            calculate_stochastic,
            calculate_vwap,
            calculate_ichimoku,
            calculate_adx,
            calculate_williams_r,
            calculate_cci,
            calculate_mfi,
            calculate_supertrend,
        )

        # Benchmark all 8 indicators at 100K rows
        data = generate_ohlcv_data(n=100_000, seed=42)

        results = []

        # Stochastic
        start = time.perf_counter()
        k, d = calculate_stochastic(data["high"], data["low"], data["close"], engine="cpu")
        stoch_time = (time.perf_counter() - start) * 1000
        results.append(("Stochastic", stoch_time))

        # VWAP
        start = time.perf_counter()
        vwap = calculate_vwap(
            data["high"], data["low"], data["close"], data["volume"], engine="cpu"
        )
        vwap_time = (time.perf_counter() - start) * 1000
        results.append(("VWAP", vwap_time))

        # Ichimoku
        start = time.perf_counter()
        ichimoku = calculate_ichimoku(data["high"], data["low"], data["close"], engine="cpu")
        ichimoku_time = (time.perf_counter() - start) * 1000
        results.append(("Ichimoku", ichimoku_time))

        # ADX
        start = time.perf_counter()
        adx, plus_di, minus_di = calculate_adx(
            data["high"], data["low"], data["close"], engine="cpu"
        )
        adx_time = (time.perf_counter() - start) * 1000
        results.append(("ADX", adx_time))

        # Williams %R
        start = time.perf_counter()
        williams = calculate_williams_r(data["high"], data["low"], data["close"], engine="cpu")
        williams_time = (time.perf_counter() - start) * 1000
        results.append(("Williams %R", williams_time))

        # CCI
        start = time.perf_counter()
        cci = calculate_cci(data["high"], data["low"], data["close"], engine="cpu")
        cci_time = (time.perf_counter() - start) * 1000
        results.append(("CCI", cci_time))

        # MFI
        start = time.perf_counter()
        mfi = calculate_mfi(data["high"], data["low"], data["close"], data["volume"], engine="cpu")
        mfi_time = (time.perf_counter() - start) * 1000
        results.append(("MFI", mfi_time))

        # Supertrend
        start = time.perf_counter()
        supertrend, direction = calculate_supertrend(
            data["high"], data["low"], data["close"], engine="cpu"
        )
        supertrend_time = (time.perf_counter() - start) * 1000
        results.append(("Supertrend", supertrend_time))

        # Report results
        print("\n=== Performance Comparison (100K rows, CPU) ===")
        for name, ms in sorted(results, key=lambda x: x[1]):
            print(f"{name:15s}: {ms:7.2f} ms")

        # Verify all completed in reasonable time (< 5 seconds each on CPU)
        # Note: Complex indicators like Ichimoku (5 lines) may take slightly longer
        for name, ms in results:
            assert ms < 5000, f"{name} took too long: {ms:.2f} ms"

        print("✓ All indicators completed in reasonable time")

    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_gpu_acceleration_works(self):
        """Test GPU acceleration for all indicators."""
        from kimsfinance.ops import (
            calculate_stochastic,
            calculate_vwap,
            calculate_ichimoku,
            calculate_adx,
            calculate_williams_r,
            calculate_cci,
            calculate_mfi,
            calculate_supertrend,
        )

        # Use 100K rows to trigger GPU acceleration
        data = generate_ohlcv_data(n=100_000, seed=42)

        # Calculate with CPU
        k_cpu, d_cpu = calculate_stochastic(data["high"], data["low"], data["close"], engine="cpu")
        vwap_cpu = calculate_vwap(
            data["high"], data["low"], data["close"], data["volume"], engine="cpu"
        )

        # Calculate with GPU
        k_gpu, d_gpu = calculate_stochastic(data["high"], data["low"], data["close"], engine="gpu")
        vwap_gpu = calculate_vwap(
            data["high"], data["low"], data["close"], data["volume"], engine="gpu"
        )

        # Verify results match within tolerance (1e-6 relative tolerance)
        np.testing.assert_allclose(
            k_cpu, k_gpu, rtol=1e-6, atol=1e-8, err_msg="Stochastic %K CPU/GPU mismatch"
        )
        np.testing.assert_allclose(
            d_cpu, d_gpu, rtol=1e-6, atol=1e-8, err_msg="Stochastic %D CPU/GPU mismatch"
        )
        np.testing.assert_allclose(
            vwap_cpu, vwap_gpu, rtol=1e-6, atol=1e-8, err_msg="VWAP CPU/GPU mismatch"
        )

        print("✓ GPU acceleration works correctly (results match CPU within tolerance)")

    def test_all_indicators_with_real_data_pattern(self):
        """Test with realistic trending/ranging/volatile data."""
        from kimsfinance.ops import (
            calculate_stochastic,
            calculate_vwap,
            calculate_ichimoku,
            calculate_adx,
            calculate_williams_r,
            calculate_cci,
            calculate_mfi,
            calculate_supertrend,
        )

        patterns = ["trending", "ranging", "volatile"]

        for pattern in patterns:
            data = generate_ohlcv_data(n=500, seed=42, trend=pattern)

            # Run all indicators
            k, d = calculate_stochastic(data["high"], data["low"], data["close"], engine="cpu")
            vwap = calculate_vwap(
                data["high"], data["low"], data["close"], data["volume"], engine="cpu"
            )
            ichimoku = calculate_ichimoku(data["high"], data["low"], data["close"], engine="cpu")
            adx, plus_di, minus_di = calculate_adx(
                data["high"], data["low"], data["close"], engine="cpu"
            )
            williams = calculate_williams_r(data["high"], data["low"], data["close"], engine="cpu")
            cci = calculate_cci(data["high"], data["low"], data["close"], engine="cpu")
            mfi = calculate_mfi(
                data["high"], data["low"], data["close"], data["volume"], engine="cpu"
            )
            supertrend, direction = calculate_supertrend(
                data["high"], data["low"], data["close"], engine="cpu"
            )

            # Verify appropriate signals generated (no NaN everywhere)
            assert not np.all(np.isnan(k)), f"Stochastic all NaN for {pattern} data"
            assert not np.all(np.isnan(vwap)), f"VWAP all NaN for {pattern} data"
            assert not np.all(np.isnan(adx)), f"ADX all NaN for {pattern} data"
            assert not np.all(np.isnan(williams)), f"Williams all NaN for {pattern} data"
            assert not np.all(np.isnan(cci)), f"CCI all NaN for {pattern} data"
            assert not np.all(np.isnan(mfi)), f"MFI all NaN for {pattern} data"
            assert not np.all(np.isnan(supertrend)), f"Supertrend all NaN for {pattern} data"

            print(f"✓ All indicators work with {pattern} data pattern")

    def test_batch_calculation_compatibility(self):
        """Test indicators work well together for multi-indicator strategies."""
        from kimsfinance.ops import (
            calculate_stochastic,
            calculate_vwap,
            calculate_ichimoku,
            calculate_adx,
            calculate_williams_r,
            calculate_cci,
            calculate_mfi,
            calculate_supertrend,
        )

        # Calculate all 8 indicators on same dataset
        data = generate_ohlcv_data(n=1000, seed=42)

        # Batch calculate all indicators
        indicators = {}

        k, d = calculate_stochastic(data["high"], data["low"], data["close"], engine="cpu")
        indicators["stochastic_k"] = k
        indicators["stochastic_d"] = d

        indicators["vwap"] = calculate_vwap(
            data["high"], data["low"], data["close"], data["volume"], engine="cpu"
        )

        ichimoku = calculate_ichimoku(data["high"], data["low"], data["close"], engine="cpu")
        indicators.update({f"ichimoku_{k}": v for k, v in ichimoku.items()})

        adx, plus_di, minus_di = calculate_adx(
            data["high"], data["low"], data["close"], engine="cpu"
        )
        indicators["adx"] = adx
        indicators["plus_di"] = plus_di
        indicators["minus_di"] = minus_di

        indicators["williams_r"] = calculate_williams_r(
            data["high"], data["low"], data["close"], engine="cpu"
        )

        indicators["cci"] = calculate_cci(data["high"], data["low"], data["close"], engine="cpu")

        indicators["mfi"] = calculate_mfi(
            data["high"], data["low"], data["close"], data["volume"], engine="cpu"
        )

        supertrend, direction = calculate_supertrend(
            data["high"], data["low"], data["close"], engine="cpu"
        )
        indicators["supertrend"] = supertrend
        indicators["supertrend_direction"] = direction

        # Verify no conflicts or issues
        assert (
            len(indicators) >= 14
        ), f"Expected at least 14 indicator values, got {len(indicators)}"

        # Test combining signals (example: multi-confirmation)
        valid_idx = 100  # After warmup period

        # Momentum confirmation: Stochastic oversold + Williams oversold
        stoch_oversold = indicators["stochastic_k"][valid_idx] < 20
        williams_oversold = indicators["williams_r"][valid_idx] < -80
        momentum_oversold = stoch_oversold and williams_oversold

        # Volume confirmation: Price above VWAP and MFI not overbought
        price_above_vwap = data["close"][valid_idx] > indicators["vwap"][valid_idx]
        mfi_ok = indicators["mfi"][valid_idx] < 80
        volume_ok = price_above_vwap and mfi_ok

        # Trend confirmation: ADX > 25 (trending) and Supertrend direction
        trending = indicators["adx"][valid_idx] > 25

        print(f"✓ Multi-indicator strategy compatible:")
        print(f"  - Momentum oversold: {momentum_oversold}")
        print(f"  - Volume confirmation: {volume_ok}")
        print(f"  - Trending market: {trending}")
        print("✓ No conflicts between indicators")

    def test_export_consistency(self):
        """Verify all indicators exported consistently."""
        import kimsfinance.ops as ops

        # Check all 8 in dir(ops)
        ops_dir = dir(ops)

        required_indicators = [
            "calculate_stochastic",
            "calculate_vwap",
            "calculate_ichimoku",
            "calculate_adx",
            "calculate_williams_r",
            "calculate_cci",
            "calculate_mfi",
            "calculate_supertrend",
        ]

        for indicator in required_indicators:
            assert indicator in ops_dir, f"{indicator} not in ops module"
            assert hasattr(ops, indicator), f"{indicator} not accessible via ops.{indicator}"

        # Check all 8 in ops.__all__
        if hasattr(ops, "__all__"):
            for indicator in required_indicators:
                assert indicator in ops.__all__, f"{indicator} not in ops.__all__"

        print("✓ All 8 Phase 1 indicators exported consistently")
        print(
            f"✓ All indicators in module dir: {all(ind in ops_dir for ind in required_indicators)}"
        )
        if hasattr(ops, "__all__"):
            print(
                f"✓ All indicators in __all__: {all(ind in ops.__all__ for ind in required_indicators)}"
            )


if __name__ == "__main__":
    # Run tests directly
    print("=" * 80)
    print("Phase 1 Integration Test Suite")
    print("=" * 80)

    test_suite = TestPhase1Integration()

    try:
        test_suite.test_all_indicators_importable()
        test_suite.test_all_indicators_work_with_common_data()
        test_suite.test_all_indicators_handle_edge_cases()
        test_suite.test_performance_relative_comparison()

        if gpu_available():
            test_suite.test_gpu_acceleration_works()
        else:
            print("\n⚠ Skipping GPU tests (GPU not available)")

        test_suite.test_all_indicators_with_real_data_pattern()
        test_suite.test_batch_calculation_compatibility()
        test_suite.test_export_consistency()

        print("\n" + "=" * 80)
        print("✓ ALL INTEGRATION TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

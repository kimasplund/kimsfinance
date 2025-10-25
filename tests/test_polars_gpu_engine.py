#!/usr/bin/env python3
"""
Test Polars GPU Engine Integration

Validates:
1. GPU engine detection
2. GPU engine usage when available
3. Graceful fallback to CPU when GPU unavailable
4. Result correctness (GPU == CPU results)
"""

from __future__ import annotations

import pytest
import polars as pl
import numpy as np


def check_gpu_available() -> bool:
    """
    Check if Polars GPU engine is available.

    Returns:
        bool: True if GPU engine works, False otherwise
    """
    try:
        test_df = pl.LazyFrame({"test": [1, 2, 3]})
        test_df.collect(engine="gpu")
        return True
    except Exception:
        return False


GPU_AVAILABLE = check_gpu_available()


def generate_test_data(n_rows: int = 1000) -> pl.LazyFrame:
    """Generate test OHLCV data."""
    np.random.seed(42)
    return pl.LazyFrame(
        {
            "symbol": np.random.choice(["AAPL", "GOOGL"], n_rows),
            "timestamp": np.arange(n_rows),
            "open": np.random.random(n_rows) * 100 + 100,
            "high": np.random.random(n_rows) * 110 + 100,
            "low": np.random.random(n_rows) * 90 + 100,
            "close": np.random.random(n_rows) * 100 + 100,
            "volume": np.random.randint(1000, 100000, n_rows),
        }
    )


def test_gpu_detection():
    """Test GPU availability detection."""
    print(f"\nGPU Available: {GPU_AVAILABLE}")
    assert isinstance(GPU_AVAILABLE, bool)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_engine_basic():
    """Test basic GPU engine functionality."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = lf.collect(engine="gpu")

    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]
    assert result["b"].to_list() == [4, 5, 6]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_engine_groupby():
    """Test GPU engine with groupby aggregations."""
    lf = generate_test_data(n_rows=1000)

    gpu_result = (
        lf.group_by("symbol")
        .agg(
            [
                pl.col("open").mean().alias("open_mean"),
                pl.col("volume").sum().alias("volume_sum"),
            ]
        )
        .collect(engine="gpu")
        .sort("symbol")
    )

    cpu_result = (
        lf.group_by("symbol")
        .agg(
            [
                pl.col("open").mean().alias("open_mean"),
                pl.col("volume").sum().alias("volume_sum"),
            ]
        )
        .collect(engine=None)
        .sort("symbol")
    )

    assert gpu_result.shape == cpu_result.shape
    np.testing.assert_array_almost_equal(
        gpu_result["open_mean"].to_numpy(), cpu_result["open_mean"].to_numpy(), decimal=6
    )
    np.testing.assert_array_equal(
        gpu_result["volume_sum"].to_numpy(), cpu_result["volume_sum"].to_numpy()
    )


def test_cpu_fallback():
    """Test graceful fallback to CPU when GPU unavailable."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    result = lf.collect(engine="gpu" if GPU_AVAILABLE else None)

    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_engine_complex_aggregation():
    """Test GPU engine with complex aggregations."""
    lf = generate_test_data(n_rows=10000)

    result = (
        lf.group_by("symbol")
        .agg(
            [
                pl.col("open").mean().alias("open_mean"),
                pl.col("high").max().alias("high_max"),
                pl.col("low").min().alias("low_min"),
                pl.col("close").std().alias("close_std"),
                pl.col("volume").sum().alias("volume_sum"),
                pl.count().alias("count"),
            ]
        )
        .collect(engine="gpu")
    )

    assert result.shape[1] == 7
    assert "open_mean" in result.columns
    assert "high_max" in result.columns
    assert result["count"].sum() == 10000


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_engine_performance():
    """Test GPU engine performance vs CPU (informational)."""
    import time

    lf = generate_test_data(n_rows=100000)

    start = time.perf_counter()
    cpu_result = (
        lf.group_by("symbol")
        .agg(
            [
                pl.col("open").mean(),
                pl.col("volume").sum(),
            ]
        )
        .collect(engine=None)
    )
    cpu_time = time.perf_counter() - start

    start = time.perf_counter()
    gpu_result = (
        lf.group_by("symbol")
        .agg(
            [
                pl.col("open").mean(),
                pl.col("volume").sum(),
            ]
        )
        .collect(engine="gpu")
    )
    gpu_time = time.perf_counter() - start

    speedup = cpu_time / gpu_time
    print(f"\nGPU Speedup: {speedup:.2f}x (CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s)")

    assert speedup > 0

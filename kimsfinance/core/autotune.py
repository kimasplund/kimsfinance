"""
Auto-tuning for GPU Crossover Thresholds
=========================================

This module provides functionality to empirically determine the optimal
crossover points for CPU vs. GPU execution for various operations.
"""

import json
import timeit
from pathlib import Path

import numpy as np
import polars as pl

from .types import Engine


DEFAULT_THRESHOLDS = {
    "atr": 100_000,
    "rsi": 100_000,
    "stochastic": 500_000,
    "default": 100_000,
    "batch_indicators": 15_000,
}

CACHE_FILE = Path.home() / ".kimsfinance" / "threshold_cache.json"


def _get_test_data(size: int) -> pl.DataFrame:
    """Generate a sample OHLCV DataFrame for benchmarking."""
    return pl.DataFrame(
        {
            "open": np.random.rand(size),
            "high": np.random.rand(size),
            "low": np.random.rand(size),
            "close": np.random.rand(size),
            "volume": np.random.randint(100, 1000, size=size),
        }
    )


def _benchmark_operation(func, data: pl.DataFrame, engine: Engine, number: int = 10) -> float:
    """Benchmark a given function on a specific engine."""
    from .engine import EngineManager  # Local import to avoid circular dependency

    timer = timeit.Timer(lambda: func(data, engine=engine))
    return timer.timeit(number=number) / number


def _get_operation_func(operation: str):
    """Return a real indicator function for a given operation."""
    from kimsfinance.ops.indicators import (
        calculate_atr,
        calculate_rsi,
        calculate_stochastic_oscillator,
    )

    if operation == "atr":
        return lambda df, engine: calculate_atr(df["high"], df["low"], df["close"], engine=engine)
    elif operation == "rsi":
        return lambda df, engine: calculate_rsi(df["close"], engine=engine)
    elif operation == "stochastic":
        return lambda df, engine: calculate_stochastic_oscillator(
            df["high"], df["low"], df["close"], engine=engine
        )
    else:
        # Default to a simple operation for unknown indicators
        return lambda df, engine: df.select(pl.mean("close")).collect(engine=engine)


def find_crossover(operation: str, sizes=None) -> int:
    """
    Find the crossover point for a given operation by benchmarking.
    """
    if sizes is None:
        sizes = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

    operation_func = _get_operation_func(operation)

    for size in sizes:
        data = _get_test_data(size)
        try:
            cpu_time = _benchmark_operation(operation_func, data, "cpu")
            gpu_time = _benchmark_operation(operation_func, data, "gpu")

            if gpu_time < cpu_time:
                return size
        except Exception as e:
            print(f"  - Error benchmarking at size {size}: {e}")
            continue

    return DEFAULT_THRESHOLDS.get(operation, DEFAULT_THRESHOLDS["default"])


def run_autotune(operations: list[str] | None = None, save: bool = True) -> dict[str, int]:
    """
    Run the auto-tuning process to find the optimal crossover thresholds.
    """
    if operations is None:
        operations = list(DEFAULT_THRESHOLDS.keys())

    tuned_thresholds = {}
    for op in operations:
        print(f"Tuning operation: {op}...")
        tuned_thresholds[op] = find_crossover(op)
        print(f"  -> Found crossover at: {tuned_thresholds[op]}")

    if save:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(tuned_thresholds, f, indent=4)
        print(f"Saved tuned thresholds to: {CACHE_FILE}")

    return tuned_thresholds


def load_tuned_thresholds() -> dict[str, int]:
    """Load tuned thresholds from the cache file."""
    if not CACHE_FILE.exists():
        return DEFAULT_THRESHOLDS

    with open(CACHE_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return DEFAULT_THRESHOLDS

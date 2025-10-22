"""
Auto-tuning for GPU Crossover Thresholds
=========================================

This module provides functionality to empirically determine the optimal
crossover points for CPU vs. GPU execution for various operations.
"""

import json
import timeit
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Callable, Any

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

# Global lock for file I/O operations
_autotune_lock = threading.Lock()


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


def _benchmark_operation(
    func: Callable[[pl.DataFrame, Engine], Any], data: pl.DataFrame, engine: Engine, number: int = 10
) -> float:
    """Benchmark a given function on a specific engine."""
    from .engine import EngineManager  # Local import to avoid circular dependency

    timer = timeit.Timer(lambda: func(data, engine=engine))
    return timer.timeit(number=number) / number


def _get_operation_func(operation: str) -> Callable[[pl.DataFrame, Engine], Any]:
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


def find_crossover(operation: str, sizes: list[int] | None = None) -> int:
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
    Run the auto-tuning process to find the optimal crossover thresholds (thread-safe).

    Thread-safe: Yes (uses lock for file operations)

    Args:
        operations: List of operations to tune (None = all)
        save: If True, save results to cache file

    Returns:
        dict: Tuned threshold values
    """
    if operations is None:
        operations = list(DEFAULT_THRESHOLDS.keys())

    tuned_thresholds = {}
    for op in operations:
        print(f"Tuning operation: {op}...")
        tuned_thresholds[op] = find_crossover(op)
        print(f"  -> Found crossover at: {tuned_thresholds[op]}")

    if save:
        _save_tuned_thresholds(tuned_thresholds)

    return tuned_thresholds


def _save_tuned_thresholds(thresholds: dict[str, int]) -> None:
    """
    Save thresholds with atomic write (thread-safe).

    Uses atomic file operations:
    1. Write to temp file
    2. Atomic rename (POSIX guarantees atomicity)
    3. Set secure permissions (user read/write only)

    Thread-safe: Yes (uses global lock + atomic rename)
    """
    with _autotune_lock:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write using temp file + rename
        with tempfile.NamedTemporaryFile(
            mode="w", dir=CACHE_FILE.parent, delete=False, suffix=".tmp", prefix=".threshold_"
        ) as tf:
            json.dump(thresholds, tf, indent=4)
            temp_path = tf.name

        try:
            # Atomic rename (POSIX guarantees atomicity)
            shutil.move(temp_path, CACHE_FILE)

            # Set permissions (user read/write only)
            CACHE_FILE.chmod(0o600)

            print(f"Saved tuned thresholds to: {CACHE_FILE}")
        except Exception as e:
            # Cleanup temp file on error
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"Failed to save tuned thresholds: {e}") from e


def load_tuned_thresholds() -> dict[str, int]:
    """
    Load tuned thresholds from the cache file (thread-safe).

    Thread-safe: Yes (uses lock for file read)

    Returns:
        dict: Tuned thresholds or default if not available
    """
    with _autotune_lock:
        if not CACHE_FILE.exists():
            return DEFAULT_THRESHOLDS.copy()

        try:
            with open(CACHE_FILE, "r") as f:
                loaded = json.load(f)
                # Validate loaded data
                if not isinstance(loaded, dict):
                    return DEFAULT_THRESHOLDS.copy()
                # Return copy to prevent external modification
                return loaded
        except (json.JSONDecodeError, OSError, IOError):
            return DEFAULT_THRESHOLDS.copy()

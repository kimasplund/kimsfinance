"""
Engine Management for kimsfinance
========================================

Intelligent CPU/GPU engine selection with automatic fallback and
performance heuristics.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal, TypeVar, ParamSpec

import polars as pl

from .types import Engine
from .exceptions import GPUNotAvailableError, ConfigurationError


__all__ = [
    "EngineManager",
    "with_engine_fallback",
    "GPUNotAvailableError",
    "GPU_CROSSOVER_THRESHOLDS",
]


P = ParamSpec('P')
R = TypeVar('R')


# Empirical GPU crossover thresholds (rows)
# Below these thresholds, CPU is faster due to data transfer overhead
# Above these thresholds, GPU provides significant speedup
GPU_CROSSOVER_THRESHOLDS = {
    "atr": 100_000,
    "rsi": 100_000,
    "stochastic": 500_000,
    "bollinger": 100_000,
    "obv": 100_000,
    "macd": 100_000,
    "batch_indicators": 15_000,  # Much lower for batch! Amortizes data transfer overhead
}


class EngineManager:
    """
    Manages execution engine selection and GPU availability detection.

    The EngineManager provides intelligent engine selection based on:
    - GPU availability
    - Data size
    - Operation type
    - User preferences
    """

    _gpu_available: bool | None = None  # Cache GPU availability check

    @classmethod
    def check_gpu_available(cls) -> bool:
        """
        Check if GPU acceleration is available.

        Performs a simple test operation using Polars GPU engine.
        Result is cached for performance.

        Returns:
            bool: True if GPU is available and functional
        """
        if cls._gpu_available is not None:
            return cls._gpu_available

        try:
            # Try to execute a simple GPU operation
            test_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
            test_df.lazy().select(pl.col("x")).collect(engine="gpu")
            cls._gpu_available = True
            return True
        except Exception:
            cls._gpu_available = False
            return False

    @classmethod
    def reset_gpu_cache(cls) -> None:
        """Reset the GPU availability cache (useful for testing)."""
        cls._gpu_available = None

    @classmethod
    def select_engine(cls, engine: Engine) -> Literal["cpu", "gpu"]:
        """
        Select the appropriate execution engine.

        Args:
            engine: Requested engine ("cpu", "gpu", or "auto")

        Returns:
            Actual engine to use ("cpu" or "gpu")

        Raises:
            GPUNotAvailableError: If GPU requested but not available
            ConfigurationError: If invalid engine specified
        """
        match engine:
            case "cpu":
                return "cpu"

            case "gpu":
                if not cls.check_gpu_available():
                    raise GPUNotAvailableError()
                return "gpu"

            case "auto":
                return "gpu" if cls.check_gpu_available() else "cpu"

            case _:
                raise ConfigurationError(
                    f"Invalid engine: {engine!r}. Must be 'cpu', 'gpu', or 'auto'."
                )

    @classmethod
    def select_engine_smart(
        cls,
        engine: Engine,
        operation: str | None = None,
        data_size: int | None = None
    ) -> Literal["cpu", "gpu"]:
        """
        Select execution engine with intelligent size-based defaults.

        This method extends select_engine() with automatic threshold-based
        selection when engine="auto". It uses empirical GPU crossover thresholds
        to determine when GPU acceleration is beneficial.

        Args:
            engine: User-specified engine ("cpu", "gpu", or "auto")
            operation: Operation name for threshold lookup (e.g., "atr", "rsi", "macd")
            data_size: Dataset size in rows for automatic threshold check

        Returns:
            Actual engine to use ("cpu" or "gpu")

        Raises:
            GPUNotAvailableError: If GPU requested but not available
            ConfigurationError: If invalid engine specified

        Examples:
            >>> # Explicit CPU selection (always returns "cpu")
            >>> EngineManager.select_engine_smart("cpu")
            "cpu"

            >>> # Explicit GPU selection (checks availability)
            >>> EngineManager.select_engine_smart("gpu")
            "gpu"  # or raises GPUNotAvailableError

            >>> # Auto with operation and size (intelligent selection)
            >>> EngineManager.select_engine_smart("auto", "atr", 50_000)
            "cpu"  # Below 100K threshold

            >>> EngineManager.select_engine_smart("auto", "atr", 150_000)
            "gpu"  # Above 100K threshold (if GPU available)

            >>> # Auto without operation/size (conservative default)
            >>> EngineManager.select_engine_smart("auto")
            "cpu"  # Conservative default when no context

            >>> # Unknown operation uses default threshold
            >>> EngineManager.select_engine_smart("auto", "custom_indicator", 200_000)
            "gpu"  # Uses default 100K threshold

        Performance Rationale:
            GPU crossover thresholds are empirically derived from benchmarks:
            - Below threshold: CPU faster (data transfer overhead dominates)
            - Above threshold: GPU faster (parallel computation benefits)
            - Default threshold: 100K rows (conservative, works for most operations)
            - Stochastic: 500K rows (more complex computation pattern)

        See Also:
            - GPU_CROSSOVER_THRESHOLDS: Empirical threshold values
            - select_engine(): Basic engine selection without smart defaults
            - get_optimal_engine(): Advanced heuristics with more operations
        """
        # Explicit CPU selection
        if engine == "cpu":
            return "cpu"

        # Explicit GPU selection
        if engine == "gpu":
            if not cls.check_gpu_available():
                raise GPUNotAvailableError()
            return "gpu"

        # Auto selection with intelligent defaults
        if engine == "auto":
            # Check GPU availability first
            if not cls.check_gpu_available():
                return "cpu"

            # If operation and data_size provided, use threshold-based selection
            if operation is not None and data_size is not None:
                # Get threshold for operation (default to 100K if unknown)
                threshold = GPU_CROSSOVER_THRESHOLDS.get(operation, 100_000)
                return "gpu" if data_size >= threshold else "cpu"

            # Conservative default: CPU when no context available
            return "cpu"

        # Invalid engine
        raise ConfigurationError(
            f"Invalid engine: {engine!r}. Must be 'cpu', 'gpu', or 'auto'."
        )

    @classmethod
    def get_optimal_engine(
        cls,
        operation: str,
        data_size: int,
        *,
        force_cpu: bool = False
    ) -> Engine:
        """
        Get the optimal engine for a specific operation and data size.

        Uses performance heuristics to determine when GPU is beneficial.

        Args:
            operation: Operation name (e.g., "nanmin", "moving_average", "least_squares")
            data_size: Number of rows in dataset
            force_cpu: If True, always return "cpu"

        Returns:
            Recommended engine

        Performance Heuristics:
            - Moving averages: GPU not beneficial (data transfer overhead)
            - NaN operations: GPU beneficial for >10K rows (40-80x speedup)
            - Least squares: GPU beneficial for >1K rows (30-50x speedup)
            - ATR: GPU beneficial for >5K rows (10-24x speedup)
            - Small data (<1K): Always use CPU (overhead dominates)
        """
        if force_cpu or not cls.check_gpu_available():
            return "cpu"

        # Always use CPU for very small datasets
        if data_size < 1_000:
            return "cpu"

        # Operation-specific heuristics
        match operation:
            case "moving_average" | "sma" | "ema":
                # GPU not beneficial for moving averages
                return "cpu"

            case "nanmin" | "nanmax" | "isnan" | "nan_bounds":
                # GPU beneficial for NaN operations with >10K rows
                return "gpu" if data_size >= 10_000 else "cpu"

            case "least_squares" | "trend_line":
                # GPU beneficial for linear algebra with >1K rows
                return "gpu" if data_size >= 1_000 else "cpu"

            case "atr" | "rsi" | "indicators":
                # GPU beneficial for indicators with >5K rows
                return "gpu" if data_size >= 5_000 else "cpu"

            case "volume_sum" | "aggregations":
                # GPU beneficial for aggregations with >5K rows
                return "gpu" if data_size >= 5_000 else "cpu"

            case "pnf" | "renko" | "transformations":
                # GPU beneficial for transformations with >10K rows
                return "gpu" if data_size >= 10_000 else "cpu"

            case _:
                # Unknown operation: Conservative heuristic
                return "gpu" if data_size >= 10_000 else "cpu"

    @classmethod
    def get_info(cls) -> dict[str, str | bool]:
        """
        Get information about engine availability and configuration.

        Returns:
            Dict with engine information
        """
        gpu_available = cls.check_gpu_available()

        info: dict[str, str | bool] = {
            "cpu_available": True,
            "gpu_available": gpu_available,
            "default_engine": "auto",
        }

        if gpu_available:
            try:
                import cudf
                info["cudf_version"] = str(cudf.__version__)
            except ImportError:
                info["cudf_version"] = "Not installed"

        return info


# Convenience functions for common patterns
def with_engine_fallback(func: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator that provides automatic CPU fallback on GPU errors.

    Usage:
        @with_engine_fallback
        def my_operation(data, *, engine: Engine = "auto"):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: object, engine: Engine = "auto", **kwargs: object) -> R:
        selected_engine = EngineManager.select_engine(engine)

        try:
            return func(*args, engine=selected_engine, **kwargs)
        except Exception:
            if selected_engine == "gpu":
                # Fallback to CPU on GPU errors
                return func(*args, engine="cpu", **kwargs)
            else:
                # Re-raise if already on CPU
                raise

    return wrapper

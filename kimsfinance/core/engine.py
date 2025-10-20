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
from .autotune import load_tuned_thresholds, run_autotune


__all__ = [
    "EngineManager",
    "with_engine_fallback",
    "GPUNotAvailableError",
    "GPU_CROSSOVER_THRESHOLDS",
    "run_autotune",
]


P = ParamSpec("P")
R = TypeVar("R")


# Load thresholds on module import
GPU_CROSSOVER_THRESHOLDS = load_tuned_thresholds()

# Advanced heuristics for get_optimal_engine()
OPERATION_HEURISTICS = {
    "nan_ops": {"threshold": 10_000, "ops": ["nanmin", "nanmax", "isnan"]},
    "linear_algebra": {"threshold": 1_000, "ops": ["least_squares", "trend_line"]},
    "indicators": {"threshold": 5_000, "ops": ["atr", "rsi"]},
    "aggregations": {"threshold": 5_000, "ops": ["volume_sum"]},
    "transformations": {"threshold": 10_000, "ops": ["pnf", "renko"]},
    "moving_averages": {"threshold": float("inf"), "ops": ["sma", "ema"]},  # Always CPU
}


class EngineManager:
    """
    Manages execution engine selection and GPU availability detection.
    """

    _gpu_available: bool | None = None  # Cache GPU availability check

    @classmethod
    def check_gpu_available(cls) -> bool:
        """
        Check if GPU acceleration is available (lightweight check).
        """
        if cls._gpu_available is not None:
            return cls._gpu_available

        try:
            import cudf

            cls._gpu_available = True
            return True
        except ImportError:
            cls._gpu_available = False
            return False

    @classmethod
    def reset_gpu_cache(cls) -> None:
        """Reset the GPU availability cache."""
        cls._gpu_available = None

    @classmethod
    def select_engine(
        cls, engine: Engine, operation: str | None = None, data_size: int | None = None
    ) -> Literal["cpu", "gpu"]:
        """
        Select the appropriate execution engine with intelligent defaults.

        Args:
            engine: Requested engine ("cpu", "gpu", or "auto")
            operation: Operation name for threshold-based selection
            data_size: Dataset size for threshold check

        Returns:
            Selected engine ("cpu" or "gpu")
        """
        if engine not in ("cpu", "gpu", "auto"):
            raise ConfigurationError(f"Invalid engine: {engine!r}")

        if engine == "cpu":
            return "cpu"

        gpu_available = cls.check_gpu_available()

        if engine == "gpu":
            if not gpu_available:
                raise GPUNotAvailableError()
            return "gpu"

        # Auto selection
        if not gpu_available:
            return "cpu"

        if operation and data_size is not None:
            threshold = GPU_CROSSOVER_THRESHOLDS.get(operation, GPU_CROSSOVER_THRESHOLDS["default"])
            return "gpu" if data_size >= threshold else "cpu"

        return "cpu"  # Conservative default for "auto"

    @classmethod
    def get_optimal_engine(
        cls, operation: str, data_size: int, *, force_cpu: bool = False
    ) -> Engine:
        """
        Get the optimal engine using advanced performance heuristics.

        Args:
            operation: Operation name
            data_size: Number of rows in dataset
            force_cpu: If True, always return "cpu"
        """
        if force_cpu or not cls.check_gpu_available() or data_size < 1_000:
            return "cpu"

        for details in OPERATION_HEURISTICS.values():
            if operation in details["ops"]:
                return "gpu" if data_size >= details["threshold"] else "cpu"

        # Default for unknown operations
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

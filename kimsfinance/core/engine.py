"""
Engine Management for kimsfinance
========================================

Intelligent CPU/GPU engine selection with automatic fallback and
performance heuristics.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Literal, TypeVar, ParamSpec

import polars as pl

from .types import Engine
from .exceptions import GPUNotAvailableError, ConfigurationError
from .autotune import load_tuned_thresholds, run_autotune
from ..config.gpu_thresholds import get_threshold


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
# Thresholds are now loaded from gpu_thresholds.py for consistency
OPERATION_HEURISTICS = {
    "nan_ops": {"threshold": get_threshold("nan_ops"), "ops": ["nanmin", "nanmax", "isnan"]},
    "linear_algebra": {"threshold": get_threshold("linear_algebra"), "ops": ["least_squares", "trend_line"]},
    "indicators": {"threshold": get_threshold("aggregation"), "ops": ["atr", "rsi"]},
    "aggregations": {"threshold": get_threshold("aggregation"), "ops": ["volume_sum"]},
    "transformations": {"threshold": get_threshold("transformation"), "ops": ["pnf", "renko"]},
    "moving_averages": {"threshold": float("inf"), "ops": ["sma", "ema"]},  # Always CPU
}


class EngineManager:
    """
    Manages execution engine selection and GPU availability detection (thread-safe).
    """

    _gpu_available: bool | None = None  # Cache GPU availability check
    _gpu_check_lock = threading.Lock()  # Lock for GPU availability check

    @classmethod
    def check_gpu_available(cls) -> bool:
        """
        Check if GPU acceleration is available (thread-safe, double-checked locking).

        Thread-safe: Yes (double-checked locking pattern)

        Returns:
            bool: True if GPU is available, False otherwise
        """
        # Fast path: already checked (no lock needed)
        if cls._gpu_available is not None:
            return cls._gpu_available

        # Slow path: need to check (acquire lock)
        with cls._gpu_check_lock:
            # Double-check inside lock (another thread may have already checked)
            if cls._gpu_available is not None:
                return cls._gpu_available

            # Perform actual GPU check
            try:
                import cudf
                import cupy as cp

                # Test GPU functionality (ensure it actually works)
                _ = cp.array([1, 2, 3])
                cls._gpu_available = True
            except (ImportError, Exception):
                cls._gpu_available = False

            return cls._gpu_available

    @classmethod
    def reset_gpu_cache(cls) -> None:
        """
        Reset the GPU availability cache (thread-safe).

        Thread-safe: Yes (uses lock)
        """
        with cls._gpu_check_lock:
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

        match engine:
            case "cpu":
                return "cpu"
            case "gpu":
                gpu_available = cls.check_gpu_available()
                if not gpu_available:
                    raise GPUNotAvailableError()
                return "gpu"
            case "auto":
                # Auto selection
                gpu_available = cls.check_gpu_available()
                if not gpu_available:
                    return "cpu"

                if operation and data_size is not None:
                    threshold = GPU_CROSSOVER_THRESHOLDS.get(operation, GPU_CROSSOVER_THRESHOLDS["default"])
                    return "gpu" if data_size >= threshold else "cpu"

                return "cpu"  # Conservative default for "auto"
            case _:
                # This should never be reached due to validation above
                raise ConfigurationError(f"Invalid engine: {engine!r}")

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
        # Minimum size for any GPU operation (linear algebra operations have lowest threshold)
        min_threshold = get_threshold("linear_algebra")
        if force_cpu or not cls.check_gpu_available() or data_size < min_threshold:
            return "cpu"

        for details in OPERATION_HEURISTICS.values():
            if operation in details["ops"]:
                return "gpu" if data_size >= details["threshold"] else "cpu"

        # Default for unknown operations
        default_threshold = get_threshold("default")
        return "gpu" if data_size >= default_threshold else "cpu"

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

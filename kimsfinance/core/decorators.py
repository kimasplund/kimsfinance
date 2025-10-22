"""
GPU Acceleration Decorator and Infrastructure
==============================================

Eliminates 70% code duplication in indicator implementations by providing:
- Automatic NumPy/CuPy array handling
- GPU availability checking and fallback
- Smart engine selection with size-based optimization
- Unified error handling

Usage:
    @gpu_accelerated(operation_type="rolling_window", min_gpu_size=100_000)
    def calculate_stochastic(high, low, close, k_period=14, *, engine="auto"):
        # Single implementation - decorator handles GPU/CPU switching
        highest_high = xp.max(...)  # xp = numpy or cupy
        return k_percent, d_percent

This reduces indicator code from 100 lines to 20-30 lines per indicator.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, ParamSpec, TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
    from cupy import ndarray as CupyArray

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    CupyArray = None

if TYPE_CHECKING:
    from types import ModuleType

from .types import ArrayLike, Engine
from .engine import EngineManager, GPUNotAvailableError
from ..config.gpu_thresholds import get_threshold


P = ParamSpec("P")
R = TypeVar("R")


def _to_numpy_array(data: ArrayLike) -> NDArray[np.float64]:
    """
    Convert array-like input to numpy array.

    Handles: np.ndarray, pd.Series, pd.DataFrame columns, pl.Series, lists.
    """
    if isinstance(data, np.ndarray):
        return data.astype(np.float64, copy=False)
    elif hasattr(data, "to_numpy"):  # pandas, polars
        arr = data.to_numpy()
        return np.asarray(arr, dtype=np.float64)
    elif hasattr(data, "values"):  # pandas
        return np.asarray(data.values, dtype=np.float64)
    else:
        return np.array(data, dtype=np.float64)


def gpu_accelerated(
    *,
    operation_type: str = "general",
    min_gpu_size: int | None = None,
    validate_inputs: bool = True,
    requires_volume: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for GPU-accelerated indicator calculations.

    Eliminates boilerplate by handling:
    - Array conversion and validation
    - Engine selection (CPU/GPU/auto)
    - GPU availability checking
    - Automatic NumPy/CuPy dispatch
    - Error handling and fallback

    Args:
        operation_type: Operation category for smart engine selection
                       ("rolling_window", "ewma", "cumulative", "general")
        min_gpu_size: Minimum dataset size for GPU benefit (overrides smart selection)
        validate_inputs: Whether to validate input array lengths (default: True)
        requires_volume: Whether indicator requires volume data (default: False)

    Returns:
        Decorated function with GPU acceleration

    Example:
        @gpu_accelerated(operation_type="rolling_window", min_gpu_size=100_000)
        def calculate_stochastic(high, low, close, k_period=14, d_period=3, *, engine="auto"):
            # xp is numpy or cupy depending on engine selection
            xp = get_array_module(high)

            # Single implementation works for both CPU and GPU
            highest_high = rolling_max(high, k_period)
            lowest_low = rolling_min(low, k_period)

            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d_percent = rolling_mean(k_percent, d_period)

            return k_percent, d_percent

    Reduces code from:
        - 100 lines (manual GPU/CPU branches, error handling, conversion)
        - To 20-30 lines (pure algorithm implementation)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Extract engine parameter (must be keyword-only)
            engine_raw = kwargs.get("engine", "auto")
            if not isinstance(engine_raw, str):
                raise TypeError(f"engine must be str, got {type(engine_raw)}")
            engine: Engine = cast(Engine, engine_raw)

            # Convert all array-like positional arguments to numpy
            converted_args: list[NDArray[np.float64] | object] = []
            array_lengths: list[int] = []

            for arg in args:
                # Check if this looks like array-like data
                if hasattr(arg, "__len__") and not isinstance(arg, str):
                    arr = _to_numpy_array(arg)
                    converted_args.append(arr)
                    array_lengths.append(len(arr))
                else:
                    # Non-array argument (e.g., period parameter)
                    converted_args.append(arg)

            # Validate array lengths match
            if validate_inputs and len(array_lengths) > 1:
                if not all(length == array_lengths[0] for length in array_lengths):
                    raise ValueError(
                        f"All input arrays must have same length. " f"Got lengths: {array_lengths}"
                    )

            # Determine data size for engine selection
            data_size = array_lengths[0] if array_lengths else 0

            # Validate minimum data requirements
            if data_size == 0:
                raise ValueError("Input arrays cannot be empty")

            # Smart engine selection with size-based override
            if engine == "auto":
                # Automatically select based on data size and GPU availability
                # Use provided threshold or get default from config
                threshold = min_gpu_size if min_gpu_size is not None else get_threshold("default")
                if data_size >= threshold and EngineManager.check_gpu_available():
                    exec_engine = "gpu"
                else:
                    exec_engine = "cpu"
            else:
                # Use explicit engine selection (cpu or gpu)
                exec_engine = EngineManager.select_engine(engine)

            # GPU execution
            if exec_engine == "gpu":
                if not CUPY_AVAILABLE or cp is None:
                    if engine == "gpu":
                        raise GPUNotAvailableError(
                            "CuPy not installed. Install with: pip install cupy-cuda12x"
                        )
                    # Fallback to CPU for auto mode
                    exec_engine = "cpu"
                else:
                    try:
                        # Transfer arrays to GPU
                        gpu_args: list[Any] = []
                        for arg in converted_args:
                            if isinstance(arg, np.ndarray):
                                gpu_args.append(cp.asarray(arg, dtype=cp.float64))
                            else:
                                gpu_args.append(arg)

                        # Call function with GPU arrays
                        result = func(*gpu_args, **kwargs)

                        # Transfer result(s) back to CPU
                        if isinstance(result, tuple):
                            cpu_results = tuple(
                                cp.asnumpy(r) if isinstance(r, cp.ndarray) else r for r in result
                            )
                            return cast(R, cpu_results)
                        elif cp is not None and isinstance(result, cp.ndarray):
                            return cast(R, cp.asnumpy(result))
                        else:
                            return result

                    except Exception as e:
                        if engine == "gpu":
                            raise GPUNotAvailableError(f"GPU operation failed: {e}")
                        # Fallback to CPU for auto mode
                        warnings.warn(
                            f"GPU execution failed, falling back to CPU: {e}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        exec_engine = "cpu"

            # CPU execution (or fallback)
            if exec_engine == "cpu":
                result = func(*converted_args, **kwargs)
                return result

            # Should never reach here
            raise RuntimeError(f"Invalid execution engine: {exec_engine}")

        return wrapper

    return decorator


def get_array_module(arr: NDArray[Any] | Any) -> type[np.ndarray] | ModuleType:
    """
    Get the appropriate array module (numpy or cupy) for an array.

    This allows writing generic code that works with both NumPy and CuPy:

    Example:
        xp = get_array_module(arr)
        result = xp.maximum(arr, 0)  # Works for both numpy and cupy

    Args:
        arr: Array to check

    Returns:
        numpy or cupy module
    """
    if CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp
    else:
        return np


def to_gpu(arr: NDArray[np.float64]) -> NDArray[np.float64] | Any:
    """
    Transfer numpy array to GPU if CuPy is available.

    Args:
        arr: NumPy array

    Returns:
        CuPy array if available, otherwise original NumPy array
    """
    if CUPY_AVAILABLE and cp is not None:
        return cp.asarray(arr, dtype=cp.float64)
    return arr


def to_cpu(arr: NDArray[Any] | Any) -> NDArray[np.float64]:
    """
    Transfer array to CPU (convert CuPy to NumPy if needed).

    Args:
        arr: NumPy or CuPy array

    Returns:
        NumPy array
    """
    if CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return cast(NDArray[np.float64], arr)


# Re-export for convenience
__all__ = [
    "gpu_accelerated",
    "get_array_module",
    "to_gpu",
    "to_cpu",
]

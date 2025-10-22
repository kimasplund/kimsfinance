"""
Type System for kimsfinance
==================================

Python 3.13+ type aliases and protocols for the library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypeGuard, Any

import numpy as np
from numpy.typing import NDArray
import polars as pl
import pandas as pd

# Import at module level to avoid circular import issues
from ..config.gpu_thresholds import get_threshold


# ============================================================================
# Result Types
# ============================================================================

type ArrayResult = NDArray[np.float64]
type SeriesResult = pl.Series
type DataFrameResult = pl.DataFrame


# ============================================================================
# Input Types
# ============================================================================

type DataFrameInput = pl.DataFrame | pl.LazyFrame | pd.DataFrame
type ArrayLike = NDArray[Any] | pl.Series | pd.Series | Sequence[float] | list[float]


# ============================================================================
# Parameter Types
# ============================================================================

type WindowSize = int | Sequence[int]
type ShiftPeriods = int | Sequence[int] | None
type Engine = Literal["cpu", "gpu", "auto"]


# ============================================================================
# Financial Data Types
# ============================================================================


class OHLCProtocol(Protocol):
    """Protocol for OHLC data structures."""

    def __getitem__(self, key: str) -> ArrayLike: ...
    def __len__(self) -> int: ...


# ============================================================================
# Operation Results
# ============================================================================

type MovingAverageResult = list[ArrayResult]
type BoundsResult = tuple[float, float]
type LinearFitResult = tuple[float, float]  # (slope, intercept)
type MACDResult = tuple[ArrayResult, ArrayResult, ArrayResult]  # (macd, signal, histogram)


# ============================================================================
# Configuration Types
# ============================================================================


class EngineConfig:
    """Engine configuration settings."""

    def __init__(
        self,
        engine: Engine = "auto",
        *,
        gpu_min_rows: int | None = None,
        gpu_operations: set[str] | None = None,
        fallback_on_error: bool = True,
    ):
        self.engine = engine
        # Use config threshold if not explicitly provided
        self.gpu_min_rows = gpu_min_rows if gpu_min_rows is not None else get_threshold("default")
        self.gpu_operations = gpu_operations or {
            "nanmin",
            "nanmax",
            "isnan",
            "least_squares",
            "atr",
            "volume_sum",
        }
        self.fallback_on_error = fallback_on_error


# ============================================================================
# Type Guards
# ============================================================================


def is_polars_dataframe(obj: object) -> TypeGuard[pl.DataFrame | pl.LazyFrame]:
    """Check if object is a Polars DataFrame or LazyFrame."""
    return isinstance(obj, (pl.DataFrame, pl.LazyFrame))


def is_pandas_dataframe(obj: object) -> TypeGuard[pd.DataFrame]:
    """Check if object is a pandas DataFrame."""
    return isinstance(obj, pd.DataFrame)


def is_array_like(
    obj: object,
) -> TypeGuard[NDArray[Any] | pl.Series | pd.Series | list[Any] | tuple[Any, ...]]:
    """Check if object is array-like."""
    return isinstance(obj, (np.ndarray, pl.Series, pd.Series, list, tuple))


def is_numpy_array(obj: object) -> TypeGuard[NDArray[Any]]:
    """Check if object is a NumPy array."""
    return isinstance(obj, np.ndarray)


def is_polars_series(obj: object) -> TypeGuard[pl.Series]:
    """Check if object is a Polars Series."""
    return isinstance(obj, pl.Series)


def is_pandas_series(obj: object) -> TypeGuard[pd.Series]:
    """Check if object is a pandas Series."""
    return isinstance(obj, pd.Series)

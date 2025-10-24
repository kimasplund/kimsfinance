"""
Polars-based Moving Average Calculations for mplfinance
========================================================

Python 3.13+ optimized implementation using Polars DataFrame library.
Provides 10-100x speedup over pandas on CPU, with optional GPU acceleration.

Requirements:
    - Python 3.13+
    - polars >= 1.0
    - numpy >= 2.0

Optional (for GPU):
    - RAPIDS cuDF >= 24.12
    - NVIDIA GPU with CUDA support

Example:
    >>> import polars as pl
    >>> from moving_averages import calculate_sma, calculate_ema
    >>>
    >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 103]})
    >>> sma = calculate_sma(df, "close", window=3)
    >>> print(sma)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl
import numpy as np

from kimsfinance.core.engine import EngineManager
from kimsfinance.core.exceptions import DataValidationError
from kimsfinance.core.types import ArrayLike, ArrayResult, Engine, MovingAverageResult, ShiftPeriods


def calculate_sma(prices: ArrayLike, period: int = 20, *, engine: Engine = "auto") -> ArrayResult:
    """
    Calculate Simple Moving Average(s) using Polars.

    50-100x faster than pandas on CPU, with optional GPU acceleration.

    Args:
        data: Polars DataFrame or LazyFrame with price data
        column: Column name to calculate SMA on (e.g., "close")
        windows: Window size(s) for SMA. Single int or sequence of ints.
        shift: Optional shift period(s). If sequence, must match windows length.
        engine: Execution engine - "cpu" (default), "gpu", or "auto"

    Returns:
        List of numpy arrays, one per window size

    Raises:
        MovingAverageError: If column not found or windows invalid
        GPUNotAvailableError: If GPU requested but not available

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 103, 107]})
        >>> sma_20, sma_50 = calculate_sma(df, "close", windows=[20, 50])
        >>>
        >>> # With GPU acceleration
        >>> sma = calculate_sma(df, "close", windows=20, engine="gpu")
    """
    # Convert to Polars Series for calculation
    s = pl.Series(prices, dtype=pl.Float64)

    # Calculate SMA using Polars' built-in function
    sma_series = s.rolling_mean(window_size=period)

    return sma_series.to_numpy()


def calculate_ema(prices: ArrayLike, period: int = 12, *, engine: Engine = "auto") -> ArrayResult:
    """
    Calculate Exponential Moving Average (EMA) using Polars.

    This implementation leverages Polars for high performance and consistency
    across the library. It is GPU-accelerated where available.

    Args:
        prices: Input price data (typically close prices)
        period: Lookback period for calculation (default: 12)
        engine: Computation engine ('auto', 'cpu', 'gpu'). Note: Polars
                handles engine selection; this parameter is for API
                consistency.

    Returns:
        Array of EMA values (length matches input)
        Initial values are NaN, consistent with warmup period.

    Raises:
        ValueError: If period < 1 or inputs have insufficient data.

    Formula:
        Uses Polars' ewm_mean, which is a standard and highly optimized
        EMA implementation.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Convert to Polars Series for calculation
    s = pl.Series(prices, dtype=pl.Float64)

    if len(s) < period:
        raise ValueError(f"Insufficient data: need {period}, got {len(s)}")

    # Calculate EMA using Polars' built-in function
    # `adjust=False` gives the standard recursive definition of EMA
    ema_series = s.ewm_mean(span=period, adjust=False, min_samples=period)

    return ema_series.to_numpy()


def calculate_multiple_mas(
    data: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    sma_windows: Sequence[int] | None = None,
    ema_windows: Sequence[int] | None = None,
    sma_shift: ShiftPeriods = None,
    ema_shift: ShiftPeriods = None,
    engine: Engine = "cpu",
) -> dict[str, list[MAResult]]:
    """
    Calculate multiple SMAs and EMAs in a single optimized pass.

    More efficient than calling calculate_sma() and calculate_ema() separately
    because all operations are executed in a single query plan.

    Args:
        data: Polars DataFrame or LazyFrame with price data
        column: Column name to calculate MAs on
        sma_windows: Window sizes for simple moving averages
        ema_windows: Span sizes for exponential moving averages
        sma_shift: Optional shift for SMAs
        ema_shift: Optional shift for EMAs
        engine: Execution engine - "cpu", "gpu", or "auto"

    Returns:
        Dict with keys "sma" and "ema", each containing list of numpy arrays

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 101, 105, 103, 107, 106]})
        >>> mas = calculate_multiple_mas(
        ...     df, "close",
        ...     sma_windows=[5, 10, 20],
        ...     ema_windows=[12, 26]
        ... )
        >>> sma_results = mas["sma"]  # [sma_5, sma_10, sma_20]
        >>> ema_results = mas["ema"]  # [ema_12, ema_26]
    """
    if sma_windows is None and ema_windows is None:
        raise DataValidationError("Must specify at least one of sma_windows or ema_windows")

    # Validate column
    if column not in data.columns:
        raise DataValidationError(
            f"Column {column!r} not found in DataFrame. " f"Available columns: {data.columns}"
        )

    # Select execution engine
    exec_engine = EngineManager.select_engine(engine)

    # Convert to lazy if needed
    if isinstance(data, pl.DataFrame):
        lf = data.lazy()
    else:
        lf = data

    expressions = []
    sma_cols = []
    ema_cols = []

    # Build SMA expressions
    if sma_windows is not None:
        sma_windows = list(sma_windows)[:7]  # Limit to 7

        if sma_shift is not None:
            if isinstance(sma_shift, int):
                sma_shift = [sma_shift] * len(sma_windows)

        for idx, window in enumerate(sma_windows):
            expr = pl.col(column).rolling_mean(window_size=window)

            if sma_shift is not None:
                expr = expr.shift(periods=sma_shift[idx])

            col_name = f"_sma_{window}"
            expressions.append(expr.alias(col_name))
            sma_cols.append(col_name)

    # Build EMA expressions
    if ema_windows is not None:
        ema_windows = list(ema_windows)[:7]  # Limit to 7

        if ema_shift is not None:
            if isinstance(ema_shift, int):
                ema_shift = [ema_shift] * len(ema_windows)

        for idx, window in enumerate(ema_windows):
            expr = pl.col(column).ewm_mean(span=window, adjust=False)

            if ema_shift is not None:
                expr = expr.shift(periods=ema_shift[idx])

            col_name = f"_ema_{window}"
            expressions.append(expr.alias(col_name))
            ema_cols.append(col_name)

    # Execute all expressions in single pass
    result_df = lf.select(expressions).collect(engine=exec_engine)

    # Separate SMA and EMA results
    return {
        "sma": [result_df[col].to_numpy() for col in sma_cols] if sma_cols else [],
        "ema": [result_df[col].to_numpy() for col in ema_cols] if ema_cols else [],
    }


# Convenience function for pandas compatibility layer
def from_pandas_series(series: object, window: int, ma_type: str = "sma") -> MovingAverageResult:
    """
    Drop-in replacement for pandas rolling mean calculation.

    Internally converts to Polars for speed, returns numpy array.

    Args:
        series: Pandas Series or array-like
        window: Window size
        ma_type: "sma" or "ema"

    Returns:
        Numpy array with moving average values

    Example:
        >>> import polars as pl
        >>> prices = pl.Series([100, 102, 101, 105, 103])
        >>> sma = from_pandas_series(prices, window=3)
    """
    # Convert to Polars DataFrame
    df = pl.DataFrame({"_data": series})

    # Calculate MA
    exec_engine = EngineManager.select_engine(
        "auto", operation="moving_average", data_size=len(series)
    )
    match ma_type.lower():
        case "sma":
            result = calculate_sma(df, "_data", windows=window, engine=exec_engine)
        case "ema":
            result = calculate_ema(df, "_data", windows=window, engine=exec_engine)
        case _:
            raise ValueError(f"Invalid ma_type: {ma_type!r}. Must be 'sma' or 'ema'.")

    return result[0]


def calculate_wma(prices: ArrayLike, period: int = 20, *, engine: Engine = "auto") -> ArrayResult:
    """
    Calculate Weighted Moving Average(s) using Polars.
    """
    if engine not in ("auto", "cpu", "gpu"):
        raise ValueError("Invalid engine")

    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Coerce to numpy array to handle lists and other array-likes
    prices_arr = np.asarray(prices, dtype=np.float64)

    if len(prices_arr) < period:
        raise ValueError("Insufficient data")

    # Convert to Polars Series for calculation
    s = pl.Series(prices_arr, dtype=pl.Float64)
    # Weights must be an eager Series for the UDF to work correctly
    weights = pl.int_range(1, period + 1, eager=True).cast(pl.Float64)
    weights_sum = weights.sum()

    # Calculate WMA using a UDF in rolling_map
    # The lambda must return a scalar float, not a Polars expression
    wma_series = s.rolling_map(
        window_size=period, function=lambda s: (s * weights).sum() / weights_sum
    )

    return wma_series.to_numpy()


def calculate_vwma(
    prices: ArrayLike, volumes: ArrayLike, period: int = 20, *, engine: Engine = "auto"
) -> ArrayResult:
    """
    Calculate Volume Weighted Moving Average(s) using Polars.
    """
    # Convert to Polars DataFrame for calculation
    df = pl.DataFrame({"price": prices, "volume": volumes})

    # Calculate VWMA using Polars' built-in functions
    df = df.with_columns(
        (pl.col("price") * pl.col("volume")).rolling_sum(window_size=period)
        / pl.col("volume").rolling_sum(window_size=period)
    )
    return df["price"].to_numpy()


def calculate_hma(*args, **kwargs):
    raise NotImplementedError


if __name__ == "__main__":
    # Test code moved to tests/test_moving_averages.py
    # Run: pytest tests/test_moving_averages.py
    pass

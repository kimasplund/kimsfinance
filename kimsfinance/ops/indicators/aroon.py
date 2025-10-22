from __future__ import annotations

import numpy as np
import polars as pl

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ...config.gpu_thresholds import get_threshold
from ...core import (
    ArrayLike,
    ArrayResult,
    DataFrameInput,
    MACDResult,
    Engine,
    EngineManager,
    GPUNotAvailableError,
)
from ...utils.array_utils import to_numpy_array


def calculate_aroon(
    highs: ArrayLike, lows: ArrayLike, period: int = 25, *, engine: Engine = "auto"
) -> tuple[ArrayResult, ArrayResult]:
    """
    Calculate Aroon Indicator (Aroon Up and Aroon Down).

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    The Aroon indicator measures the time since the highest high and lowest low
    over a given period. It is used to identify trend changes and strength.
    Values range from 0 to 100.

    Aroon Up measures the time since the period high, indicating uptrend strength.
    Aroon Down measures the time since the period low, indicating downtrend strength.

    Interpretation:
    - Aroon Up > 50 and Aroon Down < 50: Uptrend
    - Aroon Down > 50 and Aroon Up < 50: Downtrend
    - Both near 50: Consolidation or weak trend
    - Aroon Up crosses above Aroon Down: Bullish signal
    - Aroon Down crosses above Aroon Up: Bearish signal

    Args:
        highs: High prices
        lows: Low prices
        period: Lookback period for calculation (default: 25)
        engine: Computation engine ('auto', 'cpu', 'gpu')
            auto: Intelligently selects GPU for large datasets (>500K rows)

    Returns:
        Tuple of (aroon_up, aroon_down) arrays:
        - aroon_up: Time since period high, scaled 0-100
        - aroon_down: Time since period low, scaled 0-100
        First (period-1) values are NaN due to warmup

    Raises:
        ValueError: If period < 1 or inputs have mismatched lengths

    Formula:
        Aroon Up = ((period - periods since highest high) / period) * 100
        Aroon Down = ((period - periods since lowest low) / period) * 100

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> aroon_up, aroon_down = calculate_aroon(df['High'], df['Low'], period=25)
        >>> # Detect uptrend
        >>> uptrend = (aroon_up > 70) & (aroon_down < 30)

    Performance:
        < 500K rows: CPU optimal
        500K-1M rows: GPU beneficial
        1M+ rows: GPU strong benefit

    References:
        - https://www.investopedia.com/terms/a/aroon.asp
        - Developed by Tushar Chande in 1995
    """
    # Validate inputs
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Convert to numpy arrays
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)

    if len(highs_arr) != len(lows_arr):
        raise ValueError(
            f"highs and lows must have same length: got {len(highs_arr)} and {len(lows_arr)}"
        )

    if len(highs_arr) < period:
        raise ValueError(f"Insufficient data: need {period}, got {len(highs_arr)}")

    # Engine routing
    threshold = get_threshold("iterative")
    match engine:
        case "auto":
            use_gpu = len(highs_arr) >= threshold and CUPY_AVAILABLE
        case "gpu":
            use_gpu = CUPY_AVAILABLE
        case "cpu":
            use_gpu = False
        case _:
            raise ValueError(f"Invalid engine: {engine}")

    # Dispatch to CPU or GPU
    if use_gpu:
        return _calculate_aroon_gpu(highs_arr, lows_arr, period)
    else:
        return _calculate_aroon_cpu(highs_arr, lows_arr, period)


def _calculate_aroon_cpu(
    highs: np.ndarray, lows: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """CPU implementation of Aroon using NumPy (fully vectorized with sliding windows)."""

    n = len(highs)
    aroon_up = np.full(n, np.nan, dtype=np.float64)
    aroon_down = np.full(n, np.nan, dtype=np.float64)

    # Create rolling windows using stride tricks (zero-copy views)
    from numpy.lib.stride_tricks import sliding_window_view

    high_windows = sliding_window_view(highs, period)
    low_windows = sliding_window_view(lows, period)

    # Find maximum/minimum values in each window (vectorized)
    max_values = np.max(high_windows, axis=1)
    min_values = np.min(low_windows, axis=1)

    # Find the LAST (most recent) occurrence of max/min in each window
    # Strategy: reverse each window, find first occurrence, then convert back to original index
    # This is fully vectorized using broadcasting
    high_windows_reversed = high_windows[:, ::-1]
    low_windows_reversed = low_windows[:, ::-1]

    # Create boolean masks for matches
    high_matches = high_windows_reversed == max_values[:, np.newaxis]
    low_matches = low_windows_reversed == min_values[:, np.newaxis]

    # Find first True in each row (argmax on boolean returns first occurrence)
    reversed_max_indices = np.argmax(high_matches, axis=1)
    reversed_min_indices = np.argmax(low_matches, axis=1)

    # Convert reversed indices back to original window indices
    max_indices = period - 1 - reversed_max_indices
    min_indices = period - 1 - reversed_min_indices

    # Calculate periods since high/low
    periods_since_high = period - 1 - max_indices
    periods_since_low = period - 1 - min_indices

    # Calculate Aroon values (0-100) - fully vectorized
    aroon_up_values = ((period - periods_since_high) / period) * 100.0
    aroon_down_values = ((period - periods_since_low) / period) * 100.0

    # Place results in correct positions (starting at period-1)
    aroon_up[period - 1 :] = aroon_up_values
    aroon_down[period - 1 :] = aroon_down_values

    return (aroon_up, aroon_down)


def _calculate_aroon_gpu(
    highs: np.ndarray, lows: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """GPU implementation of Aroon using CuPy."""

    try:
        import cupy as cp
    except ImportError:
        # Fallback to CPU if CuPy not available
        return _calculate_aroon_cpu(highs, lows, period)

    # Transfer to GPU
    highs_gpu = cp.asarray(highs, dtype=cp.float64)
    lows_gpu = cp.asarray(lows, dtype=cp.float64)

    n = len(highs_gpu)
    aroon_up_gpu = cp.full(n, cp.nan, dtype=cp.float64)
    aroon_down_gpu = cp.full(n, cp.nan, dtype=cp.float64)

    # Calculate Aroon for each valid position
    # Note: This is still sequential on GPU due to argmax limitation
    # For large datasets, the memory bandwidth benefits still provide speedup
    for i in range(period - 1, n):
        window_start = i - period + 1
        high_window = highs_gpu[window_start : i + 1]
        low_window = lows_gpu[window_start : i + 1]

        # Find periods since highest high
        max_val = cp.max(high_window)
        periods_since_high = period - 1 - cp.where(high_window == max_val)[0][-1]

        # Find periods since lowest low
        min_val = cp.min(low_window)
        periods_since_low = period - 1 - cp.where(low_window == min_val)[0][-1]

        # Convert to Aroon values (0-100)
        aroon_up_gpu[i] = ((period - periods_since_high) / period) * 100.0
        aroon_down_gpu[i] = ((period - periods_since_low) / period) * 100.0

    # Transfer back to CPU
    return (cp.asnumpy(aroon_up_gpu), cp.asnumpy(aroon_down_gpu))

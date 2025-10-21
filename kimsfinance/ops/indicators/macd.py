from __future__ import annotations

import numpy as np
import polars as pl

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

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


def calculate_macd(
    prices: ArrayLike,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    engine: Engine = "auto",
) -> MACDResult:
    """
    GPU-accelerated MACD (Moving Average Convergence Divergence).
    Automatically uses GPU for datasets > 100,000 rows when engine='auto'.

    MACD is a trend-following momentum indicator.

    Args:
        prices: Price data (typically close prices)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        engine: Execution engine

    Returns:
        Tuple of (macd_line, signal_line, histogram)

    Formula:
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal

    Example:
        >>> prices = np.array([...])  # Close prices
        >>> macd, signal, histogram = calculate_macd(prices)
    """
    # Import EMA from moving_averages module
    from .moving_averages import calculate_ema
    import polars as pl

    prices_arr = to_numpy_array(prices)

    # Calculate fast and slow EMAs in one pass
    ema_fast = calculate_ema(
        prices_arr, period=fast_period, engine=engine
    )
    ema_slow = calculate_ema(
        prices_arr, period=slow_period, engine=engine
    )

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line (EMA of MACD)
    signal_line = calculate_ema(
        macd_line, period=signal_period, engine=engine
    )

    # Calculate histogram
    histogram = macd_line - signal_line

    return (macd_line, signal_line, histogram)

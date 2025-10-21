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
from .moving_averages import calculate_wma


def calculate_hma(
    prices: ArrayLike,
    period: int = 20,
    *,
    engine: Engine = "auto"
) -> ArrayResult:
    """
    Calculate Hull Moving Average (HMA).

    The Hull Moving Average (HMA) is an extremely responsive moving average with
    minimal lag.

    Formula:
        Half Period WMA = WMA(prices, period/2)
        Full Period WMA = WMA(prices, period)
        Raw HMA = 2 * Half Period WMA - Full Period WMA
        HMA = WMA(Raw HMA, sqrt(period))

    Args:
        prices: Input price data (typically close prices)
        period: Lookback period for calculation (default: 20)
        engine: Computation engine ('auto', 'cpu', 'gpu'). This is for
                API consistency and is passed to the underlying WMA function.

    Returns:
        Array of HMA values.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Calculate half period (integer division)
    half_period = period // 2

    # Calculate sqrt period (rounded to integer)
    sqrt_period = int(np.round(np.sqrt(period)))

    # Step 1: Calculate WMA with half period
    wma_half = calculate_wma(prices, period=half_period, engine=engine)

    # Step 2: Calculate WMA with full period
    wma_full = calculate_wma(prices, period=period, engine=engine)

    # Step 3: Calculate raw HMA = 2 * WMA(half) - WMA(full)
    raw_hma = 2.0 * wma_half - wma_full

    # Step 4: Calculate final HMA by applying WMA to raw HMA with sqrt(period)
    hma = calculate_wma(raw_hma, period=sqrt_period, engine=engine)

    return hma

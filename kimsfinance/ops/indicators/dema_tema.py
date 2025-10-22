from __future__ import annotations

import numpy as np
from ...core import (
    ArrayLike,
    ArrayResult,
    Engine,
)
from .moving_averages import calculate_ema


def calculate_dema(prices: ArrayLike, period: int = 20, *, engine: Engine = "auto") -> ArrayResult:
    """
    Calculate Double Exponential Moving Average (DEMA).

    DEMA is a faster-moving average that reduces lag by applying a double
    smoothing technique. It was developed by Patrick Mulloy and published
    in Technical Analysis of Stocks & Commodities magazine in 1994.

    Formula:
        DEMA = 2 * EMA - EMA(EMA)

    Args:
        prices: Input price data (typically close prices)
        period: Lookback period for EMA calculation (default: 20)
        engine: Computation engine ('auto', 'cpu', 'gpu'). This is for
                API consistency and is passed to the underlying EMA function.

    Returns:
        Array of DEMA values.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Calculate first EMA
    ema1 = calculate_ema(prices, period=period, engine=engine)

    # Calculate EMA of EMA
    ema2 = calculate_ema(ema1, period=period, engine=engine)

    # Calculate DEMA = 2 * EMA - EMA(EMA)
    dema = 2 * ema1 - ema2

    return dema


def calculate_tema(prices: ArrayLike, period: int = 20, *, engine: Engine = "auto") -> ArrayResult:
    """
    Calculate Triple Exponential Moving Average (TEMA).

    TEMA is an even faster-moving average that further reduces lag by applying
    triple smoothing.

    Formula:
        TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))

    Args:
        prices: Input price data (typically close prices)
        period: Lookback period for EMA calculation (default: 20)
        engine: Computation engine ('auto', 'cpu', 'gpu'). This is for
                API consistency and is passed to the underlying EMA function.

    Returns:
        Array of TEMA values.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Calculate first EMA
    ema1 = calculate_ema(prices, period=period, engine=engine)

    # Calculate EMA of EMA
    ema2 = calculate_ema(ema1, period=period, engine=engine)

    # Calculate EMA of EMA of EMA
    ema3 = calculate_ema(ema2, period=period, engine=engine)

    # Calculate TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    tema = 3 * ema1 - 3 * ema2 + ema3

    return tema

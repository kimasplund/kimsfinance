from __future__ import annotations

import numpy as np
from ...core import (
    ArrayLike,
    ArrayResult,
    Engine,
)
from .moving_averages import calculate_ema


def _to_numpy(arr):
    """Convert array to NumPy, handling CuPy arrays."""
    if hasattr(arr, "get"):  # CuPy array
        return arr.get()
    return np.asarray(arr)


def calculate_elder_ray(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 13,
    *,
    engine: Engine = "auto",
) -> tuple[ArrayResult, ArrayResult]:
    """
    Calculate Elder Ray (Bull Power and Bear Power).

    Elder Ray measures buying and selling pressure relative to an exponential
    moving average.

    Formula:
        EMA = Exponential Moving Average of close prices
        Bull Power = High - EMA
        Bear Power = Low - EMA

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: EMA period for calculation (default: 13)
        engine: Computation engine ('auto', 'cpu', 'gpu'). This is for
                API consistency and is passed to the underlying EMA function.

    Returns:
        Tuple of (bull_power, bear_power) as NumPy arrays
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    highs_arr = np.asarray(highs)
    lows_arr = np.asarray(lows)
    closes_arr = np.asarray(closes)

    if not (len(highs_arr) == len(lows_arr) == len(closes_arr)):
        raise ValueError("highs, lows, and closes must have same length")

    # Calculate EMA of close prices
    ema = calculate_ema(closes_arr, period=period, engine=engine)

    # Calculate Bull Power = High - EMA
    bull_power = highs_arr - ema

    # Calculate Bear Power = Low - EMA
    bear_power = lows_arr - ema

    # Convert back to NumPy if GPU was used
    bull_power = _to_numpy(bull_power)
    bear_power = _to_numpy(bear_power)

    return (bull_power, bear_power)

"""
ATR (Average True Range) - GPU-Accelerated Implementation
===========================================================

The ATR is a volatility indicator that shows how much an asset moves on average.
It is calculated using a smoothed average of the True Range values.

Formula:
    1. Calculate True Range (TR)
    2. Smooth TR using Wilder's smoothing (same as in ADX)

Interpretation:
    - High ATR: High volatility
    - Low ATR: Low volatility
    - Not directional, only measures magnitude of volatility
"""

from __future__ import annotations

from ..core import gpu_accelerated, ArrayLike, ArrayResult, Engine
from ..core.decorators import get_array_module
from .indicator_utils import true_range, _wilder_smoothing


@gpu_accelerated(operation_type="rolling_window", min_gpu_size=100_000)
def calculate_atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
    *,
    engine: Engine = "auto"
) -> ArrayResult:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period for smoothing (default: 14)
        engine: Execution engine ("cpu", "gpu", or "auto")

    Returns:
        ATR values as a NumPy or CuPy array.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    xp = get_array_module(high)

    # Calculate True Range
    tr = true_range(high, low, close)

    # Pad with NaN at the beginning to align with original data length
    tr_full = xp.concatenate([xp.array([xp.nan]), tr])

    # Smooth the True Range to get ATR
    atr = _wilder_smoothing(tr_full, period)

    return atr

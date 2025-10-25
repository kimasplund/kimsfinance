"""
Common Indicator Utility Functions
===================================

Reusable building blocks for technical indicators.

Eliminates code duplication by extracting common calculations:
- True Range (for ATR, ADX)
- Gain/Loss separation (for RSI, Stochastic RSI)
- Typical Price (for CCI, MFI)
- Money Flow (for MFI)
- Directional Movement (for ADX)

All functions support both NumPy and CuPy via array module polymorphism.
"""

from __future__ import annotations

import numpy as np
from typing import Any

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        """Fallback decorator when Numba is not available."""

        def decorator(func):  # type: ignore
            return func

        return decorator


from ..core.decorators import get_array_module


EPSILON = 1e-10
LAMBERT_CONSTANT = 0.015


def validate_period(period: int, name: str = "period") -> None:
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")


def validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def validate_non_negative(value: float | int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def true_range(
    high: np.ndarray | Any, low: np.ndarray | Any, close: np.ndarray | Any
) -> np.ndarray | Any:
    """
    Calculate True Range for volatility indicators.

    True Range is the greatest of:
    - Current High - Current Low
    - Abs(Current High - Previous Close)
    - Abs(Current Low - Previous Close)

    Used by: ATR, ADX, Supertrend

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        True Range values (length = len(input) - 1)

    Example:
        >>> highs = np.array([102, 105, 104, 107, 106])
        >>> lows = np.array([100, 101, 102, 104, 103])
        >>> closes = np.array([101, 103, 102, 106, 104])
        >>> tr = true_range(highs, lows, closes)
        >>> tr.shape
        (4,)  # One less than input (uses previous close)
    """
    xp = get_array_module(high)

    # Calculate three components of True Range
    high_low = high[1:] - low[1:]
    high_close_prev = xp.abs(high[1:] - close[:-1])
    low_close_prev = xp.abs(low[1:] - close[:-1])

    # True Range = max of the three
    tr = xp.maximum(high_low, xp.maximum(high_close_prev, low_close_prev))

    return tr


def gain_loss_separation(prices: np.ndarray | Any) -> tuple[np.ndarray | Any, np.ndarray | Any]:
    """
    Separate price changes into gains and losses.

    Used by: RSI, Stochastic RSI

    Args:
        prices: Price series

    Returns:
        Tuple of (gains, losses) where:
        - gains[i] = max(0, price[i] - price[i-1])
        - losses[i] = max(0, price[i-1] - price[i])
        Both arrays have length = len(prices) - 1

    Example:
        >>> prices = np.array([100, 102, 101, 105, 103])
        >>> gains, losses = gain_loss_separation(prices)
        >>> gains
        array([2., 0., 4., 0.])
        >>> losses
        array([0., 1., 0., 2.])
    """
    xp = get_array_module(prices)

    # Calculate price changes
    deltas = xp.diff(prices)

    # Separate into gains and losses
    gains = xp.where(deltas > 0, deltas, 0.0)
    losses = xp.where(deltas < 0, -deltas, 0.0)

    return gains, losses


def typical_price(
    high: np.ndarray | Any, low: np.ndarray | Any, close: np.ndarray | Any
) -> np.ndarray | Any:
    """
    Calculate Typical Price (HLC/3).

    Used by: CCI, MFI, VWAP

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Typical Price = (High + Low + Close) / 3

    Example:
        >>> highs = np.array([102, 105, 104])
        >>> lows = np.array([100, 101, 102])
        >>> closes = np.array([101, 103, 102])
        >>> tp = typical_price(highs, lows, closes)
        >>> tp
        array([101., 103., 102.67])
    """
    xp = get_array_module(high)
    return (high + low + close) / 3.0


def money_flow(
    high: np.ndarray | Any, low: np.ndarray | Any, close: np.ndarray | Any, volume: np.ndarray | Any
) -> np.ndarray | Any:
    """
    Calculate Raw Money Flow for volume indicators.

    Used by: MFI (Money Flow Index)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume

    Returns:
        Raw Money Flow = Typical Price * Volume

    Example:
        >>> highs = np.array([102, 105])
        >>> lows = np.array([100, 101])
        >>> closes = np.array([101, 103])
        >>> volumes = np.array([1000, 1500])
        >>> mf = money_flow(highs, lows, closes, volumes)
        >>> mf
        array([101000., 154500.])
    """
    tp = typical_price(high, low, close)
    return tp * volume


def positive_negative_money_flow(
    high: np.ndarray | Any, low: np.ndarray | Any, close: np.ndarray | Any, volume: np.ndarray | Any
) -> tuple[np.ndarray | Any, np.ndarray | Any]:
    """
    Separate money flow into positive and negative flows.

    Used by: MFI

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume

    Returns:
        Tuple of (positive_flow, negative_flow) where:
        - positive_flow[i] = money_flow[i] if tp[i] > tp[i-1] else 0
        - negative_flow[i] = money_flow[i] if tp[i] < tp[i-1] else 0

    Example:
        >>> highs = np.array([102, 105, 104])
        >>> lows = np.array([100, 101, 102])
        >>> closes = np.array([101, 103, 102])
        >>> volumes = np.array([1000, 1500, 1200])
        >>> pos, neg = positive_negative_money_flow(highs, lows, closes, volumes)
    """
    xp = get_array_module(high)

    # Calculate typical price and money flow
    tp = typical_price(high, low, close)
    mf = money_flow(high, low, close, volume)

    # Determine direction of typical price
    tp_direction = xp.diff(tp)

    # Separate into positive and negative flows
    # Note: Result is one element shorter (uses diff)
    positive_flow = xp.where(tp_direction > 0, mf[1:], 0.0)
    negative_flow = xp.where(tp_direction < 0, mf[1:], 0.0)

    return positive_flow, negative_flow


def directional_movement(
    high: np.ndarray | Any, low: np.ndarray | Any
) -> tuple[np.ndarray | Any, np.ndarray | Any]:
    """
    Calculate Directional Movement (+DM, -DM) for ADX.

    Used by: ADX (Average Directional Index)

    Args:
        high: High prices
        low: Low prices

    Returns:
        Tuple of (plus_dm, minus_dm) where:
        - plus_dm[i] = max(0, high[i] - high[i-1])
        - minus_dm[i] = max(0, low[i-1] - low[i])
        When both are positive, only the larger is kept

    Example:
        >>> highs = np.array([102, 105, 104, 107])
        >>> lows = np.array([100, 101, 102, 104])
        >>> plus_dm, minus_dm = directional_movement(highs, lows)
        >>> plus_dm
        array([3., 0., 3.])
        >>> minus_dm
        array([0., 0., 0.])
    """
    xp = get_array_module(high)

    # Calculate raw directional movements
    high_diff = xp.diff(high)
    low_diff = -xp.diff(low)  # Negative because we want low[i-1] - low[i]

    # Initialize +DM and -DM
    plus_dm = xp.where((high_diff > 0) & (high_diff > low_diff), high_diff, 0.0)

    minus_dm = xp.where((low_diff > 0) & (low_diff > high_diff), low_diff, 0.0)

    return plus_dm, minus_dm


def percentage_change(arr: np.ndarray | Any, periods: int = 1) -> np.ndarray | Any:
    """
    Calculate percentage change over N periods.

    Used by: ROC (Rate of Change)

    Args:
        arr: Input array
        periods: Number of periods for change calculation

    Returns:
        Percentage change = (arr[i] - arr[i-periods]) / arr[i-periods] * 100

    Example:
        >>> prices = np.array([100, 105, 102, 110])
        >>> pct_change = percentage_change(prices, periods=1)
        >>> pct_change
        array([nan, 5., -2.86, 7.84])
    """
    xp = get_array_module(arr)

    result = xp.full(len(arr), xp.nan, dtype=xp.float64)

    if len(arr) > periods:
        xp.copyto(result[periods:], (arr[periods:] - arr[:-periods]) / arr[:-periods] * 100.0)

    return result


def median_price(high: np.ndarray | Any, low: np.ndarray | Any) -> np.ndarray | Any:
    """
    Calculate Median Price (HL/2).

    Used by: Some VWAP variants, Ichimoku

    Args:
        high: High prices
        low: Low prices

    Returns:
        Median Price = (High + Low) / 2

    Example:
        >>> highs = np.array([102, 105, 104])
        >>> lows = np.array([100, 101, 102])
        >>> mp = median_price(highs, lows)
        >>> mp
        array([101., 103., 103.])
    """
    return (high + low) / 2.0


def weighted_close(
    high: np.ndarray | Any, low: np.ndarray | Any, close: np.ndarray | Any
) -> np.ndarray | Any:
    """
    Calculate Weighted Close (HLCC/4).

    Used by: Some oscillators

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Weighted Close = (High + Low + 2*Close) / 4

    Example:
        >>> highs = np.array([102, 105, 104])
        >>> lows = np.array([100, 101, 102])
        >>> closes = np.array([101, 103, 102])
        >>> wc = weighted_close(highs, lows, closes)
        >>> wc
        array([101., 103., 102.5])
    """
    return (high + low + 2 * close) / 4.0


@njit(cache=True, fastmath=True)
def _wilder_smoothing_jit(arr: np.ndarray, period: int, alpha: float) -> np.ndarray:
    """
    JIT-compiled Wilder's smoothing for CPU path.

    Provides 5-15x speedup over pure NumPy implementation by eliminating
    Python loop overhead in the exponential smoothing calculation.

    Args:
        arr: Input array (NumPy only)
        period: Smoothing period
        alpha: Pre-computed alpha = 1.0 / period

    Returns:
        Smoothed array
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    # Initialize with mean of first period values (handle NaN)
    sum_val = 0.0
    count = 0
    for i in range(period):
        if not np.isnan(arr[i]):
            sum_val += arr[i]
            count += 1

    if count > 0:
        result[period - 1] = sum_val / count
    else:
        return result

    # Apply exponential smoothing
    for i in range(period, n):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]  # Carry forward previous value
        else:
            result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]

    return result


# Re-export for convenience
def _wilder_smoothing(arr: np.ndarray | Any, period: int) -> np.ndarray | Any:
    """
    Apply Wilder's smoothing (EWM with alpha=1/period).

    Handles NaN values in initialization by using nanmean.

    Args:
        arr: Input array (may contain NaN at start)
        period: Smoothing period

    Returns:
        Smoothed array
    """
    xp = get_array_module(arr)

    # Fast path: Use JIT for CPU NumPy arrays
    if NUMBA_AVAILABLE and isinstance(arr, np.ndarray):
        alpha = 1.0 / period
        return _wilder_smoothing_jit(arr, period, alpha)

    # Slow path: Generic implementation for CuPy or non-Numba
    result = xp.full(len(arr), xp.nan, dtype=xp.float64)

    if len(arr) < period:
        return result

    # Wilder's alpha
    alpha = 1.0 / period

    # Initialize with nanmean of first period values (skipping NaN)
    if xp.__name__ == "cupy":
        # CuPy doesn't have nanmean, so convert to numpy temporarily
        import numpy as np_module

        first_valid = np_module.nanmean(xp.asnumpy(arr[:period]))
        result[period - 1] = first_valid
    else:
        result[period - 1] = xp.nanmean(arr[:period])

    # Apply exponential smoothing
    for i in range(period, len(arr)):
        if xp.isnan(arr[i]):
            result[i] = result[i - 1]  # Carry forward previous value
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]

    return result


__all__ = [
    "EPSILON",
    "LAMBERT_CONSTANT",
    "validate_period",
    "validate_positive",
    "validate_non_negative",
    "true_range",
    "gain_loss_separation",
    "typical_price",
    "money_flow",
    "positive_negative_money_flow",
    "directional_movement",
    "percentage_change",
    "median_price",
    "weighted_close",
    "_wilder_smoothing",
]

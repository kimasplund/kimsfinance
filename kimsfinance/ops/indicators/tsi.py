from __future__ import annotations

import numpy as np

from ...core import (
    ArrayLike,
    ArrayResult,
    Engine,
)
from .moving_averages import calculate_ema


def calculate_tsi(
    prices: ArrayLike, long_period: int = 25, short_period: int = 13, *, engine: Engine = "auto"
) -> ArrayResult:
    """
    Calculate True Strength Index (TSI).

    This implementation uses a high-performance, Polars-based EMA calculation
    that can be optionally GPU-accelerated.

    TSI is a double-smoothed momentum oscillator that shows both trend direction
    and overbought/oversold conditions. It uses double exponential smoothing of
    price momentum to filter out noise and provide clearer signals than traditional
    momentum indicators.

    The TSI oscillates around zero, where:
    - Positive values indicate bullish momentum
    - Negative values indicate bearish momentum
    - Zero-line crosses signal potential trend changes
    - Extreme readings suggest overbought/oversold conditions

    Formula:
        PC = Price Change = Close - Close[1]
        Double Smoothed PC = EMA(EMA(PC, long_period), short_period)
        Double Smoothed |PC| = EMA(EMA(|PC|, long_period), short_period)
        TSI = 100 * (Double Smoothed PC / Double Smoothed |PC|)

    Args:
        prices: Input price data (typically close prices)
        long_period: Long EMA period for first smoothing (default: 25)
        short_period: Short EMA period for second smoothing (default: 13)
        engine: Computation engine ('auto', 'cpu', 'gpu'). Passed to the
                underlying EMA calculation.

    Returns:
        Array of TSI values (range: -100 to +100)
        Initial values are NaN due to the double smoothing warmup period.

    Raises:
        ValueError: If long_period < 1 or short_period < 1 or insufficient data
    """
    # 1. VALIDATE INPUTS
    if long_period < 1:
        raise ValueError(f"long_period must be >= 1, got {long_period}")
    if short_period < 1:
        raise ValueError(f"short_period must be >= 1, got {short_period}")

    # 2. CONVERT TO NUMPY (standardize input)
    data_array = np.asarray(prices, dtype=np.float64)

    # The EMA function itself will raise a more specific error if needed.
    min_required = long_period + short_period
    if len(data_array) < min_required:
        raise ValueError(f"Insufficient data: need at least {min_required}, got {len(data_array)}")

    # 3. CALCULATE PRICE CHANGES
    # Calculate price changes (no prepend - first value will be NaN in result)
    price_change_raw = np.diff(data_array)
    abs_price_change_raw = np.abs(price_change_raw)

    # 4. FIRST SMOOTHING (long period EMA)
    # The `engine` parameter is passed down to the high-performance EMA function.
    smoothed_pc = calculate_ema(price_change_raw, period=long_period, engine=engine)
    smoothed_abs_pc = calculate_ema(abs_price_change_raw, period=long_period, engine=engine)

    # 5. SECOND SMOOTHING (short period EMA)
    # Note: The EMA function has issues with leading NaN values
    # We need to find the first valid index and only process from there
    first_valid_idx = np.where(~np.isnan(smoothed_pc))[0]

    if len(first_valid_idx) == 0:
        # No valid values after first smoothing, return all NaN
        return np.full_like(data_array, np.nan)

    first_valid = first_valid_idx[0]

    # Extract valid portions for second smoothing
    smoothed_pc_valid = smoothed_pc[first_valid:]
    smoothed_abs_pc_valid = smoothed_abs_pc[first_valid:]

    # Apply second smoothing
    double_smoothed_pc_valid = calculate_ema(smoothed_pc_valid, period=short_period, engine=engine)
    double_smoothed_abs_pc_valid = calculate_ema(smoothed_abs_pc_valid, period=short_period, engine=engine)

    # 6. CALCULATE TSI for valid portion
    tsi_valid = 100.0 * (double_smoothed_pc_valid / (double_smoothed_abs_pc_valid + 1e-10))

    # 7. RECONSTRUCT FULL TSI ARRAY
    # Prepend NaN for: first price (diff loses 1) + warmup period
    tsi = np.full(len(data_array), np.nan)
    tsi[first_valid + 1:first_valid + 1 + len(tsi_valid)] = tsi_valid

    return tsi

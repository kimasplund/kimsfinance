"""Average Directional Index (ADX) - Trend strength indicator.

The ADX measures the strength of a trend, regardless of direction.
It's part of the Directional Movement System which includes +DI and -DI.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Tuple

try:
    import cupy as cp  # noqa: F401  # availability probe

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ...core import (
    ArrayLike,
    ArrayResult,
    Engine,
    EngineManager,
)
from ...utils.array_utils import to_numpy_array


def calculate_adx(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
    *,
    engine: Engine = "auto",
) -> Tuple[ArrayResult, ArrayResult, ArrayResult]:
    """
    GPU-accelerated Average Directional Index (ADX) calculation.

    Automatically uses GPU for datasets > 100,000 rows when engine="auto".

    ADX measures trend strength (not direction). It's derived from the
    Directional Movement System which also produces +DI and -DI indicators
    that show trend direction.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ADX period (default: 14)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>100K rows)

    Returns:
        Tuple of (adx, plus_di, minus_di) arrays:
        - adx: Trend strength indicator (0-100 range)
        - plus_di: Positive directional indicator
        - minus_di: Negative directional indicator

    Formula:
        1. +DM = High[i] - High[i-1] (if positive and > -DM, else 0)
        2. -DM = Low[i-1] - Low[i] (if positive and > +DM, else 0)
        3. TR = max(High - Low, |High - Close[prev]|, |Low - Close[prev]|)
        4. +DI = 100 × Smoothed +DM / Smoothed TR
        5. -DI = 100 × Smoothed -DM / Smoothed TR
        6. DX = 100 × |+DI - -DI| / (+DI + -DI)
        7. ADX = Smoothed DX (typically 14-period)

    Interpretation:
        ADX Values:
        - < 25: Weak or absent trend
        - 25-50: Strong trend
        - > 50: Very strong trend

        Directional Indicators:
        - +DI > -DI: Uptrend
        - +DI < -DI: Downtrend
        - +DI crossing above -DI: Potential buy signal
        - +DI crossing below -DI: Potential sell signal

    Example:
        >>> highs = np.array([102, 105, 104, 107, 106])
        >>> lows = np.array([100, 101, 102, 104, 103])
        >>> closes = np.array([101, 103, 102, 106, 104])
        >>> adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)
        >>> # adx > 25: strong trend
        >>> # plus_di > minus_di: uptrend

    Performance:
        < 100K rows: CPU optimal (1-5ms)
        100K-1M rows: GPU beneficial (potential speedup)
        1M+ rows: GPU strong benefit

    Note:
        ADX requires 2 × period - 1 bars before producing valid values.
        Initial values will be NaN.
    """
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)

    if period < 1:
        raise ValueError("period must be >= 1")

    if not (len(highs_arr) == len(lows_arr) == len(closes_arr)):
        raise ValueError("highs, lows, and closes must have same length")

    # Need at least `period` bars to compute. With fewer than 2*period bars the
    # result is simply an all-NaN warmup window rather than an error.
    if len(highs_arr) < period:
        raise ValueError(
            f"Data length ({len(highs_arr)}) must be >= period ({period}) " f"for ADX calculation"
        )

    # Create Polars DataFrame for calculation
    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
        }
    )

    # Calculate directional movement
    # +DM: positive when current high > previous high
    # -DM: positive when previous low > current low
    high_diff = pl.col("high").diff()
    low_diff = -pl.col("low").diff()  # Negative because we want prev_low - curr_low

    # +DM is high_diff when it's positive and greater than low_diff
    plus_dm = pl.when((high_diff > 0) & (high_diff > low_diff)).then(high_diff).otherwise(0)

    # -DM is low_diff when it's positive and greater than high_diff
    minus_dm = pl.when((low_diff > 0) & (low_diff > high_diff)).then(low_diff).otherwise(0)

    # Calculate True Range
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )

    df = df.with_columns(
        [
            plus_dm.alias("plus_dm"),
            minus_dm.alias("minus_dm"),
            tr.alias("tr"),
        ]
    )

    # Apply Wilder's smoothing (EMA with span = 2 * period - 1)
    # This is the standard smoothing method for ADX
    span = 2 * period - 1

    smooth_plus_dm = pl.col("plus_dm").ewm_mean(span=span, adjust=False)
    smooth_minus_dm = pl.col("minus_dm").ewm_mean(span=span, adjust=False)
    smooth_tr = pl.col("tr").ewm_mean(span=span, adjust=False)

    # Calculate directional indicators
    # Add small epsilon to avoid division by zero
    plus_di = (100 * smooth_plus_dm / (smooth_tr + 1e-10)).alias("plus_di")
    minus_di = (100 * smooth_minus_dm / (smooth_tr + 1e-10)).alias("minus_di")

    df = df.lazy().select([plus_di, minus_di])

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="adx_di", data_size=len(highs_arr)
    )
    if polars_engine is not None:
        df = df.collect(engine=polars_engine)
    else:
        df = df.collect()

    # Calculate DX (Directional Index)
    # DX = 100 × |+DI - -DI| / (+DI + -DI)
    dx_df = pl.DataFrame(
        {
            "plus_di": df["plus_di"].to_numpy(),
            "minus_di": df["minus_di"].to_numpy(),
        }
    )

    di_sum = pl.col("plus_di") + pl.col("minus_di")
    di_diff = (pl.col("plus_di") - pl.col("minus_di")).abs()

    # Calculate DX with protection against division by zero
    dx = (100 * di_diff / (di_sum + 1e-10)).alias("dx")

    dx_result = dx_df.lazy().select(dx)

    # Execute DX calculation with selected engine
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="adx_dx", data_size=len(highs_arr)
    )
    if polars_engine is not None:
        dx_result = dx_result.collect(engine=polars_engine)
    else:
        dx_result = dx_result.collect()

    # Calculate ADX (smoothed DX)
    # ADX is the exponential moving average of DX
    adx_df = pl.DataFrame({"dx": dx_result["dx"].to_numpy()})

    adx_expr = pl.col("dx").ewm_mean(span=span, adjust=False).alias("adx")

    adx_result = adx_df.lazy().select(adx_expr)

    # Execute ADX calculation with selected engine
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="adx", data_size=len(highs_arr)
    )
    if polars_engine is not None:
        adx_result = adx_result.collect(engine=polars_engine)
    else:
        adx_result = adx_result.collect()

    adx_out = adx_result["adx"].to_numpy().astype(float)
    plus_di_out = df["plus_di"].to_numpy().astype(float)
    minus_di_out = df["minus_di"].to_numpy().astype(float)

    # Mask the warmup window with NaN. +DI/-DI need `period` bars for the first
    # Wilder smoothing; ADX needs a further `period` bars (2*period total).
    # Without this the EMA seed leaks pseudo-values into the warmup region,
    # contradicting the documented "initial values will be NaN" contract.
    di_warmup = min(period, len(plus_di_out))
    adx_warmup = min(2 * period, len(adx_out))
    plus_di_out[:di_warmup] = np.nan
    minus_di_out[:di_warmup] = np.nan
    adx_out[:adx_warmup] = np.nan

    return adx_out, plus_di_out, minus_di_out

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


def calculate_cmf(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    volumes: ArrayLike,
    period: int = 20,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    Calculate Chaikin Money Flow (CMF).

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    Chaikin Money Flow is a volume-weighted average of accumulation and distribution
    over a specified period. It measures the amount of Money Flow Volume over a
    specific period, providing insight into buying and selling pressure.

    CMF oscillates between -1 and +1, where:
    - Positive values indicate buying pressure (accumulation)
    - Negative values indicate selling pressure (distribution)
    - Values near 0 indicate equilibrium

    Formula:
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = Money Flow Multiplier * Volume
        CMF = Sum(Money Flow Volume, period) / Sum(Volume, period)

    Common usage:
        - CMF > 0: Buying pressure dominates (bullish)
        - CMF < 0: Selling pressure dominates (bearish)
        - CMF crossing 0: Potential trend change
        - Divergences with price: Potential reversal signals

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        period: Lookback period for Money Flow calculation (default: 20)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>500K rows)

    Returns:
        Array of CMF values (range: -1 to +1)
        First (period-1) values are NaN due to warmup

    Raises:
        ValueError: If period < 1 or inputs have mismatched lengths

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> cmf = calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'], period=20)

        >>> # Detect buying/selling pressure
        >>> buying_pressure = cmf > 0
        >>> selling_pressure = cmf < 0

        >>> # Detect strong signals
        >>> strong_buying = cmf > 0.25
        >>> strong_selling = cmf < -0.25

    References:
        - https://en.wikipedia.org/wiki/Money_flow_index
        - Marc Chaikin, developer of the indicator
        - https://www.investopedia.com/terms/c/chaikinoscillator.asp

    Performance:
        < 500K rows: CPU optimal
        500K-1M rows: GPU beneficial (1.2-1.5x speedup)
        1M+ rows: GPU strong benefit (up to 2.0x speedup)
    """
    # Validate inputs
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Convert to numpy arrays
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)
    volumes_arr = to_numpy_array(volumes)

    # Validate array lengths
    if not (len(highs_arr) == len(lows_arr) == len(closes_arr) == len(volumes_arr)):
        raise ValueError("highs, lows, closes, and volumes must have same length")

    if len(closes_arr) < period:
        raise ValueError(f"Insufficient data: need {period}, got {len(closes_arr)}")

    # Create Polars DataFrame
    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
            "volume": volumes_arr,
        }
    )

    # Select execution engine for Polars
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="cmf", data_size=len(closes_arr)
    )

    # Calculate Money Flow Multiplier
    # MF Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    # Simplified: = (2*Close - High - Low) / (High - Low)
    mf_multiplier = (2 * pl.col("close") - pl.col("high") - pl.col("low")) / (
        pl.col("high") - pl.col("low") + 1e-10
    )  # Add small epsilon to avoid division by zero

    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * pl.col("volume")

    # Calculate CMF
    # CMF = Sum(MF Volume, period) / Sum(Volume, period)
    cmf_expr = mf_volume.rolling_sum(window_size=period) / (
        pl.col("volume").rolling_sum(window_size=period) + 1e-10
    )

    # Execute calculation
    result = df.lazy().select(cmf=cmf_expr).collect(engine=polars_engine)

    return result["cmf"].to_numpy()

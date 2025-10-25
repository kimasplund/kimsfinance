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


def calculate_williams_r(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    GPU-accelerated Williams %R.
    This is a momentum indicator that is the inverse of the Stochastic Oscillator.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Lookback period (default: 14)
        engine: Execution engine ("cpu", "gpu", "auto")

    Returns:
        Array of Williams %R values (range: -100 to 0)
    """
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)

    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
        }
    )

    # Calculate highest high and lowest low over the period
    highest_high = pl.col("high").rolling_max(window_size=period)
    lowest_low = pl.col("low").rolling_min(window_size=period)

    # Calculate Williams %R
    wr_expr = -100 * ((highest_high - pl.col("close")) / (highest_high - lowest_low + 1e-10))

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="williams_r", data_size=len(highs_arr)
    )
    result = df.lazy().select(wr=wr_expr).collect(engine=polars_engine)

    return result["wr"].to_numpy()

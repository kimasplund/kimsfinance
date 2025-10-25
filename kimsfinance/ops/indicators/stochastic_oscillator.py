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


def calculate_stochastic_oscillator(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
    *,
    engine: Engine = "auto",
) -> tuple[ArrayResult, ArrayResult]:
    """
    Calculate the Stochastic Oscillator.

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    Compares a closing price to its price range over a period.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Oscillator period (default: 14)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>500K rows)

    Returns:
        Tuple of (%K, %D)

    Formula:
        %K = 100 * (close - low(period)) / (high(period) - low(period))
        %D = SMA(%K, 3)

    Performance:
        < 500K rows: CPU optimal
        500K-1M rows: GPU beneficial (1.1x speedup)
        1M+ rows: GPU strong benefit (up to 2.9x speedup)
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

    # Calculate rolling high and low
    rolling_low = pl.col("low").rolling_min(window_size=period)
    rolling_high = pl.col("high").rolling_max(window_size=period)

    # Calculate %K
    k_percent = 100 * ((pl.col("close") - rolling_low) / (rolling_high - rolling_low + 1e-10))

    # Calculate %D (3-period SMA of %K)
    d_percent = k_percent.rolling_mean(window_size=3)

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="stochastic", data_size=len(highs_arr)
    )
    result = df.lazy().select(k=k_percent, d=d_percent).collect(engine=polars_engine)

    return (result["k"].to_numpy(), result["d"].to_numpy())

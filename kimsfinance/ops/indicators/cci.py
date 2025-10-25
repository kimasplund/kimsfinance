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


def calculate_cci(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 20,
    constant: float = 0.015,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    GPU-accelerated Commodity Channel Index (CCI).
    Measures the deviation of a security's price from its statistical mean.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Lookback period (default: 20)
        constant: Lambert's constant (default: 0.015)
        engine: Execution engine ("cpu", "gpu", "auto")

    Returns:
        Array of CCI values
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

    # Calculate Typical Price
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3

    # Calculate SMA of Typical Price
    sma_tp = tp.rolling_mean(window_size=period)

    # Calculate Mean Deviation
    mean_deviation = (tp - sma_tp).abs().rolling_mean(window_size=period)

    # Calculate CCI
    cci_expr = (tp - sma_tp) / (constant * mean_deviation + 1e-10)

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="cci", data_size=len(highs_arr)
    )
    result = df.lazy().select(cci=cci_expr).collect(engine=polars_engine)

    return result["cci"].to_numpy()

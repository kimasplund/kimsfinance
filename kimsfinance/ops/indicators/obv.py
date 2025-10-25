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


def calculate_obv(closes: ArrayLike, volumes: ArrayLike, *, engine: Engine = "auto") -> ArrayResult:
    """
    Calculate On-Balance Volume (OBV).
    Automatically uses GPU for datasets > 100,000 rows when engine='auto'.

    Relates price and volume to identify momentum.

    Args:
        closes: Close prices
        volumes: Volume data
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>100K rows)

    Returns:
        Array of OBV values
    """
    closes_arr = to_numpy_array(closes)
    volumes_arr = to_numpy_array(volumes)

    df = pl.DataFrame(
        {
            "close": closes_arr,
            "volume": volumes_arr,
        }
    )

    # Determine direction of price change
    price_change = pl.col("close").diff()

    # Calculate OBV
    obv = (
        pl.when(price_change > 0)
        .then(pl.col("volume"))
        .when(price_change < 0)
        .then(-pl.col("volume"))
        .otherwise(0)
        .cum_sum()
        .alias("obv")
    )

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="obv", data_size=len(closes_arr)
    )
    result = df.lazy().select(obv).collect(engine=polars_engine)

    return result["obv"].to_numpy()

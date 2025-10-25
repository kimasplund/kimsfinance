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


def calculate_vwap(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    volumes: ArrayLike,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    GPU-accelerated Volume Weighted Average Price (VWAP).
    Automatically uses GPU for datasets > 100,000 rows when engine='auto'.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        engine: Execution engine ("cpu", "gpu", "auto")

    Returns:
        Array of VWAP values
    """
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)
    volumes_arr = to_numpy_array(volumes)

    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
            "volume": volumes_arr,
        }
    )

    # Calculate Typical Price and cumulative sums
    vwap_expr = (
        (pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")
    ).cum_sum() / pl.col("volume").cum_sum()

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="vwap", data_size=len(highs_arr)
    )
    result = df.lazy().select(vwap=vwap_expr).collect(engine=polars_engine)

    return result["vwap"].to_numpy()


def calculate_vwap_anchored(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    volumes: ArrayLike,
    anchor_indices: ArrayLike,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    GPU-accelerated Anchored VWAP.
    Resets VWAP calculation at specified anchor points (e.g., session start).

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        anchor_indices: Boolean array indicating reset points (True = reset)
        engine: Execution engine ("cpu", "gpu", "auto")

    Returns:
        Array of anchored VWAP values
    """
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)
    volumes_arr = to_numpy_array(volumes)
    anchors_arr = to_numpy_array(anchor_indices)

    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
            "volume": volumes_arr,
            "anchor": anchors_arr,
        }
    )

    # Create a 'session_id' that increments at each anchor point
    session_id = pl.col("anchor").cum_sum()

    # Calculate VWAP within each session
    vwap_expr = (
        (pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")
    ).cum_sum().over(session_id) / pl.col("volume").cum_sum().over(session_id)

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="vwap_anchored", data_size=len(highs_arr)
    )
    result = df.lazy().select(anchored_vwap=vwap_expr).collect(engine=polars_engine)

    return result["anchored_vwap"].to_numpy()

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


def calculate_atr(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    GPU-accelerated Average True Range (ATR) calculation.

    Automatically uses GPU for datasets > 100,000 rows when engine="auto".

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR period (default: 14)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>100K rows)

    Returns:
        Array of ATR values (same length as input)

    Formula:
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = Wilder's smoothing of TR over period

    Example:
        >>> highs = np.array([102, 105, 104, 107, 106])
        >>> lows = np.array([100, 101, 102, 104, 103])
        >>> closes = np.array([101, 103, 102, 106, 104])
        >>> atr = calculate_atr(highs, lows, closes, period=3)

    Performance:
        < 100K rows: CPU optimal (0.5-3ms)
        100K-1M rows: GPU beneficial (1.1-1.3x speedup)
        1M+ rows: GPU strong benefit (up to 1.5x speedup)

    Target in mplfinance:
        _utils.py:116-134 - Original implementation uses Python loop
    """
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)

    if not (len(highs_arr) == len(lows_arr) == len(closes_arr)):
        raise ValueError("highs, lows, and closes must have same length")

    if len(highs_arr) < period:
        raise ValueError(f"Data length ({len(highs_arr)}) must be >= period ({period})")

    # Create Polars DataFrame for calculation
    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
        }
    )

    # Calculate True Range using Polars expressions
    df = df.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("tr")
    )

    # Wilder's smoothing is EMA with alpha = 1 / period
    # span = 2 * period - 1
    atr_expr = pl.col("tr").ewm_mean(span=2 * period - 1, adjust=False)

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="atr", data_size=len(highs_arr)
    )
    result = df.lazy().select(atr=atr_expr).collect(engine=polars_engine)

    return result["atr"].to_numpy()

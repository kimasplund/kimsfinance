from __future__ import annotations

import numpy as np
import polars as pl

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ...config.gpu_thresholds import get_threshold
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


def calculate_rsi(prices: ArrayLike, period: int = 14, *, engine: Engine = "auto") -> ArrayResult:
    """
    GPU-accelerated Relative Strength Index (RSI).

    Automatically uses GPU for datasets > 100,000 rows when engine="auto".

    RSI is a momentum oscillator that measures speed and magnitude
    of price changes.

    Args:
        prices: Price data (typically close prices)
        period: RSI period (default: 14)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>100K rows)

    Returns:
        Array of RSI values (0-100 range)

    Formula:
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

    Performance:
        < 100K rows: CPU optimal
        100K-1M rows: GPU beneficial (up to 2.1x speedup)
        1M+ rows: GPU strong benefit

    Example:
        >>> prices = np.array([100, 102, 101, 105, 103, 107, 106])
        >>> rsi = calculate_rsi(prices, period=3)
    """
    prices_arr = to_numpy_array(prices)

    if len(prices_arr) < period + 1:
        raise ValueError(f"Data length must be > period ({period})")

    df = pl.DataFrame({"price": prices_arr})

    # Calculate price changes
    delta = pl.col("price").diff()

    # Separate gains and losses
    df = df.with_columns(
        gain=pl.when(delta > 0).then(delta).otherwise(0),
        loss=pl.when(delta < 0).then(-delta).otherwise(0),
    )

    # Wilder's smoothing for average gain/loss
    # span = 2 * period - 1
    avg_gain = pl.col("gain").ewm_mean(span=2 * period - 1, adjust=False)
    avg_loss = pl.col("loss").ewm_mean(span=2 * period - 1, adjust=False)

    # Calculate RS and RSI
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_expr = (100 - (100 / (1 + rs))).alias("rsi")

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="rsi", data_size=len(prices_arr)
    )
    result = df.lazy().select(rsi=rsi_expr).collect(engine=polars_engine)

    return result["rsi"].to_numpy()

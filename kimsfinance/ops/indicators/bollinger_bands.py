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


def calculate_bollinger_bands(
    prices: ArrayLike, period: int = 20, num_std: float = 2.0, *, engine: Engine = "auto"
) -> tuple[ArrayResult, ArrayResult, ArrayResult]:
    """
    GPU-accelerated Bollinger Bands.
    Automatically uses GPU for datasets > 100,000 rows when engine='auto'.

    Bollinger Bands measure volatility and provide dynamic support/resistance levels.

    Args:
        prices: Price data
        period: SMA period for middle band (default: 20)
        num_std: Number of standard deviations for upper/lower bands (default: 2.0)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>100K rows)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)

    Formula:
        Middle Band = SMA(period)
        Upper Band = Middle Band + (num_std * std_dev)
        Lower Band = Middle Band - (num_std * std_dev)

    Example:
        >>> prices = np.array([...])
        >>> upper, middle, lower = calculate_bollinger_bands(prices)
    """
    from .moving_averages import calculate_sma
    import polars as pl

    prices_arr = to_numpy_array(prices)
    df = pl.DataFrame({"price": prices_arr})

    # Select execution engine
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="bollinger", data_size=len(prices_arr)
    )

    # Calculate middle band (SMA) and rolling standard deviation in one pass
    result = (
        df.lazy()
        .select(
            middle=pl.col("price").rolling_mean(window_size=period),
            std_dev=pl.col("price").rolling_std(window_size=period),
        )
        .collect(engine=polars_engine)
    )

    middle_band = result["middle"].to_numpy()
    std_dev = result["std_dev"].to_numpy()

    # Calculate upper and lower bands
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)

    return (upper_band, middle_band, lower_band)

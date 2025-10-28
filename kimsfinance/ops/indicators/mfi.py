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
    Engine,
    EngineManager,
)
from ...utils.array_utils import to_numpy_array


def calculate_mfi(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    period: int = 14,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    Calculate Money Flow Index (MFI).

    MFI is a volume-weighted momentum indicator that measures buying/selling
    pressure. It's often called the "volume-weighted RSI".

    Automatically uses GPU for datasets > 100,000 rows when engine="auto".

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: Lookback period (default: 14)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>100K rows)

    Returns:
        Array of MFI values (0-100 range)

    Formula:
        1. Typical Price = (High + Low + Close) / 3
        2. Raw Money Flow = Typical Price × Volume
        3. Money Flow Ratio = (Positive Money Flow) / (Negative Money Flow)
        4. MFI = 100 - (100 / (1 + Money Flow Ratio))

    Interpretation:
        > 80: Overbought condition
        < 20: Oversold condition
        50: Neutral

    Performance:
        < 100K rows: CPU optimal
        100K-1M rows: GPU beneficial
        1M+ rows: GPU strong benefit

    Example:
        >>> high = np.array([105, 107, 106, 110, 108])
        >>> low = np.array([100, 102, 101, 105, 103])
        >>> close = np.array([103, 106, 104, 108, 106])
        >>> volume = np.array([1000, 1200, 900, 1500, 1100])
        >>> mfi = calculate_mfi(high, low, close, volume, period=3)
    """
    # Validate period
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")

    # Convert inputs to numpy arrays
    high_arr = to_numpy_array(high)
    low_arr = to_numpy_array(low)
    close_arr = to_numpy_array(close)
    volume_arr = to_numpy_array(volume)

    # Validate inputs
    if len(high_arr) < period + 1:
        raise ValueError(f"Data length must be > period ({period})")

    if not (len(high_arr) == len(low_arr) == len(close_arr) == len(volume_arr)):
        raise ValueError("All input arrays must have the same length")

    # Create DataFrame with all inputs
    df = pl.DataFrame(
        {
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": volume_arr,
        }
    )

    # Calculate typical price
    typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0

    # Calculate raw money flow
    raw_money_flow = typical_price * pl.col("volume")

    # Calculate typical price change
    tp_change = typical_price.diff()

    # Separate positive and negative money flow
    positive_flow = pl.when(tp_change > 0).then(raw_money_flow).otherwise(0)
    negative_flow = pl.when(tp_change < 0).then(raw_money_flow).otherwise(0)

    # Calculate rolling sums for the period
    pos_mf = positive_flow.rolling_sum(window_size=period, min_samples=period)
    neg_mf = negative_flow.rolling_sum(window_size=period, min_samples=period)

    # Calculate Money Flow Ratio and MFI
    # Add small epsilon to prevent division by zero
    money_ratio = pos_mf / (neg_mf + 1e-10)

    # Calculate MFI and clip to [0, 100] range
    # Note: When neg_mf is 0, money_ratio is very large, and MFI approaches 100
    mfi_value = 100.0 - (100.0 / (1.0 + money_ratio))

    # Clip to valid range [0, 100] to handle edge cases
    mfi_expr = mfi_value.clip(0.0, 100.0).alias("mfi")

    # Execute with selected Polars engine (GPU if available)
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="mfi", data_size=len(high_arr)
    )

    # Polars requires non-None engine parameter, default to "cpu" if None
    if polars_engine is None:
        polars_engine = "cpu"

    result = df.lazy().select(mfi=mfi_expr).collect(engine=polars_engine)

    return result["mfi"].to_numpy()

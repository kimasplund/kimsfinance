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


def calculate_keltner_channels(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 20,
    multiplier: float = 2.0,
    *,
    engine: Engine = "auto",
) -> tuple[ArrayResult, ArrayResult, ArrayResult]:
    """
    GPU-accelerated Keltner Channels.

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    Keltner Channels are volatility-based envelopes set above and below an
    exponential moving average. They use the Average True Range (ATR) to set
    channel distance and provide dynamic support/resistance levels.

    The channels expand during volatile markets and contract during calm periods,
    making them useful for identifying trending vs ranging markets and potential
    breakout points.

    Formula:
        Middle Line = EMA(close, period)
        Upper Channel = Middle + (multiplier * ATR(period))
        Lower Channel = Middle - (multiplier * ATR(period))

    Common usage:
        - Price above upper band: potential overbought/strong uptrend
        - Price below lower band: potential oversold/strong downtrend
        - Price within bands: normal trading range
        - Band expansion: increasing volatility
        - Band contraction: decreasing volatility

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Lookback period for EMA and ATR (default: 20)
        multiplier: ATR multiplier for channel width (default: 2.0)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>500K rows)

    Returns:
        Tuple of (upper_channel, middle_line, lower_channel)
        All arrays have same length as input, with first (period-1) values as NaN

    Raises:
        ValueError: If period < 1 or inputs have mismatched lengths

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> upper, middle, lower = calculate_keltner_channels(
        ...     df['High'], df['Low'], df['Close'], period=20
        ... )

        >>> # Detect breakout signals
        >>> breakout_up = df['Close'] > upper
        >>> breakout_down = df['Close'] < lower

    References:
        - https://en.wikipedia.org/wiki/Keltner_channel
        - Chester W. Keltner, "How to Make Money in Commodities" (1960)

    Performance:
        < 500K rows: CPU optimal
        500K-1M rows: GPU beneficial (1.2-1.5x speedup)
        1M+ rows: GPU strong benefit (up to 2.0x speedup)
    """
    # Validate inputs
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    if multiplier < 0:
        raise ValueError(f"multiplier must be >= 0, got {multiplier}")

    # Convert to numpy arrays
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)

    # Validate array lengths
    if not (len(highs_arr) == len(lows_arr) == len(closes_arr)):
        raise ValueError("highs, lows, and closes must have same length")

    if len(closes_arr) < period:
        raise ValueError(f"Insufficient data: need {period}, got {len(closes_arr)}")

    # Create Polars DataFrame
    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
        }
    )

    # Select execution engine
    exec_engine = EngineManager.select_engine(
        engine, operation="keltner", data_size=len(closes_arr)
    )

    # Calculate EMA of close (middle line)
    middle_expr = pl.col("close").ewm_mean(span=period, adjust=False)

    # Calculate ATR
    # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr_expr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )

    # ATR is Wilder's smoothing (EMA with span = 2 * period - 1)
    atr_expr = tr_expr.ewm_mean(span=2 * period - 1, adjust=False)

    # Calculate upper and lower channels
    upper_expr = middle_expr + (multiplier * atr_expr)
    lower_expr = middle_expr - (multiplier * atr_expr)

    # Execute all calculations in single pass
    result = (
        df.lazy()
        .select(upper=upper_expr, middle=middle_expr, lower=lower_expr)
        .collect(engine=exec_engine)
    )

    return (result["upper"].to_numpy(), result["middle"].to_numpy(), result["lower"].to_numpy())

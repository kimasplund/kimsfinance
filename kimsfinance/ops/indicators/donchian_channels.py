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


def calculate_donchian_channels(
    highs: ArrayLike, lows: ArrayLike, period: int = 20, *, engine: Engine = "auto"
) -> tuple[ArrayResult, ArrayResult, ArrayResult]:
    """
    Calculate Donchian Channels.

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    Donchian Channels identify the highest high and lowest low over a specified
    period. They were popularized by Richard Donchian and became famous as part
    of the Turtle Traders system. The channels provide dynamic support and
    resistance levels based on recent price action.

    The indicator consists of three bands:
    - Upper Channel: Highest high over the period
    - Middle Channel: Average of upper and lower channels
    - Lower Channel: Lowest low over the period

    Breakouts above the upper channel or below the lower channel are often
    interpreted as potential trend continuation signals. The Turtle Traders
    used a 20-period Donchian Channel for entries and a 10-period for exits.

    Formula:
        Upper Channel = MAX(High, period)
        Lower Channel = MIN(Low, period)
        Middle Channel = (Upper Channel + Lower Channel) / 2

    Common usage:
        - Price breaks above upper channel: Bullish breakout signal
        - Price breaks below lower channel: Bearish breakout signal
        - Price within channels: Ranging market
        - Channel width: Measure of volatility (wider = more volatile)
        - 20-period standard for entries, 10-period for exits (Turtle Traders)

    Args:
        highs: High prices
        lows: Low prices
        period: Lookback period for highest/lowest calculation (default: 20)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>500K rows)

    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
        All arrays have same length as input, with first (period-1) values as NaN

    Raises:
        ValueError: If period < 1 or inputs have mismatched lengths

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> upper, middle, lower = calculate_donchian_channels(
        ...     df['High'], df['Low'], period=20
        ... )

        >>> # Turtle Traders entry system
        >>> long_entry = df['Close'] > upper
        >>> short_entry = df['Close'] < lower

        >>> # 20-period for entry, 10-period for exit
        >>> entry_upper, _, entry_lower = calculate_donchian_channels(
        ...     df['High'], df['Low'], period=20
        ... )
        >>> exit_upper, _, exit_lower = calculate_donchian_channels(
        ...     df['High'], df['Low'], period=10
        ... )

    References:
        - https://en.wikipedia.org/wiki/Donchian_channel
        - Richard Donchian, "Donchian's 5- and 20-Day Moving Averages"
        - Curtis Faith, "Way of the Turtle" (Turtle Traders system)
        - https://www.investopedia.com/terms/d/donchianchannels.asp

    Performance:
        < 500K rows: CPU optimal
        500K-1M rows: GPU beneficial (1.2-1.5x speedup)
        1M+ rows: GPU strong benefit (up to 2.0x speedup)
    """
    # Validate inputs
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Convert to numpy arrays
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)

    # Validate array lengths
    if len(highs_arr) != len(lows_arr):
        raise ValueError("highs and lows must have same length")

    if len(highs_arr) < period:
        raise ValueError(f"Insufficient data: need {period}, got {len(highs_arr)}")

    # Create Polars DataFrame
    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
        }
    )

    # Select execution engine for Polars
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="donchian", data_size=len(highs_arr)
    )

    # Calculate upper and lower channels using rolling max/min
    upper_expr = pl.col("high").rolling_max(window_size=period)
    lower_expr = pl.col("low").rolling_min(window_size=period)

    # Calculate middle channel as average of upper and lower
    middle_expr = (upper_expr + lower_expr) / 2

    # Execute all calculations in single pass
    result = (
        df.lazy()
        .select(upper=upper_expr, middle=middle_expr, lower=lower_expr)
        .collect(engine=polars_engine)
    )

    return (result["upper"].to_numpy(), result["middle"].to_numpy(), result["lower"].to_numpy())

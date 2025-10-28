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


def calculate_supertrend(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    atr_period: int = 10,
    multiplier: float = 3.0,
    *,
    engine: Engine = "auto",
) -> tuple[ArrayResult, ArrayResult]:
    """
    GPU-accelerated Supertrend indicator calculation.

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    Supertrend is a trend-following indicator that provides dynamic support/resistance
    levels based on the Average True Range (ATR). It generates clear buy/sell signals
    by identifying trend direction changes.

    The indicator plots above the price during downtrends and below during uptrends,
    making it useful for:
    - Identifying trend direction
    - Setting stop-loss levels
    - Generating entry/exit signals
    - Filtering trades in trending markets

    Formula:
        1. Calculate ATR (Average True Range)
        2. Basic Upper Band = (High + Low) / 2 + (Multiplier × ATR)
        3. Basic Lower Band = (High + Low) / 2 - (Multiplier × ATR)
        4. Final Upper Band = min(Basic Upper Band, Previous Final Upper Band)
           if Close[prev] <= Final Upper Band[prev], else Basic Upper Band
        5. Final Lower Band = max(Basic Lower Band, Previous Final Lower Band)
           if Close[prev] >= Final Lower Band[prev], else Basic Lower Band
        6. Supertrend = Final Upper Band if in downtrend, else Final Lower Band
        7. Signal: 1 = uptrend (bullish), -1 = downtrend (bearish)

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        atr_period: Period for ATR calculation (default: 10)
        multiplier: ATR multiplier for band calculation (default: 3.0)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>500K rows)

    Returns:
        Tuple of (supertrend, signal)
        - supertrend: Supertrend line values (array same length as input)
        - signal: Trend direction (1 = uptrend/bullish, -1 = downtrend/bearish)
        First (atr_period) values will be NaN for supertrend, 0 for signal

    Raises:
        ValueError: If atr_period < 1, multiplier < 0, or inputs have mismatched lengths

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> supertrend, signal = calculate_supertrend(
        ...     df['High'], df['Low'], df['Close'], atr_period=10, multiplier=3.0
        ... )

        >>> # Identify buy/sell signals
        >>> buy_signal = (signal == 1) & (signal.shift(1) == -1)
        >>> sell_signal = (signal == -1) & (signal.shift(1) == 1)

        >>> # Use as dynamic stop-loss
        >>> long_stop = supertrend[signal == 1]
        >>> short_stop = supertrend[signal == -1]

    Trading Interpretation:
        - Signal changes from -1 to 1: Buy signal (trend reversal to uptrend)
        - Signal changes from 1 to -1: Sell signal (trend reversal to downtrend)
        - Signal == 1 (uptrend): Hold long positions, use supertrend as trailing stop
        - Signal == -1 (downtrend): Hold short positions or stay out

    Common Parameters:
        - Conservative (fewer signals): atr_period=14, multiplier=3.0
        - Standard: atr_period=10, multiplier=3.0
        - Aggressive (more signals): atr_period=7, multiplier=2.0

    References:
        - Olivier Seban's Supertrend indicator
        - https://tradingqna.com/t/supertrend-indicator-explained/5609

    Performance:
        < 500K rows: CPU optimal
        500K-1M rows: GPU beneficial (1.2-1.5x speedup)
        1M+ rows: GPU strong benefit (up to 2.0x speedup)
    """
    # Validate inputs
    if atr_period < 1:
        raise ValueError(f"atr_period must be >= 1, got {atr_period}")

    if multiplier < 0:
        raise ValueError(f"multiplier must be >= 0, got {multiplier}")

    # Convert to numpy arrays
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)
    closes_arr = to_numpy_array(closes)

    # Validate array lengths
    if not (len(highs_arr) == len(lows_arr) == len(closes_arr)):
        raise ValueError("highs, lows, and closes must have same length")

    if len(closes_arr) < atr_period:
        raise ValueError(f"Insufficient data: need {atr_period}, got {len(closes_arr)}")

    n = len(closes_arr)

    # Create Polars DataFrame
    df = pl.DataFrame(
        {
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
        }
    )

    # Select execution engine for Polars
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="supertrend", data_size=len(closes_arr)
    )

    # Calculate ATR
    # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr_expr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )

    # ATR is Wilder's smoothing (EMA with span = 2 * period - 1)
    atr_expr = tr_expr.ewm_mean(span=2 * atr_period - 1, adjust=False)

    # Calculate HL average (middle line)
    hl_avg_expr = (pl.col("high") + pl.col("low")) / 2.0

    # Calculate basic bands
    basic_upper_expr = hl_avg_expr + (multiplier * atr_expr)
    basic_lower_expr = hl_avg_expr - (multiplier * atr_expr)

    # Execute Polars calculations
    lazy_df = df.lazy().select(
        hl_avg=hl_avg_expr,
        basic_upper=basic_upper_expr,
        basic_lower=basic_lower_expr,
        atr=atr_expr,
        close=pl.col("close"),
    )

    # Only pass engine parameter if not None
    if polars_engine is not None:
        result = lazy_df.collect(engine=polars_engine)
    else:
        result = lazy_df.collect()

    # Extract arrays for iterative calculation
    basic_upper = result["basic_upper"].to_numpy()
    basic_lower = result["basic_lower"].to_numpy()
    close = result["close"].to_numpy()

    # Calculate final bands (requires iterative logic)
    final_upper = np.copy(basic_upper)
    final_lower = np.copy(basic_lower)

    for i in range(atr_period, n):
        # Upper band: keep previous if close was above it, otherwise use new basic upper
        if not np.isnan(final_upper[i - 1]):
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

        # Lower band: keep previous if close was below it, otherwise use new basic lower
        if not np.isnan(final_lower[i - 1]):
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

    # Calculate Supertrend and signal
    supertrend = np.full(n, np.nan)
    signal = np.zeros(n, dtype=np.int8)

    # Initialize at atr_period
    if not np.isnan(final_upper[atr_period]) and not np.isnan(final_lower[atr_period]):
        if close[atr_period] <= final_upper[atr_period]:
            supertrend[atr_period] = final_upper[atr_period]
            signal[atr_period] = -1
        else:
            supertrend[atr_period] = final_lower[atr_period]
            signal[atr_period] = 1

    # Calculate subsequent values
    for i in range(atr_period + 1, n):
        if np.isnan(supertrend[i - 1]):
            continue

        # Determine trend based on previous supertrend position
        if supertrend[i - 1] == final_upper[i - 1]:
            # Was in downtrend
            if close[i] <= final_upper[i]:
                # Stay in downtrend
                supertrend[i] = final_upper[i]
                signal[i] = -1
            else:
                # Switch to uptrend
                supertrend[i] = final_lower[i]
                signal[i] = 1
        else:
            # Was in uptrend
            if close[i] >= final_lower[i]:
                # Stay in uptrend
                supertrend[i] = final_lower[i]
                signal[i] = 1
            else:
                # Switch to downtrend
                supertrend[i] = final_upper[i]
                signal[i] = -1

    return supertrend, signal

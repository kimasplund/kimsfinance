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


def calculate_ichimoku(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    conversion_period: int = 9,
    base_period: int = 26,
    leading_span_b_period: int = 52,
    displacement: int = 26,
    *,
    engine: Engine = "auto",
) -> dict[str, ArrayResult]:
    """
    Calculate Ichimoku Cloud indicator (Ichimoku Kinko Hyo).

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    Ichimoku Cloud is a comprehensive indicator that provides a complete view of
    support/resistance levels, momentum, and trend direction at a glance. It was
    developed by Goichi Hosoda in Japan and consists of five lines that form a
    "cloud" visualization when plotted.

    The indicator consists of five components:
    - Tenkan-sen (Conversion Line): Fast-moving line
    - Kijun-sen (Base Line): Standard line
    - Senkou Span A (Leading Span A): First cloud boundary (shifted forward)
    - Senkou Span B (Leading Span B): Second cloud boundary (shifted forward)
    - Chikou Span (Lagging Span): Close price (shifted backward)

    The "cloud" (Kumo) is the area between Senkou Span A and Senkou Span B, which
    represents support/resistance zones.

    Formula:
        Tenkan-sen = (9-period high + 9-period low) / 2
        Kijun-sen = (26-period high + 26-period low) / 2
        Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
        Senkou Span B = (52-period high + 52-period low) / 2, plotted 26 periods ahead
        Chikou Span = Close price, plotted 26 periods behind

    Trading signals:
        - Price above cloud: Bullish trend
        - Price below cloud: Bearish trend
        - Price in cloud: Consolidation/uncertainty
        - Tenkan-sen crosses above Kijun-sen: Bullish signal
        - Tenkan-sen crosses below Kijun-sen: Bearish signal
        - Senkou Span A above Span B: Bullish cloud
        - Senkou Span A below Span B: Bearish cloud

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        conversion_period: Period for Tenkan-sen (default: 9)
        base_period: Period for Kijun-sen (default: 26)
        leading_span_b_period: Period for Senkou Span B (default: 52)
        displacement: Number of periods to shift Senkou spans forward (default: 26)
        engine: Execution engine ("cpu", "gpu", "auto")
            auto: Intelligently selects GPU for large datasets (>500K rows)

    Returns:
        Dictionary with keys:
        - 'tenkan_sen': Conversion line
        - 'kijun_sen': Base line
        - 'senkou_span_a': Leading span A (cloud boundary, shifted forward)
        - 'senkou_span_b': Leading span B (cloud boundary, shifted forward)
        - 'chikou_span': Lagging span (close price, shifted backward)

        All arrays have same length as input. NaN values appear where insufficient
        data is available or where values are shifted.

    Raises:
        ValueError: If periods < 1 or inputs have mismatched lengths

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> ichimoku = calculate_ichimoku(
        ...     df['High'], df['Low'], df['Close']
        ... )
        >>> tenkan = ichimoku['tenkan_sen']
        >>> kijun = ichimoku['kijun_sen']

        >>> # Detect bullish crossover
        >>> bullish_signal = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))

        >>> # Check if price is above cloud
        >>> span_a = ichimoku['senkou_span_a']
        >>> span_b = ichimoku['senkou_span_b']
        >>> cloud_top = np.maximum(span_a, span_b)
        >>> price_above_cloud = df['Close'] > cloud_top

    References:
        - https://en.wikipedia.org/wiki/Ichimoku_Kinkō_Hyō
        - Goichi Hosoda, "Ichimoku Kinko Studies" (1969)
        - https://www.investopedia.com/terms/i/ichimoku-cloud.asp
        - https://www.babypips.com/learn/forex/ichimoku-kinko-hyo

    Performance:
        < 500K rows: CPU optimal
        500K-1M rows: GPU beneficial (1.2-1.5x speedup)
        1M+ rows: GPU strong benefit (up to 2.0x speedup)
    """
    # Validate inputs
    if conversion_period < 1:
        raise ValueError(f"conversion_period must be >= 1, got {conversion_period}")
    if base_period < 1:
        raise ValueError(f"base_period must be >= 1, got {base_period}")
    if leading_span_b_period < 1:
        raise ValueError(f"leading_span_b_period must be >= 1, got {leading_span_b_period}")
    if displacement < 0:
        raise ValueError(f"displacement must be >= 0, got {displacement}")

    # Convert to numpy arrays
    high_arr = to_numpy_array(high)
    low_arr = to_numpy_array(low)
    close_arr = to_numpy_array(close)

    # Validate array lengths
    if not (len(high_arr) == len(low_arr) == len(close_arr)):
        raise ValueError("high, low, and close must have same length")

    n = len(high_arr)

    # Create Polars DataFrame
    df = pl.DataFrame(
        {
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
        }
    )

    # Select execution engine for Polars
    polars_engine = EngineManager.select_polars_engine(
        engine, operation="ichimoku", data_size=n
    )

    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_high = pl.col("high").rolling_max(window_size=conversion_period)
    tenkan_low = pl.col("low").rolling_min(window_size=conversion_period)
    tenkan_expr = (tenkan_high + tenkan_low) / 2

    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_high = pl.col("high").rolling_max(window_size=base_period)
    kijun_low = pl.col("low").rolling_min(window_size=base_period)
    kijun_expr = (kijun_high + kijun_low) / 2

    # Calculate Senkou Span B base: (52-period high + 52-period low) / 2
    span_b_high = pl.col("high").rolling_max(window_size=leading_span_b_period)
    span_b_low = pl.col("low").rolling_min(window_size=leading_span_b_period)
    span_b_base_expr = (span_b_high + span_b_low) / 2

    # Execute calculations in single pass
    result = (
        df.lazy()
        .select(
            tenkan_sen=tenkan_expr,
            kijun_sen=kijun_expr,
            span_b_base=span_b_base_expr,
            close=pl.col("close"),
        )
        .collect(engine=polars_engine)
    )

    tenkan_sen = result["tenkan_sen"].to_numpy()
    kijun_sen = result["kijun_sen"].to_numpy()
    span_b_base = result["span_b_base"].to_numpy()

    # Calculate Senkou Span A base: (Tenkan-sen + Kijun-sen) / 2
    span_a_base = (tenkan_sen + kijun_sen) / 2

    # Shift Senkou Span A forward by displacement periods
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if i + displacement < n:
            senkou_span_a[i + displacement] = span_a_base[i]

    # Shift Senkou Span B forward by displacement periods
    senkou_span_b = np.full(n, np.nan)
    for i in range(n):
        if i + displacement < n:
            senkou_span_b[i + displacement] = span_b_base[i]

    # Calculate Chikou Span: Close price, shifted backward by displacement
    chikou_span = np.full(n, np.nan)
    for i in range(displacement, n):
        chikou_span[i - displacement] = close_arr[i]

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }

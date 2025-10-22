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


def calculate_pivot_points(
    high: float, low: float, close: float, *, engine: Engine = "auto"
) -> dict[str, float]:
    """
    Calculate Pivot Points and support/resistance levels.

    Pivot Points are used by intraday traders to identify key price levels
    for potential support and resistance. They are calculated from the previous
    period's high, low, and close prices.

    Automatically uses GPU for large-scale batch calculations when engine="auto"
    (though for single calculations, CPU is always used).

    Args:
        high: Previous period's high price
        low: Previous period's low price
        close: Previous period's close price
        engine: Execution engine ("cpu", "gpu", "auto")
            Note: For single scalar calculations, CPU is always used regardless
            of engine parameter. GPU routing is reserved for future batch operations.

    Returns:
        Dictionary containing:
            - PP: Pivot Point (central level)
            - R1, R2, R3: Resistance levels 1, 2, 3 (above PP)
            - S1, S2, S3: Support levels 1, 2, 3 (below PP)

    Formula:
        PP = (H + L + C) / 3
        R1 = 2*PP - L
        R2 = PP + (H - L)
        R3 = H + 2*(PP - L)
        S1 = 2*PP - H
        S2 = PP - (H - L)
        S3 = L - 2*(H - PP)

    Raises:
        ValueError: If high < low or any input is NaN/inf

    Examples:
        >>> # Calculate pivot points from previous day's data
        >>> pivots = calculate_pivot_points(high=110.5, low=108.2, close=109.8)
        >>> print(f"Pivot Point: {pivots['PP']:.2f}")
        >>> print(f"Resistance: R1={pivots['R1']:.2f}, R2={pivots['R2']:.2f}")
        >>> print(f"Support: S1={pivots['S1']:.2f}, S2={pivots['S2']:.2f}")

    References:
        - https://en.wikipedia.org/wiki/Pivot_point_(technical_analysis)
        - Standard Pivot Points (Floor Pivot Points)
    """
    # Validate inputs
    if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
        raise ValueError("All inputs must be finite numbers (not NaN or inf)")

    if high < low:
        raise ValueError(f"high ({high}) must be >= low ({low})")

    # Calculate Pivot Point (central level)
    pp = (high + low + close) / 3.0

    # Calculate range
    price_range = high - low

    # Calculate resistance levels
    r1 = 2.0 * pp - low
    r2 = pp + price_range
    r3 = high + 2.0 * (pp - low)

    # Calculate support levels
    s1 = 2.0 * pp - high
    s2 = pp - price_range
    s3 = low - 2.0 * (high - pp)

    return {
        "PP": pp,
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "S1": s1,
        "S2": s2,
        "S3": s3,
    }

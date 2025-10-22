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


def calculate_fibonacci_retracement(
    high: float, low: float, *, engine: Engine = "auto"
) -> dict[str, float]:
    """
    Calculate Fibonacci Retracement levels.

    Fibonacci Retracement identifies potential support and resistance levels
    based on the Fibonacci sequence ratios. These levels are calculated between
    a significant high and low price point.

    The standard Fibonacci ratios are:
    - 0.0% (High): Starting point (highest price)
    - 23.6%: First retracement level
    - 38.2%: Second retracement level
    - 50.0%: Mid-point retracement (not a Fibonacci ratio, but commonly used)
    - 61.8%: Third retracement level (golden ratio)
    - 100.0% (Low): Ending point (lowest price)

    Args:
        high: The highest price point in the range
        low: The lowest price point in the range
        engine: Computation engine ('auto', 'cpu', 'gpu') - included for
                consistency with other indicators, but not used as this is
                a simple scalar calculation

    Returns:
        Dictionary mapping Fibonacci level labels to price values:
        {
            '0.0%': high,
            '23.6%': retracement_value,
            '38.2%': retracement_value,
            '50.0%': retracement_value,
            '61.8%': retracement_value,
            '100.0%': low
        }

    Raises:
        ValueError: If high <= low (invalid price range)

    Examples:
        >>> # Calculate Fibonacci levels for a price range
        >>> levels = calculate_fibonacci_retracement(high=150.0, low=100.0)
        >>> print(f"61.8% retracement: {levels['61.8%']}")
        61.8% retracement: 119.1

        >>> # Use in trading strategy
        >>> current_price = 125.0
        >>> if current_price <= levels['38.2%']:
        ...     print("Price at 38.2% support level")

    References:
        - https://www.investopedia.com/terms/f/fibonacciretracement.asp
        - https://en.wikipedia.org/wiki/Fibonacci_retracement
    """
    # Validate inputs
    if high <= low:
        raise ValueError(f"high must be > low, got high={high}, low={low}")

    # Calculate the range
    price_range = high - low

    # Standard Fibonacci retracement ratios
    ratios = {
        "0.0%": 0.0,
        "23.6%": 0.236,
        "38.2%": 0.382,
        "50.0%": 0.500,
        "61.8%": 0.618,
        "100.0%": 1.0,
    }

    # Calculate retracement levels
    # Formula: level = high - (high - low) * ratio
    levels = {}
    for label, ratio in ratios.items():
        levels[label] = high - (price_range * ratio)

    return levels

"""
Swing High/Low Detection
========================

Functions for identifying significant swing points in price series.
"""

from __future__ import annotations

import numpy as np
from ..core import ArrayLike

def find_swing_points(
    data: ArrayLike,
    n: int = 10,
    is_high: bool = True
) -> np.ndarray:
    """
    Find swing highs or lows in a price series.

    A swing high is a peak that is higher than the `n` prices before and after it.
    A swing low is a trough that is lower than the `n` prices before and after it.

    Args:
        data: The price series to search (e.g., high or low prices)
        n: The number of periods to look back and forward
        is_high: If True, find swing highs. If False, find swing lows.

    Returns:
        An array of indices for the detected swing points.
    """
    swing_points = []
    for i in range(n, len(data) - n):
        is_swing_point = True
        for j in range(1, n + 1):
            if is_high:
                if data[i] < data[i - j] or data[i] <= data[i + j]:
                    is_swing_point = False
                    break
            else:
                if data[i] > data[i - j] or data[i] >= data[i + j]:
                    is_swing_point = False
                    break
        if is_swing_point:
            swing_points.append(i)
    return np.array(swing_points)

"""
Pick's Theorem Momentum Ratio Indicator by kimsfinance
========================================================

A speculative indicator based on Pick's Theorem, which relates the area of a
simple polygon to the number of integer points on its boundary and in its interior.

This version uses the Shoelace formula to calculate the area and the GCD
method to find boundary points, then applies Pick's Theorem to solve for the
interior points. This is both mathematically correct and computationally efficient.
"""

from __future__ import annotations

import numpy as np
import math
from ..core import ArrayLike
from .atr import calculate_atr
from .swing import find_swing_points


def _calculate_polygon_properties(polygon: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Calculates the interior (I) and boundary (B) points for a simple polygon
    with integer vertices.
    """
    (x1, y1), (x2, y2), (x3, y3) = polygon

    # 1. Calculate Boundary Points (B) using GCD
    # The number of integer points on a line segment between two integer points
    # (excluding endpoints) is gcd(|dx|, |dy|) - 1. The total number on the
    # segment is gcd(|dx|, |dy|) + 1. Summing the segments and subtracting
    # the 3 double-counted vertices gives:
    # B = (gcd1+1) + (gcd2+1) + (gcd3+1) - 3 = gcd1 + gcd2 + gcd3
    b1 = math.gcd(abs(x2 - x1), abs(y2 - y1))
    b2 = math.gcd(abs(x3 - x2), abs(y3 - y2))
    b3 = math.gcd(abs(x1 - x3), abs(y1 - y3))
    boundary_points = b1 + b2 + b3

    # 2. Calculate Area (A) using the Shoelace Formula
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    # 3. Solve for Interior Points (I) using Pick's Theorem: A = I + B/2 - 1
    # Rearranged: I = A - B/2 + 1
    # The result must be an integer; rounding handles potential float precision errors.
    interior_points = int(round(area - boundary_points / 2 + 1))

    return interior_points, boundary_points


def calculate_picks_momentum_ratio(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    n: int = 10,
    atr_period: int = 14,
    atr_multiplier: float = 0.5,
) -> tuple[np.ndarray, list]:
    """
    Calculates the Pick's Momentum Ratio (PMR) for a given price series.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        n: The number of periods to look back and forward for swing points
        atr_period: The period for the ATR calculation
        atr_multiplier: The multiplier for the ATR to determine the grid size

    Returns:
        A tuple containing:
        - An array of PMR values
        - A list of polygon vertices (in integer grid coordinates)
    """
    # 1. Calculate ATR
    atr = calculate_atr(high, low, close, period=atr_period)

    # 2. Find swing points
    swing_highs_idx = find_swing_points(high, n=n, is_high=True)
    swing_lows_idx = find_swing_points(low, n=n, is_high=False)

    pmr_values = np.full_like(high, np.nan, dtype=float)
    polygons = []

    # Combine and sort swing points by their index in the price series
    swing_points = sorted(
        [(idx, "high") for idx in swing_highs_idx] + [(idx, "low") for idx in swing_lows_idx]
    )

    # Iterate through swing points to form polygons (triangles)
    for i in range(len(swing_points) - 2):
        p1_idx, p1_type = swing_points[i]
        p2_idx, p2_type = swing_points[i + 1]
        p3_idx, p3_type = swing_points[i + 2]

        # Ensure we have an alternating sequence (e.g., high-low-high)
        if p1_type == p2_type or p2_type == p3_type:
            continue

        # Define the price quantum (PQ) for this polygon based on ATR at its start
        pq = atr[p1_idx] * atr_multiplier
        if pq == 0 or np.isnan(pq):
            continue

        # Get prices for vertices
        y1_price = high[p1_idx] if p1_type == "high" else low[p1_idx]
        y2_price = high[p2_idx] if p2_type == "high" else low[p2_idx]
        y3_price = high[p3_idx] if p3_type == "high" else low[p3_idx]

        # Create vertices and discretize them into integer grid coordinates
        v1 = (p1_idx, int(round(y1_price / pq)))
        v2 = (p2_idx, int(round(y2_price / pq)))
        v3 = (p3_idx, int(round(y3_price / pq)))

        polygon = [v1, v2, v3]
        polygons.append(polygon)

        # 4. Calculate I and B using the corrected helper function
        I, B = _calculate_polygon_properties(polygon)

        # 5. Calculate PMR = I / B
        if B > 0:
            pmr = I / B
            pmr_values[p3_idx] = pmr

    return pmr_values, polygons

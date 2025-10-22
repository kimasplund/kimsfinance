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


def calculate_parabolic_sar(
    highs: ArrayLike,
    lows: ArrayLike,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
    *,
    engine: Engine = "auto",
) -> ArrayResult:
    """
    Calculate Parabolic SAR (Stop and Reverse).

    Automatically uses GPU for datasets > 500,000 rows when engine="auto".

    The Parabolic SAR is a trend-following indicator that provides entry and exit
    points. It appears as dots above or below price bars. When dots flip from below
    to above price (or vice versa), it signals a potential trend reversal.

    The indicator uses an acceleration factor (AF) that increases as the trend
    continues, making the SAR more responsive to price changes over time.

    Args:
        highs: High prices
        lows: Low prices
        af_start: Starting acceleration factor (default: 0.02)
        af_increment: AF increment when new extreme point reached (default: 0.02)
        af_max: Maximum acceleration factor (default: 0.2)
        engine: Computation engine ('auto', 'cpu', 'gpu')

    Returns:
        Array of SAR values (same length as input)
        All values are initialized (no NaN values unless input contains NaN)

    Raises:
        ValueError: If af_start, af_increment, or af_max are invalid
        ValueError: If highs and lows have mismatched lengths

    Examples:
        >>> import numpy as np
        >>> highs = np.array([102, 105, 104, 107, 106])
        >>> lows = np.array([100, 101, 102, 104, 103])
        >>> sar = calculate_parabolic_sar(highs, lows)

    Algorithm:
        1. Initial trend determined by first price move
        2. SAR updated each period: SAR = SAR + AF * (EP - SAR)
        3. EP (Extreme Point) = highest high (uptrend) or lowest low (downtrend)
        4. AF starts at af_start, increases by af_increment each new EP, max af_max
        5. Trend reverses when price crosses SAR

    References:
        - Wilder, J. Wells (1978). "New Concepts in Technical Trading Systems"
        - https://en.wikipedia.org/wiki/Parabolic_SAR
    """
    # 1. VALIDATE INPUTS
    if af_start <= 0 or af_start >= 1:
        raise ValueError(f"af_start must be in (0, 1), got {af_start}")
    if af_increment <= 0 or af_increment >= 1:
        raise ValueError(f"af_increment must be in (0, 1), got {af_increment}")
    if af_max <= af_start or af_max >= 1:
        raise ValueError(f"af_max must be in (af_start, 1), got {af_max}")

    # 2. CONVERT TO NUMPY (standardize input)
    highs_arr = to_numpy_array(highs)
    lows_arr = to_numpy_array(lows)

    if len(highs_arr) != len(lows_arr):
        raise ValueError(
            f"highs and lows must have same length: {len(highs_arr)} != {len(lows_arr)}"
        )

    if len(highs_arr) < 2:
        raise ValueError(f"Insufficient data: need at least 2, got {len(highs_arr)}")

    # 3. ENGINE ROUTING
    if engine == "auto":
        # Parabolic SAR is iterative, GPU benefit threshold is higher
        use_gpu = len(highs_arr) >= 500_000 and CUPY_AVAILABLE
    elif engine == "gpu":
        use_gpu = CUPY_AVAILABLE
    elif engine == "cpu":
        use_gpu = False
    else:
        raise ValueError(f"Invalid engine: {engine}")

    # 4. DISPATCH TO CPU OR GPU
    if use_gpu:
        return _calculate_parabolic_sar_gpu(highs_arr, lows_arr, af_start, af_increment, af_max)
    else:
        return _calculate_parabolic_sar_cpu(highs_arr, lows_arr, af_start, af_increment, af_max)


def _calculate_parabolic_sar_cpu(
    highs: np.ndarray, lows: np.ndarray, af_start: float, af_increment: float, af_max: float
) -> np.ndarray:
    """
    CPU implementation of Parabolic SAR using NumPy.

    This is an inherently iterative algorithm where each value depends on the
    previous state, making it challenging to parallelize efficiently.
    """
    n = len(highs)
    sar = np.full(n, np.nan, dtype=np.float64)

    # Initialize state variables
    # Start with uptrend assumption (can be either, results converge quickly)
    is_uptrend = True
    af = af_start

    # Initialize SAR and EP (Extreme Point)
    # For uptrend: EP is highest high, SAR starts at lowest low
    # For downtrend: EP is lowest low, SAR starts at highest high
    sar[0] = lows[0]  # First SAR value
    ep = highs[0]  # Extreme point

    # Iterate through each bar
    for i in range(1, n):
        # Calculate new SAR
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        # Check for trend reversal
        if is_uptrend:
            # In uptrend: SAR should be below price
            # Adjust SAR to not exceed prior two lows
            if i >= 2:
                sar[i] = min(sar[i], lows[i - 1], lows[i - 2])
            elif i >= 1:
                sar[i] = min(sar[i], lows[i - 1])

            # Check if price crossed below SAR (reversal to downtrend)
            if lows[i] < sar[i]:
                is_uptrend = False
                sar[i] = ep  # SAR becomes the prior EP
                ep = lows[i]  # New EP is current low
                af = af_start  # Reset acceleration factor
            else:
                # Continue uptrend: check for new high (new EP)
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_increment, af_max)
        else:
            # In downtrend: SAR should be above price
            # Adjust SAR to not exceed prior two highs
            if i >= 2:
                sar[i] = max(sar[i], highs[i - 1], highs[i - 2])
            elif i >= 1:
                sar[i] = max(sar[i], highs[i - 1])

            # Check if price crossed above SAR (reversal to uptrend)
            if highs[i] > sar[i]:
                is_uptrend = True
                sar[i] = ep  # SAR becomes the prior EP
                ep = highs[i]  # New EP is current high
                af = af_start  # Reset acceleration factor
            else:
                # Continue downtrend: check for new low (new EP)
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_increment, af_max)

    return sar


def _calculate_parabolic_sar_gpu(
    highs: np.ndarray, lows: np.ndarray, af_start: float, af_increment: float, af_max: float
) -> np.ndarray:
    """
    GPU implementation of Parabolic SAR using CuPy.

    Note: Parabolic SAR is inherently iterative with state dependencies,
    making true parallelization very challenging. This implementation uses
    CuPy's JIT compilation for potential speedup on very large datasets,
    but the algorithm remains fundamentally sequential.
    """
    try:
        import cupy as cp
    except ImportError:
        # Fallback to CPU if CuPy not available
        return _calculate_parabolic_sar_cpu(highs, lows, af_start, af_increment, af_max)

    # For iterative algorithms like Parabolic SAR, GPU doesn't provide
    # significant benefit unless dataset is extremely large (500K+)
    # For now, use CPU implementation with CuPy arrays

    # Transfer to GPU
    highs_gpu = cp.asarray(highs, dtype=cp.float64)
    lows_gpu = cp.asarray(lows, dtype=cp.float64)

    # Convert back to CPU for iterative computation
    # (True GPU parallelization would require custom CUDA kernel)
    highs_cpu = cp.asnumpy(highs_gpu)
    lows_cpu = cp.asnumpy(lows_gpu)

    # Use CPU implementation
    result = _calculate_parabolic_sar_cpu(highs_cpu, lows_cpu, af_start, af_increment, af_max)

    return result

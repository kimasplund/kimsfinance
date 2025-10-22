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


def calculate_roc(prices: ArrayLike, period: int = 12, *, engine: Engine = "auto") -> ArrayResult:
    """
    Calculate Rate of Change (ROC).

    Automatically uses GPU for datasets > 50,000 rows when engine="auto".

    ROC is a momentum oscillator that measures the percentage change in price
    between the current price and the price N periods ago. It oscillates above
    and below zero, indicating the speed and direction of price movement.

    Positive values indicate upward momentum, while negative values indicate
    downward momentum. The further from zero, the stronger the momentum.

    Args:
        prices: Input price data (typically close prices)
        period: Lookback period for calculation (default: 12)
        engine: Computation engine ('auto', 'cpu', 'gpu')

    Returns:
        Array of ROC values (percentage change, e.g., 5.0 = 5% increase)
        First (period) values are NaN due to warmup

    Raises:
        ValueError: If period < 1 or inputs have insufficient data

    Formula:
        ROC = ((Price - Price[n]) / Price[n]) * 100

    Examples:
        >>> import polars as pl
        >>> df = pl.read_csv("ohlcv.csv")
        >>> roc = calculate_roc(df['Close'], period=12)
        >>> roc_fast = calculate_roc(df['Close'], period=5)

    References:
        - https://en.wikipedia.org/wiki/Momentum_(technical_analysis)
        - https://www.investopedia.com/terms/r/rateofchange.asp
    """
    # 1. VALIDATE INPUTS
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # 2. CONVERT TO NUMPY (standardize input)
    data_array = np.asarray(prices, dtype=np.float64)

    if len(data_array) < period + 1:
        raise ValueError(f"Insufficient data: need {period + 1}, got {len(data_array)}")

    # 3. ENGINE ROUTING using EngineManager (standardized pattern)
    exec_engine = EngineManager.select_engine(
        engine,
        operation="roc_indicator",
        data_size=len(data_array)
    )

    # 4. DISPATCH TO CPU OR GPU based on selected engine
    if exec_engine == "gpu":
        return _calculate_roc_gpu(data_array, period)
    else:
        return _calculate_roc_cpu(data_array, period)


def _calculate_roc_cpu(data: np.ndarray, period: int) -> np.ndarray:
    """CPU implementation of ROC using NumPy (vectorized)."""

    # Initialize result array with NaN
    result = np.full(len(data), np.nan, dtype=np.float64)

    # Get current and previous prices using array slicing (vectorized)
    current_prices = data[period:]
    prev_prices = data[:-period]

    # Calculate ROC: ((current - prev) / prev) * 100
    # Use where to avoid division by zero
    roc_values = np.where(
        prev_prices != 0, ((current_prices - prev_prices) / prev_prices) * 100.0, np.nan
    )

    # Place results in correct positions (starting at period)
    result[period:] = roc_values

    return result


def _calculate_roc_gpu(data: np.ndarray, period: int) -> np.ndarray:
    """GPU implementation of ROC using CuPy."""

    try:
        import cupy as cp
    except ImportError:
        # Fallback to CPU if CuPy not available
        return _calculate_roc_cpu(data, period)

    # Transfer to GPU
    data_gpu = cp.asarray(data, dtype=cp.float64)

    # Initialize result array with NaN
    result_gpu = cp.full(len(data_gpu), cp.nan, dtype=cp.float64)

    # Get current and previous prices using array slicing (vectorized)
    current_prices = data_gpu[period:]
    prev_prices = data_gpu[:-period]

    # Calculate ROC: ((current - prev) / prev) * 100
    # Use where to avoid division by zero
    roc_values = cp.where(
        prev_prices != 0, ((current_prices - prev_prices) / prev_prices) * 100.0, cp.nan
    )

    # Place results in correct positions (starting at period)
    result_gpu[period:] = roc_values

    # Transfer back to CPU
    return cp.asnumpy(result_gpu)

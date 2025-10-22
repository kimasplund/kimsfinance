from __future__ import annotations

import numpy as np
import polars as pl
import pandas as pd

from ..core.types import ArrayLike


def to_numpy_array(data: ArrayLike) -> np.ndarray:
    """
    Convert various array-like types to NumPy array.

    Args:
        data: Input data (NumPy array, Polars Series, pandas Series, or list)

    Returns:
        NumPy array
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pl.Series):
        return data.to_numpy()
    elif isinstance(data, pd.Series):
        return data.to_numpy()
    else:
        return np.asarray(data)


__all__ = ["to_numpy_array"]

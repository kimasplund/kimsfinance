import numpy as np
import polars as pl
import pytest
from kimsfinance.ops.moving_averages import calculate_sma


def test_calculate_sma_basic():
    """Verify basic SMA calculation."""
    df = pl.DataFrame({"close": [1, 2, 3, 4, 5]})
    window = 3
    expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])

    result = calculate_sma(df, "close", windows=window)[0]

    np.testing.assert_allclose(result, expected, equal_nan=True)

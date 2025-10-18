import numpy as np
import pytest
from kimsfinance.ops.indicators import calculate_atr

@pytest.fixture
def sample_data():
    """Generates sample OHLC data for testing."""
    high = np.array([10, 12, 11, 13, 14])
    low = np.array([8, 9, 10, 11, 12])
    close = np.array([9, 11, 10, 12, 13])
    return high, low, close

def test_calculate_atr(sample_data):
    """Test the calculate_atr function with a simple case."""
    high, low, close = sample_data
    period = 3
    atr = calculate_atr(high, low, close, period=period)

    # New Polars-based implementation uses Polars' native ewm_mean
    # with span=2*period-1, which produces different values than
    # the old custom _wilder_smoothing function
    # The new implementation does not have leading NaN values
    expected_atr = np.array([2.0, 2.33333333, 1.88888889, 2.25925926, 2.17283951])

    assert isinstance(atr, np.ndarray)
    assert len(atr) == len(high)
    assert np.allclose(atr, expected_atr, equal_nan=True, atol=1e-5)
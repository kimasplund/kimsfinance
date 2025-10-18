import numpy as np
import pytest
from kimsfinance.ops.swing import find_swing_points

@pytest.fixture
def sample_data():
    """Generates sample OHLC data for testing."""
    high = np.array([10, 12, 11, 13, 14, 13, 12, 15, 14, 13, 12, 11])
    low = np.array([8, 9, 10, 11, 12, 11, 10, 9, 8, 9, 10, 9])
    return high, low

def test_find_swing_points(sample_data):
    """Test the find_swing_points function."""
    high, low = sample_data
    n = 2
    swing_highs = find_swing_points(high, n=n, is_high=True)
    swing_lows = find_swing_points(low, n=n, is_high=False)

    # Expected swing points based on n=2
    # High at index 4 (14) is higher than [11, 13] and [13, 12]
    # High at index 7 (15) is higher than [13, 12] and [14, 13]
    # Low at index 8 (8) is lower than [10, 9] and [9, 10]
    expected_swing_highs = np.array([4, 7])
    expected_swing_lows = np.array([8])

    assert np.array_equal(swing_highs, expected_swing_highs)
    assert np.array_equal(swing_lows, expected_swing_lows)
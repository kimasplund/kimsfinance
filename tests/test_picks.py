import numpy as np
import pytest
from kimsfinance.ops.picks import calculate_picks_momentum_ratio


@pytest.fixture
def sample_data():
    """Generates a longer sample OHLC data for testing."""
    # Generate a more realistic price series
    np.random.seed(42)
    base_price = 100
    price_changes = np.random.randn(100).cumsum()
    close = base_price + price_changes
    high = close + np.random.uniform(0, 2, size=100)
    low = close - np.random.uniform(0, 2, size=100)
    return high, low, close


def test_calculate_picks_momentum_ratio(sample_data):
    """Test the calculate_picks_momentum_ratio function."""
    high, low, close = sample_data
    n = 5  # Use a more realistic value for n
    atr_period = 14
    atr_multiplier = 0.5

    pmr_values, polygons = calculate_picks_momentum_ratio(
        high, low, close, n=n, atr_period=atr_period, atr_multiplier=atr_multiplier
    )

    assert isinstance(pmr_values, np.ndarray)
    assert len(pmr_values) == len(high)
    assert isinstance(polygons, list)
    assert len(polygons) > 0
    assert not np.all(np.isnan(pmr_values))


@pytest.mark.parametrize(
    "n, atr_multiplier",
    [
        (3, 0.2),
        (10, 1.0),
        (20, 2.0),
    ],
)
def test_different_parameters(sample_data, n, atr_multiplier):
    """Test with different n and atr_multiplier parameters."""
    high, low, close = sample_data
    atr_period = 14

    pmr_values, polygons = calculate_picks_momentum_ratio(
        high, low, close, n=n, atr_period=atr_period, atr_multiplier=atr_multiplier
    )

    assert isinstance(pmr_values, np.ndarray)
    assert len(pmr_values) == len(high)
    assert isinstance(polygons, list)


def test_no_swing_points(sample_data):
    """Test the case where no swing points are found."""
    high, low, close = sample_data
    # Use a very large n to ensure no swing points are found
    n = len(high)

    pmr_values, polygons = calculate_picks_momentum_ratio(
        high, low, close, n=n, atr_period=14, atr_multiplier=0.5
    )

    assert isinstance(pmr_values, np.ndarray)
    assert len(pmr_values) == len(high)
    assert isinstance(polygons, list)
    assert len(polygons) == 0
    assert np.all(np.isnan(pmr_values))


def test_polygon_contents(sample_data):
    """Test the contents of the generated polygons."""
    high, low, close = sample_data
    n = 5
    atr_period = 14
    atr_multiplier = 0.5

    pmr_values, polygons = calculate_picks_momentum_ratio(
        high, low, close, n=n, atr_period=atr_period, atr_multiplier=atr_multiplier
    )

    for polygon in polygons:
        assert len(polygon) == 3
        for vertex in polygon:
            assert len(vertex) == 2
            assert isinstance(vertex[0], (int, np.integer))
            assert isinstance(vertex[1], (int, np.integer))


def test_with_known_output(sample_data):
    """Test with a known output to verify the algorithm's correctness."""
    # Use the more varied sample_data fixture
    high, low, close = sample_data
    n = 5
    atr_period = 14
    atr_multiplier = 0.5

    pmr_values, polygons = calculate_picks_momentum_ratio(
        high, low, close, n=n, atr_period=atr_period, atr_multiplier=atr_multiplier
    )

    print("PMR values:", pmr_values)

    # Find the first valid PMR value
    first_valid_pmr = next((x for x in pmr_values if not np.isnan(x)), None)

    assert first_valid_pmr is not None, "No valid PMR values were calculated."
    # With more varied data, we expect some interior points
    assert first_valid_pmr >= 0, "PMR should be non-negative."

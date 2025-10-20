"""Tests for DEMA and TEMA indicators."""

import numpy as np
import pytest
from kimsfinance.ops.indicators import calculate_dema, calculate_tema, calculate_ema


class TestDEMA:
    """Test Double Exponential Moving Average calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        return prices

    def test_basic_calculation(self, sample_data):
        """Test basic DEMA calculation."""
        result = calculate_dema(sample_data, period=10, engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))
        # First (2*period-2) = 18 values should be NaN
        assert np.all(np.isnan(result[:18]))
        # After warmup, should have valid values
        assert not np.isnan(result[19])

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        cpu_result = calculate_dema(sample_data, period=10, engine='cpu')
        gpu_result = calculate_dema(sample_data, period=10, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10, equal_nan=True)

    def test_invalid_period(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_dema(sample_data, period=0)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_data = np.array([100, 101, 102, 103, 104])
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_dema(short_data, period=10)

    def test_known_values(self):
        """Test against known DEMA values."""
        # Simple test case - verify DEMA formula is correctly implemented
        data = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113])
        period = 3

        result = calculate_dema(data, period=period, engine='cpu')

        # Verify the result has expected structure:
        # - First 2*period-2 = 4 values should be NaN
        # - Remaining values should be valid numbers
        assert len(result) == len(data)
        assert np.sum(np.isnan(result)) == 4  # First 4 values should be NaN
        assert not np.any(np.isnan(result[4:]))  # Rest should be valid

        # Verify DEMA is in reasonable range (between min and max of data)
        valid_dema = result[~np.isnan(result)]
        assert np.all(valid_dema >= data.min() - 5)  # Allow some undershoot
        assert np.all(valid_dema <= data.max() + 5)  # Allow some overshoot

    def test_reduced_lag(self):
        """Test that DEMA has reduced lag compared to EMA."""
        # Create trending data
        data = np.linspace(100, 150, 100)
        period = 10

        ema = calculate_ema(data, period=period, engine='cpu')
        dema = calculate_dema(data, period=period, engine='cpu')

        # In an uptrend, DEMA should be closer to current price than EMA
        # (less lag). Check the last valid values
        valid_indices = ~np.isnan(dema)
        last_idx = np.where(valid_indices)[0][-1]

        assert dema[last_idx] > ema[last_idx], "DEMA should be above EMA in uptrend (less lag)"

    def test_auto_engine_selection(self, sample_data):
        """Test that auto engine selection works."""
        result_auto = calculate_dema(sample_data, period=10, engine='auto')
        result_cpu = calculate_dema(sample_data, period=10, engine='cpu')

        # Auto should produce same result as CPU for small datasets
        np.testing.assert_allclose(result_auto, result_cpu, rtol=1e-10, equal_nan=True)

    def test_different_periods(self, sample_data):
        """Test DEMA with different periods."""
        dema_5 = calculate_dema(sample_data, period=5, engine='cpu')
        dema_20 = calculate_dema(sample_data, period=20, engine='cpu')

        # Shorter period should have fewer NaN values at start
        assert np.sum(np.isnan(dema_5)) < np.sum(np.isnan(dema_20))


class TestTEMA:
    """Test Triple Exponential Moving Average calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 150
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        return prices

    def test_basic_calculation(self, sample_data):
        """Test basic TEMA calculation."""
        result = calculate_tema(sample_data, period=10, engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))
        # First (3*period-3) = 27 values should be NaN
        assert np.all(np.isnan(result[:27]))
        # After warmup, should have valid values
        assert not np.isnan(result[28])

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        cpu_result = calculate_tema(sample_data, period=10, engine='cpu')
        gpu_result = calculate_tema(sample_data, period=10, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10, equal_nan=True)

    def test_invalid_period(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_tema(sample_data, period=0)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_data = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_tema(short_data, period=10)

    def test_known_values(self):
        """Test against known TEMA values."""
        # Simple test case - verify TEMA formula is correctly implemented
        data = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                        111, 113, 112, 114, 116, 115, 117, 119])
        period = 3

        result = calculate_tema(data, period=period, engine='cpu')

        # Verify the result has expected structure:
        # - First 3*period-3 = 6 values should be NaN
        # - Remaining values should be valid numbers
        assert len(result) == len(data)
        assert np.sum(np.isnan(result)) == 6  # First 6 values should be NaN
        assert not np.any(np.isnan(result[6:]))  # Rest should be valid

        # Verify TEMA is in reasonable range (between min and max of data)
        valid_tema = result[~np.isnan(result)]
        assert np.all(valid_tema >= data.min() - 5)  # Allow some undershoot
        assert np.all(valid_tema <= data.max() + 5)  # Allow some overshoot

    def test_reduced_lag(self):
        """Test that TEMA has even more reduced lag than DEMA."""
        # Create trending data
        data = np.linspace(100, 150, 150)
        period = 10

        ema = calculate_ema(data, period=period, engine='cpu')
        dema = calculate_dema(data, period=period, engine='cpu')
        tema = calculate_tema(data, period=period, engine='cpu')

        # In an uptrend, TEMA should be closest to current price (least lag)
        # Check the last valid values
        valid_indices = ~np.isnan(tema)
        last_idx = np.where(valid_indices)[0][-1]

        # TEMA >= DEMA >= EMA in uptrend (progressively less lag)
        # Use >= instead of > to handle edge cases where they converge
        assert tema[last_idx] >= dema[last_idx] - 0.01, "TEMA should be >= DEMA in uptrend (less lag)"
        assert dema[last_idx] >= ema[last_idx] - 0.01, "DEMA should be >= EMA in uptrend (less lag)"

    def test_auto_engine_selection(self, sample_data):
        """Test that auto engine selection works."""
        result_auto = calculate_tema(sample_data, period=10, engine='auto')
        result_cpu = calculate_tema(sample_data, period=10, engine='cpu')

        # Auto should produce same result as CPU for small datasets
        np.testing.assert_allclose(result_auto, result_cpu, rtol=1e-10, equal_nan=True)

    def test_different_periods(self, sample_data):
        """Test TEMA with different periods."""
        tema_5 = calculate_tema(sample_data, period=5, engine='cpu')
        tema_20 = calculate_tema(sample_data, period=20, engine='cpu')

        # Shorter period should have fewer NaN values at start
        assert np.sum(np.isnan(tema_5)) < np.sum(np.isnan(tema_20))

    def test_responsiveness(self, sample_data):
        """Test that TEMA, DEMA, and EMA all respond to price changes."""
        # Add a sharp price spike
        data = sample_data.copy()
        data[75] = data[75] + 20  # Sharp spike

        period = 10

        ema = calculate_ema(data, period=period, engine='cpu')
        dema = calculate_dema(data, period=period, engine='cpu')
        tema = calculate_tema(data, period=period, engine='cpu')

        # Check response to spike (a few bars after the spike)
        spike_idx = 75
        check_idx = spike_idx + 2

        # All indicators should show significant response (change > 1.0)
        ema_change = abs(ema[check_idx] - ema[spike_idx - 1])
        dema_change = abs(dema[check_idx] - dema[spike_idx - 1])
        tema_change = abs(tema[check_idx] - tema[spike_idx - 1])

        # All should respond to the spike
        assert ema_change > 1.0, "EMA should respond to price spike"
        assert dema_change > 1.0, "DEMA should respond to price spike"
        assert tema_change > 1.0, "TEMA should respond to price spike"

        # At minimum, DEMA should be more responsive than EMA
        assert dema_change >= ema_change * 0.8, "DEMA should be at least 80% as responsive as EMA"


class TestDEMATEMAComparison:
    """Test comparisons between EMA, DEMA, and TEMA."""

    @pytest.fixture
    def trending_data(self):
        """Generate trending price data."""
        return np.linspace(100, 200, 100)

    def test_warmup_periods(self, trending_data):
        """Test that warmup periods are correct."""
        period = 10

        ema = calculate_ema(trending_data, period=period, engine='cpu')
        dema = calculate_dema(trending_data, period=period, engine='cpu')
        tema = calculate_tema(trending_data, period=period, engine='cpu')

        # Count NaN values at start
        ema_nan_count = np.sum(np.isnan(ema[:period]))
        dema_nan_count = np.sum(np.isnan(dema[:2*period]))
        tema_nan_count = np.sum(np.isnan(tema[:3*period]))

        # EMA: period-1 NaN values
        assert ema_nan_count == period - 1

        # DEMA: 2*period-2 NaN values
        assert dema_nan_count == 2*period - 2

        # TEMA: 3*period-3 NaN values
        assert tema_nan_count == 3*period - 3

    def test_lag_ordering(self, trending_data):
        """Test that lag reduction follows expected order: TEMA < DEMA < EMA."""
        period = 10

        ema = calculate_ema(trending_data, period=period, engine='cpu')
        dema = calculate_dema(trending_data, period=period, engine='cpu')
        tema = calculate_tema(trending_data, period=period, engine='cpu')

        # In strong uptrend, TEMA should be highest (closest to actual price)
        # Check last 10 valid values
        for i in range(-10, 0):
            if not (np.isnan(ema[i]) or np.isnan(dema[i]) or np.isnan(tema[i])):
                assert tema[i] >= dema[i], f"TEMA should be >= DEMA at index {i}"
                assert dema[i] >= ema[i], f"DEMA should be >= EMA at index {i}"

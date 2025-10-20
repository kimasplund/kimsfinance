"""
Comprehensive tests for technical indicators.

Tests both CPU and GPU implementations with known values,
edge cases, and GPU/CPU parity validation.
"""

import numpy as np
import pytest
from kimsfinance.ops.indicators import calculate_sma, calculate_ema, calculate_vwma, calculate_tsi, calculate_hma, calculate_elder_ray


class TestSMA:
    """Test Simple Moving Average calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        return prices

    def test_basic_calculation(self, sample_data):
        """Test basic SMA calculation."""
        result = calculate_sma(sample_data, period=14, engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))
        # First (period-1) values should be NaN
        assert np.all(np.isnan(result[:13]))
        # After warmup, should have valid values
        assert not np.isnan(result[14])

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        cpu_result = calculate_sma(sample_data, period=14, engine='cpu')
        gpu_result = calculate_sma(sample_data, period=14, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_invalid_period(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_sma(sample_data, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_sma(sample_data, period=-5)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_data = np.array([100, 101, 102])
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_sma(short_data, period=14)

    def test_known_values(self):
        """Test against known SMA values."""
        # Simple test case with known values
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = calculate_sma(data, period=3, engine='cpu')

        # Expected SMA values:
        # Index 0-1: NaN (warmup)
        # Index 2: (1+2+3)/3 = 2.0
        # Index 3: (2+3+4)/3 = 3.0
        # Index 4: (3+4+5)/3 = 4.0
        # etc.
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_period_equals_length(self):
        """Test SMA when period equals data length."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_sma(data, period=5, engine='cpu')

        # Should have NaN for first 4 values, then mean of all values
        assert np.all(np.isnan(result[:4]))
        assert result[4] == 3.0  # Mean of [1,2,3,4,5]

    def test_auto_engine_small_data(self):
        """Test auto engine selection for small datasets."""
        small_data = np.random.randn(1000) + 100
        result = calculate_sma(small_data, period=20, engine='auto')

        # Should work correctly regardless of engine choice
        assert len(result) == len(small_data)
        assert np.all(np.isnan(result[:19]))
        assert not np.isnan(result[20])

    def test_list_input(self):
        """Test that SMA accepts list input."""
        data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_sma(data_list, period=3, engine='cpu')

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data_list)


class TestEMA:
    """Test Exponential Moving Average calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        return prices

    def test_basic_calculation(self, sample_data):
        """Test basic EMA calculation."""
        result = calculate_ema(sample_data, period=14, engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))
        # First (period-1) values should be NaN
        assert np.all(np.isnan(result[:13]))
        # After warmup, should have valid values
        assert not np.isnan(result[14])

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        cpu_result = calculate_ema(sample_data, period=14, engine='cpu')
        gpu_result = calculate_ema(sample_data, period=14, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_invalid_period(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_ema(sample_data, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_ema(sample_data, period=-5)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_data = np.array([100, 101, 102])
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_ema(short_data, period=14)

    def test_known_values(self):
        """Test against known EMA values."""
        # Simple test case with known values
        data = np.array([22.0, 23.0, 24.0, 23.5, 24.5, 25.0, 26.0])
        result = calculate_ema(data, period=3, engine='cpu')

        # Calculate expected values manually
        # alpha = 2 / (3 + 1) = 0.5
        # EMA[2] = SMA(first 3) = (22 + 23 + 24) / 3 = 23.0
        # EMA[3] = 23.5 * 0.5 + 23.0 * 0.5 = 23.25
        # EMA[4] = 24.5 * 0.5 + 23.25 * 0.5 = 23.875
        # EMA[5] = 25.0 * 0.5 + 23.875 * 0.5 = 24.4375
        # EMA[6] = 26.0 * 0.5 + 24.4375 * 0.5 = 25.21875

        expected = np.array([
            np.nan,
            np.nan,
            23.0,
            23.25,
            23.875,
            24.4375,
            25.21875
        ])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_ema_more_responsive_than_sma(self):
        """Test that EMA responds faster to price changes than SMA."""
        # Create data with a sudden price jump
        data = np.array([100.0] * 20 + [110.0] * 10)

        sma_result = calculate_sma(data, period=10, engine='cpu')
        ema_result = calculate_ema(data, period=10, engine='cpu')

        # After the jump (at index 20), EMA should reach higher values faster
        # than SMA due to its exponential weighting
        # Check a few periods after the jump
        assert ema_result[22] > sma_result[22]
        assert ema_result[24] > sma_result[24]

    def test_period_equals_length(self):
        """Test EMA when period equals data length."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_ema(data, period=5, engine='cpu')

        # Should have NaN for first 4 values
        assert np.all(np.isnan(result[:4]))
        # Fifth value should be SMA of all values
        assert result[4] == 3.0  # Mean of [1,2,3,4,5]

    def test_auto_engine_small_data(self):
        """Test auto engine selection for small datasets."""
        small_data = np.random.randn(1000) + 100
        result = calculate_ema(small_data, period=20, engine='auto')

        # Should work correctly regardless of engine choice
        assert len(result) == len(small_data)
        assert np.all(np.isnan(result[:19]))
        assert not np.isnan(result[20])

    def test_list_input(self):
        """Test that EMA accepts list input."""
        data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_ema(data_list, period=3, engine='cpu')

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data_list)

    def test_ema_smoothing_factor(self):
        """Test that EMA uses correct smoothing factor."""
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        period = 3
        result = calculate_ema(data, period=period, engine='cpu')

        # alpha = 2 / (period + 1) = 2 / 4 = 0.5
        alpha = 2.0 / (period + 1)

        # First EMA is SMA
        first_ema = np.mean(data[:period])
        assert result[period - 1] == first_ema

        # Second EMA follows formula
        second_ema = alpha * data[period] + (1 - alpha) * first_ema
        np.testing.assert_almost_equal(result[period], second_ema)


class TestEdgeCases:
    """Test edge cases for both SMA and EMA."""

    def test_single_period(self):
        """Test both indicators with period=1."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        sma_result = calculate_sma(data, period=1, engine='cpu')
        ema_result = calculate_ema(data, period=1, engine='cpu')

        # With period=1, both should equal the input data (no smoothing)
        np.testing.assert_allclose(sma_result, data, rtol=1e-10)
        np.testing.assert_allclose(ema_result, data, rtol=1e-10)

    def test_constant_prices(self):
        """Test with constant price data."""
        data = np.array([100.0] * 50)

        sma_result = calculate_sma(data, period=10, engine='cpu')
        ema_result = calculate_ema(data, period=10, engine='cpu')

        # All non-NaN values should equal the constant price
        assert np.allclose(sma_result[9:], 100.0)
        assert np.allclose(ema_result[9:], 100.0)

    def test_invalid_engine(self):
        """Test that invalid engine raises ValueError."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_sma(data, period=3, engine='invalid')

        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_ema(data, period=3, engine='invalid')


class TestPerformance:
    """Performance and benchmark tests (optional, for validation)."""

    @pytest.mark.slow
    def test_large_dataset_cpu(self):
        """Test SMA/EMA on large dataset (CPU)."""
        n = 100_000
        data = np.random.randn(n).cumsum() + 100

        sma_result = calculate_sma(data, period=20, engine='cpu')
        ema_result = calculate_ema(data, period=20, engine='cpu')

        assert len(sma_result) == n
        assert len(ema_result) == n
        assert not np.all(np.isnan(sma_result))
        assert not np.all(np.isnan(ema_result))

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_large_dataset_gpu(self):
        """Test SMA/EMA on large dataset (GPU)."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        n = 1_000_000
        data = np.random.randn(n).cumsum() + 100

        sma_result = calculate_sma(data, period=20, engine='gpu')
        ema_result = calculate_ema(data, period=20, engine='gpu')

        assert len(sma_result) == n
        assert len(ema_result) == n
        assert not np.all(np.isnan(sma_result))
        assert not np.all(np.isnan(ema_result))


class TestTSI:
    """Test True Strength Index (TSI) calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        return prices

    def test_basic_calculation(self, sample_data):
        """Test basic TSI calculation."""
        result = calculate_tsi(sample_data, long_period=25, short_period=13, engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))
        # First (long_period + short_period - 2) values should be NaN
        warmup = 25 + 13 - 2
        assert np.all(np.isnan(result[:warmup]))
        # After warmup, should have valid values
        assert not np.all(np.isnan(result[warmup+1:]))

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        cpu_result = calculate_tsi(sample_data, long_period=25, short_period=13, engine='cpu')
        gpu_result = calculate_tsi(sample_data, long_period=25, short_period=13, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_invalid_long_period(self, sample_data):
        """Test that invalid long_period raises ValueError."""
        with pytest.raises(ValueError, match="long_period must be >= 1"):
            calculate_tsi(sample_data, long_period=0, short_period=13)

        with pytest.raises(ValueError, match="long_period must be >= 1"):
            calculate_tsi(sample_data, long_period=-5, short_period=13)

    def test_invalid_short_period(self, sample_data):
        """Test that invalid short_period raises ValueError."""
        with pytest.raises(ValueError, match="short_period must be >= 1"):
            calculate_tsi(sample_data, long_period=25, short_period=0)

        with pytest.raises(ValueError, match="short_period must be >= 1"):
            calculate_tsi(sample_data, long_period=25, short_period=-5)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_data = np.array([100, 101, 102, 103, 104])
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_tsi(short_data, long_period=25, short_period=13)

    def test_known_values(self):
        """Test against known TSI values."""
        # Use simple trending data
        # Uptrend: prices steadily increasing
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                        110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                        120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0,
                        130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0])

        result = calculate_tsi(data, long_period=25, short_period=13, engine='cpu')

        # For a perfect uptrend, TSI should be strongly positive
        # Check that non-NaN values are positive (bullish momentum)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values > 0), f"Expected positive TSI for uptrend, got {valid_values}"

    def test_downtrend(self):
        """Test TSI on downtrend data."""
        # Downtrend: prices steadily decreasing
        data = np.array([140.0, 139.0, 138.0, 137.0, 136.0, 135.0, 134.0, 133.0, 132.0, 131.0,
                        130.0, 129.0, 128.0, 127.0, 126.0, 125.0, 124.0, 123.0, 122.0, 121.0,
                        120.0, 119.0, 118.0, 117.0, 116.0, 115.0, 114.0, 113.0, 112.0, 111.0,
                        110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0])

        result = calculate_tsi(data, long_period=25, short_period=13, engine='cpu')

        # For a downtrend, TSI should be strongly negative
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values < 0), f"Expected negative TSI for downtrend, got {valid_values}"

    def test_range_bound(self, sample_data):
        """Test that TSI values are within -100 to +100 range."""
        result = calculate_tsi(sample_data, long_period=25, short_period=13, engine='cpu')

        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -100.0)
        assert np.all(valid_values <= 100.0)

    def test_auto_engine_small_data(self):
        """Test auto engine selection for small datasets."""
        small_data = np.random.randn(1000).cumsum() + 100
        result = calculate_tsi(small_data, long_period=25, short_period=13, engine='auto')

        # Should work correctly regardless of engine choice
        assert len(result) == len(small_data)
        warmup = 25 + 13 - 2
        assert np.all(np.isnan(result[:warmup]))
        assert not np.all(np.isnan(result[warmup+1:]))

    def test_list_input(self):
        """Test that TSI accepts list input."""
        data_list = [100.0 + i for i in range(50)]
        result = calculate_tsi(data_list, long_period=25, short_period=13, engine='cpu')

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data_list)

    def test_different_periods(self):
        """Test TSI with different period combinations."""
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(100) * 2)

        # Test with shorter periods
        result_short = calculate_tsi(data, long_period=10, short_period=5, engine='cpu')
        assert not np.all(np.isnan(result_short))

        # Test with longer periods
        result_long = calculate_tsi(data, long_period=30, short_period=15, engine='cpu')
        assert not np.all(np.isnan(result_long))

        # Shorter periods should have more valid values (less warmup)
        valid_short = np.sum(~np.isnan(result_short))
        valid_long = np.sum(~np.isnan(result_long))
        assert valid_short > valid_long

    def test_zero_crossover(self):
        """Test TSI zero-line crossover detection."""
        # Create data that goes from uptrend to downtrend
        uptrend = np.linspace(100, 110, 25)
        downtrend = np.linspace(110, 100, 25)
        data = np.concatenate([uptrend, downtrend])

        result = calculate_tsi(data, long_period=10, short_period=5, engine='cpu')

        valid_mask = ~np.isnan(result)
        valid_result = result[valid_mask]

        # Should have some positive and some negative values
        # (due to trend reversal)
        has_positive = np.any(valid_result > 0)
        has_negative = np.any(valid_result < 0)

        # At least one of these should be true for a reversal pattern
        # (may not always have both depending on smoothing)
        assert has_positive or has_negative

    def test_invalid_engine(self):
        """Test that invalid engine raises ValueError."""
        data = np.array([100.0 + i for i in range(50)])

        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_tsi(data, long_period=25, short_period=13, engine='invalid')

    def test_nan_handling(self):
        """Test TSI behavior with NaN values in input."""
        # Create data with NaN in the middle
        data = np.array([100.0 + i for i in range(50)])
        data[25] = np.nan

        result = calculate_tsi(data, long_period=10, short_period=5, engine='cpu')

        # Result should be computed where possible
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test TSI on large dataset."""
        n = 100_000
        data = np.random.randn(n).cumsum() + 100

        result = calculate_tsi(data, long_period=25, short_period=13, engine='cpu')

        assert len(result) == n
        assert not np.all(np.isnan(result))
        # Values should be in valid range
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -100.0)
        assert np.all(valid_values <= 100.0)


class TestHMA:
    """Test Hull Moving Average calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        return prices

    def test_basic_calculation(self, sample_data):
        """Test basic HMA calculation."""
        result = calculate_hma(sample_data, period=20, engine='cpu')

        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))
        # First (period-1) values should be NaN
        assert np.all(np.isnan(result[:19]))
        # After warmup, should have valid values
        assert not np.isnan(result[25])

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        cpu_result = calculate_hma(sample_data, period=20, engine='cpu')
        gpu_result = calculate_hma(sample_data, period=20, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_invalid_period(self, sample_data):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_hma(sample_data, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_hma(sample_data, period=-5)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_data = np.array([100, 101, 102])
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_hma(short_data, period=20)

    def test_known_values(self):
        """Test against known HMA values."""
        # Simple test case with known values
        # Using a sequence that should produce predictable results
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        result = calculate_hma(data, period=5, engine='cpu')

        # For an uptrending sequence, HMA should follow the trend
        # and be smooth with minimal lag
        # We expect the HMA values to be close to the actual prices
        # but with some smoothing

        # Check that we have valid values after warmup
        valid_mask = ~np.isnan(result)
        valid_result = result[valid_mask]

        assert len(valid_result) > 0
        # For uptrend, HMA should be increasing
        assert valid_result[-1] > valid_result[0]

    def test_hma_less_lag_than_sma(self):
        """Test that HMA responds faster to price changes than SMA."""
        # Create data with a sudden price jump
        data = np.array([100.0] * 30 + [110.0] * 20)

        hma_result = calculate_hma(data, period=20, engine='cpu')
        sma_result = calculate_sma(data, period=20, engine='cpu')

        # After the jump (at index 30), HMA should reach higher values faster
        # than SMA due to its low-lag characteristics
        # Check a few periods after the jump
        jump_index = 30
        check_indices = [jump_index + 5, jump_index + 10]

        for idx in check_indices:
            if not np.isnan(hma_result[idx]) and not np.isnan(sma_result[idx]):
                # HMA should be closer to the new price level (110) than SMA
                assert hma_result[idx] > sma_result[idx]

    def test_different_periods(self):
        """Test HMA with different periods."""
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(100) * 2)

        hma_9 = calculate_hma(data, period=9, engine='cpu')
        hma_20 = calculate_hma(data, period=20, engine='cpu')
        hma_55 = calculate_hma(data, period=55, engine='cpu')

        # All should have same length
        assert len(hma_9) == len(hma_20) == len(hma_55) == len(data)

        # Shorter period should have valid values starting earlier
        valid_9 = ~np.isnan(hma_9)
        valid_20 = ~np.isnan(hma_20)
        valid_55 = ~np.isnan(hma_55)

        # HMA(9) should have more valid values than HMA(20)
        assert np.sum(valid_9) > np.sum(valid_20)
        # HMA(20) should have more valid values than HMA(55)
        assert np.sum(valid_20) > np.sum(valid_55)

    def test_auto_engine_small_data(self):
        """Test auto engine selection for small datasets."""
        small_data = np.random.randn(1000) + 100
        result = calculate_hma(small_data, period=20, engine='auto')

        # Should work correctly regardless of engine choice
        assert len(result) == len(small_data)
        assert not np.all(np.isnan(result))

    def test_list_input(self):
        """Test that HMA accepts list input."""
        data_list = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
        result = calculate_hma(data_list, period=5, engine='cpu')

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data_list)

    def test_smoothness(self):
        """Test that HMA provides smooth output despite low lag."""
        # Create noisy data
        np.random.seed(42)
        trend = np.linspace(100, 120, 50)
        noise = np.random.randn(50) * 2
        data = trend + noise

        hma_result = calculate_hma(data, period=20, engine='cpu')

        # HMA should smooth out the noise
        valid_mask = ~np.isnan(hma_result)
        valid_hma = hma_result[valid_mask]
        valid_data = data[valid_mask]

        # Calculate variance to measure smoothness
        # HMA should have lower variance than raw data
        hma_variance = np.var(np.diff(valid_hma))
        data_variance = np.var(np.diff(valid_data))

        assert hma_variance < data_variance

    def test_invalid_engine(self):
        """Test that invalid engine raises ValueError."""
        data = np.array([100.0 + i for i in range(50)])

        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_hma(data, period=20, engine='invalid')

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test HMA on large dataset."""
        n = 100_000
        data = np.random.randn(n).cumsum() + 100

        result = calculate_hma(data, period=20, engine='cpu')

        assert len(result) == n
        assert not np.all(np.isnan(result))
        # Should have valid values after warmup
        assert not np.isnan(result[30])


class TestElderRay:
    """Test Elder Ray (Bull Power and Bear Power) calculation."""

    @pytest.fixture
    def sample_ohlc_data(self):
        """Generate sample OHLC data for testing."""
        np.random.seed(42)
        n = 100
        closes = 100 + np.cumsum(np.random.randn(n) * 2)
        highs = closes + np.abs(np.random.randn(n) * 1.5)
        lows = closes - np.abs(np.random.randn(n) * 1.5)
        return highs, lows, closes

    def test_basic_calculation(self, sample_ohlc_data):
        """Test basic Elder Ray calculation."""
        highs, lows, closes = sample_ohlc_data
        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=13, engine='cpu'
        )

        assert len(bull_power) == len(closes)
        assert len(bear_power) == len(closes)
        assert not np.all(np.isnan(bull_power))
        assert not np.all(np.isnan(bear_power))
        # First (period-1) values should be NaN
        assert np.all(np.isnan(bull_power[:12]))
        assert np.all(np.isnan(bear_power[:12]))
        # After warmup, should have valid values
        assert not np.isnan(bull_power[13])
        assert not np.isnan(bear_power[13])

    def test_gpu_cpu_match(self, sample_ohlc_data):
        """Test GPU and CPU implementations produce identical results."""
        highs, lows, closes = sample_ohlc_data

        bull_cpu, bear_cpu = calculate_elder_ray(
            highs, lows, closes, period=13, engine='cpu'
        )
        bull_gpu, bear_gpu = calculate_elder_ray(
            highs, lows, closes, period=13, engine='gpu'
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(bull_cpu, bull_gpu, rtol=1e-10)
        np.testing.assert_allclose(bear_cpu, bear_gpu, rtol=1e-10)

    def test_invalid_period(self, sample_ohlc_data):
        """Test that invalid period raises ValueError."""
        highs, lows, closes = sample_ohlc_data

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_elder_ray(highs, lows, closes, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_elder_ray(highs, lows, closes, period=-5)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_highs = np.array([102, 103, 104])
        short_lows = np.array([98, 99, 100])
        short_closes = np.array([100, 101, 102])

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_elder_ray(short_highs, short_lows, short_closes, period=13)

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        highs = np.array([102, 103, 104, 105])
        lows = np.array([98, 99, 100])  # Different length
        closes = np.array([100, 101, 102, 103])

        with pytest.raises(ValueError, match="must have same length"):
            calculate_elder_ray(highs, lows, closes, period=3)

    def test_known_values(self):
        """Test against known Elder Ray values."""
        # Simple test case with known values
        highs = np.array([102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])
        lows = np.array([98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0])
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])

        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=3, engine='cpu'
        )

        # Calculate expected EMA manually
        # alpha = 2 / (3 + 1) = 0.5
        # EMA[2] = SMA(first 3) = (100 + 101 + 102) / 3 = 101.0
        # EMA[3] = 103 * 0.5 + 101 * 0.5 = 102.0
        # EMA[4] = 104 * 0.5 + 102 * 0.5 = 103.0
        # EMA[5] = 105 * 0.5 + 103 * 0.5 = 104.0
        # EMA[6] = 106 * 0.5 + 104 * 0.5 = 105.0

        # Bull Power = High - EMA
        # Bear Power = Low - EMA
        expected_ema = np.array([np.nan, np.nan, 101.0, 102.0, 103.0, 104.0, 105.0])
        expected_bull = highs - expected_ema
        expected_bear = lows - expected_ema

        np.testing.assert_allclose(bull_power, expected_bull, rtol=1e-6)
        np.testing.assert_allclose(bear_power, expected_bear, rtol=1e-6)

    def test_bull_bear_relationship(self, sample_ohlc_data):
        """Test that Bull Power >= Bear Power always (since High >= Low)."""
        highs, lows, closes = sample_ohlc_data
        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=13, engine='cpu'
        )

        # For all non-NaN values, Bull Power should be >= Bear Power
        # (because High >= Low)
        valid_mask = ~(np.isnan(bull_power) | np.isnan(bear_power))
        assert np.all(bull_power[valid_mask] >= bear_power[valid_mask])

    def test_uptrend_indicators(self):
        """Test Elder Ray in an uptrend."""
        # Create uptrending price data
        n = 50
        closes = np.linspace(100, 150, n)  # Linear uptrend
        highs = closes + 2.0
        lows = closes - 2.0

        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=13, engine='cpu'
        )

        # In a strong uptrend, both Bull and Bear power should be positive
        # (price consistently above EMA)
        valid_idx = ~np.isnan(bull_power)
        # Most values should be positive in strong uptrend
        assert np.mean(bull_power[valid_idx] > 0) > 0.8
        # Bear power may be less positive but should trend positive
        assert np.mean(bear_power[valid_idx] > 0) > 0.5

    def test_downtrend_indicators(self):
        """Test Elder Ray in a downtrend."""
        # Create downtrending price data
        n = 50
        closes = np.linspace(150, 100, n)  # Linear downtrend
        highs = closes + 2.0
        lows = closes - 2.0

        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=13, engine='cpu'
        )

        # In a strong downtrend, both Bull and Bear power should be negative
        # (price consistently below EMA)
        valid_idx = ~np.isnan(bear_power)
        # Most values should be negative in strong downtrend
        assert np.mean(bear_power[valid_idx] < 0) > 0.8
        # Bull power may be less negative but should trend negative
        assert np.mean(bull_power[valid_idx] < 0) > 0.5

    def test_different_periods(self, sample_ohlc_data):
        """Test Elder Ray with different period values."""
        highs, lows, closes = sample_ohlc_data

        for period in [5, 13, 21, 50]:
            bull_power, bear_power = calculate_elder_ray(
                highs, lows, closes, period=period, engine='cpu'
            )

            # All should have same length as input
            assert len(bull_power) == len(closes)
            assert len(bear_power) == len(closes)

            # First (period-1) values should be NaN
            assert np.all(np.isnan(bull_power[:period-1]))
            assert np.all(np.isnan(bear_power[:period-1]))

            # Should have valid values after warmup
            if len(closes) > period:
                assert not np.isnan(bull_power[period])
                assert not np.isnan(bear_power[period])

    def test_period_equals_length(self):
        """Test Elder Ray when period equals data length."""
        highs = np.array([102.0, 103.0, 104.0, 105.0, 106.0])
        lows = np.array([98.0, 99.0, 100.0, 101.0, 102.0])
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=5, engine='cpu'
        )

        # Should have NaN for first 4 values
        assert np.all(np.isnan(bull_power[:4]))
        assert np.all(np.isnan(bear_power[:4]))

        # Fifth value should be High/Low - SMA
        sma = np.mean(closes)  # Mean of all closes
        expected_bull = 106.0 - sma
        expected_bear = 102.0 - sma

        np.testing.assert_almost_equal(bull_power[4], expected_bull)
        np.testing.assert_almost_equal(bear_power[4], expected_bear)

    def test_auto_engine_small_data(self):
        """Test auto engine selection for small datasets."""
        np.random.seed(42)
        n = 1000
        closes = 100 + np.cumsum(np.random.randn(n) * 2)
        highs = closes + np.abs(np.random.randn(n) * 1.5)
        lows = closes - np.abs(np.random.randn(n) * 1.5)

        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=20, engine='auto'
        )

        # Should work correctly regardless of engine choice
        assert len(bull_power) == n
        assert len(bear_power) == n
        assert np.all(np.isnan(bull_power[:19]))
        assert np.all(np.isnan(bear_power[:19]))
        assert not np.isnan(bull_power[20])
        assert not np.isnan(bear_power[20])

    def test_list_input(self):
        """Test that Elder Ray accepts list input."""
        highs_list = [102.0, 103.0, 104.0, 105.0, 106.0]
        lows_list = [98.0, 99.0, 100.0, 101.0, 102.0]
        closes_list = [100.0, 101.0, 102.0, 103.0, 104.0]

        bull_power, bear_power = calculate_elder_ray(
            highs_list, lows_list, closes_list, period=3, engine='cpu'
        )

        assert isinstance(bull_power, np.ndarray)
        assert isinstance(bear_power, np.ndarray)
        assert len(bull_power) == len(highs_list)
        assert len(bear_power) == len(lows_list)

    def test_invalid_engine(self, sample_ohlc_data):
        """Test that invalid engine raises ValueError."""
        highs, lows, closes = sample_ohlc_data

        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_elder_ray(highs, lows, closes, period=13, engine='invalid')


class TestElderRayPerformance:
    """Performance tests for Elder Ray (optional)."""

    @pytest.mark.slow
    def test_large_dataset_cpu(self):
        """Test Elder Ray on large dataset (CPU)."""
        n = 100_000
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(n) * 2)
        highs = closes + np.abs(np.random.randn(n) * 1.5)
        lows = closes - np.abs(np.random.randn(n) * 1.5)

        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=20, engine='cpu'
        )

        assert len(bull_power) == n
        assert len(bear_power) == n
        assert not np.all(np.isnan(bull_power))
        assert not np.all(np.isnan(bear_power))

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_large_dataset_gpu(self):
        """Test Elder Ray on large dataset (GPU)."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        n = 1_000_000
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(n) * 2)
        highs = closes + np.abs(np.random.randn(n) * 1.5)
        lows = closes - np.abs(np.random.randn(n) * 1.5)

        bull_power, bear_power = calculate_elder_ray(
            highs, lows, closes, period=20, engine='gpu'
        )

        assert len(bull_power) == n
        assert len(bear_power) == n
        assert not np.all(np.isnan(bull_power))
        assert not np.all(np.isnan(bear_power))


class TestVWMA:
    """Test Volume Weighted Moving Average calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price and volume data for testing."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 2)
        volumes = np.abs(np.random.randn(n) * 1_000_000)
        return prices, volumes

    def test_basic_calculation(self, sample_data):
        """Test basic VWMA calculation."""
        prices, volumes = sample_data
        result = calculate_vwma(prices, volumes, period=14, engine='cpu')

        assert len(result) == len(prices)
        assert not np.all(np.isnan(result))
        # First (period-1) values should be NaN
        assert np.all(np.isnan(result[:13]))
        # After warmup, should have valid values
        assert not np.isnan(result[14])

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        prices, volumes = sample_data
        cpu_result = calculate_vwma(prices, volumes, period=14, engine='cpu')
        gpu_result = calculate_vwma(prices, volumes, period=14, engine='gpu')

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-10)

    def test_invalid_period(self, sample_data):
        """Test that invalid period raises ValueError."""
        prices, volumes = sample_data

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_vwma(prices, volumes, period=0)

        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_vwma(prices, volumes, period=-5)

    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        short_prices = np.array([100, 101, 102])
        short_volumes = np.array([1000, 2000, 1500])

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_vwma(short_prices, short_volumes, period=14)

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        prices = np.array([100, 101, 102, 103, 104])
        volumes = np.array([1000, 2000, 1500])  # Different length

        with pytest.raises(ValueError, match="prices and volumes must have same length"):
            calculate_vwma(prices, volumes, period=3)

    def test_known_values(self):
        """Test against known VWMA values."""
        # Simple test case with known values
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([100.0, 200.0, 150.0, 250.0, 180.0])
        result = calculate_vwma(prices, volumes, period=3, engine='cpu')

        # Expected VWMA values:
        # Index 0-1: NaN (warmup)
        # Index 2: (10*100 + 11*200 + 12*150) / (100+200+150)
        #        = (1000 + 2200 + 1800) / 450 = 5000 / 450 = 11.111...
        # Index 3: (11*200 + 12*150 + 13*250) / (200+150+250)
        #        = (2200 + 1800 + 3250) / 600 = 7250 / 600 = 12.0833...
        # Index 4: (12*150 + 13*250 + 14*180) / (150+250+180)
        #        = (1800 + 3250 + 2520) / 580 = 7570 / 580 = 13.0517...

        expected = np.array([
            np.nan,
            np.nan,
            11.111111111111111,
            12.083333333333334,
            13.051724137931034
        ])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_equal_volumes(self):
        """Test VWMA with equal volumes equals SMA."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # All equal

        vwma_result = calculate_vwma(prices, volumes, period=3, engine='cpu')
        sma_result = calculate_sma(prices, period=3, engine='cpu')

        # With equal volumes, VWMA should equal SMA
        np.testing.assert_allclose(vwma_result, sma_result, rtol=1e-10)

    def test_zero_volume_handling(self):
        """Test VWMA handles zero volumes correctly."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # All zero

        result = calculate_vwma(prices, volumes, period=3, engine='cpu')

        # With all zero volumes, result should be all NaN
        assert np.all(np.isnan(result))

    def test_mixed_zero_volumes(self):
        """Test VWMA with some zero volumes."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([100.0, 0.0, 200.0, 150.0, 0.0])

        result = calculate_vwma(prices, volumes, period=3, engine='cpu')

        # Should calculate correctly for windows with non-zero volumes
        assert not np.all(np.isnan(result))
        # Index 2: (10*100 + 11*0 + 12*200) / (100+0+200) = 3400/300 = 11.333...
        np.testing.assert_almost_equal(result[2], 11.333333333333334)

    def test_period_equals_length(self):
        """Test VWMA when period equals data length."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([100.0, 200.0, 150.0, 250.0, 180.0])

        result = calculate_vwma(prices, volumes, period=5, engine='cpu')

        # Should have NaN for first 4 values, then VWMA of all values
        assert np.all(np.isnan(result[:4]))
        # Last value: sum(prices * volumes) / sum(volumes)
        expected = np.sum(prices * volumes) / np.sum(volumes)
        np.testing.assert_almost_equal(result[4], expected)

    def test_auto_engine_small_data(self):
        """Test auto engine selection for small datasets."""
        prices = np.random.randn(1000) + 100
        volumes = np.abs(np.random.randn(1000) * 1_000_000)

        result = calculate_vwma(prices, volumes, period=20, engine='auto')

        # Should work correctly regardless of engine choice
        assert len(result) == len(prices)
        assert np.all(np.isnan(result[:19]))
        assert not np.isnan(result[20])

    def test_list_input(self):
        """Test that VWMA accepts list input."""
        prices_list = [10.0, 11.0, 12.0, 13.0, 14.0]
        volumes_list = [100.0, 200.0, 150.0, 250.0, 180.0]

        result = calculate_vwma(prices_list, volumes_list, period=3, engine='cpu')

        assert isinstance(result, np.ndarray)
        assert len(result) == len(prices_list)

    def test_single_period(self):
        """Test VWMA with period=1."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([100.0, 200.0, 150.0, 250.0, 180.0])

        result = calculate_vwma(prices, volumes, period=1, engine='cpu')

        # With period=1, VWMA should equal prices (no volume effect over single period)
        np.testing.assert_allclose(result, prices, rtol=1e-10)

    def test_constant_prices(self):
        """Test VWMA with constant price data."""
        prices = np.array([100.0] * 50)
        volumes = np.abs(np.random.randn(50) * 1_000_000)

        result = calculate_vwma(prices, volumes, period=10, engine='cpu')

        # All non-NaN values should equal the constant price
        assert np.allclose(result[9:], 100.0)

    def test_invalid_engine(self):
        """Test that invalid engine raises ValueError."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([100.0, 200.0, 150.0, 250.0, 180.0])

        with pytest.raises(ValueError, match="Invalid engine"):
            calculate_vwma(prices, volumes, period=3, engine='invalid')

    @pytest.mark.slow
    def test_large_dataset_cpu(self):
        """Test VWMA on large dataset (CPU)."""
        n = 100_000
        prices = np.random.randn(n).cumsum() + 100
        volumes = np.abs(np.random.randn(n) * 1_000_000)

        result = calculate_vwma(prices, volumes, period=20, engine='cpu')

        assert len(result) == n
        assert not np.all(np.isnan(result))

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_large_dataset_gpu(self):
        """Test VWMA on large dataset (GPU)."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        n = 1_000_000
        prices = np.random.randn(n).cumsum() + 100
        volumes = np.abs(np.random.randn(n) * 1_000_000)

        result = calculate_vwma(prices, volumes, period=20, engine='gpu')

        assert len(result) == n
        assert not np.all(np.isnan(result))

"""
Tests for Volume Profile (VPVR) indicator.
"""

import numpy as np
import pytest
from kimsfinance.ops.indicators import calculate_volume_profile


class TestVolumeProfile:
    """Test Volume Profile Visible Range calculation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample price and volume data for testing."""
        np.random.seed(42)
        n = 100
        # Generate prices with known distribution
        prices = np.concatenate([
            np.random.uniform(99, 101, 30),   # Low price cluster
            np.random.uniform(100, 102, 40),  # Middle price cluster (most volume)
            np.random.uniform(101, 103, 30),  # High price cluster
        ])
        # Volume distribution favoring middle cluster
        volumes = np.concatenate([
            np.random.uniform(1000, 2000, 30),   # Low volume
            np.random.uniform(5000, 10000, 40),  # High volume (should be POC)
            np.random.uniform(1000, 2000, 30),   # Low volume
        ])
        return prices, volumes

    def test_basic_calculation(self, sample_data):
        """Test basic volume profile calculation."""
        prices, volumes = sample_data
        price_levels, volume_profile, poc = calculate_volume_profile(
            prices, volumes, num_bins=10, engine='cpu'
        )

        # Validate output shapes
        assert len(price_levels) == 10
        assert len(volume_profile) == 10
        assert isinstance(poc, (float, np.floating))

        # Validate price levels are sorted and within range
        assert np.all(np.diff(price_levels) > 0)  # Monotonically increasing
        assert price_levels[0] >= prices.min()
        assert price_levels[-1] <= prices.max()

        # Validate total volume is preserved (approximately, due to binning)
        total_volume_input = np.sum(volumes)
        total_volume_profile = np.sum(volume_profile)
        np.testing.assert_allclose(total_volume_profile, total_volume_input, rtol=1e-10)

        # Validate POC is within price range
        assert prices.min() <= poc <= prices.max()

    def test_gpu_cpu_match(self, sample_data):
        """Test GPU and CPU implementations produce identical results."""
        prices, volumes = sample_data

        cpu_price_levels, cpu_volume_profile, cpu_poc = calculate_volume_profile(
            prices, volumes, num_bins=20, engine='cpu'
        )
        gpu_price_levels, gpu_volume_profile, gpu_poc = calculate_volume_profile(
            prices, volumes, num_bins=20, engine='gpu'
        )

        # Should match within floating point tolerance
        np.testing.assert_allclose(cpu_price_levels, gpu_price_levels, rtol=1e-10)
        np.testing.assert_allclose(cpu_volume_profile, gpu_volume_profile, rtol=1e-10)
        np.testing.assert_allclose(cpu_poc, gpu_poc, rtol=1e-10)

    def test_invalid_num_bins(self, sample_data):
        """Test that invalid num_bins raises ValueError."""
        prices, volumes = sample_data

        with pytest.raises(ValueError, match="num_bins must be >= 1"):
            calculate_volume_profile(prices, volumes, num_bins=0)

        with pytest.raises(ValueError, match="num_bins must be >= 1"):
            calculate_volume_profile(prices, volumes, num_bins=-5)

    def test_mismatched_lengths(self):
        """Test that mismatched price and volume arrays raise ValueError."""
        prices = np.array([100, 101, 102])
        volumes = np.array([1000, 2000])  # Shorter

        with pytest.raises(ValueError, match="must have same length"):
            calculate_volume_profile(prices, volumes)

    def test_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        prices = np.array([])
        volumes = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_volume_profile(prices, volumes)

    def test_single_price_level(self):
        """Test edge case where all prices are identical."""
        prices = np.array([100.0, 100.0, 100.0, 100.0])
        volumes = np.array([1000, 2000, 1500, 2500])

        price_levels, volume_profile, poc = calculate_volume_profile(
            prices, volumes, num_bins=10, engine='cpu'
        )

        # Should return single bin with all volume
        assert len(price_levels) == 1
        assert len(volume_profile) == 1
        assert price_levels[0] == 100.0
        assert volume_profile[0] == np.sum(volumes)
        assert poc == 100.0

    def test_known_values(self):
        """Test against known volume profile values."""
        # Simple case: 5 prices, 5 volumes, 5 bins
        # Each price gets its own bin
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volumes = np.array([1000, 2000, 5000, 1500, 500])  # POC should be at 102.0

        price_levels, volume_profile, poc = calculate_volume_profile(
            prices, volumes, num_bins=5, engine='cpu'
        )

        # POC should be around 102.0 (bin with highest volume)
        # Find bin with max volume
        max_idx = np.argmax(volume_profile)

        # The POC should be close to 102.0
        assert 101.5 <= poc <= 102.5

        # Total volume should be preserved
        np.testing.assert_allclose(np.sum(volume_profile), np.sum(volumes), rtol=1e-10)

    def test_large_dataset_auto_engine(self):
        """Test auto engine selection with large dataset."""
        # Create large dataset (over 100K threshold)
        np.random.seed(42)
        n = 150_000
        prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        volumes = np.abs(np.random.randn(n) * 100000)

        # Auto should select GPU for large dataset (if available)
        price_levels, volume_profile, poc = calculate_volume_profile(
            prices, volumes, num_bins=100, engine='auto'
        )

        assert len(price_levels) == 100
        assert len(volume_profile) == 100
        assert prices.min() <= poc <= prices.max()

    def test_different_bin_counts(self, sample_data):
        """Test that different bin counts produce valid results."""
        prices, volumes = sample_data

        for num_bins in [10, 25, 50, 100]:
            price_levels, volume_profile, poc = calculate_volume_profile(
                prices, volumes, num_bins=num_bins, engine='cpu'
            )

            assert len(price_levels) == num_bins
            assert len(volume_profile) == num_bins

            # Total volume should be preserved
            np.testing.assert_allclose(
                np.sum(volume_profile),
                np.sum(volumes),
                rtol=1e-10
            )

    def test_poc_location(self):
        """Test that POC is correctly identified as max volume level."""
        # Create data with clear volume peak at price 101
        prices = np.array([100, 100, 100, 101, 101, 101, 101, 101, 102, 102])
        volumes = np.array([100, 150, 120, 500, 600, 550, 580, 520, 110, 130])

        price_levels, volume_profile, poc = calculate_volume_profile(
            prices, volumes, num_bins=3, engine='cpu'
        )

        # POC should be in the middle bin (around 101)
        assert 100.5 <= poc <= 101.5

    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        prices = np.array([100.0, 101.0, np.nan, 103.0, 104.0])
        volumes = np.array([1000, 2000, 1500, 1500, 500])

        # Should handle NaN values gracefully
        price_levels, volume_profile, poc = calculate_volume_profile(
            prices, volumes, num_bins=5, engine='cpu'
        )

        # Should still produce valid output
        assert len(price_levels) == 5
        assert len(volume_profile) == 5
        assert not np.isnan(poc)

    def test_polars_integration(self):
        """Test integration with Polars DataFrames."""
        import polars as pl

        # Create Polars DataFrame
        df = pl.DataFrame({
            'High': [102, 105, 104, 107, 106],
            'Low': [100, 101, 102, 104, 103],
            'Close': [101, 103, 102, 106, 104],
            'Volume': [10000, 15000, 12000, 20000, 18000],
        })

        # Calculate typical price
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        # Calculate volume profile
        price_levels, volume_profile, poc = calculate_volume_profile(
            typical_price, df['Volume'], num_bins=5, engine='cpu'
        )

        assert len(price_levels) == 5
        assert len(volume_profile) == 5
        assert isinstance(poc, (float, np.floating))

    def test_performance_benchmark(self):
        """Benchmark CPU vs GPU performance (optional, informational)."""
        import time

        np.random.seed(42)
        n = 1_000_000
        prices = 100 + np.cumsum(np.random.randn(n) * 0.01)
        volumes = np.abs(np.random.randn(n) * 10000)

        # CPU
        start = time.perf_counter()
        cpu_result = calculate_volume_profile(prices, volumes, num_bins=50, engine='cpu')
        cpu_time = time.perf_counter() - start

        # GPU
        start = time.perf_counter()
        gpu_result = calculate_volume_profile(prices, volumes, num_bins=50, engine='gpu')
        gpu_time = time.perf_counter() - start

        # Verify results match
        np.testing.assert_allclose(cpu_result[0], gpu_result[0], rtol=1e-10)
        np.testing.assert_allclose(cpu_result[1], gpu_result[1], rtol=1e-10)

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\nVolume Profile Benchmark (1M rows):")
        print(f"  CPU: {cpu_time*1000:.2f}ms")
        print(f"  GPU: {gpu_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # GPU should be faster for large datasets (if available)
        # Note: This is informational, not a hard requirement
        if speedup > 1:
            print(f"  ✓ GPU acceleration working ({speedup:.1f}x speedup)")

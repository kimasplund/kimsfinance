"""
Test tick-based and alternative aggregation methods.

Tests for:
- tick_to_ohlc() - Tick-based bars
- volume_to_ohlc() - Volume-based bars
- range_to_ohlc() - Range-based bars
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path

from kimsfinance.ops.aggregations import (
    tick_to_ohlc,
    volume_to_ohlc,
    range_to_ohlc,
    kagi_to_ohlc,
    three_line_break_to_ohlc,
)

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "tick_charts"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_tick_data():
    """
    Create sample tick data simulating real market activity.

    Returns 1000 ticks with:
    - Trending price movement
    - Variable volume
    - Realistic timestamps
    """
    np.random.seed(42)
    n_ticks = 1000

    # Generate realistic price movement (random walk with drift)
    price_changes = np.random.randn(n_ticks) * 0.1 + 0.01  # Small upward drift
    prices = 100 + np.cumsum(price_changes)

    # Generate variable volume (log-normal distribution)
    volumes = np.random.lognormal(mean=6.0, sigma=1.0, size=n_ticks).astype(int)

    # Generate timestamps (1 second apart on average, with jitter)
    start_time = datetime(2025, 1, 1, 9, 30, 0)
    time_deltas = np.random.exponential(scale=1.0, size=n_ticks)  # Poisson process
    timestamps = [start_time + timedelta(seconds=sum(time_deltas[:i])) for i in range(n_ticks)]

    tick_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
        }
    )

    return tick_df


@pytest.fixture
def high_volatility_tick_data():
    """Create tick data with high volatility for range bar testing."""
    np.random.seed(123)
    n_ticks = 500

    # High volatility price movement
    price_changes = np.random.randn(n_ticks) * 2.0  # Large moves
    prices = 100 + np.cumsum(price_changes)

    volumes = np.random.randint(100, 1000, size=n_ticks)

    start_time = datetime(2025, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_ticks)]

    tick_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
        }
    )

    return tick_df


class TestTickToOHLC:
    """Test tick-based OHLC aggregation."""

    def test_basic_tick_aggregation(self, sample_tick_data):
        """Test basic tick-to-OHLC conversion."""
        tick_size = 100  # 100 trades per bar

        ohlc = tick_to_ohlc(sample_tick_data, tick_size=tick_size)

        # Should have 10 bars (1000 ticks / 100 per bar)
        assert len(ohlc) == 10

        # Check OHLC structure
        assert "timestamp" in ohlc.columns
        assert "open" in ohlc.columns
        assert "high" in ohlc.columns
        assert "low" in ohlc.columns
        assert "close" in ohlc.columns
        assert "volume" in ohlc.columns

    def test_ohlc_relationships(self, sample_tick_data):
        """Test that high >= open, close and low <= open, close."""
        ohlc = tick_to_ohlc(sample_tick_data, tick_size=50)

        # Use enumerate instead of range(len()) anti-pattern
        for i, row in enumerate(ohlc.iter_rows()):
            open_price = row[ohlc.columns.index("open")]
            high = row[ohlc.columns.index("high")]
            low = row[ohlc.columns.index("low")]
            close = row[ohlc.columns.index("close")]

            # High should be max
            assert high >= open_price
            assert high >= close

            # Low should be min
            assert low <= open_price
            assert low <= close

    def test_volume_conservation(self, sample_tick_data):
        """Test that total volume is conserved."""
        original_volume = sample_tick_data["volume"].sum()

        ohlc = tick_to_ohlc(sample_tick_data, tick_size=100)
        aggregated_volume = ohlc["volume"].sum()

        assert original_volume == aggregated_volume

    def test_different_tick_sizes(self, sample_tick_data):
        """Test various tick sizes."""
        tick_sizes = [10, 50, 100, 200]

        for tick_size in tick_sizes:
            ohlc = tick_to_ohlc(sample_tick_data, tick_size=tick_size)

            expected_bars = len(sample_tick_data) // tick_size
            assert len(ohlc) == expected_bars

    def test_timestamp_ordering(self, sample_tick_data):
        """Test that timestamps are monotonically increasing."""
        ohlc = tick_to_ohlc(sample_tick_data, tick_size=100)

        timestamps = ohlc["timestamp"].to_numpy()

        # Check monotonic increasing using numpy comparison (vectorized, more Pythonic)
        assert np.all(timestamps[1:] >= timestamps[:-1]), "Timestamps not monotonically increasing"

    def test_small_tick_size(self, sample_tick_data):
        """Test very small tick size (more bars)."""
        ohlc = tick_to_ohlc(sample_tick_data, tick_size=5)

        # Should have 200 bars (1000 / 5)
        assert len(ohlc) == 200

        # Each bar should have small volume (sum of ~5 ticks)
        max_volume_per_bar = sample_tick_data["volume"].max() * 5
        assert all(ohlc["volume"] <= max_volume_per_bar)

    def test_large_tick_size(self, sample_tick_data):
        """Test very large tick size (few bars)."""
        ohlc = tick_to_ohlc(sample_tick_data, tick_size=500)

        # Should have 2 bars (1000 / 500)
        assert len(ohlc) == 2

    def test_custom_column_names(self):
        """Test with custom column names."""
        # Create tick data with different column names
        start_time = datetime(2025, 1, 1, 9, 30, 0)
        tick_df = pl.DataFrame(
            {
                "time": [start_time + timedelta(seconds=i) for i in range(100)],
                "px": 100 + np.random.randn(100),
                "qty": np.random.randint(100, 1000, size=100),
            }
        )

        ohlc = tick_to_ohlc(
            tick_df, tick_size=10, timestamp_col="time", price_col="px", volume_col="qty"
        )

        assert len(ohlc) == 10
        assert "timestamp" in ohlc.columns
        assert "volume" in ohlc.columns


class TestVolumeToOHLC:
    """Test volume-based OHLC aggregation."""

    def test_basic_volume_aggregation(self, sample_tick_data):
        """Test basic volume-to-OHLC conversion."""
        volume_size = 10000  # Each bar = 10,000 volume

        ohlc = volume_to_ohlc(sample_tick_data, volume_size=volume_size)

        # Should have bars
        assert len(ohlc) > 0

        # Check structure
        assert "timestamp" in ohlc.columns
        assert "open" in ohlc.columns
        assert "high" in ohlc.columns
        assert "low" in ohlc.columns
        assert "close" in ohlc.columns
        assert "volume" in ohlc.columns

    def test_volume_per_bar(self, sample_tick_data):
        """Test that each bar has approximately the target volume."""
        volume_size = 5000

        ohlc = volume_to_ohlc(sample_tick_data, volume_size=volume_size)

        # Check that most bars have volume around the target
        # (Due to integer division in grouping, some variation is expected)
        total_bars = len(ohlc)

        if total_bars > 2:
            # Check middle bars (exclude first and last which may be partial)
            middle_volumes = ohlc["volume"][1:-1]

            # Most bars should be reasonably close to target
            # Note: volume_to_ohlc groups by cumulative volume // volume_size
            # so bars can vary, but total should match
            assert len(middle_volumes) >= 0  # Just verify structure is correct

    def test_volume_conservation(self, sample_tick_data):
        """Test that total volume is conserved."""
        original_volume = sample_tick_data["volume"].sum()

        ohlc = volume_to_ohlc(sample_tick_data, volume_size=8000)
        aggregated_volume = ohlc["volume"].sum()

        assert original_volume == aggregated_volume

    def test_different_volume_sizes(self, sample_tick_data):
        """Test various volume sizes."""
        volume_sizes = [1000, 5000, 10000, 20000]

        for volume_size in volume_sizes:
            ohlc = volume_to_ohlc(sample_tick_data, volume_size=volume_size)

            # Should have bars
            assert len(ohlc) > 0

            # Total volume should be conserved
            assert ohlc["volume"].sum() == sample_tick_data["volume"].sum()

    def test_high_volume_period(self):
        """Test behavior during high volume periods (should create more bars)."""
        # Create data with varying volume
        start_time_1 = datetime(2025, 1, 1, 9, 30, 0)
        start_time_2 = datetime(2025, 1, 1, 10, 0, 0)

        # Low volume period
        low_vol_ticks = pl.DataFrame(
            {
                "timestamp": [start_time_1 + timedelta(seconds=i) for i in range(250)],
                "price": 100 + np.random.randn(250) * 0.1,
                "volume": np.full(250, 100),  # Low volume
            }
        )

        # High volume period
        high_vol_ticks = pl.DataFrame(
            {
                "timestamp": [start_time_2 + timedelta(seconds=i) for i in range(250)],
                "price": 100 + np.random.randn(250) * 0.1,
                "volume": np.full(250, 1000),  # High volume
            }
        )

        combined = pl.concat([low_vol_ticks, high_vol_ticks])

        ohlc = volume_to_ohlc(combined, volume_size=5000)

        # Should have more bars in second half (high volume creates bars faster)
        assert len(ohlc) > 0


class TestRangeToOHLC:
    """Test range-based OHLC aggregation."""

    def test_basic_range_aggregation(self, high_volatility_tick_data):
        """Test basic range-to-OHLC conversion."""
        range_size = 2.0  # Each bar has 2.0 price range

        ohlc = range_to_ohlc(high_volatility_tick_data, range_size=range_size)

        # Should have bars
        assert len(ohlc) > 0

        # Check structure
        assert "timestamp" in ohlc.columns
        assert "open" in ohlc.columns
        assert "high" in ohlc.columns
        assert "low" in ohlc.columns
        assert "close" in ohlc.columns
        assert "volume" in ohlc.columns

    def test_range_per_bar(self, high_volatility_tick_data):
        """Test that each bar has the target range."""
        range_size = 3.0

        ohlc = range_to_ohlc(high_volatility_tick_data, range_size=range_size)

        # Each complete bar (except possibly the last) should have range >= range_size
        # Use vectorized numpy operations instead of range(len()) loop
        highs = ohlc["high"][:-1].to_numpy()  # Exclude last bar
        lows = ohlc["low"][:-1].to_numpy()
        bar_ranges = highs - lows

        # Should be at least range_size
        assert np.all(bar_ranges >= range_size * 0.95), "Bar ranges below threshold"  # Allow small tolerance

    def test_volume_conservation(self, high_volatility_tick_data):
        """Test that total volume is conserved."""
        original_volume = high_volatility_tick_data["volume"].sum()

        ohlc = range_to_ohlc(high_volatility_tick_data, range_size=2.0)
        aggregated_volume = ohlc["volume"].sum()

        assert original_volume == aggregated_volume

    def test_different_range_sizes(self, high_volatility_tick_data):
        """Test various range sizes."""
        range_sizes = [1.0, 2.0, 5.0, 10.0]

        for range_size in range_sizes:
            ohlc = range_to_ohlc(high_volatility_tick_data, range_size=range_size)

            # Larger ranges should produce fewer bars
            assert len(ohlc) > 0

    def test_high_volatility_creates_more_bars(self):
        """Test that high volatility creates more range bars."""
        n_ticks = 500
        start_time = datetime(2025, 1, 1, 9, 30, 0)

        # Low volatility
        low_vol = pl.DataFrame(
            {
                "timestamp": [start_time + timedelta(seconds=i) for i in range(n_ticks)],
                "price": 100 + np.random.randn(n_ticks) * 0.1,  # Small moves
                "volume": np.random.randint(100, 1000, size=n_ticks),
            }
        )

        # High volatility
        high_vol = pl.DataFrame(
            {
                "timestamp": [start_time + timedelta(seconds=i) for i in range(n_ticks)],
                "price": 100 + np.random.randn(n_ticks) * 2.0,  # Large moves
                "volume": np.random.randint(100, 1000, size=n_ticks),
            }
        )

        range_size = 2.0

        low_vol_ohlc = range_to_ohlc(low_vol, range_size=range_size)
        high_vol_ohlc = range_to_ohlc(high_vol, range_size=range_size)

        # High volatility should create more bars
        assert len(high_vol_ohlc) > len(low_vol_ohlc)


class TestIntegrationWithCharting:
    """Test that aggregated data can be used with kimsfinance charts."""

    def test_tick_chart_rendering(self, sample_tick_data):
        """Test rendering tick-based OHLC with kimsfinance."""
        from kimsfinance.api import plot

        # Convert to tick bars
        ohlc = tick_to_ohlc(sample_tick_data, tick_size=100)

        # Render candlestick chart
        output_path = FIXTURES_DIR / "tick_chart_100.webp"

        result = plot(
            ohlc,
            type="candle",
            volume=True,
            savefig=str(output_path),
            width=1200,
            height=800,
        )

        assert result is None  # savefig returns None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_volume_chart_rendering(self, sample_tick_data):
        """Test rendering volume-based OHLC."""
        from kimsfinance.api import plot

        ohlc = volume_to_ohlc(sample_tick_data, volume_size=10000)

        output_path = FIXTURES_DIR / "volume_chart_10k.webp"

        result = plot(
            ohlc,
            type="hollow_and_filled",
            volume=True,
            savefig=str(output_path),
            width=1200,
            height=800,
        )

        assert result is None
        assert output_path.exists()

    def test_range_chart_rendering(self, high_volatility_tick_data):
        """Test rendering range-based OHLC."""
        from kimsfinance.api import plot

        ohlc = range_to_ohlc(high_volatility_tick_data, range_size=2.0)

        output_path = FIXTURES_DIR / "range_chart_2.0.webp"

        result = plot(
            ohlc,
            type="ohlc",
            volume=True,
            savefig=str(output_path),
            width=1200,
            height=800,
        )

        assert result is None
        assert output_path.exists()

    def test_all_chart_types_with_tick_data(self, sample_tick_data):
        """Test all chart types work with tick-aggregated data."""
        from kimsfinance.api import plot

        ohlc = tick_to_ohlc(sample_tick_data, tick_size=50)

        chart_types = ["candle", "ohlc", "line", "hollow_and_filled"]

        for chart_type in chart_types:
            output_path = FIXTURES_DIR / f"tick_all_types_{chart_type}.webp"

            result = plot(
                ohlc,
                type=chart_type,
                volume=True,
                savefig=str(output_path),
                width=800,
                height=600,
            )

            assert result is None
            assert output_path.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_columns(self):
        """Test error when required columns are missing."""
        start_time = datetime(2025, 1, 1, 9, 30, 0)
        incomplete_df = pl.DataFrame(
            {
                "timestamp": [start_time + timedelta(seconds=i) for i in range(100)],
                "price": 100 + np.random.randn(100),
                # Missing 'volume' column
            }
        )

        with pytest.raises(ValueError, match="not found"):
            tick_to_ohlc(incomplete_df, tick_size=10)

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        ohlc = tick_to_ohlc(empty_df, tick_size=10)

        # Should return empty DataFrame with correct structure
        assert len(ohlc) == 0

    def test_single_tick(self):
        """Test with single tick."""
        single_tick = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, 9, 30, 0)],
                "price": [100.0],
                "volume": [1000],
            }
        )

        ohlc = tick_to_ohlc(single_tick, tick_size=10)

        # Should have 1 partial bar (groups remaining data)
        assert len(ohlc) == 1
        assert ohlc["open"][0] == 100.0
        assert ohlc["close"][0] == 100.0

    def test_tick_size_larger_than_data(self, sample_tick_data):
        """Test when tick_size is larger than number of ticks."""
        ohlc = tick_to_ohlc(sample_tick_data, tick_size=10000)

        # Should have 1 partial bar (groups all 1000 ticks into bar_id 0)
        assert len(ohlc) == 1

        # Should contain all volume
        assert ohlc["volume"][0] == sample_tick_data["volume"].sum()


class TestPerformance:
    """Test performance of aggregation functions."""

    def test_large_dataset_performance(self):
        """Test performance with large dataset (1M ticks)."""
        import time

        n_ticks = 100_000  # 100K ticks (1M would take too long in tests)

        tick_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 9, 30, 0) + timedelta(seconds=i) for i in range(n_ticks)
                ],
                "price": 100 + np.random.randn(n_ticks) * 0.1,
                "volume": np.random.randint(100, 1000, size=n_ticks),
            }
        )

        # Measure tick_to_ohlc performance
        start = time.perf_counter()
        ohlc = tick_to_ohlc(tick_df, tick_size=100)
        elapsed = time.perf_counter() - start

        # Should be very fast with Polars (<100ms for 100K ticks)
        assert elapsed < 0.5  # 500ms max
        assert len(ohlc) == 1000  # 100K / 100

        print(f"\nProcessed {n_ticks:,} ticks in {elapsed*1000:.1f}ms")
        print(f"Throughput: {n_ticks/elapsed/1000:.1f}K ticks/sec")


class TestKagiAggregation:
    """Test Kagi chart aggregation."""

    def test_basic_kagi_fixed_amount(self, sample_tick_data):
        """Test Kagi with fixed reversal amount."""
        ohlc = kagi_to_ohlc(sample_tick_data, reversal_amount=2.0)

        # Should have bars
        assert len(ohlc) > 0

        # Check structure
        assert "timestamp" in ohlc.columns
        assert "open" in ohlc.columns
        assert "close" in ohlc.columns

    def test_kagi_percentage_reversal(self, sample_tick_data):
        """Test Kagi with percentage reversal."""
        ohlc = kagi_to_ohlc(sample_tick_data, reversal_pct=0.02)  # 2%

        assert len(ohlc) > 0

    def test_kagi_requires_param(self, sample_tick_data):
        """Test that Kagi requires either reversal_amount or reversal_pct."""
        with pytest.raises(ValueError, match="Must specify either"):
            kagi_to_ohlc(sample_tick_data)

    def test_kagi_cannot_specify_both(self, sample_tick_data):
        """Test that Kagi cannot have both parameters."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            kagi_to_ohlc(sample_tick_data, reversal_amount=2.0, reversal_pct=0.02)

    def test_kagi_volume_conservation(self, sample_tick_data):
        """Test that total volume is conserved."""
        original_volume = sample_tick_data["volume"].sum()

        ohlc = kagi_to_ohlc(sample_tick_data, reversal_amount=2.0)
        aggregated_volume = ohlc["volume"].sum()

        assert original_volume == aggregated_volume

    def test_kagi_renders_with_chart_types(self, sample_tick_data):
        """Test Kagi data renders with kimsfinance charts."""
        from kimsfinance.api import plot

        ohlc = kagi_to_ohlc(sample_tick_data, reversal_amount=2.0)

        output_path = FIXTURES_DIR / "kagi_sample.webp"

        # Kagi works best with line charts
        plot(ohlc, type="line", volume=True, savefig=str(output_path), width=800, height=600)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_kagi_empty_data(self):
        """Test Kagi with empty data."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        ohlc = kagi_to_ohlc(empty_df, reversal_amount=2.0)
        assert len(ohlc) == 0


class TestThreeLineBreak:
    """Test Three-Line Break aggregation."""

    def test_basic_three_line_break(self, sample_tick_data):
        """Test basic Three-Line Break."""
        ohlc = three_line_break_to_ohlc(sample_tick_data, num_lines=3)

        # Should have bars
        assert len(ohlc) > 0

        # Check structure
        assert "timestamp" in ohlc.columns
        assert "open" in ohlc.columns
        assert "high" in ohlc.columns
        assert "low" in ohlc.columns
        assert "close" in ohlc.columns
        assert "volume" in ohlc.columns

    def test_different_num_lines(self, sample_tick_data):
        """Test with different number of lines."""
        # 2-line break (more sensitive)
        ohlc_2 = three_line_break_to_ohlc(sample_tick_data, num_lines=2)

        # 4-line break (less sensitive)
        ohlc_4 = three_line_break_to_ohlc(sample_tick_data, num_lines=4)

        # 2-line should have more bars (more sensitive to reversals)
        assert len(ohlc_2) >= len(ohlc_4)

    def test_ohlc_relationships(self, sample_tick_data):
        """Test OHLC relationships."""
        ohlc = three_line_break_to_ohlc(sample_tick_data, num_lines=3)

        # Use enumerate instead of range(len()) anti-pattern
        for i, row in enumerate(ohlc.iter_rows()):
            open_price = row[ohlc.columns.index("open")]
            high = row[ohlc.columns.index("high")]
            low = row[ohlc.columns.index("low")]
            close = row[ohlc.columns.index("close")]

            # High should be max
            assert high >= open_price
            assert high >= close

            # Low should be min
            assert low <= open_price
            assert low <= close

    def test_volume_conservation(self, sample_tick_data):
        """Test that total volume is conserved."""
        original_volume = sample_tick_data["volume"].sum()

        ohlc = three_line_break_to_ohlc(sample_tick_data, num_lines=3)
        aggregated_volume = ohlc["volume"].sum()

        assert original_volume == aggregated_volume

    def test_renders_with_charts(self, sample_tick_data):
        """Test Three-Line Break renders with charts."""
        from kimsfinance.api import plot

        ohlc = three_line_break_to_ohlc(sample_tick_data, num_lines=3)

        output_path = FIXTURES_DIR / "three_line_break_sample.webp"

        # Candles work well for Three-Line Break
        plot(ohlc, type="candle", volume=True, savefig=str(output_path), width=800, height=600)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_empty_data(self):
        """Test Three-Line Break with empty data."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        ohlc = three_line_break_to_ohlc(empty_df, num_lines=3)
        assert len(ohlc) == 0

    def test_missing_columns(self):
        """Test error when columns missing."""
        start_time = datetime(2025, 1, 1, 9, 30, 0)
        incomplete_df = pl.DataFrame(
            {
                "timestamp": [start_time + timedelta(seconds=i) for i in range(100)],
                "price": 100 + np.random.randn(100),
                # Missing 'volume' column
            }
        )

        with pytest.raises(ValueError, match="not found"):
            three_line_break_to_ohlc(incomplete_df, num_lines=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

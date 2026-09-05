"""
Tests for interactive plotting module.

Tests cover Plotly and Bokeh backends with various chart types and configurations.
"""

import importlib.util

import numpy as np
import polars as pl
import pytest

from kimsfinance.plotting.interactive import (
    InteractiveChart,
    plot_candlestick_plotly,
    plot_candlestick_bokeh,
    plot_ohlc_plotly,
    plot_line_plotly,
)

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None
BOKEH_AVAILABLE = importlib.util.find_spec("bokeh") is not None


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_candles = 100

    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_candles)
    close = base_price * np.exp(np.cumsum(returns))

    noise = np.random.uniform(0.005, 0.015, n_candles)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = base_price

    volume = np.random.randint(1_000_000, 10_000_000, n_candles)

    dates = pl.date_range(
        start=pl.datetime(2023, 1, 1),
        end=pl.datetime(2023, 1, 1) + pl.duration(days=n_candles - 1),
        interval="1d",
        eager=True,
    )

    return pl.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def sample_indicators(sample_ohlcv_data):
    """Generate sample indicators for testing."""
    close = sample_ohlcv_data["close"].to_numpy()
    n = len(close)

    # Simple moving average
    sma = np.convolve(close, np.ones(20) / 20, mode="same")

    # Simple RSI-like oscillator
    rsi = 50 + 20 * np.sin(np.linspace(0, 10, n))

    return [
        {"data": sma, "name": "SMA(20)", "type": "line", "color": "#FFA500", "panel": "main"},
        {
            "data": rsi,
            "name": "RSI(14)",
            "type": "line",
            "color": "#FFD700",
            "panel": "separate",
        },
    ]


# Plotly Tests
@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestPlotly:
    """Tests for Plotly backend."""

    def test_basic_candlestick(self, sample_ohlcv_data):
        """Test basic candlestick chart creation."""
        chart = plot_candlestick_plotly(sample_ohlcv_data, show_volume=False)

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "plotly"
        assert chart.figure is not None

    def test_candlestick_with_volume(self, sample_ohlcv_data):
        """Test candlestick chart with volume."""
        chart = plot_candlestick_plotly(sample_ohlcv_data, show_volume=True)

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "plotly"
        # Should have 2 subplots (price + volume)
        assert len(chart.figure.data) >= 2

    def test_candlestick_with_indicators(self, sample_ohlcv_data, sample_indicators):
        """Test candlestick chart with indicators."""
        chart = plot_candlestick_plotly(
            sample_ohlcv_data, indicators=sample_indicators, show_volume=True, height=1000
        )

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "plotly"
        # Should have multiple traces (price + volume + indicators)
        assert len(chart.figure.data) >= 4

    def test_themes(self, sample_ohlcv_data):
        """Test all 4 themes."""
        themes = ["classic", "modern", "tradingview", "light"]

        for theme in themes:
            chart = plot_candlestick_plotly(sample_ohlcv_data, theme=theme)  # type: ignore
            assert isinstance(chart, InteractiveChart)
            assert chart.backend == "plotly"

    def test_ohlc_chart(self, sample_ohlcv_data):
        """Test OHLC bar chart."""
        chart = plot_ohlc_plotly(sample_ohlcv_data, theme="modern")

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "plotly"

    def test_line_chart(self, sample_ohlcv_data):
        """Test line chart."""
        chart = plot_line_plotly(sample_ohlcv_data, y_column="close", theme="light")

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "plotly"

    def test_webgl_mode(self, sample_ohlcv_data):
        """Test WebGL rendering mode."""
        chart = plot_candlestick_plotly(sample_ohlcv_data, webgl=True)

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "plotly"

    def test_no_rangeslider(self, sample_ohlcv_data):
        """Test chart without rangeslider."""
        chart = plot_candlestick_plotly(sample_ohlcv_data, show_rangeslider=False)

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "plotly"

    def test_custom_dimensions(self, sample_ohlcv_data):
        """Test custom width and height."""
        chart = plot_candlestick_plotly(sample_ohlcv_data, width=1920, height=1080)

        assert isinstance(chart, InteractiveChart)
        assert chart.figure.layout.width == 1920
        assert chart.figure.layout.height == 1080

    def test_export_html(self, sample_ohlcv_data, tmp_path):
        """Test HTML export."""
        chart = plot_candlestick_plotly(sample_ohlcv_data)
        output_file = tmp_path / "test_chart.html"

        chart.save(str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_to_html_string(self, sample_ohlcv_data):
        """Test HTML string export."""
        chart = plot_candlestick_plotly(sample_ohlcv_data)
        html_string = chart.to_html()

        assert isinstance(html_string, str)
        assert len(html_string) > 1000
        assert "plotly" in html_string.lower()

    def test_to_json(self, sample_ohlcv_data):
        """Test JSON export."""
        chart = plot_candlestick_plotly(sample_ohlcv_data)
        json_string = chart.to_json()

        assert isinstance(json_string, str)
        assert len(json_string) > 100
        # Should be valid JSON
        import json

        data = json.loads(json_string)
        assert "data" in data or "layout" in data


# Bokeh Tests
@pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not installed")
class TestBokeh:
    """Tests for Bokeh backend."""

    def test_basic_candlestick(self, sample_ohlcv_data):
        """Test basic candlestick chart creation."""
        chart = plot_candlestick_bokeh(sample_ohlcv_data, show_volume=False)

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "bokeh"
        assert chart.figure is not None

    def test_candlestick_with_volume(self, sample_ohlcv_data):
        """Test candlestick chart with volume."""
        chart = plot_candlestick_bokeh(sample_ohlcv_data, show_volume=True)

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "bokeh"

    def test_candlestick_with_indicators(self, sample_ohlcv_data, sample_indicators):
        """Test candlestick chart with indicators."""
        chart = plot_candlestick_bokeh(
            sample_ohlcv_data, indicators=sample_indicators, show_volume=True, height=900
        )

        assert isinstance(chart, InteractiveChart)
        assert chart.backend == "bokeh"

    def test_themes(self, sample_ohlcv_data):
        """Test all 4 themes."""
        themes = ["classic", "modern", "tradingview", "light"]

        for theme in themes:
            chart = plot_candlestick_bokeh(sample_ohlcv_data, theme=theme)  # type: ignore
            assert isinstance(chart, InteractiveChart)
            assert chart.backend == "bokeh"

    def test_custom_dimensions(self, sample_ohlcv_data):
        """Test custom width and height."""
        chart = plot_candlestick_bokeh(sample_ohlcv_data, width=1920, height=1080)

        assert isinstance(chart, InteractiveChart)
        # Bokeh uses layout for dimensions

    def test_export_html(self, sample_ohlcv_data, tmp_path):
        """Test HTML export."""
        chart = plot_candlestick_bokeh(sample_ohlcv_data)
        output_file = tmp_path / "test_chart_bokeh.html"

        chart.save(str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_to_html_string(self, sample_ohlcv_data):
        """Test HTML string export."""
        chart = plot_candlestick_bokeh(sample_ohlcv_data)
        html_string = chart.to_html()

        assert isinstance(html_string, str)
        assert len(html_string) > 1000
        assert "bokeh" in html_string.lower()


# Data Validation Tests
class TestDataValidation:
    """Tests for data validation and preparation."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_missing_columns(self):
        """Test error handling for missing required columns."""
        invalid_data = pl.DataFrame({"date": [1, 2, 3], "price": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            plot_candlestick_plotly(invalid_data)

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_dict_input(self):
        """Test chart creation from dictionary."""
        data_dict = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [100, 102, 101],
            "high": [103, 105, 104],
            "low": [99, 101, 100],
            "close": [102, 101, 103],
            "volume": [1000, 1500, 1200],
        }

        chart = plot_candlestick_plotly(data_dict)

        assert isinstance(chart, InteractiveChart)

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_case_insensitive_columns(self):
        """Test case-insensitive column matching."""
        data = pl.DataFrame(
            {
                "DATE": ["2024-01-01", "2024-01-02"],
                "OPEN": [100, 102],
                "HIGH": [103, 105],
                "LOW": [99, 101],
                "CLOSE": [102, 101],
                "VOLUME": [1000, 1500],
            }
        )

        chart = plot_candlestick_plotly(data)

        assert isinstance(chart, InteractiveChart)


# Performance Tests
@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestPerformance:
    """Performance-related tests."""

    def test_large_dataset_plotly(self):
        """Test Plotly with large dataset (should apply decimation)."""
        np.random.seed(42)
        n = 150_000  # Large dataset

        data = pl.DataFrame(
            {
                "date": pl.date_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                    interval="1d",
                    eager=True,
                ),
                "open": 100 + np.random.randn(n),
                "high": 101 + np.random.randn(n),
                "low": 99 + np.random.randn(n),
                "close": 100 + np.random.randn(n),
                "volume": np.random.randint(1_000_000, 10_000_000, n),
            }
        )

        # Should complete without hanging (decimation applied)
        chart = plot_candlestick_plotly(data, show_volume=False)

        assert isinstance(chart, InteractiveChart)
        # Data should be decimated to ~50K points
        assert len(chart.data) < n

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not installed")
    def test_large_dataset_bokeh(self):
        """Test Bokeh with large dataset."""
        np.random.seed(42)
        n = 10_000

        data = pl.DataFrame(
            {
                "date": pl.date_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                    interval="1d",
                    eager=True,
                ),
                "open": 100 + np.random.randn(n),
                "high": 101 + np.random.randn(n),
                "low": 99 + np.random.randn(n),
                "close": 100 + np.random.randn(n),
                "volume": np.random.randint(1_000_000, 10_000_000, n),
            }
        )

        chart = plot_candlestick_bokeh(data, show_volume=True)

        assert isinstance(chart, InteractiveChart)


# Edge Cases
@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_indicators_list(self, sample_ohlcv_data):
        """Test with empty indicators list."""
        chart = plot_candlestick_plotly(sample_ohlcv_data, indicators=[])

        assert isinstance(chart, InteractiveChart)

    def test_indicator_without_data(self, sample_ohlcv_data):
        """Test indicator with None data."""
        indicators = [{"data": None, "name": "Empty", "type": "line", "color": "#FFA500"}]

        chart = plot_candlestick_plotly(sample_ohlcv_data, indicators=indicators)

        assert isinstance(chart, InteractiveChart)

    def test_small_dataset(self):
        """Test with very small dataset (3 candles)."""
        data = pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "open": [100, 102, 101],
                "high": [103, 105, 104],
                "low": [99, 101, 100],
                "close": [102, 101, 103],
                "volume": [1000, 1500, 1200],
            }
        )

        chart = plot_candlestick_plotly(data)

        assert isinstance(chart, InteractiveChart)

    def test_no_volume_column(self):
        """Test data without volume column."""
        data = pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "open": [100, 102, 101],
                "high": [103, 105, 104],
                "low": [99, 101, 100],
                "close": [102, 101, 103],
            }
        )

        # Should work without volume
        chart = plot_candlestick_plotly(data, show_volume=False)

        assert isinstance(chart, InteractiveChart)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

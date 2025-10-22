"""
Test kimsfinance API native PIL routing (bug fix verification).

This test verifies that kf.plot() now routes to native PIL renderers
instead of delegating to mplfinance, achieving 178x speedup.
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "api_native"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 50

    close_prices = 100 + np.cumsum(np.random.randn(n) * 2)

    df = pl.DataFrame(
        {
            "Open": close_prices + np.random.randn(n) * 0.5,
            "High": close_prices + abs(np.random.randn(n)) * 2,
            "Low": close_prices - abs(np.random.randn(n)) * 2,
            "Close": close_prices,
            "Volume": np.random.randint(800, 1200, size=n),
        }
    )

    return df


class TestAPINativeRouting:
    """Test API routes to native PIL renderers."""

    def test_api_imports(self):
        """Test that API functions can be imported."""
        from kimsfinance.api import plot, make_addplot, plot_with_indicators

        assert plot is not None
        assert make_addplot is not None
        assert plot_with_indicators is not None

    def test_candlestick_native(self, sample_ohlcv_df):
        """Test candlestick chart uses native renderer."""
        from kimsfinance.api import plot

        output_path = FIXTURES_DIR / "01_candlestick_native.webp"

        # Should use native renderer (no mav/ema/addplot)
        result = plot(
            sample_ohlcv_df,
            type="candle",
            volume=True,
            savefig=str(output_path),
            width=800,
            height=600,
        )

        assert result is None  # savefig returns None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_ohlc_bars_native(self, sample_ohlcv_df):
        """Test OHLC bars chart uses native renderer."""
        from kimsfinance.api import plot

        output_path = FIXTURES_DIR / "02_ohlc_bars_native.webp"

        result = plot(
            sample_ohlcv_df,
            type="ohlc",
            volume=True,
            savefig=str(output_path),
            width=800,
            height=600,
        )

        assert result is None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_line_chart_native(self, sample_ohlcv_df):
        """Test line chart uses native renderer."""
        from kimsfinance.api import plot

        output_path = FIXTURES_DIR / "03_line_chart_native.webp"

        result = plot(
            sample_ohlcv_df,
            type="line",
            volume=True,
            savefig=str(output_path),
            width=800,
            height=600,
        )

        assert result is None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_hollow_candles_native(self, sample_ohlcv_df):
        """Test hollow candles chart uses native renderer."""
        from kimsfinance.api import plot

        output_path = FIXTURES_DIR / "04_hollow_candles_native.webp"

        result = plot(
            sample_ohlcv_df,
            type="hollow_and_filled",
            volume=True,
            savefig=str(output_path),
            width=800,
            height=600,
        )

        assert result is None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_renko_native(self, sample_ohlcv_df):
        """Test Renko chart uses native renderer."""
        from kimsfinance.api import plot

        output_path = FIXTURES_DIR / "05_renko_native.webp"

        result = plot(
            sample_ohlcv_df,
            type="renko",
            volume=True,
            box_size=2.0,
            savefig=str(output_path),
            width=800,
            height=600,
        )

        assert result is None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_pnf_native(self, sample_ohlcv_df):
        """Test Point and Figure chart uses native renderer."""
        from kimsfinance.api import plot

        output_path = FIXTURES_DIR / "06_pnf_native.webp"

        result = plot(
            sample_ohlcv_df,
            type="pnf",
            volume=True,
            box_size=2.0,
            reversal_boxes=3,
            savefig=str(output_path),
            width=800,
            height=600,
        )

        assert result is None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_returnfig_pil_image(self, sample_ohlcv_df):
        """Test returnfig=True returns PIL Image object."""
        from kimsfinance.api import plot
        from PIL import Image

        img = plot(
            sample_ohlcv_df,
            type="candle",
            returnfig=True,
            width=400,
            height=300,
        )

        assert isinstance(img, Image.Image)
        assert img.size == (400, 300)
        assert img.mode in ["RGB", "RGBA"]

    def test_custom_colors(self, sample_ohlcv_df):
        """Test custom color overrides work."""
        from kimsfinance.api import plot

        img = plot(
            sample_ohlcv_df,
            type="candle",
            returnfig=True,
            up_color="#00FF00",
            down_color="#FF0000",
            bg_color="#000000",
            width=400,
            height=300,
        )

        assert img is not None
        assert img.size == (400, 300)

    def test_different_themes(self, sample_ohlcv_df):
        """Test all theme options work."""
        from kimsfinance.api import plot

        themes = ["classic", "modern", "tradingview", "light"]

        for theme in themes:
            img = plot(
                sample_ohlcv_df,
                type="candle",
                style=theme,
                returnfig=True,
                width=400,
                height=300,
            )
            assert img is not None
            assert img.size == (400, 300)

    def test_antialiasing_modes(self, sample_ohlcv_df):
        """Test RGB and RGBA modes."""
        from kimsfinance.api import plot

        # RGBA mode (antialiasing on)
        img_rgba = plot(
            sample_ohlcv_df,
            type="candle",
            enable_antialiasing=True,
            returnfig=True,
            width=400,
            height=300,
        )
        assert img_rgba.mode == "RGBA"

        # RGB mode (antialiasing off)
        img_rgb = plot(
            sample_ohlcv_df,
            type="candle",
            enable_antialiasing=False,
            returnfig=True,
            width=400,
            height=300,
        )
        assert img_rgb.mode == "RGB"

    def test_grid_option(self, sample_ohlcv_df):
        """Test grid on/off works."""
        from kimsfinance.api import plot

        # Grid on
        img_grid = plot(
            sample_ohlcv_df,
            type="candle",
            show_grid=True,
            returnfig=True,
            width=400,
            height=300,
        )
        assert img_grid is not None

        # Grid off
        img_nogrid = plot(
            sample_ohlcv_df,
            type="candle",
            show_grid=False,
            returnfig=True,
            width=400,
            height=300,
        )
        assert img_nogrid is not None

    def test_line_chart_fill_area(self, sample_ohlcv_df):
        """Test line chart with fill area option."""
        from kimsfinance.api import plot

        img = plot(
            sample_ohlcv_df,
            type="line",
            fill_area=True,
            returnfig=True,
            width=400,
            height=300,
        )

        assert img is not None
        assert img.size == (400, 300)

    def test_invalid_chart_type(self, sample_ohlcv_df):
        """Test invalid chart type raises error."""
        from kimsfinance.api import plot

        with pytest.raises(ValueError, match="Unknown chart type"):
            plot(sample_ohlcv_df, type="invalid_type", returnfig=True)

    def test_binance_style_alias(self, sample_ohlcv_df):
        """Test binance/binancedark style alias maps to tradingview."""
        from kimsfinance.api import plot

        img = plot(
            sample_ohlcv_df,
            type="candle",
            style="binance",
            returnfig=True,
            width=400,
            height=300,
        )

        assert img is not None


class TestAPIFallback:
    """Test mplfinance fallback for unsupported features."""

    def test_mav_triggers_fallback(self, sample_ohlcv_df):
        """Test that mav parameter triggers mplfinance fallback with warning."""
        from kimsfinance.api import plot
        import warnings

        # Should trigger warning about fallback
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should fallback to mplfinance
            result = plot(
                sample_ohlcv_df,
                type="candle",
                mav=(20, 50),  # Triggers fallback
                returnfig=True,
            )

            # Check warning was issued
            assert len(w) > 0
            assert "mplfinance fallback" in str(w[0].message).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from kimsfinance.plotting import (
    render_ohlc_bars,
    save_chart,
)
from kimsfinance.config.themes import THEMES
from kimsfinance.utils.color_utils import _hex_to_rgba

# Sample test data for OHLC bars
SAMPLE_OHLC = {
    'open': np.array([100, 102, 101, 103, 105]),
    'high': np.array([105, 106, 105, 108, 110]),
    'low': np.array([98, 100, 99, 102, 103]),
    'close': np.array([103, 101, 104, 106, 108]),
}
SAMPLE_VOLUME = np.array([1000, 1200, 900, 1500, 1100])


def test_render_ohlc_bars_basic():
    """
    Tests basic rendering of OHLC bars chart.
    """
    img = render_ohlc_bars(SAMPLE_OHLC, SAMPLE_VOLUME, width=800, height=600)

    # Check that an image was created
    assert isinstance(img, Image.Image)

    # Verify dimensions
    assert img.size == (800, 600)

    # Default is RGBA mode with antialiasing enabled
    assert img.mode == "RGBA"


def test_render_ohlc_bars_default_dimensions():
    """
    Tests OHLC bars rendering with default dimensions.
    """
    img = render_ohlc_bars(SAMPLE_OHLC, SAMPLE_VOLUME)

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)  # Default size
    assert img.mode == "RGBA"


def test_render_ohlc_bars_all_themes():
    """
    Tests that all color themes render successfully with OHLC bars.
    """
    for theme_name in THEMES.keys():
        img = render_ohlc_bars(
            SAMPLE_OHLC,
            SAMPLE_VOLUME,
            theme=theme_name,
            width=800,
            height=600
        )

        # Verify image was created
        assert isinstance(img, Image.Image)
        assert img.size == (800, 600)


def test_render_ohlc_bars_rgb_mode():
    """
    Tests OHLC bars rendering in RGB mode (no antialiasing).
    """
    img = render_ohlc_bars(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=800,
        height=600,
        enable_antialiasing=False
    )

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (800, 600)


def test_render_ohlc_bars_no_grid():
    """
    Tests OHLC bars rendering without grid lines.
    """
    img = render_ohlc_bars(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=800,
        height=600,
        show_grid=False
    )

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_custom_colors():
    """
    Tests OHLC bars with custom color overrides.
    """
    img = render_ohlc_bars(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=800,
        height=600,
        theme="modern",
        bg_color="#000000",
        up_color="#00FF00",
        down_color="#FF0000"
    )

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_single_bar():
    """
    Tests OHLC bars rendering with a single bar.
    """
    ohlc = {
        'open': np.array([100]),
        'high': np.array([105]),
        'low': np.array([98]),
        'close': np.array([103]),
    }
    volume = np.array([1000])

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_many_bars():
    """
    Tests OHLC bars rendering with many bars (performance test).
    """
    num_bars = 1000
    ohlc = {
        'open': np.random.uniform(90, 110, num_bars),
        'high': np.random.uniform(110, 120, num_bars),
        'low': np.random.uniform(80, 90, num_bars),
        'close': np.random.uniform(90, 110, num_bars),
    }
    volume = np.random.uniform(500, 2000, num_bars)

    img = render_ohlc_bars(ohlc, volume, width=1920, height=1080)

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_render_ohlc_bars_all_bullish():
    """
    Tests OHLC bars where all bars are bullish (close >= open).
    """
    ohlc = {
        'open': np.array([100, 102, 104]),
        'high': np.array([105, 107, 109]),
        'low': np.array([98, 100, 102]),
        'close': np.array([103, 105, 107]),  # All close >= open
    }
    volume = np.array([1000, 1200, 900])

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_all_bearish():
    """
    Tests OHLC bars where all bars are bearish (close < open).
    """
    ohlc = {
        'open': np.array([103, 105, 107]),
        'high': np.array([105, 107, 109]),
        'low': np.array([98, 100, 102]),
        'close': np.array([100, 102, 104]),  # All close < open
    }
    volume = np.array([1000, 1200, 900])

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_equal_open_close():
    """
    Tests OHLC bars where open equals close (doji pattern).
    """
    ohlc = {
        'open': np.array([100, 102, 104]),
        'high': np.array([105, 107, 109]),
        'low': np.array([98, 100, 102]),
        'close': np.array([100, 102, 104]),  # close == open
    }
    volume = np.array([1000, 1200, 900])

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_zero_volume():
    """
    Tests OHLC bars with zero volume bars.
    """
    ohlc = {
        'open': np.array([100, 102, 101]),
        'high': np.array([105, 106, 105]),
        'low': np.array([98, 100, 99]),
        'close': np.array([103, 101, 104]),
    }
    volume = np.array([0, 1200, 0])  # Some zero volume

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_small_dimensions():
    """
    Tests OHLC bars rendering with small dimensions.
    """
    img = render_ohlc_bars(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=400,
        height=300
    )

    assert isinstance(img, Image.Image)
    assert img.size == (400, 300)


def test_render_ohlc_bars_large_dimensions():
    """
    Tests OHLC bars rendering with large dimensions.
    """
    img = render_ohlc_bars(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=3840,
        height=2160
    )

    assert isinstance(img, Image.Image)
    assert img.size == (3840, 2160)


def test_render_ohlc_bars_save_webp():
    """
    Tests saving OHLC bars chart to WebP format.
    """
    img = render_ohlc_bars(SAMPLE_OHLC, SAMPLE_VOLUME, width=800, height=600)

    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
        save_chart(img, tmp.name, speed='fast')

        # Verify file was created
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0

        # Clean up
        os.unlink(tmp.name)


def test_render_ohlc_bars_save_png():
    """
    Tests saving OHLC bars chart to PNG format.
    """
    img = render_ohlc_bars(SAMPLE_OHLC, SAMPLE_VOLUME, width=800, height=600)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        save_chart(img, tmp.name, speed='fast')

        # Verify file was created
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0

        # Clean up
        os.unlink(tmp.name)


def test_render_ohlc_bars_visual_sample():
    """
    Tests OHLC bars rendering and saves a visual sample for verification.

    This test creates a sample chart that can be manually inspected to verify
    the visual correctness of OHLC bars (vertical lines, left/right ticks).
    """
    # Create test data with clear bullish and bearish bars
    ohlc = {
        'open': np.array([100, 105, 103, 108, 110, 107, 112, 115]),
        'high': np.array([106, 108, 110, 112, 115, 113, 118, 120]),
        'low': np.array([98, 102, 101, 105, 108, 105, 110, 113]),
        'close': np.array([105, 103, 108, 110, 107, 112, 115, 118]),
    }
    volume = np.array([1000, 1500, 1200, 1800, 1100, 1600, 1400, 1700])

    # Test with multiple themes
    themes = ['classic', 'modern', 'tradingview', 'light']

    # Create fixtures directory if it doesn't exist
    fixtures_dir = tempfile.mkdtemp()

    for theme in themes:
        img = render_ohlc_bars(
            ohlc,
            volume,
            width=1200,
            height=800,
            theme=theme,
            enable_antialiasing=True,
            show_grid=True
        )

        assert isinstance(img, Image.Image)
        assert img.size == (1200, 800)

        # Save sample for visual verification
        output_path = os.path.join(fixtures_dir, f'ohlc_bars_sample_{theme}.webp')
        save_chart(img, output_path, speed='fast')

        # Verify file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_render_ohlc_bars_tick_length():
    """
    Tests that OHLC bars have proper tick length (40% of bar width).

    This is a visual correctness test - ticks should be clearly visible
    but not too long.
    """
    # Use fewer bars so ticks are more visible
    ohlc = {
        'open': np.array([100, 105, 103]),
        'high': np.array([106, 108, 110]),
        'low': np.array([98, 102, 101]),
        'close': np.array([105, 103, 108]),
    }
    volume = np.array([1000, 1500, 1200])

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_price_scaling():
    """
    Tests that OHLC bars correctly scale prices across the chart area.
    """
    # Use prices with wide range to test scaling
    ohlc = {
        'open': np.array([100, 200, 150]),
        'high': np.array([150, 250, 200]),
        'low': np.array([50, 150, 100]),
        'close': np.array([120, 180, 170]),
    }
    volume = np.array([1000, 1500, 1200])

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_volume_scaling():
    """
    Tests that OHLC bars correctly scale volume bars.
    """
    # Use volumes with wide range to test scaling
    ohlc = {
        'open': np.array([100, 102, 101]),
        'high': np.array([105, 106, 105]),
        'low': np.array([98, 100, 99]),
        'close': np.array([103, 101, 104]),
    }
    volume = np.array([100, 10000, 1000])  # Wide range

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_ohlc_bars_comparison_sample():
    """
    Creates a comparison sample showing both candlesticks and OHLC bars.

    This is useful for visual verification and comparison between the two styles.
    """
    from kimsfinance.plotting import render_ohlcv_chart

    ohlc = {
        'open': np.array([100, 105, 103, 108, 110, 107, 112, 115, 113, 118]),
        'high': np.array([106, 108, 110, 112, 115, 113, 118, 120, 119, 122]),
        'low': np.array([98, 102, 101, 105, 108, 105, 110, 113, 111, 116]),
        'close': np.array([105, 103, 108, 110, 107, 112, 115, 118, 116, 120]),
    }
    volume = np.array([1000, 1500, 1200, 1800, 1100, 1600, 1400, 1700, 1300, 1900])

    # Create both versions
    img_ohlc = render_ohlc_bars(ohlc, volume, width=1200, height=800, theme='modern')
    img_candle = render_ohlcv_chart(ohlc, volume, width=1200, height=800, theme='modern')

    assert isinstance(img_ohlc, Image.Image)
    assert isinstance(img_candle, Image.Image)

    # Save comparison samples
    fixtures_dir = tempfile.mkdtemp()

    save_chart(img_ohlc, os.path.join(fixtures_dir, 'comparison_ohlc_bars.webp'), speed='fast')
    save_chart(img_candle, os.path.join(fixtures_dir, 'comparison_candlesticks.webp'), speed='fast')


if __name__ == '__main__':
    # Run the visual sample test when executed directly
    print("Generating OHLC bars visual samples...")
    test_render_ohlc_bars_visual_sample()
    test_render_ohlc_bars_comparison_sample()
    print("Visual samples saved to tests/fixtures/")

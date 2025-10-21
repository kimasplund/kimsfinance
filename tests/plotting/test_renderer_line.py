from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from kimsfinance.plotting import (
    render_line_chart,
    save_chart,
)
from kimsfinance.config.themes import THEMES
from kimsfinance.utils.color_utils import _hex_to_rgba


# Sample test data used across multiple tests
SAMPLE_OHLC = {
    "open": np.array([100, 102, 101, 103, 105]),
    "high": np.array([105, 106, 105, 108, 110]),
    "low": np.array([98, 100, 99, 102, 103]),
    "close": np.array([103, 101, 104, 106, 108]),
}
SAMPLE_VOLUME = np.array([1000, 1200, 900, 1500, 1100])


def test_render_line_chart_basic():
    """
    Tests basic line chart rendering.
    """
    # Render the chart
    img = render_line_chart(SAMPLE_OHLC, SAMPLE_VOLUME, width=800, height=600)

    # Check that an image was created
    assert isinstance(img, Image.Image)

    # Verify dimensions
    assert img.size == (800, 600)

    # Default is RGBA mode with antialiasing enabled
    assert img.mode == "RGBA"


def test_render_line_chart_default_size():
    """
    Tests line chart rendering with default dimensions.
    """
    img = render_line_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)  # Default size
    assert img.mode == "RGBA"


def test_render_line_chart_filled_area():
    """
    Tests line chart with filled area under the line.
    """
    img = render_line_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=800,
        height=600,
        fill_area=True
    )

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)
    assert img.mode == "RGBA"


def test_render_line_chart_all_themes():
    """
    Tests that all color themes render successfully.
    """
    for theme_name in THEMES.keys():
        img = render_line_chart(
            SAMPLE_OHLC,
            SAMPLE_VOLUME,
            theme=theme_name,
            width=640,
            height=480
        )

        # Verify image was created
        assert isinstance(img, Image.Image)
        assert img.size == (640, 480)


def test_render_line_chart_custom_colors():
    """
    Tests line chart with custom color overrides.
    """
    img = render_line_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        theme="modern",
        bg_color="#000000",
        line_color="#FFFF00",  # Yellow line
        width=800,
        height=600
    )

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_line_chart_custom_line_width():
    """
    Tests line chart with custom line width.
    """
    # Thin line
    img_thin = render_line_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        line_width=1,
        width=800,
        height=600
    )

    assert isinstance(img_thin, Image.Image)

    # Thick line
    img_thick = render_line_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        line_width=5,
        width=800,
        height=600
    )

    assert isinstance(img_thick, Image.Image)


def test_render_line_chart_no_grid():
    """
    Tests line chart without grid lines.
    """
    img = render_line_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        show_grid=False,
        width=800,
        height=600
    )

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_line_chart_rgb_mode():
    """
    Tests line chart in RGB mode (no antialiasing).
    """
    img = render_line_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        enable_antialiasing=False,
        width=800,
        height=600
    )

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (800, 600)


def test_render_line_chart_rgb_mode_no_fill():
    """
    Tests that fill_area is ignored in RGB mode.
    """
    # In RGB mode, fill_area should be ignored (no error)
    img = render_line_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        enable_antialiasing=False,
        fill_area=True,  # Should be ignored in RGB mode
        width=800,
        height=600
    )

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_render_line_chart_single_point():
    """
    Tests line chart with only one data point.
    """
    ohlc = {
        "open": np.array([100]),
        "high": np.array([105]),
        "low": np.array([98]),
        "close": np.array([103]),
    }
    volume = np.array([1000])

    img = render_line_chart(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_line_chart_two_points():
    """
    Tests line chart with two data points (minimal line).
    """
    ohlc = {
        "open": np.array([100, 102]),
        "high": np.array([105, 106]),
        "low": np.array([98, 100]),
        "close": np.array([103, 101]),
    }
    volume = np.array([1000, 1200])

    img = render_line_chart(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_render_line_chart_large_dataset():
    """
    Tests line chart with a large dataset.
    """
    num_points = 1000
    ohlc = {
        "open": np.random.uniform(100, 200, num_points),
        "high": np.random.uniform(150, 250, num_points),
        "low": np.random.uniform(50, 150, num_points),
        "close": np.random.uniform(100, 200, num_points),
    }
    volume = np.random.uniform(1000, 5000, num_points)

    img = render_line_chart(ohlc, volume, width=1920, height=1080)

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_render_line_chart_upward_trend():
    """
    Tests line chart with upward trending data.
    """
    ohlc = {
        "open": np.array([100, 105, 110, 115, 120]),
        "high": np.array([105, 110, 115, 120, 125]),
        "low": np.array([98, 103, 108, 113, 118]),
        "close": np.array([103, 108, 113, 118, 123]),
    }
    volume = np.array([1000, 1200, 1400, 1600, 1800])

    img = render_line_chart(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)


def test_render_line_chart_downward_trend():
    """
    Tests line chart with downward trending data.
    """
    ohlc = {
        "open": np.array([120, 115, 110, 105, 100]),
        "high": np.array([125, 120, 115, 110, 105]),
        "low": np.array([118, 113, 108, 103, 98]),
        "close": np.array([123, 118, 113, 108, 103]),
    }
    volume = np.array([1800, 1600, 1400, 1200, 1000])

    img = render_line_chart(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)


def test_render_line_chart_volatile_data():
    """
    Tests line chart with highly volatile (zigzag) data.
    """
    ohlc = {
        "open": np.array([100, 120, 90, 130, 80, 140, 70]),
        "high": np.array([125, 140, 110, 150, 100, 160, 90]),
        "low": np.array([95, 100, 85, 110, 75, 120, 65]),
        "close": np.array([120, 90, 130, 80, 140, 70, 150]),
    }
    volume = np.array([1000, 2000, 1500, 2500, 1200, 3000, 1800])

    img = render_line_chart(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)


def test_render_line_chart_save_webp():
    """
    Tests saving line chart to WebP format.
    """
    img = render_line_chart(SAMPLE_OHLC, SAMPLE_VOLUME, width=800, height=600)

    with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as f:
        output_path = f.name

    try:
        save_chart(img, output_path, speed='fast')
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify we can load the saved image
        loaded_img = Image.open(output_path)
        assert loaded_img.size == (800, 600)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_render_line_chart_save_png():
    """
    Tests saving line chart to PNG format.
    """
    img = render_line_chart(SAMPLE_OHLC, SAMPLE_VOLUME, width=800, height=600)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name

    try:
        save_chart(img, output_path, speed='fast')
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify we can load the saved image
        loaded_img = Image.open(output_path)
        assert loaded_img.size == (800, 600)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_generate_sample_line_charts():
    """
    Generates sample line charts for visual inspection.

    This test creates sample charts in the tests/fixtures/ directory.
    """
    fixtures_dir = os.path.join(
        os.path.dirname(__file__),
        'fixtures'
    )
    os.makedirs(fixtures_dir, exist_ok=True)

    # Generate sample data with realistic trend
    np.random.seed(42)
    num_points = 50
    base_price = 100
    trend = np.linspace(0, 20, num_points)
    noise = np.random.normal(0, 2, num_points)
    close_prices = base_price + trend + noise

    ohlc = {
        "open": close_prices + np.random.uniform(-1, 1, num_points),
        "high": close_prices + np.random.uniform(1, 3, num_points),
        "low": close_prices + np.random.uniform(-3, -1, num_points),
        "close": close_prices,
    }
    volume = np.random.uniform(1000, 3000, num_points)

    # Basic line chart
    img_basic = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        theme='modern'
    )
    basic_path = os.path.join(fixtures_dir, 'line_chart_basic.webp')
    save_chart(img_basic, basic_path, speed='fast')
    assert os.path.exists(basic_path)

    # Line chart with filled area
    img_filled = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        theme='modern',
        fill_area=True
    )
    filled_path = os.path.join(fixtures_dir, 'line_chart_filled.webp')
    save_chart(img_filled, filled_path, speed='fast')
    assert os.path.exists(filled_path)

    # Classic theme
    img_classic = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        theme='classic'
    )
    classic_path = os.path.join(fixtures_dir, 'line_chart_classic.webp')
    save_chart(img_classic, classic_path, speed='fast')
    assert os.path.exists(classic_path)

    # TradingView theme with filled area
    img_tradingview = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        theme='tradingview',
        fill_area=True
    )
    tradingview_path = os.path.join(fixtures_dir, 'line_chart_tradingview.webp')
    save_chart(img_tradingview, tradingview_path, speed='fast')
    assert os.path.exists(tradingview_path)

    # Light theme
    img_light = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        theme='light',
        fill_area=True
    )
    light_path = os.path.join(fixtures_dir, 'line_chart_light.webp')
    save_chart(img_light, light_path, speed='fast')
    assert os.path.exists(light_path)

    # Custom colors
    img_custom = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        bg_color='#1A1A1A',
        line_color='#FF6B35',
        line_width=3,
        fill_area=True
    )
    custom_path = os.path.join(fixtures_dir, 'line_chart_custom_colors.webp')
    save_chart(img_custom, custom_path, speed='fast')
    assert os.path.exists(custom_path)

    # No grid
    img_nogrid = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        theme='modern',
        show_grid=False
    )
    nogrid_path = os.path.join(fixtures_dir, 'line_chart_no_grid.webp')
    save_chart(img_nogrid, nogrid_path, speed='fast')
    assert os.path.exists(nogrid_path)

    # Thick line
    img_thick = render_line_chart(
        ohlc,
        volume,
        width=800,
        height=600,
        theme='modern',
        line_width=4
    )
    thick_path = os.path.join(fixtures_dir, 'line_chart_thick_line.webp')
    save_chart(img_thick, thick_path, speed='fast')
    assert os.path.exists(thick_path)

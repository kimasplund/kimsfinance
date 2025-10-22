from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from kimsfinance.plotting import (
    render_hollow_candles,
    save_chart,
)
from kimsfinance.config.themes import THEMES
from kimsfinance.utils.color_utils import _hex_to_rgba


def test_render_hollow_candles_basic():
    """
    Test basic hollow candles rendering with mix of bullish and bearish candles.
    """
    # Create test data with clear bullish and bearish candles
    ohlc = {
        "open": np.array([100, 105, 103, 102, 104]),  # Mix of bull/bear
        "high": np.array([106, 107, 105, 106, 108]),
        "low": np.array([98, 103, 101, 100, 102]),
        "close": np.array([105, 103, 104, 105, 107]),  # 105>100 (bull), 103<105 (bear), ...
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    # Verify image was created
    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)
    assert img.mode in ["RGB", "RGBA"]


def test_hollow_candles_all_bullish():
    """
    Test hollow candles with all bullish candles (all should be hollow).
    """
    ohlc = {
        "open": np.array([100, 102, 104, 106, 108]),
        "high": np.array([103, 105, 107, 109, 111]),
        "low": np.array([99, 101, 103, 105, 107]),
        "close": np.array([102, 104, 106, 108, 110]),  # All close >= open
    }
    volume = np.array([1000, 1100, 1200, 1300, 1400])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_hollow_candles_all_bearish():
    """
    Test hollow candles with all bearish candles (all should be filled).
    """
    ohlc = {
        "open": np.array([110, 108, 106, 104, 102]),
        "high": np.array([111, 109, 107, 105, 103]),
        "low": np.array([107, 105, 103, 101, 99]),
        "close": np.array([108, 106, 104, 102, 100]),  # All close < open
    }
    volume = np.array([1000, 1100, 1200, 1300, 1400])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_hollow_candles_doji_patterns():
    """
    Test hollow candles with doji patterns (open == close).
    These should be treated as bullish (hollow) since close >= open.
    """
    ohlc = {
        "open": np.array([100, 100, 100, 100, 100]),
        "high": np.array([102, 102, 102, 102, 102]),
        "low": np.array([98, 98, 98, 98, 98]),
        "close": np.array([100, 100, 100, 100, 100]),  # All doji (open == close)
    }
    volume = np.array([1000, 1000, 1000, 1000, 1000])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_hollow_candles_all_themes():
    """
    Test that hollow candles work with all available themes.
    """
    ohlc = {
        "open": np.array([100, 105, 103, 102, 104]),
        "high": np.array([106, 107, 105, 106, 108]),
        "low": np.array([98, 103, 101, 100, 102]),
        "close": np.array([105, 103, 104, 105, 107]),
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    for theme_name in THEMES.keys():
        img = render_hollow_candles(ohlc, volume, theme=theme_name, width=800, height=600)
        assert isinstance(img, Image.Image)
        assert img.size == (800, 600)


def test_hollow_candles_custom_colors():
    """
    Test hollow candles with custom color overrides.
    """
    ohlc = {
        "open": np.array([100, 105, 103]),
        "high": np.array([106, 107, 105]),
        "low": np.array([98, 103, 101]),
        "close": np.array([105, 103, 104]),
    }
    volume = np.array([1000, 1200, 900])

    img = render_hollow_candles(
        ohlc,
        volume,
        width=800,
        height=600,
        bg_color="#000000",
        up_color="#00FF00",
        down_color="#FF0000",
    )

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_hollow_candles_rgb_mode():
    """
    Test hollow candles in RGB mode (antialiasing disabled).
    """
    ohlc = {
        "open": np.array([100, 105, 103]),
        "high": np.array([106, 107, 105]),
        "low": np.array([98, 103, 101]),
        "close": np.array([105, 103, 104]),
    }
    volume = np.array([1000, 1200, 900])

    img = render_hollow_candles(ohlc, volume, width=800, height=600, enable_antialiasing=False)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_hollow_candles_rgba_mode():
    """
    Test hollow candles in RGBA mode (antialiasing enabled).
    """
    ohlc = {
        "open": np.array([100, 105, 103]),
        "high": np.array([106, 107, 105]),
        "low": np.array([98, 103, 101]),
        "close": np.array([105, 103, 104]),
    }
    volume = np.array([1000, 1200, 900])

    img = render_hollow_candles(ohlc, volume, width=800, height=600, enable_antialiasing=True)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"


def test_hollow_candles_no_grid():
    """
    Test hollow candles without grid lines.
    """
    ohlc = {
        "open": np.array([100, 105, 103]),
        "high": np.array([106, 107, 105]),
        "low": np.array([98, 103, 101]),
        "close": np.array([105, 103, 104]),
    }
    volume = np.array([1000, 1200, 900])

    img = render_hollow_candles(ohlc, volume, width=800, height=600, show_grid=False)

    assert isinstance(img, Image.Image)


def test_hollow_candles_batch_drawing_enabled():
    """
    Test hollow candles with batch drawing explicitly enabled.
    """
    # Create larger dataset to benefit from batch drawing
    np.random.seed(42)
    num_candles = 100

    open_prices = np.random.uniform(95, 105, num_candles)
    high_prices = open_prices + np.random.uniform(1, 5, num_candles)
    low_prices = open_prices - np.random.uniform(1, 5, num_candles)
    close_prices = open_prices + np.random.uniform(-3, 3, num_candles)

    ohlc = {
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
    }
    volume = np.random.uniform(1000, 2000, num_candles)

    img = render_hollow_candles(ohlc, volume, width=1920, height=1080, use_batch_drawing=True)

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_hollow_candles_batch_drawing_disabled():
    """
    Test hollow candles with batch drawing explicitly disabled.
    """
    ohlc = {
        "open": np.array([100, 105, 103, 102, 104]),
        "high": np.array([106, 107, 105, 106, 108]),
        "low": np.array([98, 103, 101, 100, 102]),
        "close": np.array([105, 103, 104, 105, 107]),
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    img = render_hollow_candles(ohlc, volume, width=800, height=600, use_batch_drawing=False)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_hollow_candles_large_dataset():
    """
    Test hollow candles with a large dataset (auto-enables batch drawing).
    """
    np.random.seed(42)
    num_candles = 5000

    open_prices = np.random.uniform(95, 105, num_candles)
    high_prices = open_prices + np.random.uniform(0.5, 2, num_candles)
    low_prices = open_prices - np.random.uniform(0.5, 2, num_candles)
    close_prices = open_prices + np.random.uniform(-1.5, 1.5, num_candles)

    ohlc = {
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
    }
    volume = np.random.uniform(1000, 2000, num_candles)

    img = render_hollow_candles(ohlc, volume, width=1920, height=1080)

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_hollow_candles_custom_wick_width():
    """
    Test hollow candles with custom wick width ratio.
    """
    ohlc = {
        "open": np.array([100, 105, 103]),
        "high": np.array([106, 107, 105]),
        "low": np.array([98, 103, 101]),
        "close": np.array([105, 103, 104]),
    }
    volume = np.array([1000, 1200, 900])

    img = render_hollow_candles(ohlc, volume, width=800, height=600, wick_width_ratio=0.2)

    assert isinstance(img, Image.Image)


def test_hollow_candles_save_webp():
    """
    Test saving hollow candles chart as WebP.
    """
    ohlc = {
        "open": np.array([100, 105, 103, 102, 104]),
        "high": np.array([106, 107, 105, 106, 108]),
        "low": np.array([98, 103, 101, 100, 102]),
        "close": np.array([105, 103, 104, 105, 107]),
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "hollow_candles.webp")
        save_chart(img, output_path, speed="fast")

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_hollow_candles_save_png():
    """
    Test saving hollow candles chart as PNG.
    """
    ohlc = {
        "open": np.array([100, 105, 103, 102, 104]),
        "high": np.array([106, 107, 105, 106, 108]),
        "low": np.array([98, 103, 101, 100, 102]),
        "close": np.array([105, 103, 104, 105, 107]),
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "hollow_candles.png")
        save_chart(img, output_path, speed="fast")

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_hollow_candles_realistic_market_data():
    """
    Test hollow candles with realistic market data patterns.
    """
    # Simulate realistic BTCUSD-like price action
    ohlc = {
        "open": np.array([42000, 42100, 42050, 41900, 41950, 42100, 42200, 42150]),
        "high": np.array([42200, 42150, 42100, 42000, 42100, 42250, 42300, 42200]),
        "low": np.array([41900, 42000, 41850, 41800, 41900, 42000, 42150, 42100]),
        "close": np.array([42100, 42050, 41900, 41950, 42100, 42200, 42150, 42180]),
    }
    volume = np.array([1500, 1800, 2200, 2000, 1700, 1600, 1400, 1500])

    img = render_hollow_candles(ohlc, volume, width=1920, height=1080, theme="tradingview")

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_hollow_candles_single_candle():
    """
    Test hollow candles with a single candle.
    """
    ohlc = {
        "open": np.array([100]),
        "high": np.array([105]),
        "low": np.array([98]),
        "close": np.array([103]),
    }
    volume = np.array([1000])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_hollow_candles_extreme_volatility():
    """
    Test hollow candles with extreme price volatility.
    """
    ohlc = {
        "open": np.array([100, 150, 80, 120, 90]),
        "high": np.array([160, 180, 130, 140, 110]),
        "low": np.array([90, 70, 60, 100, 70]),
        "close": np.array([150, 80, 120, 90, 105]),
    }
    volume = np.array([3000, 5000, 4500, 3500, 2800])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    assert isinstance(img, Image.Image)


def test_hollow_candles_comparison_dimensions():
    """
    Test that hollow candles produce same dimensions as regular candlesticks.
    """
    ohlc = {
        "open": np.array([100, 105, 103]),
        "high": np.array([106, 107, 105]),
        "low": np.array([98, 103, 101]),
        "close": np.array([105, 103, 104]),
    }
    volume = np.array([1000, 1200, 900])

    # Test various dimensions
    dimensions = [(800, 600), (1920, 1080), (3840, 2160), (640, 480)]

    for width, height in dimensions:
        img = render_hollow_candles(ohlc, volume, width=width, height=height)
        assert img.size == (width, height)


def test_hollow_candles_sample_chart():
    """
    Generate sample hollow candles chart for visual verification.

    This test creates a sample chart with a clear mix of bullish (hollow)
    and bearish (filled) candles for visual verification.
    """
    # Create data with obvious pattern
    ohlc = {
        "open": np.array([100, 105, 110, 108, 112, 115, 113, 116, 120, 118]),
        "high": np.array([107, 112, 112, 115, 117, 117, 119, 122, 122, 121]),
        "low": np.array([98, 103, 107, 106, 110, 112, 111, 114, 117, 116]),
        "close": np.array([105, 110, 108, 112, 115, 113, 116, 120, 118, 119]),
    }
    volume = np.array([1500, 1800, 2200, 2000, 1700, 2100, 1900, 1600, 1800, 2000])

    # Render with modern theme for best visual appearance
    img = render_hollow_candles(
        ohlc,
        volume,
        width=1920,
        height=1080,
        theme="modern",
        enable_antialiasing=True,
        show_grid=True,
    )

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)

    # Save sample chart for visual inspection
    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixture_dir, exist_ok=True)

    output_path = os.path.join(fixture_dir, "hollow_candles_sample.webp")
    save_chart(img, output_path, speed="balanced")

    assert os.path.exists(output_path)
    print(f"\nSample hollow candles chart saved to: {output_path}")

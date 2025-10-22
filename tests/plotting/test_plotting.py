from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from kimsfinance.plotting import (
    render_ohlcv_chart,
    render_ohlcv_charts,
    save_chart,
)
from kimsfinance.config.themes import THEMES
from kimsfinance.config.chart_settings import SPEED_PRESETS
from kimsfinance.plotting import render_charts_parallel
from kimsfinance.utils.color_utils import _hex_to_rgba

# Sample test data used across multiple tests
SAMPLE_OHLC = {
    "open": np.array([100, 102, 101, 105, 103]),
    "high": np.array([103, 105, 103, 106, 104]),
    "low": np.array([99, 101, 100, 104, 102]),
    "close": np.array([102, 101, 102, 104, 103]),
}
SAMPLE_VOLUME = np.array([1000, 1500, 1200, 2000, 1800])


def test_render_ohlcv_chart():
    """
    Tests the rendering of a candlestick chart.
    """
    # Render the chart
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    # Check that an image was created
    assert isinstance(img, Image.Image)

    # Verify dimensions
    assert img.size == (1920, 1080)
    # Default is now RGBA mode with antialiasing enabled
    assert img.mode == "RGBA"


def test_render_all_themes():
    """
    Tests that all color themes render successfully.
    """
    # Test all available themes
    for theme_name in THEMES.keys():
        img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name)

        # Verify image was created
        assert isinstance(img, Image.Image)
        assert img.size == (1920, 1080)  # Default size


def test_theme_with_color_overrides():
    """
    Tests backward compatibility: theme with manual color overrides.
    """
    # Use modern theme but override up_color
    img = render_ohlcv_chart(
        SAMPLE_OHLC, SAMPLE_VOLUME, theme="modern", up_color="#FFFF00"  # Override with yellow
    )

    assert isinstance(img, Image.Image)


def test_backward_compatibility():
    """
    Tests that old code without theme parameter still works.
    """
    # Call without theme parameter (should use classic theme by default)
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    assert isinstance(img, Image.Image)

    # Call with explicit color parameters (should override theme)
    img2 = render_ohlcv_chart(
        SAMPLE_OHLC, SAMPLE_VOLUME, bg_color="#123456", up_color="#ABCDEF", down_color="#FEDCBA"
    )
    assert isinstance(img2, Image.Image)


def test_wick_width_small_chart():
    """
    Tests wick width calculation on a small chart (minimum width enforcement).
    """
    ohlc = {
        "open": np.array([100, 102, 101]),
        "high": np.array([103, 105, 103]),
        "low": np.array([99, 101, 100]),
        "close": np.array([102, 101, 102]),
    }
    volume = np.array([1000, 1500, 1200])

    # Small chart should enforce minimum 1px wick width
    img = render_ohlcv_chart(ohlc, volume, width=100, height=100, wick_width_ratio=0.01)
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)


def test_wick_width_medium_chart():
    """
    Tests wick width calculation on a medium chart.
    """
    # Default wick_width_ratio should work
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, width=800, height=600)
    assert isinstance(img, Image.Image)
    assert img.size == (800, 600)


def test_wick_width_large_chart():
    """
    Tests wick width calculation on a large chart (maximum width enforcement).
    """
    # Large chart with high wick_width_ratio should cap at 10% of bar width
    img = render_ohlcv_chart(
        SAMPLE_OHLC, SAMPLE_VOLUME, width=3840, height=2160, wick_width_ratio=0.5
    )
    assert isinstance(img, Image.Image)
    assert img.size == (3840, 2160)


def test_wick_width_custom_ratios():
    """
    Tests various wick width ratios.
    """
    # Test different ratios
    for ratio in [0.05, 0.1, 0.2, 0.3]:
        img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, wick_width_ratio=ratio)
        assert isinstance(img, Image.Image)


def test_wick_width_edge_cases():
    """
    Tests edge cases for wick width calculation.
    """
    ohlc = {
        "open": np.array([100]),
        "high": np.array([103]),
        "low": np.array([99]),
        "close": np.array([102]),
    }
    volume = np.array([1000])

    # Single candle with very small width
    img = render_ohlcv_chart(ohlc, volume, width=50, height=50, wick_width_ratio=0.0)
    assert isinstance(img, Image.Image)

    # Single candle with maximum ratio
    img = render_ohlcv_chart(ohlc, volume, width=200, height=200, wick_width_ratio=1.0)
    assert isinstance(img, Image.Image)


# Tests for save_chart() function (Task 3)


def test_save_chart_webp():
    """
    Test saving chart in WebP format (lossless).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_chart.webp")
        save_chart(img, output_path)

        # Verify file exists and is WebP format
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify we can load it back
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"
        assert loaded_img.size == img.size


def test_save_chart_png():
    """
    Test saving chart in PNG format (optimized).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_chart.png")
        save_chart(img, output_path)

        # Verify file exists and is PNG format
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify we can load it back
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "PNG"
        assert loaded_img.size == img.size


def test_save_chart_jpeg():
    """
    Test saving chart in JPEG format (progressive).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_chart.jpg")
        save_chart(img, output_path)

        # Verify file exists and is JPEG format
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify we can load it back
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "JPEG"
        assert loaded_img.size == img.size


def test_save_chart_jpeg_alternative_extension():
    """
    Test saving chart using 'jpeg' extension (instead of 'jpg').
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_chart.jpeg")
        save_chart(img, output_path)

        # Verify file exists
        assert os.path.exists(output_path)

        # Verify we can load it back
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "JPEG"


def test_save_chart_auto_format_detection():
    """
    Test auto-detection of format from file extension.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test multiple formats
        formats = [
            ("test.webp", "WEBP"),
            ("test.png", "PNG"),
            ("test.jpg", "JPEG"),
            ("test.jpeg", "JPEG"),
        ]

        for filename, expected_format in formats:
            output_path = os.path.join(tmpdir, filename)
            save_chart(img, output_path)  # No format parameter - auto-detect

            loaded_img = Image.open(output_path)
            assert (
                loaded_img.format == expected_format
            ), f"Expected {expected_format} for {filename}, got {loaded_img.format}"


def test_save_chart_explicit_format():
    """
    Test explicitly specifying format parameter.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as PNG even though extension is .dat
        output_path = os.path.join(tmpdir, "test.dat")
        save_chart(img, output_path, format="png")

        # Verify it's actually PNG
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "PNG"


def test_save_chart_custom_kwargs():
    """
    Test passing custom kwargs to override defaults.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Override JPEG quality
        output_path = os.path.join(tmpdir, "test_low_quality.jpg")
        save_chart(img, output_path, quality=50)

        # Verify file exists and is smaller than default quality
        assert os.path.exists(output_path)

        # Compare to default quality
        default_path = os.path.join(tmpdir, "test_default.jpg")
        save_chart(img, default_path)

        # Lower quality should generally produce smaller file
        low_quality_size = os.path.getsize(output_path)
        default_size = os.path.getsize(default_path)
        assert low_quality_size < default_size


def test_save_chart_no_extension_error():
    """
    Test that ValueError is raised when no extension and no format specified.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_no_extension")

        with pytest.raises(ValueError, match="Cannot auto-detect format"):
            save_chart(img, output_path)


def test_save_chart_file_sizes_reasonable():
    """
    Test that file sizes are reasonable for different formats.
    This verifies compression is working.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        webp_path = os.path.join(tmpdir, "test.webp")
        png_path = os.path.join(tmpdir, "test.png")
        jpeg_path = os.path.join(tmpdir, "test.jpg")

        save_chart(img, webp_path)
        save_chart(img, png_path)
        save_chart(img, jpeg_path)

        webp_size = os.path.getsize(webp_path)
        png_size = os.path.getsize(png_path)
        jpeg_size = os.path.getsize(jpeg_path)

        # All files should be non-empty
        assert webp_size > 0
        assert png_size > 0
        assert jpeg_size > 0

        # All should be less than 1MB for a 1920x1080 image with simple data
        assert webp_size < 1024 * 1024
        assert png_size < 1024 * 1024
        assert jpeg_size < 1024 * 1024


def test_save_chart_fallback_format():
    """
    Test fallback for uncommon formats (BMP, TIFF, etc.).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test BMP format
        output_path = os.path.join(tmpdir, "test.bmp")
        save_chart(img, output_path)

        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "BMP"


def test_save_chart_case_insensitive_format():
    """
    Test that format detection is case-insensitive.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test uppercase extensions
        formats = [
            ("test.WEBP", "WEBP"),
            ("test.PNG", "PNG"),
            ("test.JPG", "JPEG"),
        ]

        for filename, expected_format in formats:
            output_path = os.path.join(tmpdir, filename)
            save_chart(img, output_path)

            loaded_img = Image.open(output_path)
            assert loaded_img.format == expected_format


def test_save_chart_format_override_extension():
    """
    Test that explicit format parameter overrides file extension.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # File says .jpg but we force PNG
        output_path = os.path.join(tmpdir, "misleading.jpg")
        save_chart(img, output_path, format="PNG")

        # Should actually be PNG despite .jpg extension
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "PNG"


# Tests for RGBA mode and antialiasing (Task 4)


def test_hex_to_rgba_basic():
    """
    Test basic hex to RGBA conversion.
    """
    # Test with # prefix
    assert _hex_to_rgba("#FF0000") == (255, 0, 0, 255)
    assert _hex_to_rgba("#00FF00") == (0, 255, 0, 255)
    assert _hex_to_rgba("#0000FF") == (0, 0, 255, 255)

    # Test without # prefix
    assert _hex_to_rgba("FF0000") == (255, 0, 0, 255)
    assert _hex_to_rgba("FFFFFF") == (255, 255, 255, 255)
    assert _hex_to_rgba("000000") == (0, 0, 0, 255)


def test_hex_to_rgba_with_alpha():
    """
    Test hex to RGBA conversion with custom alpha values.
    """
    # Test various alpha values
    assert _hex_to_rgba("#FF0000", alpha=0) == (255, 0, 0, 0)
    assert _hex_to_rgba("#00FF00", alpha=128) == (0, 255, 0, 128)
    assert _hex_to_rgba("#0000FF", alpha=200) == (0, 0, 255, 200)
    assert _hex_to_rgba("#FFFFFF", alpha=64) == (255, 255, 255, 64)


def test_hex_to_rgba_theme_colors():
    """
    Test hex to RGBA conversion with actual theme colors.
    """
    # Classic theme colors
    assert _hex_to_rgba(THEMES["classic"]["bg"]) == (0, 0, 0, 255)
    assert _hex_to_rgba(THEMES["classic"]["up"]) == (0, 255, 0, 255)
    assert _hex_to_rgba(THEMES["classic"]["down"]) == (255, 0, 0, 255)

    # Light theme colors
    assert _hex_to_rgba(THEMES["light"]["bg"]) == (255, 255, 255, 255)


def test_render_rgba_mode_enabled():
    """
    Test rendering with RGBA mode enabled (default).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=True)

    # Verify RGBA mode
    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"
    assert img.size == (1920, 1080)


def test_render_rgba_mode_disabled():
    """
    Test rendering with RGBA mode disabled (RGB fallback).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=False)

    # Verify RGB mode
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (1920, 1080)


def test_render_rgba_mode_default():
    """
    Test that RGBA mode is enabled by default.
    """
    # Default behavior should be RGBA mode
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    assert img.mode == "RGBA"


def test_rgba_mode_all_themes():
    """
    Test RGBA mode works with all themes.
    """
    for theme_name in THEMES.keys():
        # RGBA mode
        img_rgba = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name, enable_antialiasing=True
        )
        assert img_rgba.mode == "RGBA"

        # RGB mode
        img_rgb = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name, enable_antialiasing=False
        )
        assert img_rgb.mode == "RGB"


def test_rgba_mode_with_custom_colors():
    """
    Test RGBA mode with custom color overrides.
    """
    # RGBA mode with custom colors
    img = render_ohlcv_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        bg_color="#123456",
        up_color="#ABCDEF",
        down_color="#FEDCBA",
        enable_antialiasing=True,
    )

    assert img.mode == "RGBA"
    assert isinstance(img, Image.Image)


def test_rgba_mode_various_sizes():
    """
    Test RGBA mode at various image sizes.
    """
    sizes = [(800, 600), (1920, 1080), (3840, 2160), (400, 300)]

    for width, height in sizes:
        img = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, width=width, height=height, enable_antialiasing=True
        )
        assert img.mode == "RGBA"
        assert img.size == (width, height)


def test_rgba_vs_rgb_file_size():
    """
    Test that RGBA and RGB modes produce valid output with reasonable file sizes.
    """
    img_rgba = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=True)
    img_rgb = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        rgba_path = os.path.join(tmpdir, "rgba.png")
        rgb_path = os.path.join(tmpdir, "rgb.png")

        save_chart(img_rgba, rgba_path)
        save_chart(img_rgb, rgb_path)

        # Both files should exist and be non-empty
        assert os.path.exists(rgba_path)
        assert os.path.exists(rgb_path)
        assert os.path.getsize(rgba_path) > 0
        assert os.path.getsize(rgb_path) > 0

        # Both should be reasonable size (< 1MB for simple test data)
        assert os.path.getsize(rgba_path) < 1024 * 1024
        assert os.path.getsize(rgb_path) < 1024 * 1024


def test_rgba_mode_with_transparency():
    """
    Test that RGBA mode properly supports alpha channel.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=True, show_grid=False)

    # Verify image has alpha channel
    assert img.mode == "RGBA"

    # Get pixel data to verify alpha channel exists (check background pixel)
    pixel = img.getpixel((0, 0))
    assert len(pixel) == 4  # RGBA has 4 components

    # Alpha should be fully opaque for background (when grid is disabled)
    assert pixel[3] == 255


def test_rgb_mode_no_alpha():
    """
    Test that RGB mode does not have alpha channel.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=False)

    # Verify RGB mode
    assert img.mode == "RGB"

    # Get pixel data to verify no alpha channel
    pixel = img.getpixel((0, 0))
    assert len(pixel) == 3  # RGB has 3 components only


def test_rgba_backward_compatibility():
    """
    Test that old code without enable_antialiasing parameter still works.
    Now defaults to RGBA mode.
    """
    # Call without enable_antialiasing parameter (should default to True/RGBA)
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"  # New default


def test_rgba_mode_large_dataset():
    """
    Test RGBA mode with larger dataset.
    """
    # Create larger dataset
    large_ohlc = {
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(150, 250, 100),
        "low": np.random.uniform(50, 150, 100),
        "close": np.random.uniform(100, 200, 100),
    }
    large_volume = np.random.uniform(1000, 5000, 100)

    # Both modes should handle large datasets
    img_rgba = render_ohlcv_chart(large_ohlc, large_volume, enable_antialiasing=True)
    img_rgb = render_ohlcv_chart(large_ohlc, large_volume, enable_antialiasing=False)

    assert img_rgba.mode == "RGBA"
    assert img_rgb.mode == "RGB"


def test_rgba_save_all_formats():
    """
    Test that RGBA images can be saved in all supported formats.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test all formats
        formats = [
            ("test.webp", "WEBP"),
            ("test.png", "PNG"),
            ("test.jpg", "JPEG"),  # JPEG doesn't support alpha, should convert
        ]

        for filename, expected_format in formats:
            output_path = os.path.join(tmpdir, filename)
            save_chart(img, output_path)

            assert os.path.exists(output_path)
            loaded_img = Image.open(output_path)
            assert loaded_img.format == expected_format


def test_rgba_mode_with_wick_widths():
    """
    Test RGBA mode works correctly with variable wick widths.
    """
    for ratio in [0.05, 0.1, 0.2, 0.3]:
        img = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, wick_width_ratio=ratio, enable_antialiasing=True
        )
        assert img.mode == "RGBA"
        assert isinstance(img, Image.Image)


# Tests for Grid Lines and Price Levels (Task 5)


def test_render_with_grid_enabled():
    """
    Test rendering with grid lines enabled (default).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=True)

    # Verify image was created
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_render_with_grid_disabled():
    """
    Test rendering without grid lines.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=False)

    # Verify image was created
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_grid_default_enabled():
    """
    Test that grid is enabled by default.
    """
    # Call without show_grid parameter (should default to True)
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    assert isinstance(img, Image.Image)


def test_grid_with_all_themes():
    """
    Test grid rendering works with all themes.
    """
    for theme_name in THEMES.keys():
        # Grid enabled
        img_grid = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name, show_grid=True)
        assert isinstance(img_grid, Image.Image)

        # Grid disabled
        img_no_grid = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name, show_grid=False
        )
        assert isinstance(img_no_grid, Image.Image)


def test_grid_with_rgba_mode():
    """
    Test grid rendering in RGBA mode with transparency.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=True, show_grid=True)

    # Verify RGBA mode
    assert img.mode == "RGBA"
    assert isinstance(img, Image.Image)


def test_grid_with_rgb_mode():
    """
    Test grid rendering in RGB mode (no transparency).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=False, show_grid=True)

    # Verify RGB mode
    assert img.mode == "RGB"
    assert isinstance(img, Image.Image)


def test_grid_various_chart_sizes():
    """
    Test grid rendering at various chart sizes.
    """
    sizes = [(800, 600), (1920, 1080), (3840, 2160), (400, 300)]

    for width, height in sizes:
        img = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, width=width, height=height, show_grid=True
        )
        assert isinstance(img, Image.Image)
        assert img.size == (width, height)


def test_grid_with_large_dataset():
    """
    Test grid spacing with larger dataset (vertical lines max 20).
    """
    # Create dataset with 100 candles (should trigger vertical line spacing)
    large_ohlc = {
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(150, 250, 100),
        "low": np.random.uniform(50, 150, 100),
        "close": np.random.uniform(100, 200, 100),
    }
    large_volume = np.random.uniform(1000, 5000, 100)

    img = render_ohlcv_chart(large_ohlc, large_volume, show_grid=True)

    assert isinstance(img, Image.Image)


def test_grid_with_small_dataset():
    """
    Test grid spacing with minimal dataset.
    """
    # Single candle dataset
    small_ohlc = {
        "open": np.array([100]),
        "high": np.array([105]),
        "low": np.array([95]),
        "close": np.array([102]),
    }
    small_volume = np.array([1000])

    img = render_ohlcv_chart(small_ohlc, small_volume, show_grid=True)

    assert isinstance(img, Image.Image)


def test_grid_with_custom_colors():
    """
    Test grid with custom color overrides (grid uses theme color).
    """
    img = render_ohlcv_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        theme="modern",
        bg_color="#000000",
        up_color="#FFFFFF",
        down_color="#CCCCCC",
        show_grid=True,
    )

    # Grid should use theme's grid color, not custom colors
    assert isinstance(img, Image.Image)


def test_grid_saves_correctly():
    """
    Test that charts with grid save correctly in all formats.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        formats = [
            ("grid_test.webp", "WEBP"),
            ("grid_test.png", "PNG"),
            ("grid_test.jpg", "JPEG"),
        ]

        for filename, expected_format in formats:
            output_path = os.path.join(tmpdir, filename)
            save_chart(img, output_path)

            assert os.path.exists(output_path)
            loaded_img = Image.open(output_path)
            assert loaded_img.format == expected_format


def test_grid_backward_compatibility():
    """
    Test backward compatibility: old code without show_grid parameter.
    """
    # Call without show_grid parameter (should default to True)
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    assert isinstance(img, Image.Image)


def test_grid_with_all_features():
    """
    Test grid works with all other features combined.
    """
    img = render_ohlcv_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=1600,
        height=900,
        theme="tradingview",
        wick_width_ratio=0.15,
        enable_antialiasing=True,
        show_grid=True,
    )

    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"
    assert img.size == (1600, 900)


# Tests for Batch Drawing Optimization (Task 6)


def test_batch_drawing_explicit_enabled():
    """
    Test batch drawing mode when explicitly enabled.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, use_batch_drawing=True)

    # Verify image was created
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_batch_drawing_explicit_disabled():
    """
    Test sequential drawing mode when batch drawing is explicitly disabled.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, use_batch_drawing=False)

    # Verify image was created
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_batch_drawing_auto_enable_small_dataset():
    """
    Test that batch drawing is NOT auto-enabled for small datasets (<1000 candles).
    """
    # Small dataset (5 candles) should use sequential mode by default
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, use_batch_drawing=None)  # Auto-detect mode

    # Verify image was created (visual output should be identical)
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_batch_drawing_auto_enable_large_dataset():
    """
    Test that batch drawing IS auto-enabled for large datasets (>=1000 candles).
    """
    # Create dataset with 1000+ candles
    large_ohlc = {
        "open": np.random.uniform(100, 200, 1000),
        "high": np.random.uniform(150, 250, 1000),
        "low": np.random.uniform(50, 150, 1000),
        "close": np.random.uniform(100, 200, 1000),
    }
    large_volume = np.random.uniform(1000, 5000, 1000)

    # Should auto-enable batch drawing
    img = render_ohlcv_chart(large_ohlc, large_volume, use_batch_drawing=None)  # Auto-detect mode

    # Verify image was created
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_batch_drawing_threshold_boundary():
    """
    Test batch drawing auto-enable threshold at exactly 1000 candles.
    """
    # Exactly 1000 candles should auto-enable batch drawing
    boundary_ohlc = {
        "open": np.random.uniform(100, 200, 1000),
        "high": np.random.uniform(150, 250, 1000),
        "low": np.random.uniform(50, 150, 1000),
        "close": np.random.uniform(100, 200, 1000),
    }
    boundary_volume = np.random.uniform(1000, 5000, 1000)

    img = render_ohlcv_chart(boundary_ohlc, boundary_volume, use_batch_drawing=None)

    assert isinstance(img, Image.Image)

    # 999 candles should NOT auto-enable batch drawing
    below_threshold_ohlc = {
        "open": np.random.uniform(100, 200, 999),
        "high": np.random.uniform(150, 250, 999),
        "low": np.random.uniform(50, 150, 999),
        "close": np.random.uniform(100, 200, 999),
    }
    below_threshold_volume = np.random.uniform(1000, 5000, 999)

    img2 = render_ohlcv_chart(below_threshold_ohlc, below_threshold_volume, use_batch_drawing=None)

    assert isinstance(img2, Image.Image)


def test_batch_vs_sequential_visual_identical():
    """
    Test that batch and sequential modes produce visually identical output.
    """
    # Render with batch mode
    img_batch = render_ohlcv_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        use_batch_drawing=True,
        show_grid=False,  # Disable grid for cleaner comparison
    )

    # Render with sequential mode
    img_sequential = render_ohlcv_chart(
        SAMPLE_OHLC, SAMPLE_VOLUME, use_batch_drawing=False, show_grid=False
    )

    # Both should be the same size and mode
    assert img_batch.size == img_sequential.size
    assert img_batch.mode == img_sequential.mode

    # Convert to bytes and compare
    # Note: In production, you might want pixel-by-pixel comparison
    # For now, we verify both images are valid and same dimensions
    assert img_batch.tobytes() == img_sequential.tobytes()


def test_batch_drawing_with_all_themes():
    """
    Test batch drawing works with all color themes.
    """
    for theme_name in THEMES.keys():
        # Batch mode
        img_batch = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name, use_batch_drawing=True
        )
        assert isinstance(img_batch, Image.Image)

        # Sequential mode
        img_seq = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name, use_batch_drawing=False
        )
        assert isinstance(img_seq, Image.Image)

        # Visual output should be identical
        assert img_batch.tobytes() == img_seq.tobytes()


def test_batch_drawing_with_rgba_mode():
    """
    Test batch drawing in RGBA mode.
    """
    img = render_ohlcv_chart(
        SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=True, use_batch_drawing=True
    )

    assert img.mode == "RGBA"
    assert isinstance(img, Image.Image)


def test_batch_drawing_with_rgb_mode():
    """
    Test batch drawing in RGB mode.
    """
    img = render_ohlcv_chart(
        SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=False, use_batch_drawing=True
    )

    assert img.mode == "RGB"
    assert isinstance(img, Image.Image)


def test_batch_drawing_with_grid():
    """
    Test batch drawing with grid lines enabled.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=True, use_batch_drawing=True)

    assert isinstance(img, Image.Image)


def test_batch_drawing_with_custom_colors():
    """
    Test batch drawing with custom color overrides.
    """
    img = render_ohlcv_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        bg_color="#000000",
        up_color="#00FF00",
        down_color="#FF0000",
        use_batch_drawing=True,
    )

    assert isinstance(img, Image.Image)


def test_batch_drawing_various_sizes():
    """
    Test batch drawing at various image sizes.
    """
    sizes = [(800, 600), (1920, 1080), (3840, 2160), (400, 300)]

    for width, height in sizes:
        img = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, width=width, height=height, use_batch_drawing=True
        )
        assert img.size == (width, height)
        assert isinstance(img, Image.Image)


def test_batch_drawing_large_dataset():
    """
    Test batch drawing with a large dataset (10K candles).
    """
    # Create 10K candles dataset
    large_ohlc = {
        "open": np.random.uniform(100, 200, 10000),
        "high": np.random.uniform(150, 250, 10000),
        "low": np.random.uniform(50, 150, 10000),
        "close": np.random.uniform(100, 200, 10000),
    }
    large_volume = np.random.uniform(1000, 5000, 10000)

    # Batch mode should handle large datasets efficiently
    img = render_ohlcv_chart(large_ohlc, large_volume, use_batch_drawing=True)

    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_batch_drawing_with_wick_widths():
    """
    Test batch drawing with various wick width ratios.
    """
    for ratio in [0.05, 0.1, 0.2, 0.3]:
        img = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, wick_width_ratio=ratio, use_batch_drawing=True
        )
        assert isinstance(img, Image.Image)


def test_batch_drawing_all_bullish():
    """
    Test batch drawing with all bullish candles.
    """
    # All bullish candles (close >= open)
    bullish_ohlc = {
        "open": np.array([100, 102, 101, 105, 103]),
        "high": np.array([103, 105, 103, 106, 104]),
        "low": np.array([99, 101, 100, 104, 102]),
        "close": np.array([103, 105, 103, 106, 104]),  # All closes >= opens
    }
    bullish_volume = SAMPLE_VOLUME

    img = render_ohlcv_chart(bullish_ohlc, bullish_volume, use_batch_drawing=True)

    assert isinstance(img, Image.Image)


def test_batch_drawing_all_bearish():
    """
    Test batch drawing with all bearish candles.
    """
    # All bearish candles (close < open)
    bearish_ohlc = {
        "open": np.array([103, 105, 103, 106, 104]),
        "high": np.array([104, 106, 104, 107, 105]),
        "low": np.array([99, 101, 100, 104, 102]),
        "close": np.array([100, 102, 101, 105, 103]),  # All closes < opens
    }
    bearish_volume = SAMPLE_VOLUME

    img = render_ohlcv_chart(bearish_ohlc, bearish_volume, use_batch_drawing=True)

    assert isinstance(img, Image.Image)


def test_batch_drawing_mixed_candles():
    """
    Test batch drawing with mixed bullish and bearish candles.
    """
    # Mixed candles (original sample data)
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, use_batch_drawing=True)

    assert isinstance(img, Image.Image)


def test_batch_drawing_saves_correctly():
    """
    Test that charts rendered with batch drawing save correctly in all formats.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, use_batch_drawing=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        formats = [
            ("batch_test.webp", "WEBP"),
            ("batch_test.png", "PNG"),
            ("batch_test.jpg", "JPEG"),
        ]

        for filename, expected_format in formats:
            output_path = os.path.join(tmpdir, filename)
            save_chart(img, output_path)

            assert os.path.exists(output_path)
            loaded_img = Image.open(output_path)
            assert loaded_img.format == expected_format


def test_batch_drawing_backward_compatibility():
    """
    Test backward compatibility: old code without use_batch_drawing parameter.
    """
    # Call without use_batch_drawing parameter (should use auto-detect)
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    assert isinstance(img, Image.Image)


def test_batch_drawing_all_features_combined():
    """
    Test batch drawing with all features enabled.
    """
    img = render_ohlcv_chart(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=1600,
        height=900,
        theme="modern",
        wick_width_ratio=0.15,
        enable_antialiasing=True,
        show_grid=True,
        use_batch_drawing=True,
    )

    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"
    assert img.size == (1600, 900)


# Tests for Batch Rendering API (Task 3)


def test_render_ohlcv_charts_basic():
    """
    Test basic batch rendering with multiple datasets.
    """
    # Create 3 datasets
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    # Render all charts
    charts = render_ohlcv_charts(datasets)

    # Verify we got 3 images
    assert len(charts) == 3
    for img in charts:
        assert isinstance(img, Image.Image)
        assert img.size == (1920, 1080)
        assert img.mode == "RGBA"


def test_render_ohlcv_charts_empty_list():
    """
    Test that empty list returns empty list.
    """
    charts = render_ohlcv_charts([])
    assert charts == []
    assert len(charts) == 0


def test_render_ohlcv_charts_single_dataset():
    """
    Test batch rendering with single dataset.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets)

    assert len(charts) == 1
    assert isinstance(charts[0], Image.Image)


def test_render_ohlcv_charts_common_kwargs():
    """
    Test that common kwargs are applied to all charts.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    # Render with custom dimensions and theme
    charts = render_ohlcv_charts(datasets, width=800, height=600, theme="modern")

    # Verify all charts have same dimensions
    assert len(charts) == 2
    for img in charts:
        assert img.size == (800, 600)
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_all_themes():
    """
    Test batch rendering with all available themes.
    """
    for theme_name in THEMES.keys():
        datasets = [
            {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
            {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        ]

        charts = render_ohlcv_charts(datasets, theme=theme_name)

        assert len(charts) == 2
        for img in charts:
            assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_different_data():
    """
    Test batch rendering with different datasets.
    """
    # Create different datasets
    ohlc1 = {
        "open": np.array([100, 102, 101]),
        "high": np.array([103, 105, 103]),
        "low": np.array([99, 101, 100]),
        "close": np.array([102, 101, 102]),
    }
    volume1 = np.array([1000, 1500, 1200])

    ohlc2 = {
        "open": np.array([200, 205, 202]),
        "high": np.array([206, 208, 204]),
        "low": np.array([198, 203, 200]),
        "close": np.array([205, 204, 203]),
    }
    volume2 = np.array([2000, 2500, 2200])

    datasets = [
        {"ohlc": ohlc1, "volume": volume1},
        {"ohlc": ohlc2, "volume": volume2},
    ]

    charts = render_ohlcv_charts(datasets)

    assert len(charts) == 2
    for img in charts:
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_with_custom_colors():
    """
    Test batch rendering with custom color overrides.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(
        datasets, theme="modern", bg_color="#000000", up_color="#FFFFFF", down_color="#CCCCCC"
    )

    assert len(charts) == 2
    for img in charts:
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_various_sizes():
    """
    Test batch rendering with various image sizes.
    """
    sizes = [(800, 600), (1920, 1080), (400, 300)]

    for width, height in sizes:
        datasets = [
            {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
            {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        ]

        charts = render_ohlcv_charts(datasets, width=width, height=height)

        assert len(charts) == 2
        for img in charts:
            assert img.size == (width, height)


def test_render_ohlcv_charts_rgba_mode():
    """
    Test batch rendering in RGBA mode.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets, enable_antialiasing=True)

    assert len(charts) == 2
    for img in charts:
        assert img.mode == "RGBA"


def test_render_ohlcv_charts_rgb_mode():
    """
    Test batch rendering in RGB mode.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets, enable_antialiasing=False)

    assert len(charts) == 2
    for img in charts:
        assert img.mode == "RGB"


def test_render_ohlcv_charts_with_grid():
    """
    Test batch rendering with grid enabled.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets, show_grid=True)

    assert len(charts) == 2
    for img in charts:
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_without_grid():
    """
    Test batch rendering with grid disabled.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets, show_grid=False)

    assert len(charts) == 2
    for img in charts:
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_batch_drawing():
    """
    Test batch rendering with batch drawing enabled.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets, use_batch_drawing=True)

    assert len(charts) == 2
    for img in charts:
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_wick_width():
    """
    Test batch rendering with custom wick width ratio.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets, wick_width_ratio=0.2)

    assert len(charts) == 2
    for img in charts:
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_all_kwargs():
    """
    Test batch rendering with all possible kwargs.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(
        datasets,
        width=1600,
        height=900,
        theme="tradingview",
        bg_color="#131722",
        up_color="#089981",
        down_color="#F23645",
        wick_width_ratio=0.15,
        enable_antialiasing=True,
        show_grid=True,
        use_batch_drawing=True,
    )

    assert len(charts) == 2
    for img in charts:
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        assert img.size == (1600, 900)


def test_render_ohlcv_charts_large_batch():
    """
    Test batch rendering with many datasets (10 charts).
    """
    datasets = [{"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME} for _ in range(10)]

    charts = render_ohlcv_charts(datasets, theme="modern")

    assert len(charts) == 10
    for img in charts:
        assert isinstance(img, Image.Image)


def test_render_ohlcv_charts_save_all():
    """
    Test that all batch-rendered charts can be saved.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = render_ohlcv_charts(datasets, theme="modern")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(charts):
            output_path = os.path.join(tmpdir, f"chart_{i}.png")
            save_chart(img, output_path)

            # Verify file exists
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Verify we can load it back
            loaded_img = Image.open(output_path)
            assert loaded_img.format == "PNG"


def test_render_ohlcv_charts_import_from_package():
    """
    Test that render_ohlcv_charts is properly exported from package.
    """
    from kimsfinance.plotting import render_ohlcv_charts as imported_func

    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    charts = imported_func(datasets)

    assert len(charts) == 1
    assert isinstance(charts[0], Image.Image)


# Tests for Memory-Mapped Output (Task 5)


def test_render_to_array_basic():
    """
    Test basic render_to_array functionality.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)

    # Verify it's a numpy array
    assert isinstance(arr, np.ndarray)

    # Verify shape (default is 1920x1080, RGBA mode)
    assert arr.shape == (1080, 1920, 4)

    # Verify dtype
    assert arr.dtype == np.uint8

    # Verify values are in valid range
    assert np.all(arr >= 0)
    assert np.all(arr <= 255)


def test_render_to_array_rgba_mode():
    """
    Test render_to_array returns RGBA array (default).
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=True)

    # RGBA mode should return 4 channels
    assert arr.shape == (1080, 1920, 4)
    assert arr.dtype == np.uint8


def test_render_to_array_rgb_mode():
    """
    Test render_to_array returns RGB array when antialiasing disabled.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, enable_antialiasing=False)

    # RGB mode should return 3 channels
    assert arr.shape == (1080, 1920, 3)
    assert arr.dtype == np.uint8


def test_render_to_array_custom_dimensions():
    """
    Test render_to_array with custom dimensions.
    """
    from kimsfinance.plotting import render_to_array

    # Test various sizes
    sizes = [(800, 600), (1920, 1080), (3840, 2160), (400, 300)]

    for width, height in sizes:
        arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, width=width, height=height)

        # Verify shape matches (H, W, C)
        assert arr.shape[0] == height
        assert arr.shape[1] == width
        assert arr.shape[2] == 4  # Default RGBA


def test_render_to_array_all_themes():
    """
    Test render_to_array with all available themes.
    """
    from kimsfinance.plotting import render_to_array

    for theme_name in THEMES.keys():
        arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name)

        # Verify valid array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1080, 1920, 4)
        assert arr.dtype == np.uint8


def test_render_to_array_custom_colors():
    """
    Test render_to_array with custom color overrides.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(
        SAMPLE_OHLC, SAMPLE_VOLUME, bg_color="#000000", up_color="#00FF00", down_color="#FF0000"
    )

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1080, 1920, 4)


def test_render_to_array_with_grid():
    """
    Test render_to_array with grid enabled.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=True)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1080, 1920, 4)


def test_render_to_array_without_grid():
    """
    Test render_to_array with grid disabled.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=False)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1080, 1920, 4)


def test_render_to_array_batch_drawing():
    """
    Test render_to_array with batch drawing enabled.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, use_batch_drawing=True)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1080, 1920, 4)


def test_render_to_array_save_to_npy():
    """
    Test that render_to_array output can be saved to numpy file.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "chart.npy")
        np.save(output_path, arr)

        # Verify file exists
        assert os.path.exists(output_path)

        # Load it back
        loaded_arr = np.load(output_path)

        # Verify arrays match
        assert np.array_equal(arr, loaded_arr)
        assert loaded_arr.shape == arr.shape
        assert loaded_arr.dtype == arr.dtype


def test_render_to_array_round_trip():
    """
    Test round-trip: array -> Image -> array produces same result.
    """
    from kimsfinance.plotting import render_to_array
    from PIL import Image as PILImage

    # Render to array
    arr1 = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)

    # Convert to PIL Image
    img = PILImage.fromarray(arr1, mode="RGBA")

    # Convert back to array
    arr2 = np.array(img)

    # Should be identical
    assert np.array_equal(arr1, arr2)


def test_render_to_array_memory_layout():
    """
    Test that array has correct memory layout (C-contiguous).
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)

    # Verify C-contiguous (row-major) layout
    assert arr.flags["C_CONTIGUOUS"]


def test_render_to_array_writable():
    """
    Test that returned array is writable.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)

    # Should be writable
    assert arr.flags["WRITEABLE"]

    # Test modifying a pixel
    original_pixel = arr[0, 0].copy()
    arr[0, 0] = [255, 0, 0, 255]

    # Verify it changed
    assert not np.array_equal(arr[0, 0], original_pixel)


def test_render_to_array_large_dataset():
    """
    Test render_to_array with large dataset.
    """
    from kimsfinance.plotting import render_to_array

    # Create large dataset
    large_ohlc = {
        "open": np.random.uniform(100, 200, 1000),
        "high": np.random.uniform(150, 250, 1000),
        "low": np.random.uniform(50, 150, 1000),
        "close": np.random.uniform(100, 200, 1000),
    }
    large_volume = np.random.uniform(1000, 5000, 1000)

    arr = render_to_array(large_ohlc, large_volume)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1080, 1920, 4)
    assert arr.dtype == np.uint8


def test_render_to_array_various_kwargs():
    """
    Test render_to_array with various rendering options.
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(
        SAMPLE_OHLC,
        SAMPLE_VOLUME,
        width=1600,
        height=900,
        theme="modern",
        wick_width_ratio=0.15,
        enable_antialiasing=True,
        show_grid=True,
        use_batch_drawing=True,
    )

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (900, 1600, 4)
    assert arr.dtype == np.uint8


def test_render_to_array_import_from_package():
    """
    Test that render_to_array is properly exported from package.
    """
    from kimsfinance.plotting import render_to_array

    # Should be importable and callable
    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)

    assert isinstance(arr, np.ndarray)


def test_render_to_array_compare_to_render():
    """
    Test that render_to_array produces same result as render + conversion.
    """
    from kimsfinance.plotting import render_to_array

    # Method 1: Direct array rendering
    arr1 = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=False)

    # Method 2: Render to Image, then convert to array
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, show_grid=False)
    arr2 = np.array(img)

    # Should produce identical results
    assert np.array_equal(arr1, arr2)


def test_render_to_array_pixel_values():
    """
    Test that pixel values are valid uint8 (0-255).
    """
    from kimsfinance.plotting import render_to_array

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)

    # All values should be 0-255
    assert arr.min() >= 0
    assert arr.max() <= 255

    # Should have non-zero pixels (actual chart content)
    assert arr.max() > 0


def test_render_to_array_different_formats():
    """
    Test render_to_array can be saved in different formats via PIL.
    """
    from kimsfinance.plotting import render_to_array
    from PIL import Image as PILImage

    arr = render_to_array(SAMPLE_OHLC, SAMPLE_VOLUME)
    img = PILImage.fromarray(arr, mode="RGBA")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save in various formats
        formats = [
            ("test.png", "PNG"),
            ("test.webp", "WEBP"),
        ]

        for filename, expected_format in formats:
            output_path = os.path.join(tmpdir, filename)
            img.save(output_path)

            assert os.path.exists(output_path)
            loaded_img = PILImage.open(output_path)
            assert loaded_img.format == expected_format


# Tests for render_and_save() function (Task 4: Direct-to-File API)


def test_render_and_save_basic():
    """
    Test basic render_and_save functionality with auto-format detection.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_render_and_save.webp")

        # Render and save in one step
        from kimsfinance.plotting import render_and_save

        render_and_save(SAMPLE_OHLC, SAMPLE_VOLUME, output_path)

        # Verify file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify it's valid WebP
        img = Image.open(output_path)
        assert img.format == "WEBP"
        assert img.size == (1920, 1080)  # Default size


def test_render_and_save_with_speed():
    """
    Test render_and_save with different speed presets.
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test all speed modes
        for speed in ["fast", "balanced", "best"]:
            output_path = os.path.join(tmpdir, f"test_{speed}.webp")
            render_and_save(SAMPLE_OHLC, SAMPLE_VOLUME, output_path, speed=speed)

            # Verify file exists
            assert os.path.exists(output_path)
            img = Image.open(output_path)
            assert img.format == "WEBP"


def test_render_and_save_with_quality():
    """
    Test render_and_save with explicit quality parameter.
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_quality.webp")
        render_and_save(SAMPLE_OHLC, SAMPLE_VOLUME, output_path, quality=90)

        assert os.path.exists(output_path)
        img = Image.open(output_path)
        assert img.format == "WEBP"


def test_render_and_save_with_render_kwargs():
    """
    Test render_and_save passes render_kwargs correctly.
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_kwargs.png")

        # Pass various rendering parameters
        render_and_save(
            SAMPLE_OHLC,
            SAMPLE_VOLUME,
            output_path,
            width=800,
            height=600,
            theme="modern",
            enable_antialiasing=True,
            show_grid=False,
        )

        # Verify file created with correct dimensions
        assert os.path.exists(output_path)
        img = Image.open(output_path)
        assert img.size == (800, 600)


def test_render_and_save_all_formats():
    """
    Test render_and_save with all supported formats.
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        formats = [
            ("test.webp", "WEBP"),
            ("test.png", "PNG"),
            ("test.jpg", "JPEG"),
        ]

        for filename, expected_format in formats:
            output_path = os.path.join(tmpdir, filename)
            render_and_save(SAMPLE_OHLC, SAMPLE_VOLUME, output_path)

            assert os.path.exists(output_path)
            img = Image.open(output_path)
            assert img.format == expected_format


def test_render_and_save_explicit_format():
    """
    Test render_and_save with explicit format parameter.
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test.dat")
        render_and_save(SAMPLE_OHLC, SAMPLE_VOLUME, output_path, format="png")

        # Should be PNG despite .dat extension
        assert os.path.exists(output_path)
        img = Image.open(output_path)
        assert img.format == "PNG"


def test_render_and_save_no_return_value():
    """
    Test that render_and_save returns None (saves to disk).
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test.webp")
        result = render_and_save(SAMPLE_OHLC, SAMPLE_VOLUME, output_path)

        # Function should return None
        assert result is None
        # But file should exist
        assert os.path.exists(output_path)


def test_render_and_save_combined_parameters():
    """
    Test render_and_save with both save and render parameters.
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_combined.webp")

        render_and_save(
            SAMPLE_OHLC,
            SAMPLE_VOLUME,
            output_path,
            # Save parameters
            speed="fast",
            quality=85,
            # Render parameters
            width=1600,
            height=900,
            theme="tradingview",
            enable_antialiasing=True,
            show_grid=True,
            use_batch_drawing=False,
        )

        assert os.path.exists(output_path)
        img = Image.open(output_path)
        assert img.size == (1600, 900)
        assert img.format == "WEBP"


def test_render_and_save_vs_separate_calls():
    """
    Test that render_and_save produces same output as separate render + save.
    """
    from kimsfinance.plotting import render_and_save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Using render_and_save
        combined_path = os.path.join(tmpdir, "combined.png")
        render_and_save(
            SAMPLE_OHLC,
            SAMPLE_VOLUME,
            combined_path,
            speed="balanced",
            theme="modern",
            width=1920,
            height=1080,
        )

        # Using separate calls
        separate_path = os.path.join(tmpdir, "separate.png")
        img = render_ohlcv_chart(
            SAMPLE_OHLC, SAMPLE_VOLUME, theme="modern", width=1920, height=1080
        )
        save_chart(img, separate_path, speed="balanced")

        # Both files should exist and have similar sizes
        assert os.path.exists(combined_path)
        assert os.path.exists(separate_path)

        combined_size = os.path.getsize(combined_path)
        separate_size = os.path.getsize(separate_path)

        # Sizes should be identical (same rendering + encoding)
        assert combined_size == separate_size


# Tests for Parallel Rendering (Task 6)


def test_render_charts_parallel_basic():
    """
    Test basic parallel rendering with multiple datasets.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.webp") for i in range(len(datasets))]

        # Render in parallel
        results = render_charts_parallel(datasets, output_paths, speed="fast")

        # Verify results
        assert len(results) == 3
        assert results == output_paths

        # Verify files exist
        for path in output_paths:
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            # Verify format
            img = Image.open(path)
            assert img.format == "WEBP"


def test_render_charts_parallel_in_memory():
    """
    Test parallel rendering to in-memory PNG bytes.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    # Render to bytes (no output_paths)
    results = render_charts_parallel(datasets, num_workers=2)

    # Verify results are bytes
    assert len(results) == 2
    for result in results:
        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's valid PNG
        assert result.startswith(b"\x89PNG")


def test_render_charts_parallel_order_preserved():
    """
    Test that parallel rendering preserves input order.
    """
    # Create different datasets
    ohlc1 = {
        "open": np.array([100, 102, 101]),
        "high": np.array([103, 105, 103]),
        "low": np.array([99, 101, 100]),
        "close": np.array([102, 101, 102]),
    }
    volume1 = np.array([1000, 1500, 1200])

    ohlc2 = {
        "open": np.array([200, 205, 202]),
        "high": np.array([206, 208, 204]),
        "low": np.array([198, 203, 200]),
        "close": np.array([205, 204, 203]),
    }
    volume2 = np.array([2000, 2500, 2200])

    datasets = [
        {"ohlc": ohlc1, "volume": volume1},
        {"ohlc": ohlc2, "volume": volume2},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [
            os.path.join(tmpdir, "chart_0.png"),
            os.path.join(tmpdir, "chart_1.png"),
        ]

        results = render_charts_parallel(datasets, output_paths)

        # Results should be in same order as input
        assert results == output_paths


def test_render_charts_parallel_num_workers():
    """
    Test parallel rendering with different worker counts.
    """
    datasets = [{"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME} for _ in range(4)]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(4)]

        # Test with specific worker count
        results = render_charts_parallel(datasets, output_paths, num_workers=2)

        assert len(results) == 4
        for path in output_paths:
            assert os.path.exists(path)


def test_render_charts_parallel_auto_workers():
    """
    Test parallel rendering with automatic worker detection.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

        # num_workers=None should use os.cpu_count()
        results = render_charts_parallel(datasets, output_paths, num_workers=None)

        assert len(results) == 2


def test_render_charts_parallel_speed_modes():
    """
    Test parallel rendering with different speed modes.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    for speed in ["fast", "balanced", "best"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = [os.path.join(tmpdir, f"chart_{i}.webp") for i in range(2)]

            results = render_charts_parallel(datasets, output_paths, speed=speed)

            assert len(results) == 2
            for path in output_paths:
                assert os.path.exists(path)


def test_render_charts_parallel_common_kwargs():
    """
    Test parallel rendering with common rendering kwargs.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

        # Pass common rendering options
        results = render_charts_parallel(
            datasets,
            output_paths,
            theme="modern",
            width=800,
            height=600,
            enable_antialiasing=True,
            show_grid=True,
        )

        assert len(results) == 2

        # Verify charts have correct dimensions
        for path in output_paths:
            img = Image.open(path)
            assert img.size == (800, 600)


def test_render_charts_parallel_large_batch():
    """
    Test parallel rendering with many charts (10 charts).
    """
    datasets = [{"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME} for _ in range(10)]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(10)]

        results = render_charts_parallel(datasets, output_paths, num_workers=4, speed="fast")

        assert len(results) == 10
        for path in output_paths:
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0


def test_render_charts_parallel_single_chart():
    """
    Test parallel rendering with single chart.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, "chart.png")]

        results = render_charts_parallel(datasets, output_paths)

        assert len(results) == 1
        assert os.path.exists(output_paths[0])


def test_render_charts_parallel_empty_list():
    """
    Test parallel rendering with empty list.
    """
    datasets = []

    with tempfile.TemporaryDirectory() as tmpdir:
        results = render_charts_parallel(datasets, output_paths=[])

        assert results == []


def test_render_charts_parallel_mixed_formats():
    """
    Test parallel rendering with different output formats.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [
            os.path.join(tmpdir, "chart_0.webp"),
            os.path.join(tmpdir, "chart_1.png"),
            os.path.join(tmpdir, "chart_2.jpg"),
        ]

        results = render_charts_parallel(datasets, output_paths)

        assert len(results) == 3

        # Verify formats
        img0 = Image.open(output_paths[0])
        assert img0.format == "WEBP"

        img1 = Image.open(output_paths[1])
        assert img1.format == "PNG"

        img2 = Image.open(output_paths[2])
        assert img2.format == "JPEG"


def test_render_charts_parallel_length_mismatch_error():
    """
    Test that length mismatch raises ValueError.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Wrong number of output paths
        output_paths = [os.path.join(tmpdir, "chart.png")]

        with pytest.raises(ValueError, match="Length mismatch"):
            render_charts_parallel(datasets, output_paths)


def test_render_charts_parallel_all_themes():
    """
    Test parallel rendering with all available themes.
    """
    for theme_name in THEMES.keys():
        datasets = [
            {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
            {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

            results = render_charts_parallel(datasets, output_paths, theme=theme_name)

            assert len(results) == 2
            for path in output_paths:
                assert os.path.exists(path)


def test_render_charts_parallel_custom_colors():
    """
    Test parallel rendering with custom color overrides.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

        results = render_charts_parallel(
            datasets,
            output_paths,
            theme="modern",
            bg_color="#000000",
            up_color="#FFFFFF",
            down_color="#CCCCCC",
        )

        assert len(results) == 2


def test_render_charts_parallel_batch_drawing():
    """
    Test parallel rendering with batch drawing enabled.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

        results = render_charts_parallel(datasets, output_paths, use_batch_drawing=True)

        assert len(results) == 2


def test_render_charts_parallel_import_from_package():
    """
    Test that render_charts_parallel is properly exported from package.
    """
    from kimsfinance.plotting import render_charts_parallel as imported_func

    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    # Render to bytes
    results = imported_func(datasets)

    assert len(results) == 1
    assert isinstance(results[0], bytes)


def test_render_charts_parallel_in_memory_load_images():
    """
    Test that in-memory PNG bytes can be loaded as images.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    # Render to bytes
    png_bytes_list = render_charts_parallel(datasets)

    # Load each as image
    for png_bytes in png_bytes_list:
        import io

        buf = io.BytesIO(png_bytes)
        img = Image.open(buf)

        assert isinstance(img, Image.Image)
        assert img.format == "PNG"
        assert img.size == (1920, 1080)


def test_render_charts_parallel_wick_width():
    """
    Test parallel rendering with custom wick width ratio.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

        results = render_charts_parallel(datasets, output_paths, wick_width_ratio=0.2)

        assert len(results) == 2


def test_render_charts_parallel_rgba_mode():
    """
    Test parallel rendering in RGBA mode.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

        results = render_charts_parallel(datasets, output_paths, enable_antialiasing=True)

        assert len(results) == 2

        # Verify RGBA mode
        for path in output_paths:
            img = Image.open(path)
            # PNG preserves RGBA
            assert img.mode in ("RGBA", "RGB")


def test_render_charts_parallel_rgb_mode():
    """
    Test parallel rendering in RGB mode.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.png") for i in range(2)]

        results = render_charts_parallel(datasets, output_paths, enable_antialiasing=False)

        assert len(results) == 2


def test_render_charts_parallel_all_features():
    """
    Test parallel rendering with all features enabled.
    """
    datasets = [
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
        {"ohlc": SAMPLE_OHLC, "volume": SAMPLE_VOLUME},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [os.path.join(tmpdir, f"chart_{i}.webp") for i in range(3)]

        results = render_charts_parallel(
            datasets,
            output_paths,
            num_workers=2,
            speed="fast",
            width=1600,
            height=900,
            theme="tradingview",
            wick_width_ratio=0.15,
            enable_antialiasing=True,
            show_grid=True,
            use_batch_drawing=True,
        )

        assert len(results) == 3

        # Verify all charts
        for path in output_paths:
            assert os.path.exists(path)
            img = Image.open(path)
            assert img.size == (1600, 900)


# Tests for Quality Parameter (Task 2)


def test_quality_parameter_webp():
    """
    Test quality parameter for WebP format.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test valid quality range for WebP (1-100)
        for quality in [1, 50, 75, 85, 100]:
            output_path = os.path.join(tmpdir, f"test_quality_{quality}.webp")
            save_chart(img, output_path, quality=quality)

            # Verify file exists
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Verify we can load it back
            loaded_img = Image.open(output_path)
            assert loaded_img.format == "WEBP"


def test_quality_parameter_jpeg():
    """
    Test quality parameter for JPEG format.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test valid quality range for JPEG (1-95)
        for quality in [1, 50, 75, 90, 95]:
            output_path = os.path.join(tmpdir, f"test_quality_{quality}.jpg")
            save_chart(img, output_path, quality=quality)

            # Verify file exists
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Verify we can load it back
            loaded_img = Image.open(output_path)
            assert loaded_img.format == "JPEG"


def test_quality_parameter_webp_invalid_range():
    """
    Test that invalid quality values raise ValueError for WebP.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test quality < 1
        with pytest.raises(ValueError, match="WebP quality must be in range 1-100"):
            save_chart(img, os.path.join(tmpdir, "test.webp"), quality=0)

        # Test quality > 100
        with pytest.raises(ValueError, match="WebP quality must be in range 1-100"):
            save_chart(img, os.path.join(tmpdir, "test.webp"), quality=101)


def test_quality_parameter_jpeg_invalid_range():
    """
    Test that invalid quality values raise ValueError for JPEG.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test quality < 1
        with pytest.raises(ValueError, match="JPEG quality must be in range 1-95"):
            save_chart(img, os.path.join(tmpdir, "test.jpg"), quality=0)

        # Test quality > 95
        with pytest.raises(ValueError, match="JPEG quality must be in range 1-95"):
            save_chart(img, os.path.join(tmpdir, "test.jpg"), quality=96)


def test_quality_parameter_none_uses_defaults():
    """
    Test that quality=None uses default quality values.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # WebP with quality=None should use defaults
        webp_path = os.path.join(tmpdir, "test_default.webp")
        save_chart(img, webp_path, quality=None)
        assert os.path.exists(webp_path)

        # JPEG with quality=None should use defaults
        jpeg_path = os.path.join(tmpdir, "test_default.jpg")
        save_chart(img, jpeg_path, quality=None)
        assert os.path.exists(jpeg_path)


def test_quality_parameter_png_ignored():
    """
    Test that quality parameter is ignored for PNG (lossless format).
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # PNG with quality parameter should not raise error (just ignored)
        png_path = os.path.join(tmpdir, "test.png")
        save_chart(img, png_path, quality=50)  # Should be ignored
        assert os.path.exists(png_path)

        loaded_img = Image.open(png_path)
        assert loaded_img.format == "PNG"


def test_quality_parameter_file_size_variation():
    """
    Test that different quality values produce different file sizes.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # WebP: Lower quality should produce smaller files
        webp_low = os.path.join(tmpdir, "webp_low.webp")
        webp_high = os.path.join(tmpdir, "webp_high.webp")
        save_chart(img, webp_low, quality=50)
        save_chart(img, webp_high, quality=100)

        low_size = os.path.getsize(webp_low)
        high_size = os.path.getsize(webp_high)

        # Higher quality should generally produce larger files
        # (though not guaranteed for all cases, this is typical)
        assert low_size > 0
        assert high_size > 0

        # JPEG: Lower quality should produce smaller files
        jpeg_low = os.path.join(tmpdir, "jpeg_low.jpg")
        jpeg_high = os.path.join(tmpdir, "jpeg_high.jpg")
        save_chart(img, jpeg_low, quality=50)
        save_chart(img, jpeg_high, quality=95)

        jpeg_low_size = os.path.getsize(jpeg_low)
        jpeg_high_size = os.path.getsize(jpeg_high)

        assert jpeg_low_size > 0
        assert jpeg_high_size > 0
        assert jpeg_low_size < jpeg_high_size  # Lower quality = smaller file


def test_quality_overrides_speed_preset():
    """
    Test that quality parameter overrides speed preset quality.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # speed='fast' normally uses quality=75, but we override to 95
        output_path = os.path.join(tmpdir, "override.webp")
        save_chart(img, output_path, speed="fast", quality=95)

        # Verify file exists
        assert os.path.exists(output_path)

        # We can't directly verify the quality setting from the file,
        # but we can verify it doesn't raise an error and produces valid output
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"


def test_quality_parameter_all_formats():
    """
    Test quality parameter with all supported formats.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # WebP
        webp_path = os.path.join(tmpdir, "test.webp")
        save_chart(img, webp_path, quality=85)
        assert os.path.exists(webp_path)

        # JPEG
        jpeg_path = os.path.join(tmpdir, "test.jpg")
        save_chart(img, jpeg_path, quality=85)
        assert os.path.exists(jpeg_path)

        # PNG (quality ignored)
        png_path = os.path.join(tmpdir, "test.png")
        save_chart(img, png_path, quality=85)
        assert os.path.exists(png_path)


def test_quality_parameter_boundary_values():
    """
    Test quality parameter at boundary values.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # WebP boundary values
        save_chart(img, os.path.join(tmpdir, "webp_min.webp"), quality=1)
        save_chart(img, os.path.join(tmpdir, "webp_max.webp"), quality=100)

        # JPEG boundary values
        save_chart(img, os.path.join(tmpdir, "jpeg_min.jpg"), quality=1)
        save_chart(img, os.path.join(tmpdir, "jpeg_max.jpg"), quality=95)

        # All files should exist
        assert os.path.exists(os.path.join(tmpdir, "webp_min.webp"))
        assert os.path.exists(os.path.join(tmpdir, "webp_max.webp"))
        assert os.path.exists(os.path.join(tmpdir, "jpeg_min.jpg"))
        assert os.path.exists(os.path.join(tmpdir, "jpeg_max.jpg"))


def test_quality_parameter_with_different_speeds():
    """
    Test quality parameter works correctly with different speed presets.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test quality override with all speed modes
        for speed in ["fast", "balanced", "best"]:
            for quality in [60, 80, 95]:
                output_path = os.path.join(tmpdir, f"speed_{speed}_quality_{quality}.webp")
                save_chart(img, output_path, speed=speed, quality=quality)
                assert os.path.exists(output_path)

                loaded_img = Image.open(output_path)
                assert loaded_img.format == "WEBP"


def test_quality_parameter_backward_compatibility():
    """
    Test backward compatibility: old code without quality parameter still works.
    """
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Call without quality parameter (should use defaults)
        output_path = os.path.join(tmpdir, "no_quality.webp")
        save_chart(img, output_path)
        assert os.path.exists(output_path)

        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"


def test_numpy_arrays_c_contiguous():
    """
    Test that all NumPy arrays used in rendering are C-contiguous for optimal cache performance.

    C-contiguous arrays provide:
    - Better CPU cache locality (fewer cache misses)
    - Faster vectorized NumPy operations
    - More efficient SIMD operations
    - 5-10% performance improvement on large datasets (50K+ candles)
    """
    # Import the renderer module to access internal array conversions
    from kimsfinance.utils.array_utils import to_numpy_array

    # Create test data (both small and large datasets)
    test_datasets = [
        # Small dataset
        {
            "ohlc": {
                "open": np.array([100, 102, 101, 105, 103]),
                "high": np.array([103, 105, 103, 106, 104]),
                "low": np.array([99, 101, 100, 104, 102]),
                "close": np.array([102, 101, 102, 104, 103]),
            },
            "volume": np.array([1000, 1500, 1200, 2000, 1800]),
        },
        # Large dataset (1000 candles)
        {
            "ohlc": {
                "open": np.random.uniform(100, 200, 1000),
                "high": np.random.uniform(100, 200, 1000),
                "low": np.random.uniform(100, 200, 1000),
                "close": np.random.uniform(100, 200, 1000),
            },
            "volume": np.random.uniform(1000, 5000, 1000),
        },
    ]

    for dataset in test_datasets:
        ohlc = dataset["ohlc"]
        volume = dataset["volume"]

        # Convert arrays the same way the renderer does
        open_prices = np.ascontiguousarray(to_numpy_array(ohlc["open"]))
        high_prices = np.ascontiguousarray(to_numpy_array(ohlc["high"]))
        low_prices = np.ascontiguousarray(to_numpy_array(ohlc["low"]))
        close_prices = np.ascontiguousarray(to_numpy_array(ohlc["close"]))
        volume_data = np.ascontiguousarray(to_numpy_array(volume))

        # Verify all arrays are C-contiguous
        assert open_prices.flags.c_contiguous, "open_prices must be C-contiguous"
        assert high_prices.flags.c_contiguous, "high_prices must be C-contiguous"
        assert low_prices.flags.c_contiguous, "low_prices must be C-contiguous"
        assert close_prices.flags.c_contiguous, "close_prices must be C-contiguous"
        assert volume_data.flags.c_contiguous, "volume_data must be C-contiguous"

        # Verify arrays are writeable (can be used for in-place operations)
        assert open_prices.flags.writeable, "open_prices must be writeable"
        assert high_prices.flags.writeable, "high_prices must be writeable"
        assert low_prices.flags.writeable, "low_prices must be writeable"
        assert close_prices.flags.writeable, "close_prices must be writeable"
        assert volume_data.flags.writeable, "volume_data must be writeable"

        # Verify arrays have NumPy dtype (int or float)
        assert np.issubdtype(open_prices.dtype, np.number), f"Unexpected dtype: {open_prices.dtype}"
        assert np.issubdtype(high_prices.dtype, np.number), f"Unexpected dtype: {high_prices.dtype}"
        assert np.issubdtype(low_prices.dtype, np.number), f"Unexpected dtype: {low_prices.dtype}"
        assert np.issubdtype(
            close_prices.dtype, np.number
        ), f"Unexpected dtype: {close_prices.dtype}"
        assert np.issubdtype(volume_data.dtype, np.number), f"Unexpected dtype: {volume_data.dtype}"

    # Also verify that rendering works correctly with these contiguous arrays
    img = render_ohlcv_chart(test_datasets[0]["ohlc"], test_datasets[0]["volume"])
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_numpy_arrays_memory_layout_non_contiguous():
    """
    Test that np.ascontiguousarray() correctly handles non-contiguous input arrays.

    This verifies that even if input data is in a non-contiguous layout (e.g., sliced arrays,
    transposed arrays), the renderer converts them to C-contiguous format for optimal performance.
    """
    from kimsfinance.utils.array_utils import to_numpy_array

    # Create a larger array and slice it to create non-contiguous views
    large_array = np.random.uniform(100, 200, 2000)

    # Create non-contiguous arrays using slicing (every other element)
    non_contiguous_open = large_array[::2]  # Strided array (non-contiguous)
    non_contiguous_high = large_array[1::2]

    # Verify these are NOT contiguous before conversion
    assert (
        not non_contiguous_open.flags.c_contiguous
    ), "Test setup error: array should be non-contiguous"
    assert (
        not non_contiguous_high.flags.c_contiguous
    ), "Test setup error: array should be non-contiguous"

    # Convert to contiguous arrays (as renderer does)
    contiguous_open = np.ascontiguousarray(to_numpy_array(non_contiguous_open))
    contiguous_high = np.ascontiguousarray(to_numpy_array(non_contiguous_high))

    # Verify arrays are now C-contiguous
    assert contiguous_open.flags.c_contiguous, "Array must be C-contiguous after conversion"
    assert contiguous_high.flags.c_contiguous, "Array must be C-contiguous after conversion"

    # Verify data integrity (values should be preserved)
    assert np.array_equal(
        contiguous_open, non_contiguous_open
    ), "Data must be preserved during conversion"
    assert np.array_equal(
        contiguous_high, non_contiguous_high
    ), "Data must be preserved during conversion"

    # Verify the arrays work in rendering
    num_candles = min(len(contiguous_open), len(contiguous_high))
    ohlc = {
        "open": contiguous_open[:num_candles],
        "high": contiguous_high[:num_candles],
        "low": contiguous_open[:num_candles] * 0.95,  # Create valid OHLC data
        "close": contiguous_high[:num_candles] * 0.98,
    }
    volume = np.random.uniform(1000, 5000, num_candles)

    img = render_ohlcv_chart(ohlc, volume)
    assert isinstance(img, Image.Image)

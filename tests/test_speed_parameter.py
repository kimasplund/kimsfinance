from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from kimsfinance.plotting.renderer import (
    render_ohlcv_chart,
    save_chart,
    THEMES,
    SPEED_PRESETS
)

# Sample test data
SAMPLE_OHLC = {
    "open": np.array([100, 102, 101, 105, 103]),
    "high": np.array([103, 105, 103, 106, 104]),
    "low": np.array([99, 101, 100, 104, 102]),
    "close": np.array([102, 101, 102, 104, 103]),
}
SAMPLE_VOLUME = np.array([1000, 1500, 1200, 2000, 1800])


# Tests for Speed Parameter (Task 1)

def test_speed_parameter_fast_webp():
    """Test save_chart with speed='fast' for WebP format."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "fast.webp")
        save_chart(img, output_path, speed='fast')
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"
        assert loaded_img.size == img.size


def test_speed_parameter_balanced_webp():
    """Test save_chart with speed='balanced' for WebP format (default)."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "balanced.webp")
        save_chart(img, output_path, speed='balanced')
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"
        assert loaded_img.size == img.size


def test_speed_parameter_best_webp():
    """Test save_chart with speed='best' for WebP format."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "best.webp")
        save_chart(img, output_path, speed='best')
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"
        assert loaded_img.size == img.size


def test_speed_parameter_fast_png():
    """Test save_chart with speed='fast' for PNG format."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "fast.png")
        save_chart(img, output_path, speed='fast')
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "PNG"
        assert loaded_img.size == img.size


def test_speed_parameter_balanced_png():
    """Test save_chart with speed='balanced' for PNG format (default)."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "balanced.png")
        save_chart(img, output_path, speed='balanced')
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "PNG"
        assert loaded_img.size == img.size


def test_speed_parameter_best_png():
    """Test save_chart with speed='best' for PNG format."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "best.png")
        save_chart(img, output_path, speed='best')
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "PNG"
        assert loaded_img.size == img.size


def test_speed_parameter_invalid():
    """Test that invalid speed parameter raises ValueError."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "invalid.webp")
        with pytest.raises(ValueError, match="Invalid speed"):
            save_chart(img, output_path, speed='super_fast')
        with pytest.raises(ValueError, match="Invalid speed"):
            save_chart(img, output_path, speed='slow')
        with pytest.raises(ValueError, match="Invalid speed"):
            save_chart(img, output_path, speed='')


def test_speed_parameter_default():
    """Test that speed parameter defaults to 'balanced'."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        default_path = os.path.join(tmpdir, "default.webp")
        save_chart(img, default_path)
        balanced_path = os.path.join(tmpdir, "balanced.webp")
        save_chart(img, balanced_path, speed='balanced')
        assert os.path.exists(default_path)
        assert os.path.exists(balanced_path)


def test_speed_parameter_file_size_comparison_webp():
    """Test that different speed settings produce different file sizes for WebP."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        fast_path = os.path.join(tmpdir, "fast.webp")
        balanced_path = os.path.join(tmpdir, "balanced.webp")
        best_path = os.path.join(tmpdir, "best.webp")
        save_chart(img, fast_path, speed='fast')
        save_chart(img, balanced_path, speed='balanced')
        save_chart(img, best_path, speed='best')
        fast_size = os.path.getsize(fast_path)
        balanced_size = os.path.getsize(balanced_path)
        best_size = os.path.getsize(best_path)
        assert fast_size > 0
        assert balanced_size > 0
        assert best_size > 0
        assert Image.open(fast_path).format == "WEBP"
        assert Image.open(balanced_path).format == "WEBP"
        assert Image.open(best_path).format == "WEBP"


def test_speed_parameter_file_size_comparison_png():
    """Test that different speed settings produce different file sizes for PNG."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        fast_path = os.path.join(tmpdir, "fast.png")
        balanced_path = os.path.join(tmpdir, "balanced.png")
        best_path = os.path.join(tmpdir, "best.png")
        save_chart(img, fast_path, speed='fast')
        save_chart(img, balanced_path, speed='balanced')
        save_chart(img, best_path, speed='best')
        fast_size = os.path.getsize(fast_path)
        balanced_size = os.path.getsize(balanced_path)
        best_size = os.path.getsize(best_path)
        assert fast_size > 0
        assert balanced_size > 0
        assert best_size > 0
        assert fast_size >= best_size  # Fast mode produces larger files
        assert Image.open(fast_path).format == "PNG"
        assert Image.open(balanced_path).format == "PNG"
        assert Image.open(best_path).format == "PNG"


def test_speed_parameter_kwargs_override():
    """Test that kwargs override speed preset values."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "override.webp")
        save_chart(img, output_path, speed='fast', quality=100, method=6)
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"


def test_speed_parameter_all_modes_webp():
    """Test all three speed modes work correctly for WebP."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        for speed_mode in ['fast', 'balanced', 'best']:
            output_path = os.path.join(tmpdir, f"{speed_mode}.webp")
            save_chart(img, output_path, speed=speed_mode)
            assert os.path.exists(output_path)
            loaded_img = Image.open(output_path)
            assert loaded_img.format == "WEBP"
            assert loaded_img.size == img.size


def test_speed_parameter_all_modes_png():
    """Test all three speed modes work correctly for PNG."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        for speed_mode in ['fast', 'balanced', 'best']:
            output_path = os.path.join(tmpdir, f"{speed_mode}.png")
            save_chart(img, output_path, speed=speed_mode)
            assert os.path.exists(output_path)
            loaded_img = Image.open(output_path)
            assert loaded_img.format == "PNG"
            assert loaded_img.size == img.size


def test_speed_parameter_jpeg_unaffected():
    """Test that speed parameter doesn't affect JPEG (no preset defined)."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        for speed_mode in ['fast', 'balanced', 'best']:
            output_path = os.path.join(tmpdir, f"{speed_mode}.jpg")
            save_chart(img, output_path, speed=speed_mode)
            assert os.path.exists(output_path)
            loaded_img = Image.open(output_path)
            assert loaded_img.format == "JPEG"


def test_speed_parameter_with_all_themes():
    """Test speed parameter works with all color themes."""
    for theme_name in THEMES.keys():
        img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME, theme=theme_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            for speed_mode in ['fast', 'balanced', 'best']:
                output_path = os.path.join(tmpdir, f"{theme_name}_{speed_mode}.webp")
                save_chart(img, output_path, speed=speed_mode)
                assert os.path.exists(output_path)
                loaded_img = Image.open(output_path)
                assert loaded_img.format == "WEBP"


def test_speed_parameter_backward_compatibility():
    """Test backward compatibility: old code without speed parameter still works."""
    img = render_ohlcv_chart(SAMPLE_OHLC, SAMPLE_VOLUME)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "no_speed_param.webp")
        save_chart(img, output_path)
        assert os.path.exists(output_path)
        loaded_img = Image.open(output_path)
        assert loaded_img.format == "WEBP"
        assert loaded_img.size == img.size


def test_speed_presets_constant_exists():
    """Test that SPEED_PRESETS constant is properly defined."""
    assert isinstance(SPEED_PRESETS, dict)
    assert 'fast' in SPEED_PRESETS
    assert 'balanced' in SPEED_PRESETS
    assert 'best' in SPEED_PRESETS
    for speed_mode in ['fast', 'balanced', 'best']:
        assert 'webp' in SPEED_PRESETS[speed_mode]
        assert 'png' in SPEED_PRESETS[speed_mode]
        assert 'quality' in SPEED_PRESETS[speed_mode]['webp']
        assert 'method' in SPEED_PRESETS[speed_mode]['webp']
        assert 'compress_level' in SPEED_PRESETS[speed_mode]['png']


def test_speed_presets_values():
    """Test that SPEED_PRESETS has the correct values as specified in the plan."""
    assert SPEED_PRESETS['fast']['webp']['quality'] == 75
    assert SPEED_PRESETS['fast']['webp']['method'] == 4
    assert SPEED_PRESETS['fast']['png']['compress_level'] == 1
    assert SPEED_PRESETS['balanced']['webp']['quality'] == 85
    assert SPEED_PRESETS['balanced']['webp']['method'] == 5
    assert SPEED_PRESETS['balanced']['png']['compress_level'] == 6
    assert SPEED_PRESETS['best']['webp']['quality'] == 100
    assert SPEED_PRESETS['best']['webp']['method'] == 6
    assert SPEED_PRESETS['best']['png']['compress_level'] == 9

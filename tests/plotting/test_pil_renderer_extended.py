#!/usr/bin/env python3
"""
Extended Test Suite for PIL Renderer in kimsfinance.

This suite is designed to comprehensively test the `pil_renderer.py` module,
addressing low test coverage by focusing on:
1.  Validation helper functions (`_validate_numeric_params`).
2.  The `save_chart` function with its various formats and presets.
3.  Rendering of chart types not covered elsewhere (OHLC, Renko, P&F, Hollow).
4.  Specialty functions like `render_to_array` and `render_and_save`.
5.  Correctness of JIT-compiled and fallback coordinate calculations.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from PIL import Image

# Module to be tested
from kimsfinance.plotting import pil_renderer

# Helper to create sample data
def create_sample_data(num_candles: int = 50) -> dict:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    base_price = 100.0
    prices = []
    current_price = base_price
    for _ in range(num_candles):
        change = np.random.randn() * 2
        current_price += change
        open_price = current_price
        close_price = current_price + np.random.randn() * 3
        high_price = max(open_price, close_price) + abs(np.random.randn() * 2)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 2)
        prices.append({
            "open": open_price, "high": high_price,
            "low": low_price, "close": close_price,
            "volume": np.random.randint(1000, 10000),
        })
        current_price = close_price

    # Convert list of dicts to dict of numpy arrays
    ohlcv = {key: np.array([p[key] for p in prices]) for key in prices[0]}
    return ohlcv

# === Test Helper and Validation Functions ===

def test_validate_numeric_params():
    """Test validation of numeric rendering parameters."""
    # Valid cases
    pil_renderer._validate_numeric_params(1920, 1080)
    pil_renderer._validate_numeric_params(3840, 2160, line_width=5)
    pil_renderer._validate_numeric_params(100, 100, box_size=1.0, reversal_boxes=2)

    # Invalid cases
    with pytest.raises(ValueError, match="width must be between 100 and 8192"):
        pil_renderer._validate_numeric_params(50, 1080)
    with pytest.raises(ValueError, match="height must be between 100 and 8192"):
        pil_renderer._validate_numeric_params(1920, 9999)
    with pytest.raises(ValueError, match="line_width must be between 0.1 and 20.0"):
        pil_renderer._validate_numeric_params(1920, 1080, line_width=0.05)
    with pytest.raises(ValueError, match="box_size must be positive"):
        pil_renderer._validate_numeric_params(1920, 1080, box_size=-2.0)
    with pytest.raises(ValueError, match="reversal_boxes must be between 1 and 10"):
        pil_renderer._validate_numeric_params(1920, 1080, reversal_boxes=11)

def test_save_chart_functionality():
    """Test the save_chart function with different formats and presets."""
    img = Image.new("RGBA", (200, 100), "blue")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test PNG saving
        png_path = tmp_path / "test.png"
        pil_renderer.save_chart(img, str(png_path), speed='fast')
        assert png_path.exists()

        # Test JPEG saving (and RGBA -> RGB conversion)
        jpeg_path = tmp_path / "test.jpg"
        pil_renderer.save_chart(img, str(jpeg_path), quality=90)
        assert jpeg_path.exists()
        with Image.open(jpeg_path) as saved_img:
            assert saved_img.mode == "RGB"

        # Test WebP saving
        webp_path = tmp_path / "test.webp"
        pil_renderer.save_chart(img, str(webp_path), speed='best')
        assert webp_path.exists()

        # Test invalid speed
        with pytest.raises(ValueError, match="Invalid speed"):
            pil_renderer.save_chart(img, str(tmp_path / "invalid.png"), speed='wrong')

# === Test Untested Chart Types ===

def test_render_ohlc_bars():
    """Test rendering of OHLC bars."""
    data = create_sample_data(50)
    img = pil_renderer.render_ohlc_bars(data, data['volume'])
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)

def test_render_renko_chart():
    """Test rendering of Renko charts."""
    data = create_sample_data(100)
    img = pil_renderer.render_renko_chart(data, data['volume'], box_size=2.0)
    assert isinstance(img, Image.Image)

    # Test empty bricks case
    empty_data = {k: np.array([]) for k in data}
    img_empty = pil_renderer.render_renko_chart(empty_data, np.array([]))
    assert isinstance(img_empty, Image.Image)

def test_render_pnf_chart():
    """Test rendering of Point & Figure charts."""
    data = create_sample_data(100)
    img = pil_renderer.render_pnf_chart(data, data['volume'], box_size=2.0)
    assert isinstance(img, Image.Image)

    # Test empty columns case
    empty_data = create_sample_data(1)
    img_empty = pil_renderer.render_pnf_chart(empty_data, empty_data['volume'])
    assert isinstance(img_empty, Image.Image)

def test_render_hollow_candles():
    """Test rendering of hollow candles."""
    data = create_sample_data(50)
    img = pil_renderer.render_hollow_candles(data, data['volume'])
    assert isinstance(img, Image.Image)
    # A more detailed test could analyze pixel colors to ensure hollowness

# === Test Specialty Rendering Functions ===

def test_render_to_array():
    """Test rendering directly to a NumPy array."""
    data = create_sample_data(30)
    arr = pil_renderer.render_to_array(data, data['volume'], width=400, height=300)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (300, 400, 4)  # H, W, C (RGBA)
    assert arr.dtype == np.uint8

def test_render_and_save():
    """Test the convenience render_and_save function."""
    data = create_sample_data(20)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "chart.webp"
        pil_renderer.render_and_save(data, data['volume'], str(output_path), speed='fast')
        assert output_path.exists()

# === Test Coordinate Calculation Logic ===

def test_coordinate_calculation_fallback():
    """Test the NumPy fallback for coordinate calculation."""
    data = create_sample_data(50)
    coords = pil_renderer._calculate_coordinates_numpy(
        num_candles=50,
        candle_width=10,
        spacing=2,
        bar_width=8,
        high_prices=data['high'],
        low_prices=data['low'],
        open_prices=data['open'],
        close_prices=data['close'],
        volume_data=data['volume'],
        price_min=np.min(data['low']),
        price_range=np.max(data['high']) - np.min(data['low']),
        volume_range=np.max(data['volume']),
        chart_height=756,
        volume_height=324,
        height=1080,
    )
    # Check if it returns the correct number of arrays
    assert len(coords) == 11
    # Check if the arrays have the correct length
    assert len(coords[0]) == 50  # x_start

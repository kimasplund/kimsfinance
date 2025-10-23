#!/usr/bin/env python3
"""
Extended Test Suite for SVG Renderer in kimsfinance.

This suite focuses on edge cases, helper functions, and chart types
not fully covered in the primary `test_svg_export.py`.

Specifically, it tests:
1.  Helper function `_validate_save_path` for security and correctness.
2.  SVGZ compression via `_save_svg_or_svgz`.
3.  Error handling when `svgwrite` is not installed.
4.  Rendering of OHLC, Renko, P&F, and Hollow Candlestick charts.
5.  Edge cases like empty data for Renko/P&F charts.
"""

import gzip
import numpy as np
import polars as pl
import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch

from kimsfinance.plotting.svg_renderer import (
    _validate_save_path,
    _save_svg_or_svgz,
    render_candlestick_svg,
    render_ohlc_bars_svg,
    render_renko_chart_svg,
    render_pnf_chart_svg,
    render_hollow_candles_svg,
)


# Sample data generation from test_svg_export
def create_sample_data(num_candles: int = 50) -> pl.DataFrame:
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
        prices.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": np.random.randint(1000, 10000),
            }
        )
        current_price = close_price
    return pl.DataFrame(prices)


# === Test Helper Functions ===


def test_validate_save_path_security():
    """Test _validate_save_path against directory traversal and invalid paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set current working directory to temp dir to have a safe base
        with patch("pathlib.Path.cwd", return_value=Path(tmpdir).resolve()):
            # Test empty path
            with pytest.raises(ValueError):
                _validate_save_path("")

            # Test directory traversal
            with pytest.raises(ValueError):
                _validate_save_path("../../../etc/passwd")

            # Test writing to system directory
            with pytest.raises(ValueError):
                _validate_save_path("/etc/hosts")

            # Test valid path
            valid_path = Path(tmpdir) / "charts" / "my_chart.svg"
            result = _validate_save_path(str(valid_path))
            assert result.parent.exists()
            assert "my_chart.svg" in str(result)


def test_save_svgz_compression():
    """Test saving to a compressed SVGZ file."""

    # A simple mock for the svgwrite.Drawing object
    class MockDrawing:
        def tostring(self):
            return '<svg width="100" height="100"></svg>'

        def saveas(self, filename):
            with open(filename, "w") as f:
                f.write(self.tostring())

    dwg = MockDrawing()
    with tempfile.TemporaryDirectory() as tmpdir:
        svgz_path = Path(tmpdir) / "test_chart.svgz"
        _save_svg_or_svgz(dwg, str(svgz_path))

        assert svgz_path.exists()
        # Verify it's a valid gzip file
        with gzip.open(svgz_path, "rt") as f:
            content = f.read()
            assert content == '<svg width="100" height="100"></svg>'


# === Test Error Handling ===


@patch("kimsfinance.plotting.svg_renderer.SVGWRITE_AVAILABLE", False)
def test_svg_functions_raise_import_error_if_svgwrite_missing():
    """Verify all render functions raise ImportError if svgwrite is not available."""
    df = create_sample_data(10)
    ohlc = {
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
    }
    with pytest.raises(ImportError, match="svgwrite is required"):
        render_candlestick_svg(ohlc)
    with pytest.raises(ImportError, match="svgwrite is required"):
        render_ohlc_bars_svg(ohlc)
    with pytest.raises(ImportError, match="svgwrite is required"):
        render_renko_chart_svg(ohlc)
    with pytest.raises(ImportError, match="svgwrite is required"):
        render_pnf_chart_svg(ohlc)
    with pytest.raises(ImportError, match="svgwrite is required"):
        render_hollow_candles_svg(ohlc)


# === Test Additional Chart Types ===


def test_render_ohlc_bars_svg():
    """Test rendering of OHLC bars."""
    df = create_sample_data(20)
    ohlc = {
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
    }
    svg_content = render_ohlc_bars_svg(ohlc)
    assert '<g id="ohlc_bars">' in svg_content
    # OHLC bars are made of lines
    assert "<line" in svg_content


def test_render_hollow_candles_svg():
    """Test rendering of hollow candles."""
    df = create_sample_data(20)
    ohlc = {
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
    }
    svg_content = render_hollow_candles_svg(ohlc)
    assert '<g id="candles">' in svg_content
    # Bullish candles are hollow (stroke, no fill)
    assert 'fill="none"' in svg_content
    # Bearish candles are filled
    assert 'fill="#' in svg_content


def test_render_renko_chart_svg():
    """Test rendering of a Renko chart."""
    df = create_sample_data(50)
    ohlc = {
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
    }
    svg_content = render_renko_chart_svg(ohlc, box_size=2.0)
    assert '<g id="bricks">' in svg_content
    assert "<rect" in svg_content


def test_render_pnf_chart_svg():
    """Test rendering of a Point & Figure chart."""
    df = create_sample_data(50)
    ohlc = {
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
    }
    svg_content = render_pnf_chart_svg(ohlc, box_size=2.0)
    assert '<g id="pnf_symbols">' in svg_content
    # P&F uses paths for 'X' and circles for 'O'
    assert "<path" in svg_content or "<circle" in svg_content


# === Test Edge Cases ===


def test_render_renko_empty_data():
    """Test Renko chart with data that produces no bricks."""
    # Prices move less than the box size
    ohlc = {
        "open": [100, 100.1, 100.2],
        "high": [100.3, 100.2, 100.3],
        "low": [99.9, 100.0, 100.1],
        "close": [100.1, 100.2, 100.1],
    }
    svg_content = render_renko_chart_svg(ohlc, box_size=1.0)
    # Should produce a valid SVG with a background but no brick group
    assert svg_content.startswith("<svg")
    assert "<g id='bricks'>" not in svg_content


def test_render_pnf_empty_data():
    """Test P&F chart with data that produces no columns."""
    ohlc = {"open": [100], "high": [100], "low": [100], "close": [100]}
    svg_content = render_pnf_chart_svg(ohlc, box_size=1.0)
    # Should produce a valid SVG with a background but no symbols
    assert svg_content.startswith("<svg")
    assert "<g id='pnf_symbols'>" not in svg_content

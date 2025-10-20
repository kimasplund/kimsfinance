"""Test SVG export for hollow candlestick charts."""

import os
import tempfile
import numpy as np
import pytest


def test_render_hollow_candles_svg_basic():
    """Test basic hollow candles SVG rendering."""
    from kimsfinance.plotting.renderer import render_hollow_candles_svg

    # Create sample OHLC data
    ohlc_dict = {
        'open': np.array([100.0, 102.0, 101.0, 103.0, 102.0]),
        'high': np.array([103.0, 105.0, 104.0, 106.0, 105.0]),
        'low': np.array([99.0, 101.0, 100.0, 102.0, 101.0]),
        'close': np.array([102.0, 104.0, 103.0, 105.0, 104.0])
    }
    volume_data = np.array([1000, 1500, 1200, 1800, 1300])

    # Render to string (no file output)
    svg_string = render_hollow_candles_svg(
        ohlc_dict,
        volume_data,
        width=800,
        height=600,
        theme='classic'
    )

    # Verify SVG content
    assert svg_string is not None
    assert '<svg' in svg_string
    assert 'width="800"' in svg_string
    assert 'height="600"' in svg_string
    assert '<g id="candles"' in svg_string
    assert '<g id="volume"' in svg_string


def test_render_hollow_candles_svg_file_output():
    """Test hollow candles SVG rendering with file output."""
    from kimsfinance.plotting.renderer import render_hollow_candles_svg

    # Create sample OHLC data with both bullish and bearish candles
    ohlc_dict = {
        'open': np.array([100.0, 102.0, 104.0, 103.0]),  # Mix of bullish and bearish
        'high': np.array([103.0, 105.0, 104.0, 105.0]),
        'low': np.array([99.0, 101.0, 100.0, 101.0]),
        'close': np.array([102.0, 104.0, 101.0, 104.0])  # bull, bull, bear, bull
    }
    volume_data = np.array([1000, 1500, 1200, 1800])

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Render to file
        svg_string = render_hollow_candles_svg(
            ohlc_dict,
            volume_data,
            width=1920,
            height=1080,
            theme='tradingview',
            output_path=tmp_path
        )

        # Verify file exists
        assert os.path.exists(tmp_path)

        # Verify file content
        with open(tmp_path, 'r') as f:
            file_content = f.read()

        assert '<svg' in file_content
        assert 'width="1920"' in file_content
        assert 'height="1080"' in file_content

        # Verify hollow candles rendering (fill='none' for bullish)
        assert 'fill="none"' in file_content  # Hollow bullish candles

        # Verify filled candles (bearish candles should have fill color)
        assert '#F23645' in file_content or '#089981' in file_content  # TradingView colors

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_render_hollow_candles_svg_no_volume():
    """Test hollow candles SVG rendering without volume panel."""
    from kimsfinance.plotting.renderer import render_hollow_candles_svg

    ohlc_dict = {
        'open': np.array([100.0, 102.0, 101.0]),
        'high': np.array([103.0, 105.0, 104.0]),
        'low': np.array([99.0, 101.0, 100.0]),
        'close': np.array([102.0, 104.0, 103.0])
    }

    # Render without volume
    svg_string = render_hollow_candles_svg(
        ohlc_dict,
        volume=None,
        width=800,
        height=600
    )

    assert svg_string is not None
    assert '<svg' in svg_string
    # Should not have volume group
    assert '<g id="volume"' not in svg_string


def test_render_hollow_candles_svg_custom_colors():
    """Test hollow candles SVG rendering with custom colors."""
    from kimsfinance.plotting.renderer import render_hollow_candles_svg

    ohlc_dict = {
        'open': np.array([100.0, 102.0]),
        'high': np.array([103.0, 105.0]),
        'low': np.array([99.0, 101.0]),
        'close': np.array([102.0, 104.0])
    }

    svg_string = render_hollow_candles_svg(
        ohlc_dict,
        volume=None,
        bg_color='#FFFFFF',
        up_color='#00FF00',
        down_color='#FF0000'
    )

    assert svg_string is not None
    assert '#FFFFFF' in svg_string  # Background color
    assert '#00FF00' in svg_string  # Up color
    # Note: down color may not appear if all candles are bullish


def test_hollow_candles_svg_via_plot_api():
    """Test hollow candles SVG export via plot() API."""
    from kimsfinance.api import plot
    import polars as pl

    # Create sample dataframe
    df = pl.DataFrame({
        'Open': [100.0, 102.0, 101.0, 103.0, 102.0],
        'High': [103.0, 105.0, 104.0, 106.0, 105.0],
        'Low': [99.0, 101.0, 100.0, 102.0, 101.0],
        'Close': [102.0, 104.0, 103.0, 105.0, 104.0],
        'Volume': [1000, 1500, 1200, 1800, 1300]
    })

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Render via plot() API
        result = plot(
            df,
            type='hollow_and_filled',
            volume=True,
            savefig=tmp_path,
            width=800,
            height=600
        )

        # Should return None when saving
        assert result is None

        # Verify file exists
        assert os.path.exists(tmp_path)

        # Verify file content
        with open(tmp_path, 'r') as f:
            file_content = f.read()

        assert '<svg' in file_content
        assert 'width="800"' in file_content
        assert '<g id="candles"' in file_content
        assert '<g id="volume"' in file_content

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_hollow_vs_filled_candles_svg():
    """Test that bullish candles are hollow and bearish are filled."""
    from kimsfinance.plotting.renderer import render_hollow_candles_svg

    # Create data with known bullish and bearish candles
    ohlc_dict = {
        'open': np.array([100.0, 100.0]),  # Same open
        'high': np.array([105.0, 105.0]),
        'low': np.array([95.0, 95.0]),
        'close': np.array([102.0, 98.0])   # First bullish (close > open), second bearish (close < open)
    }

    svg_string = render_hollow_candles_svg(
        ohlc_dict,
        volume=None,
        theme='classic'
    )

    # Should have both hollow (fill='none') and filled rectangles
    assert 'fill="none"' in svg_string  # Hollow bullish candle
    assert 'fill="#00FF00"' in svg_string or 'fill="#FF0000"' in svg_string  # Filled candle


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

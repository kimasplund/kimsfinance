#!/usr/bin/env python3
"""
Test SVG export functionality for kimsfinance.

This script tests the new SVG export capability using svgwrite.
It verifies that:
1. SVG files are created successfully
2. SVG content is valid XML
3. SVG contains expected chart elements
4. File sizes are reasonable
5. Both high-level plot() API and low-level renderer work
"""

import numpy as np
import polars as pl
from pathlib import Path
import xml.etree.ElementTree as ET
import tempfile

from kimsfinance.api import plot
from kimsfinance.plotting import render_candlestick_svg, render_line_chart_svg


def create_sample_data(num_candles: int = 50) -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate realistic price data
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
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': np.random.randint(1000, 10000)
        })

        current_price = close_price

    return pl.DataFrame(prices)


def validate_svg_file(svg_path: Path, chart_type: str = 'candle') -> dict:
    """
    Validate SVG file and extract metadata.

    Args:
        svg_path: Path to the SVG file
        chart_type: Type of chart ('candle', 'ohlc', 'line')

    Returns dict with validation results.
    """
    results = {
        'exists': False,
        'valid_xml': False,
        'has_background': False,
        'has_candles': False,
        'has_line': False,
        'has_volume': False,
        'has_grid': False,
        'file_size_kb': 0,
        'num_candles': 0,
        'num_volume_bars': 0,
    }

    # Check file exists
    if not svg_path.exists():
        print(f"❌ SVG file not found: {svg_path}")
        return results

    results['exists'] = True
    results['file_size_kb'] = svg_path.stat().st_size / 1024

    # Read SVG content
    svg_content = svg_path.read_text()

    # Parse XML
    try:
        root = ET.fromstring(svg_content)
        results['valid_xml'] = True
        print(f"✓ Valid XML/SVG structure")
    except ET.ParseError as e:
        print(f"❌ Invalid XML: {e}")
        return results

    # Check for SVG namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Check for background rect
    bg_rects = root.findall(".//svg:rect[@width='100%']", ns)
    if bg_rects:
        results['has_background'] = True
        print(f"✓ Background rectangle found")

    # Check for candles group
    candles_group = root.find(".//svg:g[@id='candles']", ns)
    if candles_group is not None:
        results['has_candles'] = True
        # Count candlestick elements (each candle = 1 line + 1 rect)
        num_rects = len(candles_group.findall(".//svg:rect", ns))
        results['num_candles'] = num_rects
        print(f"✓ Candles group found with {num_rects} candle bodies")

    # Check for volume group
    volume_group = root.find(".//svg:g[@id='volume']", ns)
    if volume_group is not None:
        results['has_volume'] = True
        num_vol_bars = len(volume_group.findall(".//svg:rect", ns))
        results['num_volume_bars'] = num_vol_bars
        print(f"✓ Volume group found with {num_vol_bars} volume bars")

    # Check for grid group
    grid_group = root.find(".//svg:g[@id='grid']", ns)
    if grid_group is not None:
        results['has_grid'] = True
        num_grid_lines = len(grid_group.findall(".//svg:line", ns))
        print(f"✓ Grid group found with {num_grid_lines} grid lines")

    # Check for line chart group
    line_group = root.find(".//svg:g[@id='line']", ns)
    if line_group is not None:
        results['has_line'] = True
        # Count polyline elements
        num_polylines = len(line_group.findall(".//svg:polyline", ns))
        print(f"✓ Line group found with {num_polylines} polyline(s)")

    return results


def test_low_level_svg_renderer():
    """Test render_candlestick_svg() directly."""
    print("\n" + "="*60)
    print("TEST 1: Low-level SVG renderer (render_candlestick_svg)")
    print("="*60)

    # Create sample data
    df = create_sample_data(50)

    # Prepare OHLC dict
    ohlc_dict = {
        'open': df['Open'].to_numpy(),
        'high': df['High'].to_numpy(),
        'low': df['Low'].to_numpy(),
        'close': df['Close'].to_numpy(),
    }
    volume_array = df['Volume'].to_numpy()

    # Test 1: Classic theme with volume
    output_path = Path(tempfile.mkdtemp()) / 'test_svg_classic.svg'
    print(f"\n📊 Rendering classic theme SVG to: {output_path}")

    svg_content = render_candlestick_svg(
        ohlc_dict,
        volume_array,
        width=1920,
        height=1080,
        theme='classic',
        output_path=str(output_path),
    )

    print(f"✓ SVG content generated ({len(svg_content)} bytes)")

    # Validate
    results = validate_svg_file(output_path)
    print(f"\n📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"
    assert results['has_background'], "Missing background"
    assert results['has_candles'], "Missing candles"
    assert results['has_volume'], "Missing volume bars"
    assert results['has_grid'], "Missing grid"
    assert results['num_candles'] == 50, f"Expected 50 candles, got {results['num_candles']}"

    print("✅ All low-level renderer tests passed!")

    return output_path


def test_high_level_plot_api():
    """Test plot() API with SVG format."""
    print("\n" + "="*60)
    print("TEST 2: High-level plot() API with SVG format")
    print("="*60)

    # Create sample data
    df = create_sample_data(100)

    # Test different themes
    themes = ['classic', 'modern', 'tradingview', 'light']
    temp_dir = tempfile.mkdtemp()

    for theme in themes:
        output_path = Path(temp_dir) / f'test_svg_{theme}.svg'
        print(f"\n📊 Rendering {theme} theme via plot() API to: {output_path}")

        result = plot(
            df,
            type='candle',
            style=theme,
            volume=True,
            savefig=str(output_path),
            width=1920,
            height=1080,
        )

        assert result is None, "plot() with savefig should return None"

        # Validate
        results = validate_svg_file(output_path)
        print(f"📈 File size: {results['file_size_kb']:.2f} KB")

        assert results['exists'], f"SVG file not created for {theme}"
        assert results['valid_xml'], f"SVG is not valid XML for {theme}"
        assert results['has_candles'], f"Missing candles for {theme}"

    print("\n✅ All high-level API tests passed!")


def test_svg_without_volume():
    """Test SVG rendering without volume panel."""
    print("\n" + "="*60)
    print("TEST 3: SVG without volume panel")
    print("="*60)

    df = create_sample_data(30)

    ohlc_dict = {
        'open': df['Open'].to_numpy(),
        'high': df['High'].to_numpy(),
        'low': df['Low'].to_numpy(),
        'close': df['Close'].to_numpy(),
    }

    output_path = Path(tempfile.mkdtemp()) / 'test_svg_no_volume.svg'
    print(f"\n📊 Rendering SVG without volume to: {output_path}")

    svg_content = render_candlestick_svg(
        ohlc_dict,
        volume=None,  # No volume
        width=1200,
        height=800,
        theme='modern',
        output_path=str(output_path),
    )

    # Validate
    results = validate_svg_file(output_path)
    print(f"📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"
    assert results['has_candles'], "Missing candles"
    assert not results['has_volume'], "Should not have volume panel"

    print("✅ No-volume test passed!")


def test_svg_custom_colors():
    """Test SVG with custom color overrides."""
    print("\n" + "="*60)
    print("TEST 4: SVG with custom colors")
    print("="*60)

    df = create_sample_data(25)

    ohlc_dict = {
        'open': df['Open'].to_numpy(),
        'high': df['High'].to_numpy(),
        'low': df['Low'].to_numpy(),
        'close': df['Close'].to_numpy(),
    }
    volume_array = df['Volume'].to_numpy()

    output_path = Path(tempfile.mkdtemp()) / 'test_svg_custom_colors.svg'
    print(f"\n📊 Rendering custom color SVG to: {output_path}")

    svg_content = render_candlestick_svg(
        ohlc_dict,
        volume_array,
        width=1600,
        height=900,
        theme='classic',
        bg_color='#1A1A2E',
        up_color='#16C784',
        down_color='#EA3943',
        output_path=str(output_path),
    )

    # Check colors in SVG content
    assert '#1A1A2E' in svg_content, "Custom background color not found"
    assert '#16C784' in svg_content or '#EA3943' in svg_content, "Custom candle colors not found"

    # Validate structure
    results = validate_svg_file(output_path)
    print(f"📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"

    print("✅ Custom colors test passed!")


def test_svg_large_dataset():
    """Test SVG with larger dataset to check file size."""
    print("\n" + "="*60)
    print("TEST 5: SVG with large dataset (500 candles)")
    print("="*60)

    df = create_sample_data(500)

    output_path = Path(tempfile.mkdtemp()) / 'test_svg_large.svg'
    print(f"\n📊 Rendering large SVG to: {output_path}")

    result = plot(
        df,
        type='candle',
        style='tradingview',
        volume=True,
        savefig=str(output_path),
        width=3840,  # 4K width
        height=2160,
    )

    # Validate
    results = validate_svg_file(output_path)
    print(f"📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"
    assert results['file_size_kb'] < 500, f"SVG file too large: {results['file_size_kb']:.2f} KB"

    print("✅ Large dataset test passed!")


def test_line_chart_svg_basic():
    """Test line chart SVG rendering (basic)."""
    print("\n" + "="*60)
    print("TEST 6: Line chart SVG (basic)")
    print("="*60)

    # Create sample data
    df = create_sample_data(50)

    # Prepare OHLC dict
    ohlc_dict = {
        'open': df['Open'].to_numpy(),
        'high': df['High'].to_numpy(),
        'low': df['Low'].to_numpy(),
        'close': df['Close'].to_numpy(),
    }
    volume_array = df['Volume'].to_numpy()

    output_path = Path(tempfile.mkdtemp()) / 'test_svg_line_basic.svg'
    print(f"\n📊 Rendering line chart SVG to: {output_path}")

    svg_content = render_line_chart_svg(
        ohlc_dict,
        volume_array,
        width=1920,
        height=1080,
        theme='classic',
        output_path=str(output_path),
    )

    print(f"✓ SVG content generated ({len(svg_content)} bytes)")

    # Validate
    results = validate_svg_file(output_path, chart_type='line')
    print(f"\n📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"
    assert results['has_background'], "Missing background"
    assert results['has_line'], "Missing line group"
    assert results['has_volume'], "Missing volume bars"
    assert results['has_grid'], "Missing grid"

    print("✅ Line chart basic test passed!")


def test_line_chart_svg_with_fill():
    """Test line chart SVG with filled area."""
    print("\n" + "="*60)
    print("TEST 7: Line chart SVG with filled area")
    print("="*60)

    df = create_sample_data(100)

    ohlc_dict = {
        'open': df['Open'].to_numpy(),
        'high': df['High'].to_numpy(),
        'low': df['Low'].to_numpy(),
        'close': df['Close'].to_numpy(),
    }
    volume_array = df['Volume'].to_numpy()

    output_path = Path(tempfile.mkdtemp()) / 'test_svg_line_filled.svg'
    print(f"\n📊 Rendering filled line chart SVG to: {output_path}")

    svg_content = render_line_chart_svg(
        ohlc_dict,
        volume_array,
        width=1920,
        height=1080,
        theme='modern',
        fill_area=True,
        line_width=3,
        output_path=str(output_path),
    )

    # Check that polygon element exists (for fill)
    assert '<polygon' in svg_content, "Missing polygon element for filled area"

    # Validate
    results = validate_svg_file(output_path, chart_type='line')
    print(f"📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"
    assert results['has_line'], "Missing line group"

    print("✅ Line chart filled area test passed!")


def test_line_chart_svg_via_plot_api():
    """Test line chart SVG via high-level plot() API."""
    print("\n" + "="*60)
    print("TEST 8: Line chart SVG via plot() API")
    print("="*60)

    df = create_sample_data(75)

    output_path = Path(tempfile.mkdtemp()) / 'test_svg_line_api.svg'
    print(f"\n📊 Rendering line chart via plot() API to: {output_path}")

    result = plot(
        df,
        type='line',
        style='tradingview',
        volume=True,
        savefig=str(output_path),
        width=1600,
        height=900,
        line_width=2,
        fill_area=False,
    )

    assert result is None, "plot() with savefig should return None"

    # Validate
    results = validate_svg_file(output_path, chart_type='line')
    print(f"📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"
    assert results['has_line'], "Missing line group"

    print("✅ Line chart API test passed!")


def test_line_chart_svg_custom_colors():
    """Test line chart SVG with custom colors."""
    print("\n" + "="*60)
    print("TEST 9: Line chart SVG with custom colors")
    print("="*60)

    df = create_sample_data(50)

    ohlc_dict = {
        'open': df['Open'].to_numpy(),
        'high': df['High'].to_numpy(),
        'low': df['Low'].to_numpy(),
        'close': df['Close'].to_numpy(),
    }
    volume_array = df['Volume'].to_numpy()

    output_path = Path(tempfile.mkdtemp()) / 'test_svg_line_custom.svg'
    print(f"\n📊 Rendering custom color line chart SVG to: {output_path}")

    svg_content = render_line_chart_svg(
        ohlc_dict,
        volume_array,
        width=1920,
        height=1080,
        theme='classic',
        bg_color='#0D1117',
        line_color='#58A6FF',
        line_width=2,
        output_path=str(output_path),
    )

    # Check colors in SVG content
    assert '#0D1117' in svg_content, "Custom background color not found"
    assert '#58A6FF' in svg_content, "Custom line color not found"

    # Validate structure
    results = validate_svg_file(output_path, chart_type='line')
    print(f"📈 File size: {results['file_size_kb']:.2f} KB")

    assert results['exists'], "SVG file not created"
    assert results['valid_xml'], "SVG is not valid XML"
    assert results['has_line'], "Missing line group"

    print("✅ Line chart custom colors test passed!")


def main():
    """Run all SVG export tests."""
    print("\n" + "🚀"*30)
    print("SVG Export Test Suite for kimsfinance")
    print("🚀"*30)

    try:
        # Run all tests
        test_low_level_svg_renderer()
        test_high_level_plot_api()
        test_svg_without_volume()
        test_svg_custom_colors()
        test_svg_large_dataset()

        # Line chart tests
        test_line_chart_svg_basic()
        test_line_chart_svg_with_fill()
        test_line_chart_svg_via_plot_api()
        test_line_chart_svg_custom_colors()

        # Summary
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nSVG export functionality is working correctly!")
        print("\nGenerated test files:")
        print("  Candlestick charts:")
        print("    - test_svg_classic.svg")
        print("    - test_svg_modern.svg")
        print("    - test_svg_tradingview.svg")
        print("    - test_svg_light.svg")
        print("    - test_svg_no_volume.svg")
        print("    - test_svg_custom_colors.svg")
        print("    - test_svg_large.svg")
        print("  Line charts:")
        print("    - test_svg_line_basic.svg")
        print("    - test_svg_line_filled.svg")
        print("    - test_svg_line_api.svg")
        print("    - test_svg_line_custom.svg")
        print("\nYou can open these files in a web browser to verify visual quality.")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from kimsfinance.data.renko import calculate_renko_bricks
from kimsfinance.plotting import (
    render_renko_chart,
    render_renko_chart_svg,
    save_chart,
)
from kimsfinance.config.themes import THEMES

# Check if svgwrite is available
try:
    import svgwrite
    SVGWRITE_AVAILABLE = True
except ImportError:
    SVGWRITE_AVAILABLE = False


# Sample test data for Renko charts
def create_trending_data(start: float = 100, end: float = 130, num_points: int = 50):
    """Create synthetic trending data for Renko testing."""
    # Create uptrend with some noise
    close_prices = np.linspace(start, end, num_points)
    noise = np.random.normal(0, 1, num_points)
    close_prices = close_prices + noise

    # Create OHLC from close prices
    high_prices = close_prices + np.abs(np.random.normal(0, 0.5, num_points))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.5, num_points))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start

    return {
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
    }


def test_calculate_renko_bricks_basic():
    """
    Test basic Renko brick calculation with fixed box size.
    """
    ohlc = {
        'open': np.array([100, 102, 105, 103, 100, 98]),
        'high': np.array([101, 104, 107, 104, 101, 99]),
        'low': np.array([99, 101, 104, 102, 99, 97]),
        'close': np.array([100, 103, 106, 103, 100, 98]),
    }

    bricks = calculate_renko_bricks(ohlc, box_size=2.0, reversal_boxes=1)

    # Verify bricks were created
    assert len(bricks) > 0, "Should generate at least one brick"

    # Verify brick structure
    for brick in bricks:
        assert 'price' in brick, "Brick should have 'price' key"
        assert 'direction' in brick, "Brick should have 'direction' key"
        assert brick['direction'] in [1, -1], "Direction should be 1 (up) or -1 (down)"
        assert isinstance(brick['price'], (int, float)), "Price should be numeric"

    # Print bricks for manual inspection
    print(f"\nGenerated {len(bricks)} bricks:")
    for i, brick in enumerate(bricks):
        direction_str = "UP" if brick['direction'] == 1 else "DOWN"
        print(f"  Brick {i+1}: {direction_str} at ${brick['price']:.2f}")


def test_calculate_renko_bricks_uptrend():
    """
    Test Renko bricks with strong uptrend.
    """
    ohlc = {
        'open': np.array([100, 102, 104, 106, 108]),
        'high': np.array([102, 104, 106, 108, 110]),
        'low': np.array([100, 102, 104, 106, 108]),
        'close': np.array([102, 104, 106, 108, 110]),
    }

    bricks = calculate_renko_bricks(ohlc, box_size=2.0, reversal_boxes=1)

    # Should generate mostly up bricks (may have 1 initial down brick)
    assert len(bricks) > 0
    up_bricks = [b for b in bricks if b['direction'] == 1]
    down_bricks = [b for b in bricks if b['direction'] == -1]
    assert len(up_bricks) >= len(down_bricks), "Uptrend should have more up bricks than down"


def test_calculate_renko_bricks_downtrend():
    """
    Test Renko bricks with strong downtrend.
    """
    ohlc = {
        'open': np.array([110, 108, 106, 104, 102]),
        'high': np.array([110, 108, 106, 104, 102]),
        'low': np.array([108, 106, 104, 102, 100]),
        'close': np.array([108, 106, 104, 102, 100]),
    }

    bricks = calculate_renko_bricks(ohlc, box_size=2.0, reversal_boxes=1)

    # Should generate mostly down bricks (may have 1 initial up brick)
    assert len(bricks) > 0
    up_bricks = [b for b in bricks if b['direction'] == 1]
    down_bricks = [b for b in bricks if b['direction'] == -1]
    assert len(down_bricks) >= len(up_bricks), "Downtrend should have more down bricks than up"


def test_calculate_renko_bricks_auto_box_size():
    """
    Test automatic box size calculation using ATR.
    """
    ohlc = create_trending_data(start=100, end=130, num_points=50)

    # Test with auto box size (None)
    bricks = calculate_renko_bricks(ohlc, box_size=None, reversal_boxes=1)

    # Should generate bricks with auto-calculated box size
    assert len(bricks) > 0, "Should generate bricks with auto box size"
    print(f"\nAuto box size generated {len(bricks)} bricks")


def test_calculate_renko_bricks_reversal_filter():
    """
    Test reversal_boxes parameter for noise filtering.
    """
    ohlc = {
        'open': np.array([100, 102, 104, 103, 105, 107]),
        'high': np.array([102, 104, 106, 104, 107, 109]),
        'low': np.array([100, 102, 103, 102, 105, 107]),
        'close': np.array([102, 104, 103, 104, 107, 109]),
    }

    # Test with reversal_boxes=1 (default, responsive)
    bricks_rev1 = calculate_renko_bricks(ohlc, box_size=2.0, reversal_boxes=1)

    # Test with reversal_boxes=2 (more filtering)
    bricks_rev2 = calculate_renko_bricks(ohlc, box_size=2.0, reversal_boxes=2)

    # With higher reversal threshold, should filter out small reversals
    print(f"\nreversal_boxes=1: {len(bricks_rev1)} bricks")
    print(f"reversal_boxes=2: {len(bricks_rev2)} bricks")


def test_calculate_renko_bricks_empty_data():
    """
    Test handling of empty data.
    """
    ohlc = {
        'open': np.array([]),
        'high': np.array([]),
        'low': np.array([]),
        'close': np.array([]),
    }

    bricks = calculate_renko_bricks(ohlc, box_size=2.0)
    assert len(bricks) == 0, "Empty data should return empty brick list"


def test_calculate_renko_bricks_invalid_box_size():
    """
    Test handling of invalid box size.
    """
    ohlc = create_trending_data()

    with pytest.raises(ValueError, match="box_size must be positive"):
        calculate_renko_bricks(ohlc, box_size=0)

    with pytest.raises(ValueError, match="box_size must be positive"):
        calculate_renko_bricks(ohlc, box_size=-1)


def test_render_renko_chart_basic():
    """
    Test basic Renko chart rendering.
    """
    ohlc = create_trending_data(start=100, end=130, num_points=50)
    volume = np.random.randint(800, 1200, size=50)

    img = render_renko_chart(ohlc, volume, width=1200, height=800, box_size=2.0)

    # Verify image was created
    assert isinstance(img, Image.Image)
    assert img.size == (1200, 800)
    assert img.mode == "RGBA"  # Default antialiasing enabled


def test_render_renko_chart_all_themes():
    """
    Test rendering with all available themes.
    """
    ohlc = create_trending_data(start=100, end=120, num_points=30)
    volume = np.random.randint(800, 1200, size=30)

    for theme_name in THEMES.keys():
        img = render_renko_chart(
            ohlc, volume,
            width=800, height=600,
            theme=theme_name,
            box_size=1.5
        )

        assert isinstance(img, Image.Image)
        assert img.size == (800, 600)
        print(f"Theme '{theme_name}': OK")


def test_render_renko_chart_custom_colors():
    """
    Test Renko chart with custom color overrides.
    """
    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    img = render_renko_chart(
        ohlc, volume,
        width=1000, height=700,
        bg_color="#1A1A1A",
        up_color="#00FF00",
        down_color="#FF0000",
        box_size=2.0
    )

    assert isinstance(img, Image.Image)
    assert img.size == (1000, 700)


def test_render_renko_chart_no_antialiasing():
    """
    Test Renko chart without antialiasing (RGB mode).
    """
    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    img = render_renko_chart(
        ohlc, volume,
        width=800, height=600,
        enable_antialiasing=False,
        box_size=2.0
    )

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"  # No antialiasing


def test_render_renko_chart_no_grid():
    """
    Test Renko chart without grid lines.
    """
    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    img = render_renko_chart(
        ohlc, volume,
        width=800, height=600,
        show_grid=False,
        box_size=2.0
    )

    assert isinstance(img, Image.Image)


def test_render_renko_chart_auto_box_size():
    """
    Test Renko chart with automatic box size calculation.
    """
    ohlc = create_trending_data(start=100, end=150, num_points=100)
    volume = np.random.randint(800, 1200, size=100)

    # Render with auto box size (None)
    img = render_renko_chart(
        ohlc, volume,
        width=1200, height=800,
        box_size=None  # Auto-calculate using ATR
    )

    assert isinstance(img, Image.Image)
    assert img.size == (1200, 800)
    print("\nRendered Renko chart with auto box size")


def test_render_renko_chart_different_reversal_thresholds():
    """
    Test Renko chart with different reversal_boxes values.
    """
    ohlc = create_trending_data(start=100, end=130, num_points=80)
    volume = np.random.randint(800, 1200, size=80)

    # Test reversal_boxes=1 (default, responsive)
    img1 = render_renko_chart(
        ohlc, volume,
        width=1000, height=700,
        box_size=1.5,
        reversal_boxes=1
    )
    assert isinstance(img1, Image.Image)

    # Test reversal_boxes=2 (more filtering)
    img2 = render_renko_chart(
        ohlc, volume,
        width=1000, height=700,
        box_size=1.5,
        reversal_boxes=2
    )
    assert isinstance(img2, Image.Image)

    # Test reversal_boxes=3 (aggressive filtering)
    img3 = render_renko_chart(
        ohlc, volume,
        width=1000, height=700,
        box_size=1.5,
        reversal_boxes=3
    )
    assert isinstance(img3, Image.Image)


def test_render_renko_chart_empty_bricks():
    """
    Test rendering when no bricks are generated.
    """
    # Create data with very small movements (no bricks with large box size)
    ohlc = {
        'open': np.array([100.0, 100.1, 100.05, 100.08]),
        'high': np.array([100.15, 100.15, 100.12, 100.12]),
        'low': np.array([99.95, 100.0, 100.0, 100.05]),
        'close': np.array([100.1, 100.05, 100.08, 100.1]),
    }
    volume = np.array([1000, 1100, 1050, 1080])

    # Use very large box size that won't generate any bricks
    img = render_renko_chart(ohlc, volume, box_size=10.0)

    # Should return empty chart (background only)
    assert isinstance(img, Image.Image)
    assert img.size == (1920, 1080)


def test_render_and_save_renko_chart():
    """
    Test rendering and saving Renko chart to file.
    """
    ohlc = create_trending_data(start=100, end=130, num_points=50)
    volume = np.random.randint(800, 1200, size=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "renko_test.webp")

        # Render chart
        img = render_renko_chart(ohlc, volume, width=1200, height=800, box_size=2.0)

        # Save chart
        save_chart(img, output_path, speed='fast')

        # Verify file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify can be loaded
        loaded_img = Image.open(output_path)
        assert loaded_img.size == (1200, 800)


def test_render_renko_chart_sample_output():
    """
    Generate sample Renko chart for visual inspection.
    """
    # Create realistic trending data with volatility
    np.random.seed(42)  # For reproducibility
    ohlc = create_trending_data(start=100, end=140, num_points=100)
    volume = np.random.randint(800, 1500, size=100)

    # Render Renko chart with classic theme
    img = render_renko_chart(
        ohlc, volume,
        width=1920, height=1080,
        theme="classic",
        box_size=2.0,
        reversal_boxes=1,
        enable_antialiasing=True,
        show_grid=True
    )

    # Save to fixtures directory for visual verification
    output_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'renko_chart_sample.webp')

    save_chart(img, output_path, speed='fast')
    print(f"\nSample Renko chart saved to: {output_path}")

    assert os.path.exists(output_path)
    assert img.size == (1920, 1080)


def test_renko_chart_performance():
    """
    Test Renko chart rendering performance (basic timing).
    """
    import time

    ohlc = create_trending_data(start=100, end=150, num_points=500)
    volume = np.random.randint(800, 1200, size=500)

    # Time brick calculation
    start_time = time.perf_counter()
    bricks = calculate_renko_bricks(ohlc, box_size=2.0)
    brick_time = (time.perf_counter() - start_time) * 1000

    print(f"\nBrick calculation: {brick_time:.2f}ms ({len(bricks)} bricks)")

    # Time rendering
    start_time = time.perf_counter()
    img = render_renko_chart(ohlc, volume, width=1920, height=1080, box_size=2.0)
    render_time = (time.perf_counter() - start_time) * 1000

    print(f"Renko rendering: {render_time:.2f}ms")

    # Performance targets:
    # - Brick calculation: <5ms for 1000 candles (we're testing 500)
    # - Total rendering: <10ms for typical charts
    assert isinstance(img, Image.Image)


def test_renko_bricks_price_consistency():
    """
    Test that brick prices are consistent with box_size.
    """
    ohlc = create_trending_data(start=100, end=130, num_points=50)
    box_size = 2.0

    bricks = calculate_renko_bricks(ohlc, box_size=box_size, reversal_boxes=1)

    if len(bricks) > 1:
        # Check that consecutive bricks differ by box_size
        for i in range(1, len(bricks)):
            prev_brick = bricks[i-1]
            curr_brick = bricks[i]

            price_diff = abs(curr_brick['price'] - prev_brick['price'])

            # Consecutive bricks should differ by box_size
            assert abs(price_diff - box_size) < 0.01, \
                f"Brick price difference {price_diff} should be close to box_size {box_size}"


def test_renko_chart_dimensions():
    """
    Test various image dimensions.
    """
    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    dimensions = [
        (800, 600),
        (1920, 1080),
        (1200, 800),
        (3840, 2160),  # 4K
    ]

    for width, height in dimensions:
        img = render_renko_chart(
            ohlc, volume,
            width=width, height=height,
            box_size=2.0
        )

        assert img.size == (width, height), f"Image size should be {width}x{height}"


# ============================================================================
# SVG Export Tests
# ============================================================================

@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_basic():
    """
    Test basic Renko chart SVG rendering.
    """
    ohlc = create_trending_data(start=100, end=130, num_points=50)
    volume = np.random.randint(800, 1200, size=50)

    svg_content = render_renko_chart_svg(
        ohlc, volume,
        width=1200, height=800,
        box_size=2.0
    )

    # Verify SVG content
    assert isinstance(svg_content, str)
    assert '<svg' in svg_content
    assert '</svg>' in svg_content
    assert 'id="bricks"' in svg_content
    assert 'id="volume"' in svg_content


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_all_themes():
    """
    Test SVG rendering with all available themes.
    """
    ohlc = create_trending_data(start=100, end=120, num_points=30)
    volume = np.random.randint(800, 1200, size=30)

    for theme_name in THEMES.keys():
        svg_content = render_renko_chart_svg(
            ohlc, volume,
            width=800, height=600,
            theme=theme_name,
            box_size=1.5
        )

        assert isinstance(svg_content, str)
        assert '<svg' in svg_content
        print(f"SVG Theme '{theme_name}': OK")


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_custom_colors():
    """
    Test SVG Renko chart with custom color overrides.
    """
    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    svg_content = render_renko_chart_svg(
        ohlc, volume,
        width=1000, height=700,
        bg_color="#1A1A1A",
        up_color="#00FF00",
        down_color="#FF0000",
        box_size=2.0
    )

    assert isinstance(svg_content, str)
    assert '<svg' in svg_content
    # Check custom colors are in the SVG
    assert '#1A1A1A' in svg_content or '#1a1a1a' in svg_content
    assert '#00FF00' in svg_content or '#00ff00' in svg_content
    assert '#FF0000' in svg_content or '#ff0000' in svg_content


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_no_grid():
    """
    Test SVG Renko chart without grid lines.
    """
    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    svg_content = render_renko_chart_svg(
        ohlc, volume,
        width=800, height=600,
        show_grid=False,
        box_size=2.0
    )

    assert isinstance(svg_content, str)
    assert '<svg' in svg_content
    # Should not have grid group
    assert 'id="grid"' not in svg_content


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_auto_box_size():
    """
    Test SVG Renko chart with automatic box size calculation.
    """
    ohlc = create_trending_data(start=100, end=150, num_points=100)
    volume = np.random.randint(800, 1200, size=100)

    # Render with auto box size (None)
    svg_content = render_renko_chart_svg(
        ohlc, volume,
        width=1200, height=800,
        box_size=None  # Auto-calculate using ATR
    )

    assert isinstance(svg_content, str)
    assert '<svg' in svg_content
    print("\nRendered SVG Renko chart with auto box size")


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_empty_bricks():
    """
    Test SVG rendering when no bricks are generated.
    """
    # Create data with very small movements (no bricks with large box size)
    ohlc = {
        'open': np.array([100.0, 100.1, 100.05, 100.08]),
        'high': np.array([100.15, 100.15, 100.12, 100.12]),
        'low': np.array([99.95, 100.0, 100.0, 100.05]),
        'close': np.array([100.1, 100.05, 100.08, 100.1]),
    }
    volume = np.array([1000, 1100, 1050, 1080])

    # Use very large box size that won't generate any bricks
    svg_content = render_renko_chart_svg(
        ohlc, volume,
        box_size=10.0
    )

    # Should return empty chart SVG (background only)
    assert isinstance(svg_content, str)
    assert '<svg' in svg_content


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_and_save_renko_chart_svg():
    """
    Test rendering and saving Renko chart to SVG file.
    """
    ohlc = create_trending_data(start=100, end=130, num_points=50)
    volume = np.random.randint(800, 1200, size=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "renko_test.svg")

        # Render and save SVG
        svg_content = render_renko_chart_svg(
            ohlc, volume,
            width=1200, height=800,
            box_size=2.0,
            output_path=output_path
        )

        # Verify file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify SVG structure
        assert isinstance(svg_content, str)
        assert '<svg' in svg_content


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_sample_output():
    """
    Generate sample SVG Renko chart for visual inspection.
    """
    # Create realistic trending data with volatility
    np.random.seed(42)  # For reproducibility
    ohlc = create_trending_data(start=100, end=140, num_points=100)
    volume = np.random.randint(800, 1500, size=100)

    # Render Renko chart SVG with classic theme
    svg_content = render_renko_chart_svg(
        ohlc, volume,
        width=1920, height=1080,
        theme="classic",
        box_size=2.0,
        reversal_boxes=1,
        show_grid=True
    )

    # Save to fixtures directory for visual verification
    output_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'renko_chart_sample.svg')

    # Save SVG
    render_renko_chart_svg(
        ohlc, volume,
        width=1920, height=1080,
        theme="classic",
        box_size=2.0,
        reversal_boxes=1,
        show_grid=True,
        output_path=output_path
    )

    print(f"\nSample SVG Renko chart saved to: {output_path}")
    assert os.path.exists(output_path)

    # Verify SVG structure
    assert isinstance(svg_content, str)
    assert '<svg' in svg_content


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_dimensions():
    """
    Test various SVG dimensions.
    """
    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    dimensions = [
        (800, 600),
        (1920, 1080),
        (1200, 800),
        (3840, 2160),  # 4K
    ]

    for width, height in dimensions:
        svg_content = render_renko_chart_svg(
            ohlc, volume,
            width=width, height=height,
            box_size=2.0
        )

        assert isinstance(svg_content, str)
        assert f'width="{width}"' in svg_content or f'width=\"{width}\"' in svg_content
        assert f'height="{height}"' in svg_content or f'height=\"{height}\"' in svg_content


@pytest.mark.skipif(not SVGWRITE_AVAILABLE, reason="svgwrite not installed")
def test_render_renko_chart_svg_no_volume():
    """
    Test SVG Renko chart without volume data.
    """
    ohlc = create_trending_data()

    svg_content = render_renko_chart_svg(
        ohlc, volume=None,
        width=1200, height=800,
        box_size=2.0
    )

    assert isinstance(svg_content, str)
    assert '<svg' in svg_content
    # Should not have volume group when volume is None
    assert 'id="volume"' not in svg_content


def test_render_renko_chart_svg_missing_svgwrite():
    """
    Test that appropriate error is raised when svgwrite is not available.
    """
    if SVGWRITE_AVAILABLE:
        pytest.skip("svgwrite is installed, cannot test missing dependency")

    ohlc = create_trending_data()
    volume = np.random.randint(800, 1200, size=50)

    with pytest.raises(ImportError, match="svgwrite is required"):
        render_renko_chart_svg(ohlc, volume, box_size=2.0)

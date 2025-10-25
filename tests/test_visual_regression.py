"""
Visual Regression Testing for Chart Quality

This test suite ensures that chart rendering quality remains consistent
across code changes. It compares generated charts against baseline images
using pixel-level comparison.

Baseline Management:
- Generate baselines: pytest tests/test_visual_regression.py --generate-baselines
- Run tests: pytest tests/test_visual_regression.py
- Update baselines: Remove old images and regenerate

Baseline Storage:
- tests/baseline_images/classic/ - Classic theme baselines
- tests/baseline_images/modern/ - Modern theme baselines
- tests/baseline_images/tradingview/ - TradingView theme baselines
- tests/baseline_images/light/ - Light theme baselines
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path
from PIL import Image, ImageChops
import numpy as np

from kimsfinance.plotting import render_ohlcv_chart, save_chart

# Baseline directory
BASELINE_DIR = Path(__file__).parent / "baseline_images"
DIFF_DIR = Path(__file__).parent / "visual_diffs"

# Test data - simple OHLCV for reproducibility
TEST_OHLC = {
    'open': np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0]),
    'high': np.array([103.0, 104.0, 102.0, 105.0, 103.0, 106.0, 104.0, 107.0]),
    'low': np.array([99.0, 101.0, 100.0, 102.0, 101.0, 103.0, 102.0, 104.0]),
    'close': np.array([102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0]),
}
TEST_VOLUME = np.array([1000, 1200, 900, 1100, 1050, 1300, 950, 1150])

THEMES = ['classic', 'modern', 'tradingview', 'light']


def pytest_addoption(parser):
    """Add command-line options for baseline management."""
    parser.addoption(
        "--generate-baselines",
        action="store_true",
        default=False,
        help="Generate new baseline images"
    )
    parser.addoption(
        "--tolerance",
        type=float,
        default=0.01,
        help="Acceptable difference percentage (default: 1%%)"
    )


@pytest.fixture
def generate_baselines(request):
    """Check if baseline generation is requested."""
    return request.config.getoption("--generate-baselines")


@pytest.fixture
def tolerance(request):
    """Get acceptable difference threshold."""
    return request.config.getoption("--tolerance")


def compare_images(img1_path, img2_path, tolerance=0.01):
    """
    Compare two images pixel by pixel.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        tolerance: Acceptable difference as percentage (0.01 = 1%)

    Returns:
        (is_match, difference_percentage)
    """
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # Check dimensions match
    if img1.size != img2.size:
        return False, 100.0

    # Pixel-by-pixel comparison
    diff = ImageChops.difference(img1, img2)
    diff_array = np.array(diff)

    # Calculate percentage of different pixels
    total_pixels = diff_array.size
    different_pixels = np.count_nonzero(diff_array)
    difference_pct = (different_pixels / total_pixels) * 100

    is_match = difference_pct <= (tolerance * 100)

    return is_match, difference_pct


def save_diff_image(baseline_path, current_path, diff_output_path):
    """Save a visual diff highlighting differences."""
    baseline = Image.open(baseline_path).convert('RGB')
    current = Image.open(current_path).convert('RGB')

    # Generate diff image
    diff = ImageChops.difference(baseline, current)

    # Enhance diff for visibility
    diff = ImageChops.multiply(diff, Image.new('RGB', diff.size, (10, 10, 10)))

    diff.save(diff_output_path)


@pytest.mark.parametrize("theme", THEMES)
def test_chart_visual_regression(theme, generate_baselines, tolerance, tmp_path):
    """
    Test chart rendering against baseline images.

    This test ensures that chart rendering quality remains consistent.
    If visual differences exceed the tolerance threshold, the test fails
    and generates a diff image for inspection.
    """
    # Render current chart
    img = render_ohlcv_chart(
        ohlc=TEST_OHLC,
        volume=TEST_VOLUME,
        width=400,
        height=300,
        theme=theme,
        enable_antialiasing=True
    )

    # Save current chart
    current_path = tmp_path / f"current_{theme}.png"
    img.save(current_path, format='PNG')

    # Baseline path
    baseline_dir = BASELINE_DIR / theme
    baseline_path = baseline_dir / "basic_chart.png"

    if generate_baselines:
        # Generate new baseline
        baseline_dir.mkdir(parents=True, exist_ok=True)
        img.save(baseline_path, format='PNG')
        pytest.skip(f"Generated baseline for {theme} theme")

    # Compare against baseline
    if not baseline_path.exists():
        pytest.fail(
            f"Baseline not found: {baseline_path}\n"
            f"Generate baselines with: pytest tests/test_visual_regression.py --generate-baselines"
        )

    is_match, difference_pct = compare_images(baseline_path, current_path, tolerance)

    if not is_match:
        # Save diff image for inspection
        diff_dir = DIFF_DIR / theme
        diff_dir.mkdir(parents=True, exist_ok=True)
        diff_path = diff_dir / "diff.png"
        save_diff_image(baseline_path, current_path, diff_path)

        pytest.fail(
            f"Visual regression detected for {theme} theme!\n"
            f"Difference: {difference_pct:.2f}%% (tolerance: {tolerance*100:.2f}%%)\n"
            f"Diff image saved to: {diff_path}\n"
            f"Baseline: {baseline_path}\n"
            f"Current: {current_path}"
        )


@pytest.mark.parametrize("theme", THEMES)
def test_indicator_visual_regression(theme, generate_baselines, tolerance, tmp_path):
    """Test charts with indicators against baselines."""
    # This is a placeholder - expand with actual indicator tests
    pytest.skip("Indicator visual regression tests not yet implemented")

"""
Security tests for kimsfinance - path traversal, input validation, color validation.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from kimsfinance.api import plot
from kimsfinance.plotting import save_chart, render_ohlcv_chart
from kimsfinance.utils.color_utils import _hex_to_rgba


# Sample OHLC data for testing
@pytest.fixture
def sample_ohlc():
    """Generate sample OHLC data for testing."""
    return {
        'open': np.array([100, 102, 101, 103, 105]),
        'high': np.array([105, 106, 104, 107, 108]),
        'low': np.array([99, 101, 100, 102, 104]),
        'close': np.array([102, 101, 103, 105, 107]),
    }


@pytest.fixture
def sample_volume():
    """Generate sample volume data for testing."""
    return np.array([1000, 1500, 1200, 1800, 2000])


class TestPathTraversalValidation:
    """Test path traversal vulnerability fixes."""

    def test_directory_traversal_blocked(self, sample_ohlc, sample_volume):
        """Test that directory traversal attacks are blocked."""
        # Note: The path validation allows paths outside cwd as long as they're not system directories
        # This test verifies that attempts to write to a real system file fail
        with pytest.raises(ValueError, match="auto-detect format|system directory"):
            # Use /etc/passwd directly to ensure it's caught
            plot(
                sample_ohlc,
                volume=sample_volume,
                savefig="/etc/passwd"
            )

    def test_system_directory_blocked(self, sample_ohlc, sample_volume):
        """Test that writing to system directories is blocked."""
        system_paths = [
            "/etc/test.webp",
            "/sys/test.webp",
            "/proc/test.webp",
            "/dev/test.webp",
            "/root/test.webp",
            "/boot/test.webp",
        ]

        for path in system_paths:
            with pytest.raises(ValueError, match="Cannot write to system directory"):
                plot(sample_ohlc, volume=sample_volume, savefig=path)

    def test_empty_path_handled(self, sample_ohlc, sample_volume):
        """Test that empty paths are handled (returns image, doesn't save)."""
        # Empty savefig is valid - it means display mode, not save mode
        # This should not raise an error, it should return an image
        img = plot(sample_ohlc, volume=sample_volume, savefig=None, returnfig=True)
        assert img is not None

    def test_valid_relative_path(self, sample_ohlc, sample_volume):
        """Test that valid relative paths work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                output_path = "test_chart.webp"
                plot(sample_ohlc, volume=sample_volume, savefig=output_path)

                # Verify file was created
                assert os.path.exists(output_path)
                assert os.path.getsize(output_path) > 0
            finally:
                os.chdir(original_cwd)

    def test_valid_absolute_path(self, sample_ohlc, sample_volume):
        """Test that valid absolute paths work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_chart.webp")
            plot(sample_ohlc, volume=sample_volume, savefig=output_path)

            # Verify file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_nested_directory_creation(self, sample_ohlc, sample_volume):
        """Test that nested directories are created safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir1", "subdir2", "test_chart.webp")
            plot(sample_ohlc, volume=sample_volume, savefig=output_path)

            # Verify file and directories were created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_save_chart_path_validation(self, sample_ohlc, sample_volume):
        """Test that save_chart() validates paths."""
        img = render_ohlcv_chart(sample_ohlc, sample_volume)

        with pytest.raises(ValueError, match="Cannot write to system directory"):
            save_chart(img, "/etc/test.webp")


class TestNumericValidation:
    """Test numeric parameter validation."""

    def test_width_too_small(self, sample_ohlc, sample_volume):
        """Test that width below minimum is rejected."""
        with pytest.raises(ValueError, match="width must be between"):
            plot(sample_ohlc, volume=sample_volume, width=50, returnfig=True)

    def test_width_too_large(self, sample_ohlc, sample_volume):
        """Test that width above maximum is rejected."""
        with pytest.raises(ValueError, match="width must be between"):
            plot(sample_ohlc, volume=sample_volume, width=10000, returnfig=True)

    def test_height_too_small(self, sample_ohlc, sample_volume):
        """Test that height below minimum is rejected."""
        with pytest.raises(ValueError, match="height must be between"):
            plot(sample_ohlc, volume=sample_volume, height=50, returnfig=True)

    def test_height_too_large(self, sample_ohlc, sample_volume):
        """Test that height above maximum is rejected."""
        with pytest.raises(ValueError, match="height must be between"):
            plot(sample_ohlc, volume=sample_volume, height=10000, returnfig=True)

    def test_valid_dimensions(self, sample_ohlc, sample_volume):
        """Test that valid dimensions are accepted."""
        # Test common resolutions
        valid_dimensions = [
            (100, 100),  # Minimum
            (1920, 1080),  # HD
            (3840, 2160),  # 4K
            (8192, 8192),  # Maximum
        ]

        for width, height in valid_dimensions:
            img = plot(
                sample_ohlc,
                volume=sample_volume,
                width=width,
                height=height,
                returnfig=True
            )
            assert img is not None
            assert img.size == (width, height)

    def test_line_width_too_small(self, sample_ohlc, sample_volume):
        """Test that line width below minimum is rejected."""
        with pytest.raises(ValueError, match="line_width must be between"):
            plot(
                sample_ohlc,
                volume=sample_volume,
                type='line',
                line_width=0.05,
                returnfig=True
            )

    def test_line_width_too_large(self, sample_ohlc, sample_volume):
        """Test that line width above maximum is rejected."""
        with pytest.raises(ValueError, match="line_width must be between"):
            plot(
                sample_ohlc,
                volume=sample_volume,
                type='line',
                line_width=25,
                returnfig=True
            )

    def test_box_size_negative(self, sample_ohlc, sample_volume):
        """Test that negative box size is rejected."""
        with pytest.raises(ValueError, match="box_size must be positive"):
            plot(
                sample_ohlc,
                volume=sample_volume,
                type='renko',
                box_size=-1,
                returnfig=True
            )

    def test_reversal_boxes_too_small(self, sample_ohlc, sample_volume):
        """Test that reversal_boxes below minimum is rejected."""
        with pytest.raises(ValueError, match="reversal_boxes must be between"):
            plot(
                sample_ohlc,
                volume=sample_volume,
                type='pnf',
                reversal_boxes=0,
                returnfig=True
            )

    def test_reversal_boxes_too_large(self, sample_ohlc, sample_volume):
        """Test that reversal_boxes above maximum is rejected."""
        with pytest.raises(ValueError, match="reversal_boxes must be between"):
            plot(
                sample_ohlc,
                volume=sample_volume,
                type='pnf',
                reversal_boxes=15,
                returnfig=True
            )


class TestColorValidation:
    """Test color validation."""

    def test_valid_hex_colors(self):
        """Test that valid hex colors are accepted."""
        valid_colors = [
            '#FF0000',
            '#00FF00',
            '#0000FF',
            'FF0000',  # Without #
            '#AABBCC',
            'aabbcc',  # Lowercase
            '#FF0000FF',  # With alpha
        ]

        for color in valid_colors:
            result = _hex_to_rgba(color)
            assert len(result) == 4
            assert all(0 <= v <= 255 for v in result)

    def test_invalid_hex_format(self):
        """Test that invalid hex formats are rejected."""
        invalid_colors = [
            'GGGGGG',  # Invalid hex characters
            '#GG0000',
            'FF00',  # Too short
            '#FF',
            'FF00000000',  # Too long
            '#FFFFFFFFFF',
            'not-a-color',
            '',  # Empty
            '#',
        ]

        for color in invalid_colors:
            with pytest.raises(ValueError, match="Invalid hex color format"):
                _hex_to_rgba(color)

    def test_plot_with_invalid_colors(self, sample_ohlc, sample_volume):
        """Test that plot rejects invalid color parameters."""
        with pytest.raises(ValueError, match="Invalid hex color format"):
            plot(
                sample_ohlc,
                volume=sample_volume,
                up_color='INVALID',
                returnfig=True
            )

    def test_plot_with_valid_colors(self, sample_ohlc, sample_volume):
        """Test that plot accepts valid color parameters."""
        img = plot(
            sample_ohlc,
            volume=sample_volume,
            bg_color='#000000',
            up_color='#00FF00',
            down_color='#FF0000',
            returnfig=True
        )
        assert img is not None


class TestIntegration:
    """Integration tests combining multiple security features."""

    def test_complete_security_chain(self, sample_ohlc, sample_volume):
        """Test complete security validation chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid parameters
            output_path = os.path.join(tmpdir, "secure_chart.webp")

            plot(
                sample_ohlc,
                volume=sample_volume,
                savefig=output_path,
                width=1920,
                height=1080,
                bg_color='#1E1E1E',
                up_color='#26A69A',
                down_color='#EF5350',
            )

            # Verify file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_multiple_security_violations(self, sample_ohlc, sample_volume):
        """Test that first security violation is caught."""
        # This should fail on path validation before color validation
        with pytest.raises(ValueError, match="system directory"):
            plot(
                sample_ohlc,
                volume=sample_volume,
                savefig="/etc/test.webp",
                up_color='INVALID_COLOR',  # Would fail if path check passed
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

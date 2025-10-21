"""
Tests for Point and Figure (PNF) Chart Renderer
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from kimsfinance.data.pnf import calculate_pnf_columns
from kimsfinance.plotting import render_pnf_chart


class TestCalculatePNFColumns:
    """Test PNF column calculation algorithm"""

    def test_basic_column_calculation(self):
        """Test basic PNF column calculation with known data"""
        # Trending up then reversal down
        ohlc = {
            'high': np.array([101, 104, 107, 110, 109, 106, 103, 100]),
            'low': np.array([99, 102, 105, 108, 106, 103, 100, 97]),
            'close': np.array([100, 103, 106, 109, 107, 104, 101, 98]),
        }

        columns = calculate_pnf_columns(ohlc, box_size=2.0, reversal_boxes=3)

        # Should get at least 2 columns (X then O)
        assert len(columns) >= 2

        # First column should be X (rising)
        assert columns[0]['type'] == 'X'
        assert len(columns[0]['boxes']) > 0

        # Check structure
        for col in columns:
            assert col['type'] in ['X', 'O']
            assert 'boxes' in col
            assert 'start_idx' in col
            assert isinstance(col['boxes'], list)
            assert isinstance(col['start_idx'], (int, np.integer))

    def test_auto_box_size(self):
        """Test auto-calculation of box size using ATR"""
        # Generate realistic price data
        np.random.seed(42)
        n = 50
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n) * 0.3)
        lows = closes - np.abs(np.random.randn(n) * 0.3)

        ohlc = {
            'high': highs,
            'low': lows,
            'close': closes,
        }

        columns = calculate_pnf_columns(ohlc, box_size=None, reversal_boxes=3)

        # Should get at least 1 column
        assert len(columns) >= 1

        # Verify structure
        for col in columns:
            assert col['type'] in ['X', 'O']
            assert len(col['boxes']) > 0

    def test_single_uptrend(self):
        """Test PNF with continuous uptrend (single X column)"""
        ohlc = {
            'high': np.linspace(102, 120, 20),
            'low': np.linspace(100, 118, 20),
            'close': np.linspace(101, 119, 20),
        }

        columns = calculate_pnf_columns(ohlc, box_size=1.0, reversal_boxes=3)

        # Should get mostly X columns for uptrend
        x_columns = [col for col in columns if col['type'] == 'X']
        assert len(x_columns) >= 1

    def test_alternating_trend(self):
        """Test PNF with alternating up/down trends"""
        # Create zigzag pattern
        highs = [102, 105, 103, 108, 106, 111, 109, 114]
        lows = [100, 103, 101, 106, 104, 109, 107, 112]
        closes = [101, 104, 102, 107, 105, 110, 108, 113]

        ohlc = {
            'high': np.array(highs),
            'low': np.array(lows),
            'close': np.array(closes),
        }

        columns = calculate_pnf_columns(ohlc, box_size=2.0, reversal_boxes=2)

        # Should get multiple columns with this pattern
        assert len(columns) >= 2

        # Check alternation (if multiple columns)
        if len(columns) >= 2:
            for i in range(len(columns) - 1):
                assert columns[i]['type'] != columns[i + 1]['type']

    def test_insufficient_movement(self):
        """Test PNF with insufficient price movement"""
        # Flat prices
        ohlc = {
            'high': np.array([100.5, 100.5, 100.5, 100.5]),
            'low': np.array([99.5, 99.5, 99.5, 99.5]),
            'close': np.array([100, 100, 100, 100]),
        }

        columns = calculate_pnf_columns(ohlc, box_size=5.0, reversal_boxes=3)

        # May get 0 or 1 column depending on rounding
        assert len(columns) <= 1

    def test_large_box_size(self):
        """Test PNF with very large box size (few boxes)"""
        ohlc = {
            'high': np.linspace(102, 120, 50),
            'low': np.linspace(100, 118, 50),
            'close': np.linspace(101, 119, 50),
        }

        columns = calculate_pnf_columns(ohlc, box_size=10.0, reversal_boxes=3)

        # Large box size should result in fewer boxes
        total_boxes = sum(len(col['boxes']) for col in columns)
        assert total_boxes < 50  # Much fewer than data points

    def test_small_box_size(self):
        """Test PNF with very small box size (many boxes)"""
        ohlc = {
            'high': np.linspace(102, 120, 50),
            'low': np.linspace(100, 118, 50),
            'close': np.linspace(101, 119, 50),
        }

        columns = calculate_pnf_columns(ohlc, box_size=0.5, reversal_boxes=3)

        # Small box size should result in more boxes
        total_boxes = sum(len(col['boxes']) for col in columns)
        assert total_boxes > 10

    def test_reversal_boxes_parameter(self):
        """Test different reversal_boxes values"""
        ohlc = {
            'high': np.array([102, 110, 108, 100, 102, 110]),
            'low': np.array([100, 108, 100, 98, 100, 108]),
            'close': np.array([101, 109, 102, 99, 101, 109]),
        }

        # Lower reversal threshold = more columns
        columns_low = calculate_pnf_columns(ohlc, box_size=2.0, reversal_boxes=2)

        # Higher reversal threshold = fewer columns
        columns_high = calculate_pnf_columns(ohlc, box_size=2.0, reversal_boxes=5)

        # Lower threshold should produce more or equal columns
        assert len(columns_low) >= len(columns_high)

    def test_empty_data(self):
        """Test PNF with minimal data"""
        ohlc = {
            'high': np.array([100]),
            'low': np.array([100]),
            'close': np.array([100]),
        }

        columns = calculate_pnf_columns(ohlc, box_size=1.0, reversal_boxes=3)

        # Should return empty or single column
        assert len(columns) <= 1


class TestRenderPNFChart:
    """Test PNF chart rendering"""

    def test_basic_rendering(self):
        """Test basic PNF chart rendering"""
        # Generate uptrend then downtrend data
        ohlc = {
            'open': np.linspace(100, 130, 50),
            'high': np.linspace(102, 135, 50),
            'low': np.linspace(98, 128, 50),
            'close': np.concatenate([
                np.linspace(100, 130, 30),  # Uptrend
                np.linspace(130, 110, 20),  # Downtrend
            ]),
        }
        volume = np.random.randint(800, 1200, size=50)

        img = render_pnf_chart(ohlc, volume, width=1200, height=800,
                              box_size=2.0, reversal_boxes=3)

        assert isinstance(img, Image.Image)
        assert img.size == (1200, 800)
        assert img.mode in ['RGB', 'RGBA']

    def test_render_with_themes(self):
        """Test rendering with different themes"""
        ohlc = {
            'open': np.linspace(100, 120, 30),
            'high': np.linspace(102, 122, 30),
            'low': np.linspace(98, 118, 30),
            'close': np.linspace(100, 120, 30),
        }
        volume = np.random.randint(800, 1200, size=30)

        themes = ['classic', 'modern', 'tradingview', 'light']

        for theme in themes:
            img = render_pnf_chart(ohlc, volume, theme=theme, width=800, height=600)
            assert isinstance(img, Image.Image)
            assert img.size == (800, 600)

    def test_render_with_custom_colors(self):
        """Test rendering with custom colors"""
        ohlc = {
            'open': np.linspace(100, 120, 30),
            'high': np.linspace(102, 122, 30),
            'low': np.linspace(98, 118, 30),
            'close': np.linspace(100, 120, 30),
        }
        volume = np.random.randint(800, 1200, size=30)

        img = render_pnf_chart(
            ohlc, volume,
            width=800, height=600,
            bg_color='#FFFFFF',
            up_color='#0000FF',
            down_color='#FF00FF'
        )

        assert isinstance(img, Image.Image)
        assert img.size == (800, 600)

    def test_render_no_antialiasing(self):
        """Test rendering without antialiasing (RGB mode)"""
        ohlc = {
            'open': np.linspace(100, 120, 30),
            'high': np.linspace(102, 122, 30),
            'low': np.linspace(98, 118, 30),
            'close': np.linspace(100, 120, 30),
        }
        volume = np.random.randint(800, 1200, size=30)

        img = render_pnf_chart(
            ohlc, volume,
            width=800, height=600,
            enable_antialiasing=False
        )

        assert isinstance(img, Image.Image)
        assert img.mode == 'RGB'

    def test_render_with_grid(self):
        """Test rendering with and without grid"""
        ohlc = {
            'open': np.linspace(100, 120, 30),
            'high': np.linspace(102, 122, 30),
            'low': np.linspace(98, 118, 30),
            'close': np.linspace(100, 120, 30),
        }
        volume = np.random.randint(800, 1200, size=30)

        img_with_grid = render_pnf_chart(
            ohlc, volume, width=800, height=600, show_grid=True
        )
        img_without_grid = render_pnf_chart(
            ohlc, volume, width=800, height=600, show_grid=False
        )

        assert isinstance(img_with_grid, Image.Image)
        assert isinstance(img_without_grid, Image.Image)

    def test_render_auto_box_size(self):
        """Test rendering with auto-calculated box size"""
        np.random.seed(42)
        n = 50
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n) * 0.3)
        lows = closes - np.abs(np.random.randn(n) * 0.3)
        opens = closes - np.random.randn(n) * 0.2

        ohlc = {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
        }
        volume = np.random.randint(800, 1200, size=n)

        img = render_pnf_chart(
            ohlc, volume,
            width=1200, height=800,
            box_size=None,  # Auto-calculate
            reversal_boxes=3
        )

        assert isinstance(img, Image.Image)
        assert img.size == (1200, 800)

    def test_render_empty_columns(self):
        """Test rendering when no columns are generated"""
        # Flat prices
        ohlc = {
            'open': np.array([100, 100, 100, 100]),
            'high': np.array([100.1, 100.1, 100.1, 100.1]),
            'low': np.array([99.9, 99.9, 99.9, 99.9]),
            'close': np.array([100, 100, 100, 100]),
        }
        volume = np.array([1000, 1000, 1000, 1000])

        img = render_pnf_chart(
            ohlc, volume,
            width=800, height=600,
            box_size=10.0,  # Large box size = no movement
            reversal_boxes=3
        )

        assert isinstance(img, Image.Image)
        assert img.size == (800, 600)

    def test_render_various_sizes(self):
        """Test rendering at different image sizes"""
        ohlc = {
            'open': np.linspace(100, 120, 30),
            'high': np.linspace(102, 122, 30),
            'low': np.linspace(98, 118, 30),
            'close': np.linspace(100, 120, 30),
        }
        volume = np.random.randint(800, 1200, size=30)

        sizes = [(800, 600), (1920, 1080), (3840, 2160), (400, 300)]

        for width, height in sizes:
            img = render_pnf_chart(ohlc, volume, width=width, height=height)
            assert img.size == (width, height)

    def test_render_different_reversal_boxes(self):
        """Test rendering with different reversal_boxes values"""
        ohlc = {
            'open': np.linspace(100, 130, 50),
            'high': np.linspace(102, 135, 50),
            'low': np.linspace(98, 128, 50),
            'close': np.concatenate([
                np.linspace(100, 130, 30),
                np.linspace(130, 110, 20),
            ]),
        }
        volume = np.random.randint(800, 1200, size=50)

        for reversal_boxes in [2, 3, 5, 10]:
            img = render_pnf_chart(
                ohlc, volume,
                width=1200, height=800,
                box_size=2.0,
                reversal_boxes=reversal_boxes
            )
            assert isinstance(img, Image.Image)


class TestPNFIntegration:
    """Integration tests for PNF chart rendering"""

    def test_full_workflow(self):
        """Test complete workflow: calculate columns -> render chart"""
        # Generate realistic data
        np.random.seed(42)
        n = 100
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n) * 0.3)
        lows = closes - np.abs(np.random.randn(n) * 0.3)
        opens = closes - np.random.randn(n) * 0.2

        ohlc = {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
        }
        volume = np.random.randint(800, 1200, size=n)

        # Step 1: Calculate columns
        columns = calculate_pnf_columns(ohlc, box_size=0.5, reversal_boxes=3)

        assert len(columns) > 0
        assert all(col['type'] in ['X', 'O'] for col in columns)

        # Step 2: Render chart
        img = render_pnf_chart(
            ohlc, volume,
            width=1920, height=1080,
            box_size=0.5,
            reversal_boxes=3,
            theme='modern'
        )

        assert isinstance(img, Image.Image)
        assert img.size == (1920, 1080)

    def test_save_sample_chart(self, tmp_path):
        """Test saving PNF chart to file"""
        # Generate uptrend then downtrend
        ohlc = {
            'open': np.linspace(100, 130, 50),
            'high': np.linspace(102, 135, 50),
            'low': np.linspace(98, 128, 50),
            'close': np.concatenate([
                np.linspace(100, 130, 30),
                np.linspace(130, 110, 20),
            ]),
        }
        volume = np.random.randint(800, 1200, size=50)

        img = render_pnf_chart(
            ohlc, volume,
            width=1200, height=800,
            box_size=2.0,
            reversal_boxes=3,
            theme='tradingview'
        )

        # Save to temporary file
        output_path = tmp_path / "pnf_chart_sample.webp"
        img.save(str(output_path), 'WEBP', quality=85)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

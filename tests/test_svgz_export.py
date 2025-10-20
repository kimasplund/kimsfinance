"""
Tests for SVGZ (gzipped SVG) export functionality.

Tests verify that all chart types can be exported as compressed SVGZ format,
achieving 75-85% file size reduction while maintaining vector graphics quality.
"""

import gzip
import os
import tempfile
import pytest
import polars as pl
import numpy as np

from kimsfinance.api import plot


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pl.date_range(
        start=pl.date(2024, 1, 1),
        end=pl.date(2024, 4, 10),
        interval="1d",
        eager=True
    )

    n = len(dates)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    open_prices = close + np.random.randn(n) * 0.5
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n) * 1.0)
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n) * 1.0)
    volume = np.random.randint(1000, 10000, n)

    return pl.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })


class TestSVGZExport:
    """Test SVGZ export for all chart types."""

    def test_candlestick_svgz(self, sample_ohlcv_data):
        """Test candlestick chart SVGZ export."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(sample_ohlcv_data, type='candle', savefig=svgz_path)

            # Verify file exists
            assert os.path.exists(svgz_path)

            # Verify it's gzipped
            with open(svgz_path, 'rb') as f:
                data = f.read()
                # Gzip magic number
                assert data[:2] == b'\x1f\x8b'

            # Verify it decompresses to valid SVG
            decompressed = gzip.decompress(data)
            svg_text = decompressed.decode('utf-8')
            assert svg_text.startswith('<?xml') or svg_text.startswith('<svg')
            assert '<svg' in svg_text
            assert '</svg>' in svg_text

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)

    def test_ohlc_bars_svgz(self, sample_ohlcv_data):
        """Test OHLC bars chart SVGZ export."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(sample_ohlcv_data, type='ohlc', savefig=svgz_path)

            assert os.path.exists(svgz_path)

            with open(svgz_path, 'rb') as f:
                data = f.read()
                assert data[:2] == b'\x1f\x8b'

            decompressed = gzip.decompress(data)
            svg_text = decompressed.decode('utf-8')
            assert '<svg' in svg_text

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)

    def test_line_chart_svgz(self, sample_ohlcv_data):
        """Test line chart SVGZ export."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(sample_ohlcv_data, type='line', savefig=svgz_path)

            assert os.path.exists(svgz_path)

            with open(svgz_path, 'rb') as f:
                data = f.read()
                assert data[:2] == b'\x1f\x8b'

            decompressed = gzip.decompress(data)
            svg_text = decompressed.decode('utf-8')
            assert '<svg' in svg_text
            assert 'polyline' in svg_text.lower() or 'path' in svg_text.lower()

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)

    def test_hollow_candles_svgz(self, sample_ohlcv_data):
        """Test hollow candles chart SVGZ export."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(sample_ohlcv_data, type='hollow_and_filled', savefig=svgz_path)

            assert os.path.exists(svgz_path)

            with open(svgz_path, 'rb') as f:
                data = f.read()
                assert data[:2] == b'\x1f\x8b'

            decompressed = gzip.decompress(data)
            svg_text = decompressed.decode('utf-8')
            assert '<svg' in svg_text

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)

    def test_renko_chart_svgz(self, sample_ohlcv_data):
        """Test Renko chart SVGZ export."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(sample_ohlcv_data, type='renko', savefig=svgz_path)

            assert os.path.exists(svgz_path)

            with open(svgz_path, 'rb') as f:
                data = f.read()
                assert data[:2] == b'\x1f\x8b'

            decompressed = gzip.decompress(data)
            svg_text = decompressed.decode('utf-8')
            assert '<svg' in svg_text

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)

    def test_pnf_chart_svgz(self, sample_ohlcv_data):
        """Test Point & Figure chart SVGZ export."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(sample_ohlcv_data, type='pnf', savefig=svgz_path)

            assert os.path.exists(svgz_path)

            with open(svgz_path, 'rb') as f:
                data = f.read()
                assert data[:2] == b'\x1f\x8b'

            decompressed = gzip.decompress(data)
            svg_text = decompressed.decode('utf-8')
            assert '<svg' in svg_text

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)


class TestSVGZCompression:
    """Test SVGZ compression ratios."""

    def test_compression_ratio(self, sample_ohlcv_data):
        """Verify SVGZ achieves 70%+ compression ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = os.path.join(tmpdir, 'chart.svg')
            svgz_path = os.path.join(tmpdir, 'chart.svgz')

            plot(sample_ohlcv_data, type='candle', savefig=svg_path)
            plot(sample_ohlcv_data, type='candle', savefig=svgz_path)

            svg_size = os.path.getsize(svg_path)
            svgz_size = os.path.getsize(svgz_path)

            compression_ratio = 100 * (1 - svgz_size / svg_size)

            # Verify at least 70% compression
            assert compression_ratio >= 70.0, f"Compression ratio {compression_ratio:.1f}% is below 70%"

            # Verify SVGZ is smaller
            assert svgz_size < svg_size

    def test_all_chart_types_compression(self, sample_ohlcv_data):
        """Verify all chart types achieve good compression."""
        chart_types = ['candle', 'ohlc', 'line', 'hollow_and_filled', 'renko', 'pnf']

        with tempfile.TemporaryDirectory() as tmpdir:
            for chart_type in chart_types:
                svg_path = os.path.join(tmpdir, f'{chart_type}.svg')
                svgz_path = os.path.join(tmpdir, f'{chart_type}.svgz')

                plot(sample_ohlcv_data, type=chart_type, savefig=svg_path)
                plot(sample_ohlcv_data, type=chart_type, savefig=svgz_path)

                svg_size = os.path.getsize(svg_path)
                svgz_size = os.path.getsize(svgz_path)

                compression_ratio = 100 * (1 - svgz_size / svg_size)

                # Each chart type should achieve at least 60% compression
                assert compression_ratio >= 60.0, \
                    f"{chart_type} compression {compression_ratio:.1f}% is below 60%"


class TestSVGZEquivalence:
    """Test that SVGZ decompresses to identical SVG."""

    def test_svgz_decompresses_to_svg(self, sample_ohlcv_data):
        """Verify SVGZ decompresses to equivalent SVG content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = os.path.join(tmpdir, 'chart.svg')
            svgz_path = os.path.join(tmpdir, 'chart.svgz')

            plot(sample_ohlcv_data, type='candle', savefig=svg_path)
            plot(sample_ohlcv_data, type='candle', savefig=svgz_path)

            # Read SVG
            with open(svg_path, 'r') as f:
                svg_content = f.read()

            # Read and decompress SVGZ
            with open(svgz_path, 'rb') as f:
                svgz_content = gzip.decompress(f.read()).decode('utf-8')

            # They should be identical (or at least contain same elements)
            # Note: svgwrite may format slightly differently, so check key elements
            assert svg_content.count('<rect') == svgz_content.count('<rect')
            assert svg_content.count('<line') == svgz_content.count('<line')
            assert svg_content.count('<g ') == svgz_content.count('<g ')


class TestSVGZWithOptions:
    """Test SVGZ with various rendering options."""

    def test_svgz_with_themes(self, sample_ohlcv_data):
        """Test SVGZ export with different themes."""
        themes = ['classic', 'modern', 'tradingview', 'light']

        with tempfile.TemporaryDirectory() as tmpdir:
            for theme in themes:
                svgz_path = os.path.join(tmpdir, f'chart_{theme}.svgz')
                plot(sample_ohlcv_data, type='candle', theme=theme, savefig=svgz_path)

                assert os.path.exists(svgz_path)

                # Verify it's valid gzipped SVG
                with open(svgz_path, 'rb') as f:
                    data = f.read()
                    assert data[:2] == b'\x1f\x8b'
                    decompressed = gzip.decompress(data).decode('utf-8')
                    assert '<svg' in decompressed

    def test_svgz_with_volume(self, sample_ohlcv_data):
        """Test SVGZ export with volume panel."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(sample_ohlcv_data, type='candle', volume=True, savefig=svgz_path)

            assert os.path.exists(svgz_path)

            with open(svgz_path, 'rb') as f:
                data = f.read()
                decompressed = gzip.decompress(data).decode('utf-8')
                # Volume panel should be present
                assert 'volume' in decompressed.lower() or 'opacity' in decompressed

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)

    def test_svgz_with_custom_colors(self, sample_ohlcv_data):
        """Test SVGZ export with custom colors."""
        with tempfile.NamedTemporaryFile(suffix='.svgz', delete=False) as f:
            svgz_path = f.name

        try:
            plot(
                sample_ohlcv_data,
                type='candle',
                savefig=svgz_path,
                bg_color='#FFFFFF',
                up_color='#00AA00',
                down_color='#AA0000'
            )

            assert os.path.exists(svgz_path)

            with open(svgz_path, 'rb') as f:
                data = f.read()
                decompressed = gzip.decompress(data).decode('utf-8')
                # Custom colors should be present
                assert '#FFFFFF' in decompressed or '#ffffff' in decompressed

        finally:
            if os.path.exists(svgz_path):
                os.unlink(svgz_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

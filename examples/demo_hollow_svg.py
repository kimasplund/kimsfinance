#!/usr/bin/env python3
"""
Demo script for hollow candlestick SVG export.

This demonstrates the new SVG export capability for hollow candlestick charts.
"""

import numpy as np
import polars as pl
from kimsfinance.api import plot


def main():
    """Generate sample hollow candle SVG charts."""

    # Create realistic sample data with mix of bullish/bearish candles
    np.random.seed(42)
    n_candles = 50

    # Generate random walk price data
    close_prices = np.cumsum(np.random.randn(n_candles) * 2) + 100

    # Generate OHLC
    open_prices = close_prices + np.random.randn(n_candles) * 0.5
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n_candles)) * 2
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n_candles)) * 2
    volume_data = np.abs(np.random.randn(n_candles)) * 1000 + 5000

    df = pl.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume_data,
        }
    )

    print("Generating hollow candles SVG charts...")

    # 1. Classic theme with volume
    print("  1. Classic theme (50 candles with volume)...")
    plot(
        df,
        type="hollow_and_filled",
        volume=True,
        theme="classic",
        savefig="demo_output/hollow_candles_classic.svg",
        width=1920,
        height=1080,
    )

    # 2. TradingView theme
    print("  2. TradingView theme...")
    plot(
        df,
        type="hollow",
        volume=True,
        theme="tradingview",
        savefig="demo_output/hollow_candles_tradingview.svg",
        width=1920,
        height=1080,
    )

    # 3. Light theme
    print("  3. Light theme...")
    plot(
        df,
        type="hollow_and_filled",
        volume=True,
        theme="light",
        savefig="demo_output/hollow_candles_light.svg",
        width=1920,
        height=1080,
    )

    # 4. Custom colors
    print("  4. Custom colors...")
    plot(
        df,
        type="hollow",
        volume=True,
        bg_color="#1a1a2e",
        up_color="#00ff88",
        down_color="#ff0066",
        savefig="demo_output/hollow_candles_custom.svg",
        width=1920,
        height=1080,
    )

    # 5. Small subset (10 candles) for detail inspection
    print("  5. Small chart (10 candles) for detail...")
    plot(
        df[:10],
        type="hollow_and_filled",
        volume=True,
        theme="modern",
        savefig="demo_output/hollow_candles_small.svg",
        width=800,
        height=600,
    )

    print("\nDone! Generated 5 SVG charts in demo_output/")
    print("\nSVG files can be:")
    print("  - Opened in any web browser")
    print("  - Edited in Inkscape or Adobe Illustrator")
    print("  - Scaled to any size without quality loss")
    print("  - Embedded in web pages or documents")


if __name__ == "__main__":
    import os

    os.makedirs("demo_output", exist_ok=True)
    main()

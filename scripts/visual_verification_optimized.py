#!/usr/bin/env python3
"""
Visual verification script for optimized renderers.

Generates sample outputs to ensure no visual regressions after optimization.
"""

import numpy as np
from kimsfinance.plotting.pil_renderer import render_ohlc_bars, render_hollow_candles, save_chart
from kimsfinance.plotting.svg_renderer import render_candlestick_svg, render_hollow_candles_svg


def generate_test_data(num_candles: int = 100):
    """Generate realistic OHLCV test data."""
    np.random.seed(42)
    base_price = 100.0

    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, num_candles)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, num_candles)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, num_candles)))

    # Generate volume
    volume = np.random.randint(1000, 10000, num_candles)

    ohlc = {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
    }

    return ohlc, volume


def main():
    print("=" * 70)
    print("Visual Verification for Optimized Renderers")
    print("=" * 70)

    # Generate test data
    ohlc, volume = generate_test_data(100)

    # Test PIL OHLC bars
    print("\n1. Rendering PIL OHLC bars...")
    img = render_ohlc_bars(
        ohlc=ohlc,
        volume=volume,
        width=1920,
        height=1080,
        theme='classic',
    )
    save_chart(img, '/tmp/test_ohlc_bars_optimized.webp', speed='fast')
    print("   ✓ Saved to /tmp/test_ohlc_bars_optimized.webp")

    # Test PIL hollow candles
    print("\n2. Rendering PIL hollow candles...")
    img = render_hollow_candles(
        ohlc=ohlc,
        volume=volume,
        width=1920,
        height=1080,
        theme='classic',
    )
    save_chart(img, '/tmp/test_hollow_candles_optimized.webp', speed='fast')
    print("   ✓ Saved to /tmp/test_hollow_candles_optimized.webp")

    # Test SVG candlesticks
    print("\n3. Rendering SVG candlesticks...")
    render_candlestick_svg(
        ohlc=ohlc,
        volume=volume,
        width=1920,
        height=1080,
        theme='classic',
        output_path='/tmp/test_candlestick_svg_optimized.svg',
    )
    print("   ✓ Saved to /tmp/test_candlestick_svg_optimized.svg")

    # Test SVG hollow candles
    print("\n4. Rendering SVG hollow candles...")
    render_hollow_candles_svg(
        ohlc=ohlc,
        volume=volume,
        width=1920,
        height=1080,
        theme='classic',
        output_path='/tmp/test_hollow_svg_optimized.svg',
    )
    print("   ✓ Saved to /tmp/test_hollow_svg_optimized.svg")

    print("\n" + "=" * 70)
    print("Visual verification complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - /tmp/test_ohlc_bars_optimized.webp")
    print("  - /tmp/test_hollow_candles_optimized.webp")
    print("  - /tmp/test_candlestick_svg_optimized.svg")
    print("  - /tmp/test_hollow_svg_optimized.svg")
    print("\nPlease verify visually that charts render correctly.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
SVG Export Demo for kimsfinance

Demonstrates the new SVG export capability with various configurations.
"""

import numpy as np
import polars as pl
from pathlib import Path

from kimsfinance.api import plot


def create_sample_data(num_candles: int = 100, base_price: float = 100.0) -> pl.DataFrame:
    """Create realistic OHLCV sample data."""
    np.random.seed(42)

    prices = []
    current_price = base_price

    for _ in range(num_candles):
        # Random walk with trend
        change = np.random.randn() * 2 + 0.1  # Slight upward bias

        open_price = current_price
        close_price = current_price + change + np.random.randn() * 3
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


def demo_basic_svg_export():
    """Demo 1: Basic SVG export with default settings."""
    print("\n" + "="*60)
    print("Demo 1: Basic SVG Export")
    print("="*60)

    df = create_sample_data(50)
    output = Path('demo_output/svg_basic.svg')
    output.parent.mkdir(exist_ok=True)

    plot(df, type='candle', volume=True, savefig=str(output))

    print(f"✓ Created: {output}")
    print(f"  File size: {output.stat().st_size / 1024:.2f} KB")
    print(f"  Candles: 50")
    print(f"  Theme: classic (default)")


def demo_all_themes():
    """Demo 2: Export SVG charts with all available themes."""
    print("\n" + "="*60)
    print("Demo 2: All Themes")
    print("="*60)

    df = create_sample_data(100)
    themes = ['classic', 'modern', 'tradingview', 'light']

    for theme in themes:
        output = Path(f'demo_output/svg_theme_{theme}.svg')
        plot(df, type='candle', style=theme, volume=True, savefig=str(output))

        print(f"✓ Created: {output}")
        print(f"  Theme: {theme}")
        print(f"  File size: {output.stat().st_size / 1024:.2f} KB")


def demo_custom_colors():
    """Demo 3: Custom color schemes."""
    print("\n" + "="*60)
    print("Demo 3: Custom Colors")
    print("="*60)

    df = create_sample_data(75)

    # CoinGecko colors
    output1 = Path('demo_output/svg_coingecko.svg')
    plot(
        df,
        type='candle',
        savefig=str(output1),
        bg_color='#1A1A2E',
        up_color='#16C784',
        down_color='#EA3943',
    )
    print(f"✓ Created: {output1}")
    print(f"  Style: CoinGecko colors")

    # Binance-inspired colors
    output2 = Path('demo_output/svg_binance.svg')
    plot(
        df,
        type='candle',
        savefig=str(output2),
        bg_color='#0B0E11',
        up_color='#0ECB81',
        down_color='#F6465D',
    )
    print(f"✓ Created: {output2}")
    print(f"  Style: Binance colors")


def demo_different_resolutions():
    """Demo 4: Different resolutions and aspect ratios."""
    print("\n" + "="*60)
    print("Demo 4: Different Resolutions")
    print("="*60)

    df = create_sample_data(60)

    configs = [
        ('demo_output/svg_hd.svg', 1280, 720, 'HD (720p)'),
        ('demo_output/svg_fhd.svg', 1920, 1080, 'Full HD (1080p)'),
        ('demo_output/svg_4k.svg', 3840, 2160, '4K (2160p)'),
        ('demo_output/svg_square.svg', 1080, 1080, 'Square'),
        ('demo_output/svg_wide.svg', 2560, 720, 'Wide panoramic'),
    ]

    for output_path, width, height, description in configs:
        output = Path(output_path)
        plot(df, type='candle', savefig=str(output), width=width, height=height)

        print(f"✓ Created: {output}")
        print(f"  Resolution: {width}x{height} ({description})")
        print(f"  File size: {output.stat().st_size / 1024:.2f} KB")


def demo_with_without_volume():
    """Demo 5: Charts with and without volume panel."""
    print("\n" + "="*60)
    print("Demo 5: Volume Panel Options")
    print("="*60)

    df = create_sample_data(80)

    # With volume
    output1 = Path('demo_output/svg_with_volume.svg')
    plot(df, type='candle', volume=True, savefig=str(output1))
    print(f"✓ Created: {output1}")
    print(f"  Volume panel: YES")

    # Without volume
    output2 = Path('demo_output/svg_without_volume.svg')
    plot(df, type='candle', volume=False, savefig=str(output2))
    print(f"✓ Created: {output2}")
    print(f"  Volume panel: NO")


def demo_scaling_comparison():
    """Demo 6: Compare file sizes at different dataset sizes."""
    print("\n" + "="*60)
    print("Demo 6: File Size Scaling")
    print("="*60)

    dataset_sizes = [50, 100, 250, 500]

    for size in dataset_sizes:
        df = create_sample_data(size)

        # SVG
        svg_output = Path(f'demo_output/svg_scaling_{size}.svg')
        plot(df, type='candle', savefig=str(svg_output))
        svg_size = svg_output.stat().st_size / 1024

        # WebP for comparison
        webp_output = Path(f'demo_output/svg_scaling_{size}.webp')
        plot(df, type='candle', savefig=str(webp_output))
        webp_size = webp_output.stat().st_size / 1024

        print(f"Candles: {size:3d}  |  SVG: {svg_size:6.2f} KB  |  WebP: {webp_size:6.2f} KB  |  Ratio: {svg_size/webp_size:.2f}x")


def main():
    """Run all SVG export demos."""
    print("\n" + "🎨"*30)
    print("SVG Export Demo for kimsfinance")
    print("🎨"*30)

    try:
        demo_basic_svg_export()
        demo_all_themes()
        demo_custom_colors()
        demo_different_resolutions()
        demo_with_without_volume()
        demo_scaling_comparison()

        # Summary
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        print("\nGenerated files in demo_output/:")

        output_dir = Path('demo_output')
        svg_files = sorted(output_dir.glob('*.svg'))
        total_size = sum(f.stat().st_size for f in svg_files) / 1024

        for svg_file in svg_files:
            size = svg_file.stat().st_size / 1024
            print(f"  - {svg_file.name} ({size:.2f} KB)")

        print(f"\nTotal: {len(svg_files)} SVG files, {total_size:.2f} KB")
        print("\nYou can open these SVG files in:")
        print("  - Web browsers (Firefox, Chrome, Safari, Edge)")
        print("  - Vector graphics editors (Inkscape, Illustrator, Figma)")
        print("  - Image viewers (GNOME Eye of MATE, Windows Photo Viewer)")
        print("\nSVG files are infinitely scalable without quality loss!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

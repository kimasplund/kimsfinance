#!/usr/bin/env python3
"""
Demo: Line Chart SVG Export for kimsfinance
============================================

This script demonstrates the new SVG export functionality for line charts.
It creates sample OHLCV data and exports it in various SVG formats.
"""

import numpy as np
import polars as pl
from kimsfinance.api import plot

# Create sample OHLCV data
np.random.seed(42)
num_candles = 100

prices = []
current_price = 100.0

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

df = pl.DataFrame(prices)

print("=" * 60)
print("Line Chart SVG Export Demo")
print("=" * 60)

# Demo 1: Basic line chart
print("\n1. Basic line chart (classic theme)")
plot(df, type='line', style='classic', volume=True,
     savefig='demo_line_basic.svg', width=1920, height=1080)
print("   ✓ Saved to: demo_line_basic.svg")

# Demo 2: Line chart with filled area
print("\n2. Line chart with filled area (modern theme)")
plot(df, type='line', style='modern', volume=True,
     fill_area=True, line_width=3,
     savefig='demo_line_filled.svg', width=1920, height=1080)
print("   ✓ Saved to: demo_line_filled.svg")

# Demo 3: Custom colors
print("\n3. Line chart with custom colors")
plot(df, type='line', style='classic', volume=True,
     bg_color='#0D1117', line_color='#58A6FF',
     savefig='demo_line_custom.svg', width=1920, height=1080)
print("   ✓ Saved to: demo_line_custom.svg")

# Demo 4: No volume panel
print("\n4. Line chart without volume (light theme)")
df_no_vol = df.select(['Open', 'High', 'Low', 'Close'])
plot(df_no_vol, type='line', style='light', volume=False,
     savefig='demo_line_no_volume.svg', width=1600, height=900)
print("   ✓ Saved to: demo_line_no_volume.svg")

print("\n" + "=" * 60)
print("✅ All SVG files created successfully!")
print("=" * 60)
print("\nYou can:")
print("  - Open these files in any web browser")
print("  - Scale them infinitely without quality loss")
print("  - Edit them in Inkscape/Illustrator")
print("  - Embed them in web pages")
print("\nFile sizes are typically 10-20KB (much smaller than PNG/WebP)")

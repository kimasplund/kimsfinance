# SVG Export for kimsfinance

## Overview

kimsfinance now supports **true vector SVG export** for candlestick charts. Unlike raster formats (PNG, WebP, JPEG), SVG charts are infinitely scalable without losing quality and can be edited in vector graphics software.

## Features

- **Infinitely scalable** vector graphics
- **Smaller file sizes** for charts with moderate numbers of candles (< 1000)
- **Theme support** - All kimsfinance themes work with SVG (classic, modern, tradingview, light)
- **Custom colors** - Override theme colors with hex values
- **Volume panels** - Optional volume bars below price chart
- **Grid overlay** - Optional price and time grid lines
- **Browser-friendly** - Open directly in web browsers or embed in HTML
- **Editor-compatible** - Edit in Inkscape, Adobe Illustrator, or any SVG editor

## Installation

SVG export requires the `svgwrite` library:

```bash
pip install svgwrite
```

## Usage

### High-level API (plot function)

The easiest way to export SVG is using the `plot()` API with a `.svg` file extension:

```python
from kimsfinance.api import plot
import polars as pl

# Load your OHLCV data
df = pl.read_csv("ohlcv.csv")

# Export as SVG (auto-detected from extension)
plot(df, type='candle', volume=True, savefig='chart.svg')
```

### Different themes

```python
# Classic theme (green/red candles, black background)
plot(df, type='candle', style='classic', savefig='classic.svg')

# Modern theme (teal/red candles, dark gray background)
plot(df, type='candle', style='modern', savefig='modern.svg')

# TradingView theme (similar to TradingView colors)
plot(df, type='candle', style='tradingview', savefig='tradingview.svg')

# Light theme (for light backgrounds)
plot(df, type='candle', style='light', savefig='light.svg')
```

### Custom colors

```python
plot(
    df,
    type='candle',
    savefig='custom.svg',
    bg_color='#1A1A2E',      # Dark blue background
    up_color='#16C784',       # CoinGecko green
    down_color='#EA3943',     # CoinGecko red
)
```

### Custom dimensions

```python
# 4K resolution
plot(df, type='candle', savefig='4k.svg', width=3840, height=2160)

# Square chart
plot(df, type='candle', savefig='square.svg', width=1080, height=1080)

# Wide panoramic
plot(df, type='candle', savefig='wide.svg', width=2560, height=720)
```

### Without volume panel

```python
# Chart without volume bars
plot(df, type='candle', volume=False, savefig='no_volume.svg')
```

### Low-level renderer API

For more control, use the `render_candlestick_svg()` function directly:

```python
from kimsfinance.plotting import render_candlestick_svg
import numpy as np

# Prepare OHLC data as dictionary
ohlc_dict = {
    'open': df['Open'].to_numpy(),
    'high': df['High'].to_numpy(),
    'low': df['Low'].to_numpy(),
    'close': df['Close'].to_numpy(),
}
volume_array = df['Volume'].to_numpy()

# Render and save SVG
svg_content = render_candlestick_svg(
    ohlc_dict,
    volume_array,
    width=1920,
    height=1080,
    theme='modern',
    output_path='chart.svg',
)

# Or get SVG as string without saving
svg_string = render_candlestick_svg(
    ohlc_dict,
    volume_array,
    theme='classic',
    output_path=None,  # Don't save to file
)
print(svg_string)  # Raw SVG XML
```

## File Size Comparison

SVG file sizes vary based on the number of candles:

| Candles | SVG Size | PNG Size (1920x1080) | Ratio |
|---------|----------|----------------------|-------|
| 50      | ~21 KB   | ~15 KB               | 1.4x  |
| 100     | ~41 KB   | ~18 KB               | 2.3x  |
| 500     | ~194 KB  | ~25 KB               | 7.8x  |

**Recommendations**:
- **SVG**: Best for charts with < 500 candles, presentations, web embedding, editing
- **WebP/PNG**: Best for large datasets (1000+ candles), when file size is critical

## Browser Support

SVG charts can be opened directly in all modern browsers:

```bash
# Linux
firefox chart.svg

# macOS
open chart.svg

# Windows
start chart.svg
```

## Embedding in HTML

```html
<!DOCTYPE html>
<html>
<head>
    <title>Financial Chart</title>
</head>
<body>
    <!-- Inline SVG -->
    <img src="chart.svg" alt="Candlestick Chart">

    <!-- Or embed directly -->
    <object data="chart.svg" type="image/svg+xml"></object>
</body>
</html>
```

## Editing SVG Charts

SVG charts can be edited in:

- **Inkscape** (free, open-source)
- **Adobe Illustrator**
- **Figma** (web-based)
- **Sketch** (macOS)

Example edits:
- Change colors
- Add annotations
- Adjust layouts
- Export to other formats

## Limitations

### Currently Supported
- ✅ Candlestick charts (`type='candle'`)
- ✅ All themes (classic, modern, tradingview, light)
- ✅ Volume panels
- ✅ Custom colors
- ✅ Grid overlays

### Not Yet Supported
- ❌ OHLC bar charts (`type='ohlc'`)
- ❌ Line charts (`type='line'`)
- ❌ Hollow candles (`type='hollow_and_filled'`)
- ❌ Renko charts (`type='renko'`)
- ❌ Point & Figure charts (`type='pnf'`)
- ❌ Technical indicators (RSI, MACD, etc.)

If you try to export a non-candlestick chart type as SVG, kimsfinance will:
1. Show a warning message
2. Fall back to PNG format automatically

## Performance

SVG rendering is fast for moderate datasets:

| Operation | Time |
|-----------|------|
| 50 candles | ~5ms |
| 100 candles | ~10ms |
| 500 candles | ~50ms |

Note: SVG rendering is sequential (not batched like PIL rendering), so it's slightly slower than WebP/PNG for very large datasets.

## Technical Details

### Implementation

kimsfinance uses the `svgwrite` library to create true vector graphics:

- **Background**: Single rectangle with theme color
- **Grid**: SVG `<line>` elements with 25% opacity
- **Candles**: Each candle = 1 `<line>` (wick) + 1 `<rect>` (body)
- **Volume**: SVG `<rect>` elements with 50% opacity
- **Groups**: Organized SVG structure with `<g>` groups (candles, volume, grid)

### SVG Structure

```xml
<svg width="1920" height="1080">
  <!-- Background -->
  <rect fill="#000000" width="100%" height="100%" />

  <!-- Grid -->
  <g id="grid">
    <line ... />  <!-- Horizontal/vertical grid lines -->
  </g>

  <!-- Candlesticks -->
  <g id="candles">
    <line ... />  <!-- Wick -->
    <rect ... />  <!-- Body -->
    <!-- Repeated for each candle -->
  </g>

  <!-- Volume bars -->
  <g id="volume">
    <rect ... />  <!-- Volume bar -->
    <!-- Repeated for each bar -->
  </g>
</svg>
```

## Examples

### Example 1: Create SVG charts for all themes

```python
from kimsfinance.api import plot
import polars as pl

df = pl.read_csv("btc_ohlcv.csv")

themes = ['classic', 'modern', 'tradingview', 'light']
for theme in themes:
    plot(df, type='candle', style=theme, savefig=f'btc_{theme}.svg')
```

### Example 2: Compare SVG vs WebP

```python
# Same data, different formats
plot(df, type='candle', savefig='chart.svg')    # Vector SVG
plot(df, type='candle', savefig='chart.webp')   # Raster WebP

# Check file sizes
import os
svg_size = os.path.getsize('chart.svg') / 1024
webp_size = os.path.getsize('chart.webp') / 1024
print(f"SVG: {svg_size:.2f} KB, WebP: {webp_size:.2f} KB")
```

### Example 3: Generate presentation-ready charts

```python
# High-res SVG for presentations
plot(
    df,
    type='candle',
    style='light',  # Light background for presentations
    width=2560,
    height=1440,
    savefig='presentation.svg',
)
```

## Troubleshooting

### ImportError: svgwrite not installed

```bash
pip install svgwrite
```

### SVG file is too large

For large datasets (1000+ candles), use WebP or PNG instead:

```python
# Use WebP for large datasets
plot(df, type='candle', savefig='chart.webp')
```

### Chart type not supported

Only candlestick charts support SVG export currently. Use PNG/WebP for other types:

```python
# This will warn and use PNG instead
plot(df, type='line', savefig='line.svg')  # Auto-converts to line.png
```

## Future Enhancements

Planned additions:
- Support for OHLC bars, line charts, hollow candles
- Technical indicator overlays (SMA, EMA, Bollinger Bands)
- Multi-panel charts with indicators (RSI, MACD, Stochastic)
- Text labels for prices and dates
- Axis labels and legends
- Interactive SVG with tooltips

## See Also

- [kimsfinance README](../README.md) - Main documentation
- [Native Charts Guide](../docs/implementation_plan_native_charts.md) - PIL rendering details
- [API Reference](../kimsfinance/api/plot.py) - plot() function documentation
- [Renderer Source](../kimsfinance/plotting/renderer.py) - Low-level rendering code

---

**Last Updated**: 2025-10-20
**Author**: kimsfinance contributors
**License**: AGPL-3.0 + Commercial

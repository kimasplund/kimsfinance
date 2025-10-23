# SVG Export Quick Start Guide

## Installation

```bash
pip install svgwrite
```

(Already included in kimsfinance dependencies)

## Basic Usage

### Export as SVG

```python
from kimsfinance.api import plot
import polars as pl

df = pl.read_csv("ohlcv.csv")

# Just add .svg extension!
plot(df, type='candle', volume=True, savefig='chart.svg')
```

That's it! The `.svg` extension is automatically detected.

## Quick Examples

### Different Themes

```python
# Classic (green/red, black background)
plot(df, type='candle', style='classic', savefig='classic.svg')

# Modern (teal/red, dark gray background)
plot(df, type='candle', style='modern', savefig='modern.svg')

# TradingView (similar to TradingView)
plot(df, type='candle', style='tradingview', savefig='tv.svg')

# Light (for presentations)
plot(df, type='candle', style='light', savefig='light.svg')
```

### Custom Colors

```python
# CoinGecko colors
plot(df, type='candle', savefig='coingecko.svg',
     bg_color='#1A1A2E', up_color='#16C784', down_color='#EA3943')

# Binance colors
plot(df, type='candle', savefig='binance.svg',
     bg_color='#0B0E11', up_color='#0ECB81', down_color='#F6465D')
```

### Different Sizes

```python
# 4K
plot(df, type='candle', savefig='4k.svg', width=3840, height=2160)

# Square
plot(df, type='candle', savefig='square.svg', width=1080, height=1080)

# Wide
plot(df, type='candle', savefig='wide.svg', width=2560, height=720)
```

### Without Volume

```python
plot(df, type='candle', volume=False, savefig='no_volume.svg')
```

## Viewing SVG Files

### In Browser

```bash
# Linux
firefox chart.svg

# macOS
open chart.svg

# Windows
start chart.svg
```

### In Editor

Open in:
- **Inkscape** (free, cross-platform)
- **Adobe Illustrator**
- **Figma** (web-based)

## When to Use SVG vs WebP/PNG

### Use SVG When:
- ✅ Dataset has < 500 candles
- ✅ Need infinite scalability
- ✅ Creating presentations
- ✅ Embedding in web pages
- ✅ Need to edit/customize charts
- ✅ Publishing/printing

### Use WebP/PNG When:
- ✅ Large datasets (1000+ candles)
- ✅ File size is critical
- ✅ Batch processing thousands of charts
- ✅ Need fastest possible rendering

## File Size Guide

| Candles | SVG    | WebP  |
|---------|--------|-------|
| 50      | ~21 KB | 1 KB  |
| 100     | ~41 KB | 2 KB  |
| 500     | ~194 KB| 5 KB  |

**Rule of thumb**: SVG is 15-35x larger than WebP, but infinitely scalable.

## Current Limitations

**Supported**: Only candlestick charts (`type='candle'`)

**Not yet supported**:
- OHLC bars (`type='ohlc'`)
- Line charts (`type='line'`)
- Hollow candles
- Renko/PnF charts
- Technical indicators

If you request SVG for unsupported types, kimsfinance will warn and use PNG instead.

## Full Documentation

- **User Guide**: `/home/kim/Documents/Github/kimsfinance/docs/SVG_EXPORT.md`
- **Implementation**: `/home/kim/Documents/Github/kimsfinance/SVG_EXPORT_IMPLEMENTATION_SUMMARY.md`
- **Demo Script**: `/home/kim/Documents/Github/kimsfinance/scripts/demo_svg_export.py`
- **Tests**: `/home/kim/Documents/Github/kimsfinance/tests/test_svg_export.py`

## Need Help?

```python
# Check if svgwrite is installed
import svgwrite
print("SVG export available!")

# Run demo script
python scripts/demo_svg_export.py

# Run tests
python tests/test_svg_export.py
```

---

**That's all you need to get started with SVG export!** 🎨

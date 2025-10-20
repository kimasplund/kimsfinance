# kimsfinance Sample Charts

This directory contains sample chart visualizations demonstrating the various styles and features available in kimsfinance.

All charts are generated from real Bitcoin price data (50 candles from September 12, 2025) and showcase the library's high-performance WebP output with minimal file sizes.

## Chart Samples

### 1. Classic Candlestick Chart
**File:** `01_classic_candlestick.webp` (24.4 KB)
- **Style:** Charles (classic trading view)
- **Features:** Standard candlesticks with volume panel
- **Use Case:** Traditional technical analysis, clean and professional
- **Price Range:** $114,771 - $115,879

### 2. TradingView Style ⭐
**File:** `02_tradingview_style.webp` (28.3 KB)
- **Style:** TradingView (popular platform style)
- **Features:** Familiar TradingView appearance, dark background
- **Use Case:** Users familiar with TradingView interface
- **Best For:** TradingView users, charting consistency across platforms

### 3. Modern Dark Theme
**File:** `03_modern_dark_theme.webp` (30.0 KB)
- **Style:** NightClouds (dark theme)
- **Features:** Dark background with high contrast candles
- **Use Case:** Night trading, reduced eye strain, modern UI
- **Best For:** Extended screen time, dark mode enthusiasts

### 4. Binance Dark Theme ⭐
**File:** `04_binance_dark_theme.webp` (33.6 KB)
- **Style:** Binance Dark (exchange dark mode)
- **Features:** Binance-style dark theme, crypto trader favorite
- **Use Case:** Cryptocurrency trading, dark mode preference
- **Best For:** Crypto traders, Binance users, dark mode

### 5. Minimal Clean Style
**File:** `05_minimal_clean.webp` (30.8 KB)
- **Style:** Mike (minimal design)
- **Features:** Clean layout, essential information only
- **Use Case:** Focus on price action, distraction-free analysis
- **Best For:** Quick visual assessment, presentations

### 6. Colorful High-Contrast
**File:** `06_colorful_highcontrast.webp` (16.7 KB)
- **Style:** Yahoo (vibrant colors)
- **Features:** Bold colors, high visual impact
- **Use Case:** Public presentations, marketing materials
- **Best For:** Maximum visibility, attention-grabbing visuals

### 7. Professional Trading Style
**File:** `07_professional_trading.webp` (25.9 KB)
- **Style:** Binance Light (professional exchange)
- **Features:** Clean grid, professional appearance
- **Use Case:** Institutional trading, client reports
- **Best For:** Professional environments, trading desk displays

### 8. Hollow Candlestick Chart
**File:** `08_hollow_candles.webp` (25.0 KB)
- **Style:** Hollow and filled candles
- **Features:** Hollow for up closes, filled for down closes
- **Use Case:** Advanced pattern recognition, trend visualization
- **Best For:** Experienced traders, momentum analysis

### 9. Renko Chart
**File:** `09_renko_chart.webp` (12.2 KB)
- **Style:** Renko blocks
- **Features:** Time-independent, noise-filtered price movement
- **Use Case:** Trend identification, support/resistance levels
- **Best For:** Filtering market noise, clear trend signals

### 10. Classic with SMA Indicators
**File:** `10_with_sma_indicators.webp` (28.5 KB)
- **Style:** Classic with Simple Moving Averages
- **Features:** 7-period and 20-period SMA overlay
- **Use Case:** Trend following, crossover strategies, support/resistance
- **Best For:** Traditional technical analysis, swing trading

### 11. TradingView with EMA Indicators
**File:** `11_tradingview_with_ema.webp` (32.3 KB)
- **Style:** TradingView with Exponential Moving Averages
- **Features:** 9-period and 21-period EMA (faster response than SMA)
- **Use Case:** Day trading, momentum strategies, trend confirmation
- **Best For:** Active traders, short-term momentum analysis

### 12. Binance Dark with Bollinger Bands
**File:** `12_binance_with_bollinger.webp` (35.4 KB)
- **Style:** Binance Dark with 20-period MA
- **Features:** 20-period moving average (middle Bollinger Band)
- **Use Case:** Volatility analysis, mean reversion strategies
- **Best For:** Volatility traders, range-bound strategies

### 13. HD 720p TradingView
**File:** `13_hd_720p_tradingview.webp` (44.6 KB)
- **Resolution:** 1280x720 (HD 720p, 16:9 aspect ratio)
- **Style:** TradingView
- **Use Case:** HD displays, video content, presentations
- **Best For:** YouTube thumbnails, HD monitors, video backgrounds

### 14. Full HD 1080p Binance Dark with EMA
**File:** `14_fullhd_1080p_binance_dark.webp` (96.1 KB)
- **Resolution:** 1920x1080 (Full HD, 16:9 aspect ratio)
- **Style:** Binance Dark with 9 & 21-period EMA
- **Features:** Full HD quality with technical indicators
- **Use Case:** Large displays, professional presentations, streaming
- **Best For:** 1080p monitors, projectors, high-quality screenshots

### 15. Full HD 1080p Minimal
**File:** `15_fullhd_1080p_minimal.webp` (75.3 KB)
- **Resolution:** 1920x1080 (Full HD, 16:9 aspect ratio)
- **Style:** Minimal clean design
- **Features:** Distraction-free Full HD quality
- **Use Case:** Clean presentations, focus on price action
- **Best For:** Professional displays, minimal design preference

### 16. Print Quality High DPI
**File:** `16_print_quality_high_dpi.webp` (29.9 KB)
- **Style:** Classic high-DPI
- **Features:** Enhanced DPI for print quality
- **Use Case:** Print materials, professional publications
- **Best For:** Reports, academic papers, physical media

## Technical Details

- **Data Source:** Bitcoin OHLCV data from binance-visual-ml test dataset (starting from row 600)
- **Time Period:** September 14, 2025, 15:15 - 19:20 UTC (50 candles displayed, 5-minute intervals)
- **Indicator Warmup:** 50 additional candles loaded for proper indicator calculation
- **Format:** WebP (modern, efficient image format)
- **Total Charts:** 16 different variations
  - 9 standard resolution styles
  - 3 charts with technical indicators
  - 4 high-resolution variants (HD 720p, Full HD 1080p)
- **Average File Size:** 38.7 KB (including HD variants)
- **Standard Resolution:** ~27 KB average
- **HD 720p:** ~45 KB
- **Full HD 1080p:** ~75-96 KB (still smaller than standard PNG!)
- **Generation Time:** ~20-25 seconds for all 16 charts
- **GPU Acceleration:** Compatible (auto-detects and uses GPU when available)

### Indicator Calculation
Charts with technical indicators (SMA, EMA, Bollinger Bands) load 100 candles total:
- **50 warmup candles** - Used for indicator calculation but not displayed
- **50 display candles** - Shown in the chart with fully calculated indicators

This ensures indicators like 20-period moving averages appear from the very beginning of the chart instead of starting in the middle.

## File Size Comparison

kimsfinance's WebP output is significantly smaller than traditional formats:

### Standard Resolution (~800x600)
```
WebP (current):      27 KB average
PNG equivalent:     ~160 KB (6x larger)
JPG equivalent:      ~90 KB (3x larger)
SVG equivalent:     ~220 KB (8x larger)
```

### HD 720p (1280x720)
```
WebP (current):      45 KB
PNG equivalent:     ~280 KB (6x larger)
JPG equivalent:     ~150 KB (3x larger)
```

### Full HD 1080p (1920x1080)
```
WebP (current):      75-96 KB
PNG equivalent:     ~600 KB (6-8x larger)
JPG equivalent:     ~320 KB (3-4x larger)
```

**Benefits:**
- 📉 85% smaller than PNG across all resolutions
- ⚡ Faster page loads
- 💾 Reduced storage costs (1080p chart = ~100 KB vs ~600 KB PNG)
- 🌍 Lower bandwidth usage
- 📱 Better mobile performance
- 🖥️ HD-ready without file bloat

## Usage Example

To generate similar charts in your own code:

```python
import kimsfinance as kf
import polars as pl

# Load your OHLCV data
df = pl.read_csv("your_data.csv")

# Classic candlestick chart
kf.plot(
    df,
    type="candle",
    style="charles",
    volume=True,
    savefig="output.webp"
)

# With SMA indicators (7 and 20-period)
kf.plot(
    df,
    type="candle",
    style="charles",
    mav=(7, 20),
    volume=True,
    savefig="with_sma.webp"
)

# TradingView style with EMA (9 and 21-period)
kf.plot(
    df,
    type="candle",
    style="tradingview",
    mav=(9, 21),  # Faster-responding EMAs
    volume=True,
    savefig="tradingview_ema.webp"
)

# Binance Dark with 20-period MA (Bollinger middle band)
kf.plot(
    df,
    type="candle",
    style="binancedark",
    mav=(20,),  # Single MA for Bollinger bands base
    volume=True,
    savefig="binance_bollinger.webp"
)
```

## Regenerating Samples

To regenerate these sample charts:

```bash
# From the repository root
.venv/bin/python scripts/generate_sample_charts.py
```

The script will:
1. Load 50 candles from CSV row 600 (with 50 warmup candles)
2. Generate all 16 chart variations:
   - 9 standard resolution styles
   - 3 with technical indicators
   - 4 high-resolution variants (720p, 1080p)
3. Save them as optimized WebP files
4. Display file sizes and generation statistics

## Performance Characteristics

### Generation Speed
- **CPU Mode:** ~1-2 seconds per chart (standard), ~2-3 seconds (1080p)
- **GPU Mode:** ~0.3-0.5 seconds per chart (2-4x faster)
- **Batch Generation:** ~20-25 seconds for all 16 charts
- **With Indicators:** Slightly slower due to MA calculations (~1.2-1.5x)
- **HD Resolution Impact:** 720p adds ~0.5s, 1080p adds ~1-2s per chart

### Quality Metrics
- **Resolution Options:**
  - Standard: ~800x600 pixels (default)
  - HD 720p: 1280x720 pixels (16:9)
  - Full HD 1080p: 1920x1080 pixels (16:9)
  - Custom: Up to 4K+ (3840x2160)
- **WebP Quality:** 85-95 (optimal balance)
- **Color Depth:** 24-bit (16.7 million colors)
- **Compression:** Lossless-optimized lossy
- **Aspect Ratios:** 4:3 (standard), 16:9 (HD), custom

## Supported Chart Types

kimsfinance supports all mplfinance chart types:

- `candle` - Standard candlestick charts ✓
- `hollow_and_filled` - Hollow candles ✓
- `ohlc` - OHLC bars
- `line` - Line charts
- `renko` - Renko bricks ✓
- `pnf` - Point and Figure
- `heikinashi` - Heikin-Ashi candles

## Supported Styles

Available built-in styles:

- `charles` - Classic professional ✓
- `tradingview` - TradingView style ✓ ⭐
- `nightclouds` - Modern dark theme ✓
- `binancedark` - Binance dark theme ✓ ⭐
- `binance` - Binance light theme ✓
- `mike` - Minimal clean ✓
- `yahoo` - Colorful high-contrast ✓
- `blueskies` - Blue sky theme
- `sas` - Statistical Analysis System
- `brasil` - Brazilian market style
- `classic` - Traditional style
- `checkers` - Checkerboard pattern
- `kenan` - Kenan style
- `ibd` - Investor's Business Daily
- `starsandstripes` - American flag theme

## Next Steps

- **Try the examples** - Copy the code above and modify parameters
- **Explore styles** - Test different visual themes for your use case
- **Add indicators** - Combine with technical indicators (MA, RSI, MACD, etc.)
- **Customize** - Create your own color schemes and layouts
- **Optimize** - Enable GPU acceleration for 10-50x speedup on large datasets

## Questions or Issues?

- 📖 [Documentation](https://github.com/kimasplund/kimsfinance#readme)
- 🐛 [Report Issues](https://github.com/kimasplund/kimsfinance/issues)
- 💡 [Feature Requests](https://github.com/kimasplund/kimsfinance/issues/new)

---

**Generated by:** kimsfinance v0.1.0
**Last Updated:** October 20, 2025
**License:** AGPL-3.0-or-later (Open Source) / Commercial License Available

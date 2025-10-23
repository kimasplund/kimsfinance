# API Reference

**kimsfinance v0.1.0** - High-Performance Financial Charting Library

This comprehensive API reference covers all public functions and methods in kimsfinance. The library achieves **28.8x average speedup** over mplfinance (validated range: 7.3x - 70.1x) through native PIL rendering, GPU acceleration, and optimized algorithms.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Plotting API](#core-plotting-api)
4. [Chart Types](#chart-types)
5. [Themes & Styling](#themes--styling)
6. [Technical Indicators](#technical-indicators)
7. [Batch Rendering](#batch-rendering)
8. [Output Formats](#output-formats)
9. [GPU Acceleration](#gpu-acceleration)
10. [Integration Examples](#integration-examples)
11. [Error Handling](#error-handling)
12. [Performance Tips](#performance-tips)

---

## Installation

### Basic Installation (CPU-only)

```bash
pip install kimsfinance
```

**Requirements:**
- Python 3.13+
- Pillow 12.0+
- NumPy 2.0+
- Polars 1.0+

### GPU Acceleration (Optional)

For 6.4x faster OHLCV processing and GPU-accelerated indicators:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cupy-cuda12x
```

**GPU Requirements:**
- NVIDIA GPU (RTX series, Tesla, Quadro)
- CUDA 12.x
- 4GB+ VRAM recommended

---

## Quick Start

### Basic Chart Rendering

```python
import kimsfinance as kf
import polars as pl

# Load OHLCV data
df = pl.read_csv("ohlcv_data.csv")

# Render candlestick chart
kf.plot(df, type='candle', volume=True, savefig='chart.webp')
```

### With Custom Theme

```python
# TradingView-style dark theme
kf.plot(
    df,
    type='candle',
    style='tradingview',
    volume=True,
    width=1920,
    height=1080,
    savefig='chart.webp'
)
```

### Return PIL Image

```python
# Get PIL Image object for further processing
img = kf.plot(df, type='candle', returnfig=True)
img.show()  # Display
img.save('chart.png')  # Save manually
```

---

## Core Plotting API

### `plot()`

Main plotting function for financial charts. Uses native PIL renderer achieving **28.8x average speedup** vs mplfinance (up to 70.1x at optimal conditions).

```python
kimsfinance.plot(
    data,
    *,
    type="candle",
    style="binance",
    mav=None,
    ema=None,
    volume=True,
    engine="auto",
    savefig=None,
    returnfig=False,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | Required | OHLC DataFrame (pandas or polars) with columns: Open, High, Low, Close, Volume |
| `type` | str | `"candle"` | Chart type: `'candle'`, `'ohlc'`, `'line'`, `'hollow'`, `'renko'`, `'pnf'` |
| `style` | str | `"binance"` | Visual theme: `'classic'`, `'modern'`, `'tradingview'`, `'light'`, `'binance'` |
| `mav` | list[int] | `None` | Simple moving average periods (requires mplfinance fallback) |
| `ema` | list[int] | `None` | Exponential moving average periods (requires mplfinance fallback) |
| `volume` | bool | `True` | Show volume panel below price chart |
| `engine` | str | `"auto"` | Execution engine: `"cpu"`, `"gpu"`, `"auto"` (for indicators only) |
| `savefig` | str | `None` | Output path to save chart (e.g., `'chart.webp'`). If None, returns/displays chart |
| `returnfig` | bool | `False` | If True, returns PIL Image object. If False, saves/displays chart |

#### Keyword Arguments (`**kwargs`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | `1920` | Image width in pixels (100-8192) |
| `height` | int | `1080` | Image height in pixels (100-8192) |
| `theme` | str | `style` | Alternative way to specify theme (overrides `style`) |
| `bg_color` | str | `None` | Override background color (hex: `"#000000"`) |
| `up_color` | str | `None` | Override bullish/up color (hex: `"#00FF00"`) |
| `down_color` | str | `None` | Override bearish/down color (hex: `"#FF0000"`) |
| `enable_antialiasing` | bool | `True` | Use RGBA mode for smoother rendering |
| `show_grid` | bool | `True` | Display price/time grid lines |
| `line_width` | float | `2.0` | Line width for line charts (0.1-20.0) |
| `fill_area` | bool | `False` | Fill area under line for line charts |
| `box_size` | float | `None` | Box size for Renko/PNF charts (auto-calculated if None) |
| `reversal_boxes` | int | `1` (Renko)<br>`3` (PNF) | Reversal threshold for Renko/PNF charts |
| `wick_width_ratio` | float | `0.1` | Wick width ratio for candlestick/hollow charts |
| `speed` | str | `"balanced"` | WebP encoding speed: `"fast"`, `"balanced"`, `"best"` |
| `quality` | int | `None` | WebP quality (1-100, default varies by speed preset) |

#### Returns

- **If `returnfig=True`:** PIL Image object
- **If `returnfig=False` and `savefig=None`:** PIL Image object (with warning)
- **If `savefig` is set:** `None` (saves to file)

#### Examples

**Basic Candlestick:**
```python
kf.plot(df, type='candle', volume=True, savefig='chart.webp')
```

**OHLC Bars with Custom Colors:**
```python
kf.plot(
    df,
    type='ohlc',
    up_color='#00FF00',
    down_color='#FF0000',
    bg_color='#000000',
    savefig='ohlc.webp'
)
```

**Line Chart with Fill:**
```python
kf.plot(
    df,
    type='line',
    fill_area=True,
    line_color='#2196F3',
    line_width=2.5,
    savefig='line.webp'
)
```

**Hollow Candles:**
```python
kf.plot(
    df,
    type='hollow',
    style='tradingview',
    width=3840,
    height=2160,
    savefig='hollow_4k.webp'
)
```

**Renko Chart:**
```python
kf.plot(
    df,
    type='renko',
    box_size=0.5,
    reversal_boxes=2,
    savefig='renko.webp'
)
```

**Point & Figure:**
```python
kf.plot(
    df,
    type='pnf',
    box_size=1.0,
    reversal_boxes=3,
    savefig='pnf.webp'
)
```

#### Performance

| Chart Type | Throughput | Speedup vs mplfinance |
|------------|------------|----------------------|
| Candlestick | 6,249 img/sec (peak) | 7.3x - 70.1x (avg: 28.8x) |
| OHLC Bars | 1,337 img/sec | 20-50x faster |
| Line | 2,100 img/sec | 30-60x faster |
| Hollow Candles | 5,728 img/sec | 20-50x faster |
| Renko | 3,800 img/sec | 15-40x faster |
| Point & Figure | 357 img/sec | 10-30x faster |

---

### `make_addplot()`

Create additional plot data for multi-panel charts. **Note:** Using `addplot` requires mplfinance and disables native PIL renderer (performance penalty).

```python
kimsfinance.make_addplot(data, **kwargs)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | Data to plot (e.g., indicator values) |
| `**kwargs` | dict | Additional mplfinance.make_addplot() parameters |

#### Example

```python
from kimsfinance import plot, make_addplot
import numpy as np

# Calculate custom indicator
rsi = calculate_rsi(df['close'])

# Add to plot (uses mplfinance fallback - slower!)
ap = make_addplot(rsi, panel=2, color='purple')
plot(df, type='candle', addplot=ap)
```

**Warning:** For maximum performance (28.8x average speedup), avoid `addplot` and use native renderer without multi-panel features.

---

### `plot_with_indicators()`

Plot with GPU-accelerated technical indicators. **Note:** This function uses mplfinance fallback for multi-panel support.

```python
kimsfinance.plot_with_indicators(
    data,
    *,
    type="candle",
    indicators=None,
    engine="auto",
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | Required | OHLC DataFrame |
| `type` | str | `"candle"` | Chart type |
| `indicators` | dict | `None` | Dict of indicators to add (see below) |
| `engine` | str | `"auto"` | Execution engine for indicator calculations |
| `**kwargs` | dict | - | Additional plot parameters |

#### Indicators Dictionary Format

```python
indicators = {
    'sma': [20, 50, 200],  # Simple moving averages
    'ema': [12, 26],       # Exponential moving averages
    'rsi': {               # RSI in separate panel
        'period': 14,
        'panel': 2
    },
    'macd': {              # MACD in separate panel
        'panel': 3
    }
}
```

#### Example

```python
kf.plot_with_indicators(
    df,
    type='candle',
    indicators={
        'sma': [20, 50],
        'rsi': {'period': 14, 'panel': 2}
    },
    engine='gpu',  # Use GPU for indicator calculations
    savefig='chart_with_indicators.webp'
)
```

**Performance Note:** Uses mplfinance fallback (~35 img/sec vs 6,249 img/sec native). For maximum performance, calculate indicators separately and overlay them.

---

## Chart Types

### Candlestick (`type='candle'`)

Traditional Japanese candlesticks showing Open, High, Low, Close.

```python
kf.plot(df, type='candle', volume=True, savefig='candle.webp')
```

**Features:**
- Filled body (up: green, down: red by default)
- Wicks show high/low range
- Configurable wick width ratio
- 6,249 img/sec throughput

---

### OHLC Bars (`type='ohlc'`)

Traditional OHLC bars with horizontal ticks.

```python
kf.plot(df, type='ohlc', volume=True, savefig='ohlc.webp')
```

**Features:**
- Vertical bar with left (open) and right (close) ticks
- Color based on close vs open
- 1,337 img/sec throughput

---

### Line Chart (`type='line'`)

Simple line chart connecting close prices.

```python
kf.plot(
    df,
    type='line',
    line_color='#2196F3',
    line_width=2,
    fill_area=True,
    savefig='line.webp'
)
```

**Features:**
- Smooth line connecting prices
- Optional area fill
- Configurable line width and color
- 2,100 img/sec throughput

---

### Hollow Candles (`type='hollow'` or `type='hollow_and_filled'`)

Hollow candles that indicate trend direction.

```python
kf.plot(df, type='hollow', volume=True, savefig='hollow.webp')
```

**Features:**
- Hollow when close > open (bullish)
- Filled when close < open (bearish)
- Visual momentum indicator
- 5,728 img/sec throughput

---

### Renko Chart (`type='renko'`)

Price-only chart filtering out time and noise.

```python
kf.plot(
    df,
    type='renko',
    box_size=0.5,        # Fixed price movement
    reversal_boxes=2,    # Reversal threshold
    savefig='renko.webp'
)
```

**Features:**
- Fixed-size boxes (bricks)
- Time-independent
- Trend visualization
- Auto box size calculation if not specified
- 3,800 img/sec throughput

---

### Point & Figure (`type='pnf'` or `type='pointandfigure'`)

Classic point & figure chart showing price movements.

```python
kf.plot(
    df,
    type='pnf',
    box_size=1.0,        # Price per box
    reversal_boxes=3,    # Standard 3-box reversal
    savefig='pnf.webp'
)
```

**Features:**
- X's for upward movement
- O's for downward movement
- 3-box reversal (standard)
- Time-independent
- 357 img/sec throughput

---

## Themes & Styling

### Built-in Themes

kimsfinance includes 4 professional themes:

#### Classic (`style='classic'`)
```python
kf.plot(df, style='classic', savefig='classic.webp')
```
- **Background:** Black (#000000)
- **Up Color:** Green (#00FF00)
- **Down Color:** Red (#FF0000)
- **Grid:** Dark gray (#333333)
- **Use case:** Traditional trading terminal

#### Modern (`style='modern'`)
```python
kf.plot(df, style='modern', savefig='modern.webp')
```
- **Background:** Dark gray (#1E1E1E)
- **Up Color:** Teal (#26A69A)
- **Down Color:** Red (#EF5350)
- **Grid:** Medium gray (#424242)
- **Use case:** Modern applications

#### TradingView (`style='tradingview'`)
```python
kf.plot(df, style='tradingview', savefig='tradingview.webp')
```
- **Background:** Navy (#131722)
- **Up Color:** Teal (#089981)
- **Down Color:** Pink-red (#F23645)
- **Grid:** Dark navy (#2A2E39)
- **Use case:** TradingView-like appearance

#### Light (`style='light'`)
```python
kf.plot(df, style='light', savefig='light.webp')
```
- **Background:** White (#FFFFFF)
- **Up Color:** Teal (#26A69A)
- **Down Color:** Red (#EF5350)
- **Grid:** Light gray (#E0E0E0)
- **Use case:** Print, reports, presentations

### Custom Colors

Override individual colors:

```python
kf.plot(
    df,
    type='candle',
    bg_color='#0D1117',      # GitHub dark background
    up_color='#3FB950',      # GitHub green
    down_color='#F85149',    # GitHub red
    savefig='custom.webp'
)
```

### Grid Customization

```python
# Disable grid
kf.plot(df, show_grid=False, savefig='no_grid.webp')

# Enable grid (default)
kf.plot(df, show_grid=True, savefig='with_grid.webp')
```

### Antialiasing

```python
# High-quality RGBA rendering (default)
kf.plot(df, enable_antialiasing=True, savefig='smooth.webp')

# Fast RGB rendering (faster but less smooth)
kf.plot(df, enable_antialiasing=False, savefig='fast.webp')
```

---

## Technical Indicators

kimsfinance provides **29 GPU-accelerated technical indicators** with automatic CPU/GPU engine selection.

### Indicator Categories

1. **Moving Averages** (4 indicators)
2. **Momentum Oscillators** (7 indicators)
3. **Volatility** (4 indicators)
4. **Volume** (4 indicators)
5. **Trend** (5 indicators)
6. **Support/Resistance** (5 indicators)

---

### Moving Averages

#### `calculate_sma()` - Simple Moving Average

```python
kimsfinance.calculate_sma(prices, period=20, *, engine="auto")
```

**Parameters:**
- `prices` (array-like): Price data
- `period` (int): Lookback period (default: 20)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** NumPy array of SMA values

**Example:**
```python
import kimsfinance as kf
import polars as pl

df = pl.read_csv("ohlcv.csv")
sma_20 = kf.calculate_sma(df['close'], period=20)
sma_50 = kf.calculate_sma(df['close'], period=50)
```

**Performance:** 1.1-3.3x speedup on CPU (Polars), 5-10x on GPU

---

#### `calculate_ema()` - Exponential Moving Average

```python
kimsfinance.calculate_ema(prices, period=12, *, engine="auto")
```

**Parameters:**
- `prices` (array-like): Price data
- `period` (int): Lookback period (default: 12)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** NumPy array of EMA values

**Example:**
```python
ema_12 = kf.calculate_ema(df['close'], period=12)
ema_26 = kf.calculate_ema(df['close'], period=26)
```

**Performance:** Similar to SMA (1.1-3.3x speedup)

---

#### `calculate_wma()` - Weighted Moving Average

```python
kimsfinance.calculate_wma(prices, period=20, *, engine="auto")
```

**Parameters:**
- `prices` (array-like): Price data
- `period` (int): Lookback period (default: 20)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** NumPy array of WMA values

**Example:**
```python
wma_20 = kf.calculate_wma(df['close'], period=20)
```

---

#### `calculate_vwma()` - Volume Weighted Moving Average

```python
kimsfinance.calculate_vwma(prices, volumes, period=20, *, engine="auto")
```

**Parameters:**
- `prices` (array-like): Price data
- `volumes` (array-like): Volume data
- `period` (int): Lookback period (default: 20)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** NumPy array of VWMA values

**Example:**
```python
vwma_20 = kf.calculate_vwma(df['close'], df['volume'], period=20)
```

---

### Momentum Oscillators

#### `calculate_rsi()` - Relative Strength Index

```python
kimsfinance.calculate_rsi(prices, period=14, *, engine="auto")
```

**Parameters:**
- `prices` (array-like): Price data (typically close)
- `period` (int): RSI period (default: 14)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** NumPy array of RSI values (0-100 range)

**Example:**
```python
rsi_14 = kf.calculate_rsi(df['close'], period=14, engine='gpu')
```

**Formula:**
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

**Performance:**
- < 100K rows: CPU optimal
- 100K-1M rows: GPU 1.5-2.1x speedup
- 1M+ rows: GPU strong benefit

---

#### `calculate_macd()` - Moving Average Convergence Divergence

```python
kimsfinance.calculate_macd(
    prices,
    fast_period=12,
    slow_period=26,
    signal_period=9,
    *,
    engine="auto"
)
```

**Parameters:**
- `prices` (array-like): Price data (typically close)
- `fast_period` (int): Fast EMA period (default: 12)
- `slow_period` (int): Slow EMA period (default: 26)
- `signal_period` (int): Signal line period (default: 9)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** Tuple of `(macd_line, signal_line, histogram)`

**Example:**
```python
macd, signal, histogram = kf.calculate_macd(df['close'], engine='gpu')
```

**Formula:**
```
MACD = EMA(fast) - EMA(slow)
Signal = EMA(MACD, signal_period)
Histogram = MACD - Signal
```

---

#### `calculate_stochastic_oscillator()` - Stochastic %K and %D

```python
kimsfinance.calculate_stochastic_oscillator(
    highs,
    lows,
    closes,
    period=14,
    smooth_k=3,
    smooth_d=3,
    *,
    engine="auto"
)
```

**Parameters:**
- `highs` (array-like): High prices
- `lows` (array-like): Low prices
- `closes` (array-like): Close prices
- `period` (int): Lookback period (default: 14)
- `smooth_k` (int): %K smoothing (default: 3)
- `smooth_d` (int): %D smoothing (default: 3)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** Tuple of `(k_values, d_values)` (0-100 range)

**Example:**
```python
k, d = kf.calculate_stochastic_oscillator(
    df['high'],
    df['low'],
    df['close'],
    period=14,
    engine='gpu'
)
```

**GPU Performance:** 2.0-2.9x speedup on large datasets

---

#### `calculate_williams_r()` - Williams %R

```python
kimsfinance.calculate_williams_r(
    highs,
    lows,
    closes,
    period=14,
    *,
    engine="auto"
)
```

**Returns:** NumPy array of Williams %R values (-100 to 0 range)

---

#### `calculate_cci()` - Commodity Channel Index

```python
kimsfinance.calculate_cci(
    highs,
    lows,
    closes,
    period=20,
    constant=0.015,
    *,
    engine="auto"
)
```

**Returns:** NumPy array of CCI values

---

#### `calculate_roc()` - Rate of Change

```python
kimsfinance.calculate_roc(prices, period=12, *, engine="auto")
```

**Returns:** NumPy array of ROC values (percentage)

---

#### `calculate_tsi()` - True Strength Index

```python
kimsfinance.calculate_tsi(
    prices,
    long_period=25,
    short_period=13,
    signal_period=13,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(tsi_values, signal_values)`

---

### Volatility Indicators

#### `calculate_atr()` - Average True Range

```python
kimsfinance.calculate_atr(
    highs,
    lows,
    closes,
    period=14,
    *,
    engine="auto"
)
```

**Parameters:**
- `highs` (array-like): High prices
- `lows` (array-like): Low prices
- `closes` (array-like): Close prices
- `period` (int): ATR period (default: 14)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** NumPy array of ATR values

**Example:**
```python
atr_14 = kf.calculate_atr(
    df['high'],
    df['low'],
    df['close'],
    period=14,
    engine='gpu'
)
```

**Formula:**
```
TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
ATR = Wilder's smoothing of TR over period
```

**Performance:**
- < 100K rows: CPU optimal (0.5-3ms)
- 100K-1M rows: GPU 1.1-1.3x speedup
- 1M+ rows: GPU 1.5x speedup

---

#### `calculate_bollinger_bands()` - Bollinger Bands

```python
kimsfinance.calculate_bollinger_bands(
    prices,
    period=20,
    num_std=2.0,
    *,
    engine="auto"
)
```

**Parameters:**
- `prices` (array-like): Price data
- `period` (int): SMA period (default: 20)
- `num_std` (float): Standard deviations (default: 2.0)
- `engine` (str): `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** Tuple of `(upper_band, middle_band, lower_band)`

**Example:**
```python
upper, middle, lower = kf.calculate_bollinger_bands(
    df['close'],
    period=20,
    num_std=2.0,
    engine='gpu'
)
```

**Formula:**
```
Middle Band = SMA(period)
Upper Band = Middle Band + (num_std * std_dev)
Lower Band = Middle Band - (num_std * std_dev)
```

---

#### `calculate_keltner_channels()` - Keltner Channels

```python
kimsfinance.calculate_keltner_channels(
    highs,
    lows,
    closes,
    period=20,
    multiplier=2.0,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(upper_channel, middle_channel, lower_channel)`

---

#### `calculate_donchian_channels()` - Donchian Channels

```python
kimsfinance.calculate_donchian_channels(
    highs,
    lows,
    period=20,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(upper_channel, middle_channel, lower_channel)`

---

### Volume Indicators

#### `calculate_obv()` - On-Balance Volume

```python
kimsfinance.calculate_obv(closes, volumes, *, engine="auto")
```

**Returns:** NumPy array of OBV values

---

#### `calculate_vwap()` - Volume Weighted Average Price

```python
kimsfinance.calculate_vwap(
    highs,
    lows,
    closes,
    volumes,
    *,
    engine="auto"
)
```

**Returns:** NumPy array of VWAP values

**Example:**
```python
vwap = kf.calculate_vwap(
    df['high'],
    df['low'],
    df['close'],
    df['volume'],
    engine='gpu'
)
```

---

#### `calculate_cmf()` - Chaikin Money Flow

```python
kimsfinance.calculate_cmf(
    highs,
    lows,
    closes,
    volumes,
    period=20,
    *,
    engine="auto"
)
```

**Returns:** NumPy array of CMF values (-1 to 1 range)

---

#### `calculate_mfi()` - Money Flow Index

```python
kimsfinance.calculate_mfi(
    highs,
    lows,
    closes,
    volumes,
    period=14,
    *,
    engine="auto"
)
```

**Returns:** NumPy array of MFI values (0-100 range)

---

### Trend Indicators

#### `calculate_adx()` - Average Directional Index

```python
kimsfinance.calculate_adx(
    highs,
    lows,
    closes,
    period=14,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(adx, plus_di, minus_di)`

---

#### `calculate_ichimoku()` - Ichimoku Cloud

```python
kimsfinance.calculate_ichimoku(
    highs,
    lows,
    closes,
    conversion_period=9,
    base_period=26,
    span_b_period=52,
    displacement=26,
    *,
    engine="auto"
)
```

**Returns:** Dict with keys: `'tenkan'`, `'kijun'`, `'senkou_a'`, `'senkou_b'`, `'chikou'`

---

#### `calculate_supertrend()` - SuperTrend

```python
kimsfinance.calculate_supertrend(
    highs,
    lows,
    closes,
    period=10,
    multiplier=3.0,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(supertrend, direction)`

---

#### `calculate_parabolic_sar()` - Parabolic SAR

```python
kimsfinance.calculate_parabolic_sar(
    highs,
    lows,
    acceleration=0.02,
    maximum=0.2,
    *,
    engine="auto"
)
```

**Returns:** NumPy array of SAR values

---

#### `calculate_aroon()` - Aroon Indicator

```python
kimsfinance.calculate_aroon(
    highs,
    lows,
    period=25,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(aroon_up, aroon_down, aroon_oscillator)`

---

### Support/Resistance

#### `calculate_pivot_points()` - Pivot Points

```python
kimsfinance.calculate_pivot_points(
    highs,
    lows,
    closes,
    method="standard",
    *,
    engine="auto"
)
```

**Parameters:**
- `method` (str): `"standard"`, `"fibonacci"`, `"woodie"`, `"camarilla"`, `"demark"`

**Returns:** Dict with keys: `'pivot'`, `'r1'`, `'r2'`, `'r3'`, `'s1'`, `'s2'`, `'s3'`

---

#### `calculate_fibonacci_retracement()` - Fibonacci Retracement

```python
kimsfinance.calculate_fibonacci_retracement(
    highs,
    lows,
    *,
    engine="auto"
)
```

**Returns:** Dict with Fibonacci levels: `'0'`, `'0.236'`, `'0.382'`, `'0.5'`, `'0.618'`, `'1'`

---

#### `calculate_volume_profile()` - Volume Profile

```python
kimsfinance.calculate_volume_profile(
    closes,
    volumes,
    num_bins=20,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(price_levels, volume_distribution)`

---

#### `calculate_elder_ray()` - Elder Ray Index

```python
kimsfinance.calculate_elder_ray(
    highs,
    lows,
    closes,
    period=13,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(bull_power, bear_power)`

---

#### `find_swing_points()` - Swing Highs/Lows

```python
kimsfinance.find_swing_points(
    highs,
    lows,
    left_bars=5,
    right_bars=5,
    *,
    engine="auto"
)
```

**Returns:** Tuple of `(swing_highs, swing_lows)` (boolean arrays)

---

### Advanced Indicators

#### `calculate_dema()` - Double Exponential Moving Average

```python
kimsfinance.calculate_dema(prices, period=20, *, engine="auto")
```

---

#### `calculate_tema()` - Triple Exponential Moving Average

```python
kimsfinance.calculate_tema(prices, period=20, *, engine="auto")
```

---

#### `calculate_hma()` - Hull Moving Average

```python
kimsfinance.calculate_hma(prices, period=20, *, engine="auto")
```

---

### Engine Selection for Indicators

All indicators support automatic CPU/GPU engine selection:

```python
# Auto-select based on data size (recommended)
result = kf.calculate_rsi(prices, period=14, engine='auto')

# Force CPU (predictable performance)
result = kf.calculate_rsi(prices, period=14, engine='cpu')

# Force GPU (requires CUDA, raises error if unavailable)
result = kf.calculate_rsi(prices, period=14, engine='gpu')
```

**Auto Selection Thresholds:**
- < 100K rows: CPU (lower overhead)
- 100K-1M rows: GPU (moderate benefit)
- 1M+ rows: GPU (strong benefit)

---

## Batch Rendering

### Sequential Batch Rendering

Render multiple charts with optimized batch drawing:

```python
from kimsfinance.plotting import render_ohlcv_charts

# Prepare datasets
datasets = [
    {'ohlc': ohlc_dict_1, 'volume': volume_1},
    {'ohlc': ohlc_dict_2, 'volume': volume_2},
    {'ohlc': ohlc_dict_3, 'volume': volume_3},
]

# Render all charts
images = render_ohlcv_charts(
    datasets,
    width=1920,
    height=1080,
    theme='tradingview',
    enable_antialiasing=True
)

# Save each chart
for i, img in enumerate(images):
    img.save(f'chart_{i}.webp', 'WEBP', quality=85)
```

**Performance:** 20-30% speedup over individual rendering due to batch optimizations.

---

### Parallel Batch Rendering

Render multiple charts in parallel using multiprocessing:

```python
from kimsfinance.plotting import render_charts_parallel

# Prepare datasets
datasets = [
    {'ohlc': ohlc_dict_1, 'volume': volume_1},
    {'ohlc': ohlc_dict_2, 'volume': volume_2},
    {'ohlc': ohlc_dict_3, 'volume': volume_3},
    # ... up to thousands of charts
]

# Output paths
output_paths = [f'chart_{i}.webp' for i in range(len(datasets))]

# Render in parallel (uses all CPU cores)
results = render_charts_parallel(
    datasets,
    output_paths=output_paths,
    num_workers=None,  # Auto-detect CPU count
    speed='fast',      # WebP encoding speed
    theme='tradingview',
    width=1920,
    height=1080
)

print(f"Rendered {len(results)} charts")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `datasets` | list[dict] | Required | List of dicts with `'ohlc'` and `'volume'` keys |
| `output_paths` | list[str] | `None` | Output file paths (if None, returns PNG bytes) |
| `num_workers` | int | `None` | Parallel workers (None = CPU count) |
| `speed` | str | `"fast"` | WebP encoding speed preset |
| `**common_render_kwargs` | dict | - | Common rendering options for all charts |

#### Performance

- **Efficiency:** Efficient for >10 charts or rendering time >100ms/chart
- **Scalability:** Linear scaling with CPU cores
- **Memory:** Each worker needs ~100-200MB RAM
- **Overhead:** ~100ms startup per worker process

**Example Performance:**
- 1000 charts on 24-core CPU: ~21 seconds (6,000+ charts/minute)
- Sequential: ~160 seconds for same workload

---

## Output Formats

kimsfinance supports multiple output formats with optimized encoding:

### WebP (Recommended)

**Best performance** - 61x faster encoding than PNG, 79% smaller files.

```python
kf.plot(df, savefig='chart.webp', speed='fast', quality=85)
```

#### Speed Presets

| Preset | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| `"fast"` | 80-85 | 61x faster | Batch processing, ML pipelines |
| `"balanced"` | 85-90 | 30x faster | General use (default) |
| `"best"` | 95-100 | 10x faster | Archival, presentation |

#### Custom Quality

```python
# Maximum quality (slowest)
kf.plot(df, savefig='chart.webp', quality=100)

# Balanced (default)
kf.plot(df, savefig='chart.webp', quality=85)

# Fast (batch processing)
kf.plot(df, savefig='chart.webp', quality=80)
```

**File Sizes:**
- Fast preset: ~0.5 KB (79% smaller than PNG)
- Balanced: ~0.7 KB
- Best: ~1.2 KB

---

### PNG

Standard PNG format with lossless compression.

```python
kf.plot(df, savefig='chart.png')
```

**Performance:** Slower encoding (~1,331ms per image)
**File Size:** ~2.6 KB (larger than WebP)

---

### JPEG

Lossy compression for smaller files (not recommended for charts).

```python
kf.plot(df, savefig='chart.jpg', quality=90)
```

**Warning:** JPEG compression artifacts may degrade chart quality.

---

### SVG (Vector)

Scalable vector graphics for infinite zoom.

```python
kf.plot(df, savefig='chart.svg')
```

**Supported Chart Types:**
- Candlestick
- OHLC Bars
- Line Chart
- Hollow Candles
- Renko
- Point & Figure

**Benefits:**
- Infinite scaling
- Small file size
- Editable in vector editors

**Limitations:**
- Slower rendering than raster formats
- Not all chart types supported (falls back to PNG)

---

### SVGZ (Compressed SVG)

Gzip-compressed SVG for smaller file sizes.

```python
kf.plot(df, savefig='chart.svgz')
```

**File Size:** ~60-80% smaller than uncompressed SVG

---

### In-Memory (NumPy Array)

Get chart as NumPy array for ML pipelines:

```python
from kimsfinance.plotting import render_to_array

array = render_to_array(ohlc_dict, volume_array, width=224, height=224)
# Returns: NumPy array (H, W, 3) uint8 RGB
```

**Use Cases:**
- Machine learning feature extraction
- Computer vision pipelines
- Real-time streaming

---

## GPU Acceleration

kimsfinance provides optional GPU acceleration for OHLCV processing and technical indicators.

### Check GPU Availability

```python
import kimsfinance as kf

# Check if GPU is available
if kf.gpu_available():
    print("GPU acceleration enabled!")
else:
    print("Running on CPU only")

# Get detailed engine info
info = kf.get_engine_info()
print(info)
# {
#     'cpu_available': True,
#     'gpu_available': True,
#     'cudf_version': '25.10.00',
#     'default_engine': 'auto'
# }
```

---

### Configure GPU Acceleration

```python
# Activate GPU acceleration globally
kf.activate(engine='auto', verbose=True)

# Configure thresholds
kf.configure(
    default_engine='auto',  # 'cpu', 'gpu', or 'auto'
    gpu_min_rows=100000,    # GPU threshold
    verbose=True
)

# Check activation status
if kf.is_active():
    config = kf.get_config()
    print(f"Default engine: {config['default_engine']}")
    print(f"GPU min rows: {config['gpu_min_rows']:,}")

# Deactivate (revert to CPU)
kf.deactivate()
```

---

### GPU-Accelerated Operations

#### OHLCV Processing (cuDF)

**6.4x speedup** for DataFrame operations:

```python
import polars as pl

# Load large dataset
df = pl.read_csv("large_ohlcv.csv")  # 1M+ rows

# Automatic GPU routing for large datasets
sma = kf.calculate_sma(df['close'], period=20, engine='auto')
# Uses GPU for >100K rows, CPU for smaller datasets
```

#### Technical Indicators (GPU)

Specific indicators with strong GPU benefit:

| Indicator | GPU Speedup | Threshold |
|-----------|-------------|-----------|
| Stochastic | 2.0-2.9x | 100K rows |
| RSI | 1.5-2.1x | 100K rows |
| ATR | 1.1-1.5x | 100K rows |
| MACD | 1.3-1.8x | 100K rows |

```python
# Force GPU for indicators
rsi = kf.calculate_rsi(prices, period=14, engine='gpu')
macd, signal, hist = kf.calculate_macd(prices, engine='gpu')
k, d = kf.calculate_stochastic_oscillator(
    highs, lows, closes,
    period=14,
    engine='gpu'
)
```

#### Linear Algebra (CuPy)

**30-50x speedup** for linear algebra operations:

```python
from kimsfinance.ops.linear_algebra import least_squares_fit

# GPU-accelerated linear regression
slope, intercept = least_squares_fit(x, y, engine='gpu')
```

---

### GPU Performance Tips

1. **Use `engine='auto'`** - Automatically selects optimal engine based on data size
2. **Batch processing** - Process multiple datasets to amortize GPU transfer overhead
3. **Monitor VRAM** - Each dataset consumes GPU memory (~100-200MB per million rows)
4. **Profile your workload** - Use `kf.get_engine_info()` to check GPU availability
5. **CPU for small datasets** - GPU overhead dominates for <100K rows

---

### GPU Hardware Requirements

**Minimum:**
- NVIDIA GPU with CUDA support
- 4GB VRAM
- CUDA 12.x

**Recommended:**
- NVIDIA RTX series (RTX 3060+, RTX 4000+)
- 8GB+ VRAM
- PCIe 4.0 x16

**Optimal:**
- NVIDIA RTX 4090 (24GB VRAM)
- NVIDIA RTX 6000 Ada (48GB VRAM)
- Data center GPUs (A100, H100)

---

## Integration Examples

### Pandas Integration

```python
import pandas as pd
import kimsfinance as kf

# Load pandas DataFrame
df = pd.read_csv("ohlcv.csv", index_col='Date', parse_dates=True)

# Convert to polars for kimsfinance (or use directly)
import polars as pl
df_polars = pl.from_pandas(df.reset_index())

# Plot candlestick chart
kf.plot(df_polars, type='candle', volume=True, savefig='chart.webp')

# Or use pandas directly (internally converts to polars)
kf.plot(df, type='candle', volume=True, savefig='chart.webp')
```

---

### Polars Integration (Native)

```python
import polars as pl
import kimsfinance as kf

# Load data with polars (native)
df = pl.read_csv("ohlcv.csv")

# Calculate indicators
rsi = kf.calculate_rsi(df['close'], period=14)
macd, signal, hist = kf.calculate_macd(df['close'])

# Add indicators to DataFrame
df = df.with_columns([
    pl.Series('rsi', rsi),
    pl.Series('macd', macd),
    pl.Series('signal', signal),
])

# Plot chart
kf.plot(df, type='candle', volume=True, savefig='chart.webp')
```

---

### Machine Learning Pipeline

```python
from kimsfinance.plotting import render_to_array
import numpy as np

# Render chart as NumPy array
chart_array = render_to_array(
    ohlc_dict,
    volume_array,
    width=224,
    height=224,
    theme='tradingview'
)

# Normalize for ML model
chart_normalized = chart_array.astype(np.float32) / 255.0

# Feed to model
model.predict(chart_normalized)
```

---

### Batch Processing Pipeline

```python
import polars as pl
import kimsfinance as kf
from pathlib import Path

# Load multiple datasets
files = Path("data/").glob("*.csv")
datasets = [pl.read_csv(f) for f in files]

# Prepare for parallel rendering
render_datasets = [
    {
        'ohlc': {
            'open': df['open'].to_numpy(),
            'high': df['high'].to_numpy(),
            'low': df['low'].to_numpy(),
            'close': df['close'].to_numpy(),
        },
        'volume': df['volume'].to_numpy()
    }
    for df in datasets
]

# Render in parallel
from kimsfinance.plotting import render_charts_parallel

output_paths = [f"output/chart_{i}.webp" for i in range(len(datasets))]
results = render_charts_parallel(
    render_datasets,
    output_paths=output_paths,
    num_workers=24,  # Use all cores
    speed='fast',
    theme='tradingview'
)

print(f"Rendered {len(results)} charts")
```

---

### Real-Time Streaming

```python
import kimsfinance as kf
from collections import deque

# Sliding window buffer
window_size = 500
buffer = deque(maxlen=window_size)

def update_chart(new_candle):
    """Update chart with new streaming candle."""
    buffer.append(new_candle)

    # Convert to OHLC dict
    ohlc = {
        'open': [c['open'] for c in buffer],
        'high': [c['high'] for c in buffer],
        'low': [c['low'] for c in buffer],
        'close': [c['close'] for c in buffer],
    }
    volume = [c['volume'] for c in buffer]

    # Render chart (returns PIL Image)
    img = kf.plot(
        {'ohlc': ohlc, 'volume': volume},
        type='candle',
        returnfig=True
    )

    # Stream to client (e.g., websocket)
    return img

# Process streaming data
for candle in stream:
    chart_img = update_chart(candle)
    send_to_client(chart_img)
```

---

## Error Handling

### Common Exceptions

#### `KimsFinanceError`

Base exception for all kimsfinance errors.

```python
from kimsfinance.core import KimsFinanceError

try:
    result = kf.calculate_rsi(prices, period=14)
except KimsFinanceError as e:
    print(f"Error: {e}")
```

---

#### `GPUNotAvailableError`

Raised when GPU is requested but not available.

```python
from kimsfinance.core import GPUNotAvailableError

try:
    result = kf.calculate_rsi(prices, period=14, engine='gpu')
except GPUNotAvailableError:
    print("GPU not available, falling back to CPU")
    result = kf.calculate_rsi(prices, period=14, engine='cpu')
```

---

#### `DataValidationError`

Raised for invalid input data.

```python
from kimsfinance.core import DataValidationError

try:
    rsi = kf.calculate_rsi(prices, period=100)  # period > data length
except DataValidationError as e:
    print(f"Invalid data: {e}")
```

---

#### `ValueError`

Standard Python ValueError for parameter validation.

```python
try:
    kf.plot(df, type='invalid_type', savefig='chart.webp')
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

---

### Error Handling Best Practices

```python
import kimsfinance as kf
from kimsfinance.core import (
    KimsFinanceError,
    GPUNotAvailableError,
    DataValidationError
)

def safe_plot(df, **kwargs):
    """Safe plotting with comprehensive error handling."""
    try:
        # Validate data
        if len(df) < 50:
            raise DataValidationError("Insufficient data (need at least 50 candles)")

        # Render chart
        kf.plot(df, **kwargs)

    except GPUNotAvailableError:
        # Fallback to CPU
        print("GPU unavailable, using CPU")
        kwargs['engine'] = 'cpu'
        kf.plot(df, **kwargs)

    except DataValidationError as e:
        print(f"Data validation failed: {e}")

    except ValueError as e:
        print(f"Invalid parameter: {e}")

    except KimsFinanceError as e:
        print(f"kimsfinance error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")

# Usage
safe_plot(df, type='candle', savefig='chart.webp')
```

---

## Performance Tips

### 1. Use WebP Fast Mode for Batch Processing

```python
# 61x faster encoding
kf.plot(df, savefig='chart.webp', speed='fast', quality=80)
```

**Speedup:** 61x faster than PNG, 79% smaller files

---

### 2. Disable Antialiasing for Speed

```python
# Faster RGB rendering (no alpha channel)
kf.plot(df, enable_antialiasing=False, savefig='chart.webp')
```

**Speedup:** 10-15% faster rendering

---

### 3. Use Parallel Rendering for Multiple Charts

```python
from kimsfinance.plotting import render_charts_parallel

results = render_charts_parallel(
    datasets,
    output_paths,
    num_workers=24,  # Use all CPU cores
    speed='fast'
)
```

**Speedup:** Linear scaling with CPU cores

---

### 4. Optimize Image Dimensions

```python
# Lower resolution for ML pipelines
kf.plot(df, width=224, height=224, savefig='chart.webp')

# Standard HD for viewing
kf.plot(df, width=1920, height=1080, savefig='chart.webp')

# 4K for presentations
kf.plot(df, width=3840, height=2160, savefig='chart.webp')
```

**Impact:** Memory usage scales with pixel count (width × height)

---

### 5. Use GPU for Large Datasets

```python
# Auto-select optimal engine
rsi = kf.calculate_rsi(prices, period=14, engine='auto')

# Force GPU for 1M+ rows
rsi = kf.calculate_rsi(prices, period=14, engine='gpu')
```

**Speedup:** 1.5-2.9x for indicators, 6.4x for OHLCV processing

---

### 6. Batch Indicator Calculations

```python
from kimsfinance.ops.indicators import calculate_multiple_mas

# Calculate multiple MAs in single pass
mas = calculate_multiple_mas(
    df,
    'close',
    sma_windows=[20, 50, 200],
    ema_windows=[12, 26],
    engine='auto'
)
```

**Speedup:** 2-3x faster than individual calculations

---

### 7. Pre-allocate Arrays for Streaming

```python
import numpy as np

# Pre-allocate buffer for real-time updates
ohlc = {
    'open': np.zeros(500, dtype=np.float64),
    'high': np.zeros(500, dtype=np.float64),
    'low': np.zeros(500, dtype=np.float64),
    'close': np.zeros(500, dtype=np.float64),
}

# Update in-place (avoid reallocation)
ohlc['close'][-1] = new_price
```

**Speedup:** Eliminates allocation overhead

---

### 8. Disable Grid for Minimal Charts

```python
# Minimal rendering (no grid)
kf.plot(df, show_grid=False, savefig='minimal.webp')
```

**Speedup:** 5-8% faster rendering

---

### 9. Use Direct-to-File API

```python
from kimsfinance.plotting import render_and_save

# One-shot render and save (no intermediate PIL Image)
render_and_save(
    ohlc_dict,
    volume_array,
    output_path='chart.webp',
    speed='fast'
)
```

**Speedup:** Eliminates PIL Image object overhead

---

### 10. Monitor Performance

```python
import time
import kimsfinance as kf

# Benchmark rendering
start = time.perf_counter()
kf.plot(df, type='candle', savefig='chart.webp', speed='fast')
elapsed = time.perf_counter() - start

print(f"Rendered in {elapsed*1000:.2f}ms")
print(f"Throughput: {1/elapsed:.0f} img/sec")
```

---

## API Version

**Current Version:** v0.1.0

### Version History

- **v0.1.0** (2025-10-22) - Initial release
  - Native PIL rendering (28.8x average speedup, up to 70.1x)
  - 6 chart types
  - 29 technical indicators
  - GPU acceleration (optional)
  - WebP fast mode (61x faster encoding)
  - Parallel batch rendering

---

## Support

### Documentation

- **API Reference:** This document
- **Performance Guide:** `/docs/PERFORMANCE.md`
- **GPU Optimization:** `/docs/GPU_OPTIMIZATION.md`
- **Migration Guide:** `/docs/MIGRATION.md`

### GitHub

- **Repository:** https://github.com/kimasplund/kimsfinance
- **Issues:** https://github.com/kimasplund/kimsfinance/issues
- **Pull Requests:** https://github.com/kimasplund/kimsfinance/pulls

### Contact

- **Email:** kim.asplund@kimasplund.com
- **Website:** https://www.kimasplund.com

---

## License

**Dual License:**

- **AGPL-3.0** - Open source projects
- **Commercial License** - Proprietary/commercial projects

See `LICENSE` and `COMMERCIAL-LICENSE.md` for details.

---

**Last Updated:** 2025-10-22
**Version:** 0.1.0
**Status:** Production Ready (Beta)


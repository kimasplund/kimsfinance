# Interactive Charting Guide

kimsfinance now supports **interactive HTML charts** using Plotly and Bokeh, in addition to the blazing-fast static PIL rendering.

## Overview

kimsfinance offers **two rendering approaches**:

### 1. Static PIL Rendering (Default - 28.8x faster)
- **Speed**: 2ms per chart (batch mode: 6,249 charts/sec)
- **File Size**: 79% smaller with WebP compression
- **Use Cases**: Batch rendering, backtesting, report generation
- **Pros**: Blazing fast, tiny files, production-ready
- **Cons**: Not interactive

### 2. Interactive Rendering (New!)
- **Backends**: Plotly and Bokeh
- **Features**: Hover tooltips, zoom/pan, crosshairs, indicator overlays
- **Speed**: ~50ms per chart (25x slower than PIL)
- **Use Cases**: Exploratory analysis, dashboards, web apps
- **Pros**: Fully interactive, beautiful, flexible
- **Cons**: Slower for batch operations

---

## Installation

### Basic Installation
```bash
pip install kimsfinance
```

### With Interactive Support
```bash
pip install kimsfinance[interactive]
# Or manually:
pip install plotly>=5.0 bokeh>=3.0 kaleido>=0.2
```

### With All Features
```bash
pip install kimsfinance[all]  # GPU + JIT + Interactive + Dev + Test
```

---

## Quick Start

### Plotly Candlestick Chart
```python
import polars as pl
from kimsfinance.plotting.interactive import plot_candlestick_plotly

# Your OHLCV data
df = pl.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'open': [100, 102, 101],
    'high': [103, 105, 104],
    'low': [99, 101, 100],
    'close': [102, 101, 103],
    'volume': [1000, 1500, 1200]
})

# Create interactive chart
chart = plot_candlestick_plotly(df, theme='tradingview', show_volume=True)

# Save to HTML
chart.save('chart.html')

# Or display in browser / Jupyter
chart.show()
```

### With Technical Indicators
```python
from kimsfinance.ops.indicators import calculate_rsi, calculate_sma

close_prices = df['close'].to_numpy()
rsi = calculate_rsi(close_prices, period=14)
sma_50 = calculate_sma(close_prices, period=50)

indicators = [
    {'data': sma_50, 'name': 'SMA(50)', 'type': 'line', 'color': '#FFA500', 'panel': 'main'},
    {'data': rsi, 'name': 'RSI(14)', 'type': 'line', 'color': '#FFD700', 'panel': 'separate'}
]

chart = plot_candlestick_plotly(df, indicators=indicators, height=1000)
chart.save('chart_with_indicators.html')
```

---

## API Reference

### Functions

#### `plot_candlestick_plotly()`
Create interactive candlestick chart using Plotly.

```python
chart = plot_candlestick_plotly(
    data: DataFrameInput,              # OHLCV data (DataFrame or dict)
    indicators: Optional[list[dict]] = None,  # Indicator overlays
    theme: Theme = 'tradingview',      # 'classic', 'modern', 'tradingview', 'light'
    title: str = 'Candlestick Chart',
    width: int = 1200,
    height: int = 800,
    show_volume: bool = True,
    show_rangeslider: bool = True,
    webgl: bool = False,               # Enable for >10K points
    date_column: Optional[str] = None  # Auto-detected if None
) -> InteractiveChart
```

**Performance Tips:**
- Use `webgl=True` for datasets >10K points
- Data decimation applied automatically for >100K points
- Plotly best for <10K points

#### `plot_candlestick_bokeh()`
Create interactive candlestick chart using Bokeh (better for large datasets).

```python
chart = plot_candlestick_bokeh(
    data: DataFrameInput,
    indicators: Optional[list[dict]] = None,
    theme: Theme = 'tradingview',
    title: str = 'Candlestick Chart',
    width: int = 1200,
    height: int = 800,
    show_volume: bool = True,
    date_column: Optional[str] = None
) -> InteractiveChart
```

**Performance Tips:**
- Better for datasets >100K points
- Server-side rendering option available
- Linked plots for synchronized zooming

#### `plot_ohlc_plotly()`
Create OHLC bar chart (horizontal ticks instead of candlesticks).

```python
chart = plot_ohlc_plotly(
    data: DataFrameInput,
    theme: Theme = 'tradingview',
    title: str = 'OHLC Chart',
    width: int = 1200,
    height: int = 800,
    date_column: Optional[str] = None
) -> InteractiveChart
```

#### `plot_line_plotly()`
Create simple line chart.

```python
chart = plot_line_plotly(
    data: DataFrameInput,
    y_column: str = 'close',
    theme: Theme = 'tradingview',
    title: str = 'Line Chart',
    width: int = 1200,
    height: int = 800,
    date_column: Optional[str] = None
) -> InteractiveChart
```

---

## InteractiveChart Methods

The `InteractiveChart` object provides these methods:

### `save(path: str)`
Save chart to HTML file.

```python
chart.save('chart.html')
```

### `show()`
Display chart in browser or Jupyter notebook.

```python
chart.show()  # Opens in default browser
```

### `to_html() -> str`
Export chart to HTML string.

```python
html_string = chart.to_html()
```

### `to_json() -> str`
Export chart to JSON (Plotly only).

```python
json_string = chart.to_json()
```

### `to_png(path: str, width: int, height: int)`
Export chart to PNG (requires kaleido).

```python
chart.to_png('chart.png', width=1920, height=1080)
```

---

## Indicator Configuration

Indicators are passed as a list of dictionaries:

### Indicator Dictionary Format
```python
indicator = {
    'data': np.array([...]),     # Indicator values (NumPy array)
    'name': 'RSI(14)',           # Display name
    'type': 'line',              # 'line', 'histogram', 'band'
    'color': '#FFD700',          # Hex color
    'panel': 'main'              # 'main' (overlay) or 'separate' (new panel)
}
```

### Indicator Types

#### Line Indicator
```python
{
    'data': sma_values,
    'name': 'SMA(50)',
    'type': 'line',
    'color': '#FFA500',
    'panel': 'main'  # Overlay on main price chart
}
```

#### Histogram Indicator
```python
{
    'data': macd_histogram,
    'name': 'MACD Histogram',
    'type': 'histogram',
    'color': '#1E90FF',
    'panel': 'separate'  # Separate panel below
}
```

#### Band Indicator (e.g., Bollinger Bands)
```python
{
    'data': bb_middle,
    'name': 'Bollinger Bands',
    'type': 'band',
    'color': '#9370DB',
    'upper': bb_upper,
    'lower': bb_lower,
    'panel': 'main'
}
```

---

## Themes

kimsfinance supports 4 themes matching the static renderer:

### 1. Classic Theme
```python
chart = plot_candlestick_plotly(df, theme='classic')
```
- Background: Black (`#000000`)
- Up candles: Bright green (`#00FF00`)
- Down candles: Bright red (`#FF0000`)
- Grid: Dark gray (`#333333`)

### 2. Modern Theme
```python
chart = plot_candlestick_plotly(df, theme='modern')
```
- Background: Dark gray (`#1E1E1E`)
- Up candles: Teal (`#26A69A`)
- Down candles: Red (`#EF5350`)
- Grid: Medium gray (`#424242`)

### 3. TradingView Theme (Default)
```python
chart = plot_candlestick_plotly(df, theme='tradingview')
```
- Background: Dark blue-gray (`#131722`)
- Up candles: Green (`#089981`)
- Down candles: Red (`#F23645`)
- Grid: Blue-gray (`#2A2E39`)

### 4. Light Theme
```python
chart = plot_candlestick_plotly(df, theme='light')
```
- Background: White (`#FFFFFF`)
- Up candles: Teal (`#26A69A`)
- Down candles: Red (`#EF5350`)
- Grid: Light gray (`#E0E0E0`)

---

## Performance Comparison

### Rendering Speed Benchmarks

| Backend | 1 Chart | 100 Charts | 1000 Charts |
|---------|---------|------------|-------------|
| **PIL (static)** | 2ms | 200ms | 2s |
| **Plotly** | 50ms | 5s | 50s |
| **Bokeh** | 40ms | 4s | 40s |

### File Size Comparison

| Format | Size (100 candles) | Size (1000 candles) |
|--------|-------------------|---------------------|
| **PIL + WebP** | 5-10 KB | 20-40 KB |
| **Plotly HTML** | 800 KB | 2-5 MB |
| **Bokeh HTML** | 600 KB | 1-3 MB |

### Recommendation Matrix

| Use Case | Recommended Backend | Reasoning |
|----------|-------------------|-----------|
| Backtesting (1000+ charts) | **PIL** | 25x faster, 79% smaller files |
| Report generation | **PIL** | Fast batch rendering |
| Exploratory analysis | **Plotly** | Interactive, beautiful |
| Dashboard/web app | **Plotly/Bokeh** | User needs interactivity |
| Real-time monitoring | **Bokeh** | Server-side rendering |
| Large datasets (>100K) | **Bokeh** | Better performance |
| Jupyter notebook | **Plotly** | Better integration |

---

## Advanced Usage

### WebGL Rendering for Large Datasets

Enable WebGL for improved performance with >10K data points:

```python
df_large = generate_sample_data(20_000)

chart = plot_candlestick_plotly(
    df_large,
    webgl=True,  # Enable WebGL
    show_volume=False,  # Reduce complexity
    show_rangeslider=True
)
```

### Data Decimation

For datasets >100K points, automatic decimation is applied:

```python
# Original: 200K points
# After decimation: ~50K points (using LTTB algorithm)
chart = plot_candlestick_plotly(df_very_large)
```

### Custom Themes

While not yet directly supported, you can modify theme colors:

```python
from kimsfinance.config.themes import THEMES

# View current themes
print(THEMES)

# Modify (not recommended - create custom theme instead)
# THEMES['custom'] = {'bg': '#...', 'up': '#...', 'down': '#...', 'grid': '#...'}
```

---

## Jupyter Integration

Interactive charts work seamlessly in Jupyter:

```python
# In Jupyter notebook
from kimsfinance.plotting.interactive import plot_candlestick_plotly

chart = plot_candlestick_plotly(df)
chart.show()  # Displays inline
```

See `notebooks/interactive_charting.ipynb` for complete examples.

---

## Export Formats

### HTML (Default)
```python
chart.save('chart.html')
```

### HTML String
```python
html_string = chart.to_html()
# Embed in web page, email, etc.
```

### JSON (Plotly only)
```python
json_string = chart.to_json()
# For API responses, storage, etc.
```

### PNG (requires kaleido)
```python
# Install: pip install kaleido
chart.to_png('chart.png', width=1920, height=1080)
```

---

## Error Handling

### Missing Dependencies

If Plotly or Bokeh is not installed:

```python
try:
    chart = plot_candlestick_plotly(df)
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install kimsfinance[interactive]")
```

### Invalid Data

If required OHLCV columns are missing:

```python
try:
    chart = plot_candlestick_plotly(df)
except ValueError as e:
    print(f"Data error: {e}")
    # Ensure 'open', 'high', 'low', 'close' columns exist
```

---

## Examples

See:
- `examples/interactive_charts.py` - Complete Python examples
- `notebooks/interactive_charting.ipynb` - Jupyter notebook tutorial

Run examples:
```bash
python examples/interactive_charts.py
```

---

## Comparison: Static vs Interactive

### When to Use Static PIL Rendering

✅ **Use Static PIL:**
- Batch rendering (100+ charts)
- Backtesting with thousands of iterations
- Report generation (PDF, email)
- Need maximum speed (28.8x faster)
- Need small file sizes (79% smaller)
- CI/CD pipelines
- Server-side rendering without interactivity

### When to Use Interactive Rendering

✅ **Use Interactive:**
- Exploratory data analysis
- Dashboard applications
- Web applications
- Real-time monitoring
- User needs zoom/pan/hover
- Single charts or small batches
- Jupyter notebook analysis
- Client presentations

### Hybrid Approach

You can use both in the same application:

```python
# Static for batch rendering
from kimsfinance.plotting import render_ohlcv_chart

for symbol in symbols:
    render_ohlcv_chart(data[symbol], f'{symbol}.webp', speed='fast')

# Interactive for exploratory analysis
from kimsfinance.plotting.interactive import plot_candlestick_plotly

chart = plot_candlestick_plotly(data['SPY'], indicators=indicators)
chart.show()
```

---

## Performance Tips

### Plotly Performance
1. **Enable WebGL** for >10K points: `webgl=True`
2. **Disable rangeslider** if not needed: `show_rangeslider=False`
3. **Reduce indicators** - each adds rendering time
4. **Use line mode** instead of markers for indicators
5. **Simplify themes** - fewer colors = faster rendering

### Bokeh Performance
1. **Use Bokeh for >100K points** - better than Plotly
2. **Enable server mode** for very large datasets
3. **Link plots** instead of duplicating data
4. **Use ColumnDataSource** for efficient updates
5. **Disable hover tooltips** if not needed

### General Tips
1. **Downsample data** before rendering (e.g., 1-minute -> 5-minute)
2. **Cache charts** if data doesn't change frequently
3. **Use batch mode** for multiple charts
4. **Lazy load** charts in web applications
5. **Use PIL for batch**, interactive for UI

---

## Troubleshooting

### Plotly Not Displaying in Jupyter

```python
# Ensure plotly is installed
pip install plotly

# Update plotly
pip install --upgrade plotly

# Reset Jupyter kernel if needed
```

### Bokeh Charts Not Rendering

```python
# Ensure bokeh is installed
pip install bokeh>=3.0

# Check browser console for JavaScript errors
```

### PNG Export Failing

```python
# Install kaleido
pip install kaleido

# Or use chromium-based export (Bokeh)
pip install selenium chromedriver-binary
```

### Memory Issues with Large Datasets

```python
# Downsample data first
df_downsampled = df[::10]  # Every 10th row

# Or use data decimation
chart = plot_candlestick_plotly(df, webgl=True)  # Auto-decimation
```

---

## Contributing

Contributions welcome! Areas for improvement:

1. Additional chart types (Heikin-Ashi, Renko with interactive)
2. More indicator types (scatter, area)
3. Custom theme builder
4. Animation support
5. Real-time streaming charts
6. Dashboard templates

See `CONTRIBUTING.md` for guidelines.

---

## License

AGPL-3.0-or-later (see `LICENSE` file)

Commercial licenses available - contact: hello@asplund.kim

---

## Resources

- **Documentation**: [GitHub README](https://github.com/kimasplund/kimsfinance)
- **Examples**: `examples/interactive_charts.py`
- **Jupyter Notebook**: `notebooks/interactive_charting.ipynb`
- **Plotly Docs**: https://plotly.com/python/
- **Bokeh Docs**: https://docs.bokeh.org/

---

**Version**: 0.1.0
**Last Updated**: 2025-11-03

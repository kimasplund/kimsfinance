# Interactive Charting Feature - Quick Start

## Overview

kimsfinance now supports **interactive HTML charts** using Plotly and Bokeh!

Choose your rendering approach:
- **Static PIL**: Blazing fast (28.8x vs mplfinance) for batch operations
- **Interactive**: Full interactivity for analysis and dashboards

## Installation

```bash
# Install with interactive support
pip install kimsfinance[interactive]

# Or install interactively dependencies separately
pip install plotly bokeh kaleido
```

## Quick Start

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

# Save or display
chart.save('chart.html')
chart.show()  # Opens in browser or displays in Jupyter
```

## Features

- ✅ **Chart Types**: Candlestick, OHLC, Line
- ✅ **Backends**: Plotly (best for <10K points), Bokeh (best for >100K)
- ✅ **Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA, etc.
- ✅ **Themes**: Classic, Modern, TradingView, Light
- ✅ **Interactivity**: Hover tooltips, zoom, pan, crosshair
- ✅ **Export**: HTML, PNG, JSON
- ✅ **Jupyter**: Works seamlessly in notebooks

## With Indicators

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

## Performance Comparison

| Backend | Time/Chart | File Size | Best For |
|---------|-----------|-----------|----------|
| **PIL (Static)** | 2ms | 5-40 KB | Batch rendering, backtesting |
| **Plotly** | 50ms | 800 KB - 5 MB | Interactive analysis, dashboards |
| **Bokeh** | 40ms | 600 KB - 3 MB | Large datasets (>100K points) |

## Resources

- **Complete Guide**: `docs/INTERACTIVE_CHARTS.md`
- **Implementation Report**: `rust/docs/INTERACTIVE_IMPLEMENTATION_REPORT.md`
- **Examples**: `examples/interactive_charts.py`
- **Jupyter Notebook**: `notebooks/interactive_charting.ipynb`
- **Tests**: `tests/plotting/test_interactive.py`
- **Benchmarks**: `benchmarks/benchmark_interactive.py`

## Run Examples

```bash
# Run all examples
python examples/interactive_charts.py

# Run tests
pytest tests/plotting/test_interactive.py -v

# Run benchmarks
python benchmarks/benchmark_interactive.py

# Open Jupyter notebook
jupyter notebook notebooks/interactive_charting.ipynb
```

## When to Use Which?

### Use Static PIL (render_ohlcv_chart):
- Batch rendering (100+ charts)
- Backtesting with thousands of iterations
- Need maximum speed (28.8x faster)
- Need small file sizes (79% smaller)

### Use Interactive (plot_candlestick_plotly/bokeh):
- Exploratory data analysis
- Dashboard applications
- Single charts or small batches
- Need hover tooltips and interactivity

### Hybrid Approach:
```python
# Static for batch (fast)
from kimsfinance.plotting import render_ohlcv_chart
for symbol in symbols:
    render_ohlcv_chart(data[symbol], f'{symbol}.webp', speed='fast')

# Interactive for exploration (slow but interactive)
from kimsfinance.plotting.interactive import plot_candlestick_plotly
chart = plot_candlestick_plotly(data['SPY'], indicators=indicators)
chart.show()
```

## API Reference

### Functions
- `plot_candlestick_plotly()` - Plotly candlestick chart
- `plot_candlestick_bokeh()` - Bokeh candlestick chart
- `plot_ohlc_plotly()` - OHLC bar chart
- `plot_line_plotly()` - Simple line chart

### InteractiveChart Methods
- `save(path)` - Save to HTML file
- `show()` - Display in browser/Jupyter
- `to_html()` - Export to HTML string
- `to_json()` - Export to JSON (Plotly only)
- `to_png(path)` - Export to PNG (requires kaleido)

## Support

For questions or issues:
- Email: hello@asplund.kim
- GitHub: https://github.com/kimasplund/kimsfinance

---

**Status**: Production Ready
**Version**: 0.1.0
**Date**: 2025-11-03

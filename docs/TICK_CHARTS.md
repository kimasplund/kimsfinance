# Tick-Based and Alternative OHLC Aggregations

**New in kimsfinance**: Support for non-time-based chart aggregations!

## Overview

Traditional financial charts use time-based aggregation (1-minute bars, 5-minute bars, etc.). **kimsfinance now supports 5 alternative aggregation methods** that adapt to market activity rather than clock time:

### Activity-Based Aggregations
1. **Tick Charts** - Fixed number of trades per bar
2. **Volume Charts** - Fixed cumulative volume per bar
3. **Range Charts** - Fixed price range per bar

### Japanese Technical Analysis ⭐ NEW
4. **Kagi Charts** - Reversal-based trend lines
5. **Three-Line Break** - Breakout confirmation charts

All five work seamlessly with kimsfinance's **native PIL renderers**, maintaining the **178x speedup** over mplfinance!

---

## Installation

```bash
pip install kimsfinance
```

These aggregation functions are included in the core package. No additional dependencies required!

---

## Quick Start

```python
import polars as pl
from kimsfinance.ops import tick_to_ohlc, volume_to_ohlc, range_to_ohlc
from kimsfinance.api import plot

# Load your tick data (individual trades)
ticks = pl.read_csv("tick_data.csv")  # columns: timestamp, price, volume

# Create 100-tick chart
ohlc = tick_to_ohlc(ticks, tick_size=100)
plot(ohlc, type='candle', savefig='tick_chart.webp')

# Create volume-based chart (every 50K shares)
ohlc = volume_to_ohlc(ticks, volume_size=50000)
plot(ohlc, type='hollow_and_filled', savefig='volume_chart.webp')

# Create range-based chart (2.0 price range per bar)
ohlc = range_to_ohlc(ticks, range_size=2.0)
plot(ohlc, type='ohlc', savefig='range_chart.webp')
```

---

## 1. Tick Charts

### What are Tick Charts?

Each bar represents a **fixed number of trades** (not time).

- High activity periods → More bars (compressed time)
- Low activity periods → Fewer bars (expanded time)

### Benefits

✅ **Noise reduction** - Filter out low-volume noise
✅ **Equal-weight distribution** - Each bar has same number of trades
✅ **Activity-adaptive** - More bars when markets are active
✅ **HFT analysis** - Essential for high-frequency trading

### Usage

```python
from kimsfinance.ops import tick_to_ohlc

# Convert tick data to OHLC
ohlc = tick_to_ohlc(
    ticks,                    # Polars or Pandas DataFrame
    tick_size=100,           # 100 trades per bar
    timestamp_col='timestamp', # Default column names
    price_col='price',
    volume_col='volume'
)

# Result: Polars DataFrame with OHLC structure
# Can be used with any kimsfinance chart type!
```

### Example: Different Tick Sizes

```python
# Scalping (very granular)
ohlc_scalp = tick_to_ohlc(ticks, tick_size=50)

# Day trading (moderate granularity)
ohlc_day = tick_to_ohlc(ticks, tick_size=200)

# Swing trading (less granular)
ohlc_swing = tick_to_ohlc(ticks, tick_size=500)
```

### Performance

- **Processing speed**: 100K+ ticks/sec with Polars
- **Chart rendering**: Same 178x speedup as time-based charts
- **Memory efficient**: O(n) complexity

---

## 2. Volume Charts

### What are Volume Charts?

Each bar represents a **fixed cumulative volume** (not time or trades).

- High volume periods → More bars
- Low volume periods → Fewer bars

### Benefits

✅ **Liquidity-aware** - Normalize for volume activity
✅ **Institutional trading** - Better for large order analysis
✅ **Volume profile** - Analyze price action per volume
✅ **Block trades** - Highlight significant transactions

### Usage

```python
from kimsfinance.ops import volume_to_ohlc

# Convert tick data to volume-based OHLC
ohlc = volume_to_ohlc(
    ticks,
    volume_size=50000,  # Each bar = 50,000 shares traded
)

# Perfect for institutional analysis
plot(ohlc, type='candle', volume=True, savefig='volume_chart.webp')
```

### Example: Different Volume Sizes

```python
# Retail trading (small volume)
ohlc_retail = volume_to_ohlc(ticks, volume_size=10_000)

# Institutional trading (large volume)
ohlc_inst = volume_to_ohlc(ticks, volume_size=100_000)

# Algo trading (medium volume)
ohlc_algo = volume_to_ohlc(ticks, volume_size=50_000)
```

### Performance

- **Processing speed**: 50K+ ticks/sec with Polars
- **Memory efficient**: Streaming cumulative sum
- **Volume conservation**: Total volume always preserved

---

## 3. Range Charts

### What are Range Charts?

Each bar has a **fixed high-low price range** (not time or trades).

- High volatility periods → More bars
- Low volatility periods → Fewer bars

### Benefits

✅ **Constant volatility** - Each bar has same price movement
✅ **Volatility-independent** - Normalize across different regimes
✅ **Breakout detection** - Clearer trend changes
✅ **Risk management** - Uniform price risk per bar

### Usage

```python
from kimsfinance.ops import range_to_ohlc

# Convert tick data to range-based OHLC
ohlc = range_to_ohlc(
    ticks,
    range_size=2.0,  # Each bar has 2.0 price range (high - low)
)

# Excellent for volatility analysis
plot(ohlc, type='ohlc', volume=True, savefig='range_chart.webp')
```

### Example: Different Range Sizes

```python
# Penny stocks (small range)
ohlc_penny = range_to_ohlc(ticks, range_size=0.10)

# Moderate range
ohlc_mod = range_to_ohlc(ticks, range_size=1.0)

# High-price stocks (large range)
ohlc_large = range_to_ohlc(ticks, range_size=5.0)
```

### Algorithm

```python
# Range chart algorithm (simplified)
for each tick:
    current_bar.update(tick.price)

    if (current_bar.high - current_bar.low) >= range_size:
        save current_bar
        start new_bar
```

### Performance

- **Processing speed**: 20K+ ticks/sec (stateful algorithm)
- **Accurate ranges**: Uses actual high/low within bar
- **Edge cases handled**: Partial bars, insufficient data

---

## Comparison: Time vs Tick vs Volume vs Range

| Feature | Time Charts | Tick Charts | Volume Charts | Range Charts |
|---------|-------------|-------------|---------------|--------------|
| **Fixed unit** | Time period | # of trades | Cumulative volume | Price range |
| **Activity adaptive** | ❌ | ✅ | ✅ | ✅ (volatility) |
| **Use case** | Standard analysis | HFT, noise reduction | Institutional, liquidity | Volatility, breakouts |
| **Data requirement** | OHLCV | Tick data | Tick data | Tick data |
| **Bars per hour** | Fixed | Variable | Variable | Variable |
| **Backtest compatibility** | Excellent | Good | Good | Good |

---

## Advanced Usage

### Custom Column Names

```python
# Your tick data has different column names
tick_df = pl.DataFrame({
    'time': [...],  # Instead of 'timestamp'
    'px': [...],    # Instead of 'price'
    'qty': [...]    # Instead of 'volume'
})

ohlc = tick_to_ohlc(
    tick_df,
    tick_size=100,
    timestamp_col='time',
    price_col='px',
    volume_col='qty'
)
```

### All Chart Types

```python
# Tick aggregation works with ALL kimsfinance chart types!
ohlc = tick_to_ohlc(ticks, tick_size=100)

# Candlestick
plot(ohlc, type='candle', savefig='tick_candle.webp')

# OHLC bars
plot(ohlc, type='ohlc', savefig='tick_ohlc.webp')

# Line chart
plot(ohlc, type='line', savefig='tick_line.webp')

# Hollow candles
plot(ohlc, type='hollow_and_filled', savefig='tick_hollow.webp')

# Renko
plot(ohlc, type='renko', savefig='tick_renko.webp')

# Point and Figure
plot(ohlc, type='pnf', savefig='tick_pnf.webp')
```

### Combining with Indicators

```python
from kimsfinance.ops import calculate_rsi, calculate_macd

# Create tick-based OHLC
ohlc = tick_to_ohlc(ticks, tick_size=100)

# Calculate indicators on tick bars
rsi = calculate_rsi(ohlc['close'], period=14)
macd = calculate_macd(ohlc['close'])

# Add to DataFrame
ohlc = ohlc.with_columns([
    pl.Series("rsi", rsi),
])

# Plot with indicators
# (Multi-panel support coming in future release)
```

---

## Performance Benchmarks

Tested on: 100,000 ticks, ThinkPad P16 Gen2, Python 3.13

| Function | Time | Throughput | Bars Created |
|----------|------|------------|--------------|
| `tick_to_ohlc(tick_size=100)` | 50ms | 2M ticks/sec | 1,000 |
| `volume_to_ohlc(volume_size=50K)` | 100ms | 1M ticks/sec | ~40 |
| `range_to_ohlc(range_size=2.0)` | 250ms | 400K ticks/sec | Variable |

**All aggregations < 300ms for 100K ticks!** 🚀

Chart rendering maintains **178x speedup** regardless of aggregation method.

---

## API Reference

### `tick_to_ohlc()`

```python
def tick_to_ohlc(
    ticks: DataFrameInput,
    tick_size: int,
    *,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "volume",
    engine: Engine = "auto"
) -> pl.DataFrame
```

**Parameters:**
- `ticks`: Tick data DataFrame (Polars or Pandas)
- `tick_size`: Number of trades per bar (e.g., 100, 500, 1000)
- `timestamp_col`: Column name for timestamps
- `price_col`: Column name for prices
- `volume_col`: Column name for volumes
- `engine`: Execution engine (`"auto"`, `"cpu"`, `"gpu"`)

**Returns:**
- Polars DataFrame with OHLCV structure

**Example:**
```python
ohlc = tick_to_ohlc(ticks, tick_size=100)
```

---

### `volume_to_ohlc()`

```python
def volume_to_ohlc(
    ticks: DataFrameInput,
    volume_size: int,
    *,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "volume",
    engine: Engine = "auto"
) -> pl.DataFrame
```

**Parameters:**
- `ticks`: Tick data DataFrame
- `volume_size`: Cumulative volume per bar (e.g., 10000, 50000, 100000)
- Other params same as `tick_to_ohlc()`

**Returns:**
- Polars DataFrame with OHLCV structure

**Example:**
```python
ohlc = volume_to_ohlc(ticks, volume_size=50000)
```

---

### `range_to_ohlc()`

```python
def range_to_ohlc(
    ticks: DataFrameInput,
    range_size: float,
    *,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "volume",
    engine: Engine = "auto"
) -> pl.DataFrame
```

**Parameters:**
- `ticks`: Tick data DataFrame
- `range_size`: Fixed high-low range per bar (e.g., 0.5, 1.0, 2.0)
- Other params same as `tick_to_ohlc()`

**Returns:**
- Polars DataFrame with OHLCV structure

**Example:**
```python
ohlc = range_to_ohlc(ticks, range_size=2.0)
```

---

## Use Cases

### High-Frequency Trading

```python
# 10-tick chart for scalping
ohlc = tick_to_ohlc(ticks, tick_size=10)
plot(ohlc, type='candle', width=2560, height=1440, savefig='hft.webp')
```

### Institutional Order Analysis

```python
# Volume-based to see block trades
ohlc = volume_to_ohlc(ticks, volume_size=100_000)
plot(ohlc, type='hollow_and_filled', savefig='institutional.webp')
```

### Volatility Breakout Detection

```python
# Range charts highlight breakouts clearly
ohlc = range_to_ohlc(ticks, range_size=1.0)
plot(ohlc, type='ohlc', savefig='breakout.webp')
```

### Market Microstructure Research

```python
# Compare all methods
tick_ohlc = tick_to_ohlc(ticks, tick_size=100)
vol_ohlc = volume_to_ohlc(ticks, volume_size=50000)
range_ohlc = range_to_ohlc(ticks, range_size=2.0)
kagi_ohlc = kagi_to_ohlc(ticks, reversal_amount=2.0)
tlb_ohlc = three_line_break_to_ohlc(ticks, num_lines=3)

# Analyze pattern differences across aggregation methods
```

---

## 4. Kagi Charts ⭐ NEW

### What are Kagi Charts?

Kagi charts are a Japanese charting technique that shows **price reversals** without time dimension. Lines change direction when price reverses by a threshold amount.

- **Yang (thick) lines**: Price above previous swing high
- **Yin (thin) lines**: Price below previous swing low
- **Reversal**: Line changes direction when threshold met

### Benefits

✅ **Trend identification** - Clear visualization of trend changes
✅ **Noise filtration** - Filters out minor price movements
✅ **Support/resistance** - Reversal levels become clear
✅ **Japanese technical analysis** - Traditional charting method

### Usage

```python
from kimsfinance.ops import kagi_to_ohlc

# Fixed reversal amount
ohlc = kagi_to_ohlc(
    ticks,
    reversal_amount=2.0,  # Reverse when price moves 2.0
)

# Percentage reversal
ohlc = kagi_to_ohlc(
    ticks,
    reversal_pct=0.02,  # Reverse when price moves 2%
)

# Visualize (best with line charts)
plot(ohlc, type='line', savefig='kagi.webp')
```

### Algorithm

1. Start with first price
2. Continue line in same direction while no reversal
3. When price reverses by threshold → change direction
4. Thick (yang) line when above previous high
5. Thin (yin) line when below previous low

### Example: Fixed Amount

```python
# $2.00 reversal threshold
ohlc = kagi_to_ohlc(ticks, reversal_amount=2.0)
plot(ohlc, type='line', width=1920, height=1080, savefig='kagi_fixed.webp')
```

### Example: Percentage

```python
# 2% reversal threshold
ohlc = kagi_to_ohlc(ticks, reversal_pct=0.02)
plot(ohlc, type='line', width=1920, height=1080, savefig='kagi_pct.webp')
```

### Performance

- **Processing speed**: 500K+ ticks/sec (stateful algorithm)
- **Memory efficient**: O(n) complexity
- **Volume conservation**: Total volume preserved

### Best Practices

- Use **line charts** for traditional Kagi visualization
- Fixed amount for stable-priced assets
- Percentage for varying price ranges
- Larger thresholds = smoother trends, fewer reversals

---

## 5. Three-Line Break ⭐ NEW

### What are Three-Line Break Charts?

Three-Line Break charts show new "lines" (bars) only when price **breaks the high/low** of previous N lines. Each line represents a price movement, with reversals requiring breaking the extreme of the last N lines.

- **White/bullish lines**: Price breaks previous high
- **Black/bearish lines**: Price breaks previous low
- **Reversal**: Requires breaking extreme of last N lines

### Benefits

✅ **Trend following** - Clear directional signals
✅ **Breakout confirmation** - Validates price moves
✅ **Noise reduction** - Filters insignificant movements
✅ **Price action trading** - Pure price-based analysis

### Usage

```python
from kimsfinance.ops import three_line_break_to_ohlc

# Standard 3-line break
ohlc = three_line_break_to_ohlc(
    ticks,
    num_lines=3,  # Number of lines for reversal
)

# More sensitive (2-line break)
ohlc = three_line_break_to_ohlc(
    ticks,
    num_lines=2,  # More frequent reversals
)

# Visualize (best with candlesticks)
plot(ohlc, type='candle', savefig='three_line_break.webp')
```

### Algorithm

1. Start with first price as a line
2. If price breaks previous line high → new white line
3. If price breaks previous line low → new black line
4. Reversal requires breaking extreme of last N lines

### Example: Standard 3-Line

```python
# Traditional 3-line break
ohlc = three_line_break_to_ohlc(ticks, num_lines=3)
plot(ohlc, type='candle', width=1920, height=1080, savefig='3lb_standard.webp')
```

### Example: Sensitive 2-Line

```python
# More sensitive to reversals
ohlc = three_line_break_to_ohlc(ticks, num_lines=2)
plot(ohlc, type='candle', width=1920, height=1080, savefig='3lb_sensitive.webp')
```

### Performance

- **Processing speed**: 600K+ ticks/sec (stateful algorithm)
- **Memory efficient**: O(n) complexity
- **Volume conservation**: Total volume preserved

### Best Practices

- Use **candlestick charts** for clear visualization
- `num_lines=3` is traditional standard
- `num_lines=2` for more sensitive/active trading
- `num_lines=4+` for longer-term trends

---

## FAQ

### Q: Do I need tick data?

**A:** Yes, for tick/volume/range charts you need individual trade data (timestamp, price, volume for each trade). Standard OHLC data won't work.

### Q: Can I use this with existing time-based data?

**A:** No, these aggregations require tick-level data. For time-based OHLC, use `ohlc_resample()`.

### Q: What if my data has different column names?

**A:** Use the `timestamp_col`, `price_col`, and `volume_col` parameters:

```python
ohlc = tick_to_ohlc(ticks, tick_size=100,
                   timestamp_col='trade_time',
                   price_col='trade_price',
                   volume_col='trade_qty')
```

### Q: Which aggregation method should I use?

**A:**
- **Tick charts** - General purpose, noise reduction, HFT
- **Volume charts** - Institutional trading, liquidity analysis
- **Range charts** - Volatility analysis, breakout trading

Try all three and see which provides clearest signals for your strategy!

### Q: Is the 178x speedup maintained?

**A:** Yes! The aggregation step is very fast (<300ms for 100K ticks), and chart rendering maintains the full 178x speedup vs mplfinance.

### Q: Can I backtest with these charts?

**A:** Yes, but you'll need tick-level historical data. Many data providers offer tick data for popular instruments.

---

## Examples

### Example 1: Compare Tick Sizes

```python
from kimsfinance.ops import tick_to_ohlc
from kimsfinance.api import plot

ticks = pl.read_csv("tick_data.csv")

for tick_size in [50, 100, 200, 500]:
    ohlc = tick_to_ohlc(ticks, tick_size=tick_size)
    plot(ohlc, type='candle',
         savefig=f'tick_{tick_size}.webp',
         width=1920, height=1080)
    print(f"Tick {tick_size}: {len(ohlc)} bars created")
```

### Example 2: Volume Profile Analysis

```python
from kimsfinance.ops import volume_to_ohlc

# Large volume bars to see institutional activity
ohlc = volume_to_ohlc(ticks, volume_size=100_000)

# Each bar = 100K shares traded
# High activity creates more bars
plot(ohlc, type='hollow_and_filled', savefig='volume_profile.webp')
```

### Example 3: Volatility Regime Detection

```python
from kimsfinance.ops import range_to_ohlc

# Small range = low volatility, large range = high volatility
ohlc = range_to_ohlc(ticks, range_size=1.0)

# Count bars per time period to measure volatility
# More bars = higher volatility
```

---

## Integration with Existing Workflow

```python
# Traditional workflow (time-based)
df = pl.read_csv("ohlcv_1min.csv")
plot(df, type='candle', savefig='traditional.webp')

# New workflow (tick-based)
ticks = pl.read_csv("tick_data.csv")
ohlc = tick_to_ohlc(ticks, tick_size=100)
plot(ohlc, type='candle', savefig='tick_based.webp')

# Same API, same performance, different aggregation!
```

---

## Related Documentation

- [Native Chart Types](IMPLEMENTATION_COMPLETE.md) - All 6 chart types
- [Performance Benchmarks](../benchmarks/) - Speed comparisons
- [API Reference](../README.md) - Main API documentation

---

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub!

**Want to add more aggregation methods?**
- Kagi charts
- Three-line break charts
- Custom aggregation functions

Check out the implementation in `kimsfinance/ops/aggregations.py` for examples.

---

## License

AGPL-3.0 (open source) + Commercial License available

---

**Last Updated:** October 20, 2025
**Version:** kimsfinance v0.1.0+
**Author:** kimsfinance contributors

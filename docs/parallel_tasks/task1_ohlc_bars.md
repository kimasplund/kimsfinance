# Task 1: Implement OHLC Bars Renderer

## Objective
Implement native PIL-based OHLC bars chart renderer achieving 178x speedup vs mplfinance.

## Visual Specification

OHLC (Open-High-Low-Close) bar consists of:
- **Vertical line**: From high to low price
- **Left tick**: Open price (horizontal line extending left from vertical)
- **Right tick**: Close price (horizontal line extending right from vertical)

**Colors**:
- Bullish (close ≥ open): Green/up_color
- Bearish (close < open): Red/down_color

## Implementation Location
**File**: `kimsfinance/plotting/renderer.py`
**Function**: `render_ohlc_bars(ohlc, volume, width, height, theme, **kwargs)`

## Function Signature
```python
def render_ohlc_bars(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    enable_antialiasing: bool = True,
    show_grid: bool = True,
) -> Image.Image:
    """
    Render OHLC bars chart with volume using native PIL.

    Args:
        ohlc: Dict with 'open', 'high', 'low', 'close' arrays
        volume: Volume data array
        width: Image width in pixels
        height: Image height in pixels
        theme: Color theme ('classic', 'modern', 'tradingview', 'light')
        bg_color: Override background color (hex)
        up_color: Override bullish bar color (hex)
        down_color: Override bearish bar color (hex)
        enable_antialiasing: Use RGBA mode for smoother rendering
        show_grid: Display price/time grid lines

    Returns:
        PIL Image object
    """
```

## Drawing Algorithm

```python
# For each OHLC bar:
num_bars = len(ohlc['close'])
bar_width = width / (num_bars + 1)
tick_length = bar_width * 0.4  # 40% of bar width for ticks

for i in range(num_bars):
    x_center = int(i * bar_width + bar_width / 2)
    y_high = scale_price(ohlc['high'][i])
    y_low = scale_price(ohlc['low'][i])
    y_open = scale_price(ohlc['open'][i])
    y_close = scale_price(ohlc['close'][i])

    # Determine color
    color = up_color if ohlc['close'][i] >= ohlc['open'][i] else down_color

    # 1. Draw vertical line (high to low)
    draw.line([(x_center, y_high), (x_center, y_low)], fill=color, width=1)

    # 2. Draw left tick (open)
    x_left = int(x_center - tick_length)
    draw.line([(x_left, y_open), (x_center, y_open)], fill=color, width=1)

    # 3. Draw right tick (close)
    x_right = int(x_center + tick_length)
    draw.line([(x_center, y_close), (x_right, y_close)], fill=color, width=1)
```

## Optimization Opportunities

1. **Vectorize coordinate calculation**:
   ```python
   indices = np.arange(num_bars)
   x_centers = ((indices + 0.5) * bar_width).astype(np.int32)
   x_lefts = (x_centers - tick_length).astype(np.int32)
   x_rights = (x_centers + tick_length).astype(np.int32)

   y_highs = scale_prices_vectorized(ohlc['high'])
   y_lows = scale_prices_vectorized(ohlc['low'])
   y_opens = scale_prices_vectorized(ohlc['open'])
   y_closes = scale_prices_vectorized(ohlc['close'])
   ```

2. **Batch drawing by color**:
   - Group all bullish bars together
   - Group all bearish bars together
   - Draw all lines of same color in one pass

## Reusable Components

Copy/adapt from existing `render_ohlcv_chart()`:
- Theme color setup (lines 540-565)
- Price scaling logic (lines 586-598)
- Grid drawing (call `_draw_grid()`)
- Volume bars (lines 760-850, reuse as-is)

## Testing

Create test function:
```python
def test_render_ohlc_bars():
    ohlc = {
        'open': np.array([100, 102, 101, 103, 105]),
        'high': np.array([105, 106, 105, 108, 110]),
        'low': np.array([98, 100, 99, 102, 103]),
        'close': np.array([103, 101, 104, 106, 108]),
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    img = render_ohlc_bars(ohlc, volume, width=800, height=600)

    assert img.size == (800, 600)
    assert img.mode in ['RGB', 'RGBA']

    img.save('tests/fixtures/ohlc_bars_sample.webp')
```

## Performance Target
- **Rendering**: >5000 charts/sec
- **Speedup**: 150-200x vs mplfinance OHLC bars

## Deliverables
1. ✅ `render_ohlc_bars()` function in `renderer.py`
2. ✅ Test function in `tests/test_renderer_ohlc.py`
3. ✅ Sample chart saved to verify visual correctness

## Complexity: LOW
Estimated time: 30-45 minutes

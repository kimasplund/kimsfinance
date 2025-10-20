# Task 3: Implement Hollow Candles Renderer

## Objective
Implement native PIL-based hollow candles chart renderer achieving 178x speedup vs mplfinance.

## Visual Specification

Hollow candles differentiate bullish/bearish by fill:
- **Bullish (close ≥ open)**: Hollow body (outline only), filled with background color
- **Bearish (close < open)**: Solid filled body (same as regular candlestick)
- **Wicks**: Same as regular candlesticks (high-low lines)

**Visual advantage**: Easier to spot trend direction at a glance.

## Implementation Location
**File**: `kimsfinance/plotting/renderer.py`
**Function**: `render_hollow_candles(ohlc, volume, width, height, theme, **kwargs)`

## Function Signature
```python
def render_hollow_candles(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    wick_width_ratio: float = 0.1,
    enable_antialiasing: bool = True,
    show_grid: bool = True,
    use_batch_drawing: bool | None = None,
) -> Image.Image:
    """
    Render hollow candles chart with volume using native PIL.

    Bullish candles (close >= open) are drawn hollow (outline only).
    Bearish candles (close < open) are drawn solid (filled).

    Args:
        ohlc: Dict with 'open', 'high', 'low', 'close' arrays
        volume: Volume data array
        width: Image width in pixels
        height: Image height in pixels
        theme: Color theme
        bg_color: Override background color (hex)
        up_color: Override bullish candle outline color (hex)
        down_color: Override bearish candle fill color (hex)
        wick_width_ratio: Wick width as ratio of body width
        enable_antialiasing: Use RGBA mode
        show_grid: Display grid
        use_batch_drawing: Batch drawing optimization

    Returns:
        PIL Image object
    """
```

## Drawing Algorithm

```python
# Very similar to regular candlesticks, but change body drawing logic:

for i in range(num_candles):
    # Calculate coordinates (same as candlestick)
    x_start, x_end = calculate_x_coords(i)
    y_high, y_low = scale_prices(high[i], low[i])
    y_open, y_close = scale_prices(open[i], close[i])

    body_top = min(y_open, y_close)
    body_bottom = max(y_open, y_close)

    # Determine if bullish or bearish
    is_bullish = close[i] >= open[i]

    # Draw wicks (same as candlestick)
    x_center = (x_start + x_end) // 2
    draw.line([(x_center, y_high), (x_center, body_top)], fill=color, width=wick_width)
    draw.line([(x_center, body_bottom), (x_center, y_low)], fill=color, width=wick_width)

    # Draw body (DIFFERENT from candlestick)
    if is_bullish:
        # HOLLOW: Draw outline only, no fill
        draw.rectangle(
            [x_start, body_top, x_end, body_bottom],
            outline=up_color,
            fill=None,  # No fill = hollow!
            width=1
        )
    else:
        # FILLED: Draw solid rectangle
        draw.rectangle(
            [x_start, body_top, x_end, body_bottom],
            fill=down_color,
            outline=down_color
        )
```

## Vectorized Optimization

```python
# Reuse coordinate calculation from regular candlestick
coords = _calculate_coordinates_jit(...)  # or _calculate_coordinates_numpy

# Separate bullish and bearish candles
bullish_mask = close_prices >= open_prices
bearish_mask = ~bullish_mask

# Batch draw wicks (same for both)
for i in bullish_indices:
    # Draw wicks
    ...

# Batch draw hollow bodies (bullish)
for i in bullish_indices:
    draw.rectangle([x_start[i], body_top[i], x_end[i], body_bottom[i]],
                   outline=up_color, fill=None, width=1)

# Batch draw filled bodies (bearish)
for i in bearish_indices:
    draw.rectangle([x_start[i], body_top[i], x_end[i], body_bottom[i]],
                   fill=down_color, outline=down_color)
```

## Reusable Components

**FROM `render_ohlcv_chart()`**:
- ✅ Theme setup (lines 540-565)
- ✅ Coordinate calculation (_calculate_coordinates_jit or _calculate_coordinates_numpy)
- ✅ Price scaling
- ✅ Grid drawing
- ✅ Volume bars

**MODIFY**:
- Body drawing logic (hollow vs filled)

## Key Difference from Regular Candlestick

```python
# Regular candlestick:
if is_bullish:
    draw.rectangle([...], fill=up_color, outline=up_color)  # FILLED GREEN
else:
    draw.rectangle([...], fill=down_color, outline=down_color)  # FILLED RED

# Hollow candlestick:
if is_bullish:
    draw.rectangle([...], fill=None, outline=up_color, width=1)  # HOLLOW OUTLINE ONLY
else:
    draw.rectangle([...], fill=down_color, outline=down_color)  # FILLED RED (same)
```

## Testing

```python
def test_render_hollow_candles():
    ohlc = {
        'open': np.array([100, 105, 103, 102, 104]),  # Mix of bull/bear
        'high': np.array([106, 107, 105, 106, 108]),
        'low': np.array([98, 103, 101, 100, 102]),
        'close': np.array([105, 103, 104, 105, 107]),  # 105>100 (bull), 103<105 (bear), ...
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    img = render_hollow_candles(ohlc, volume, width=800, height=600)

    assert img.size == (800, 600)
    assert img.mode in ['RGB', 'RGBA']

    # Verify: candles 0, 2, 3, 4 are bullish (hollow), candle 1 is bearish (filled)
    img.save('tests/fixtures/hollow_candles_sample.webp')

    # Visual check: Candle 1 should be filled red, others should be hollow green
```

## Performance Target
- **Rendering**: >5000 charts/sec (similar to regular candlesticks)
- **Speedup**: 150-200x vs mplfinance hollow candles

## Deliverables
1. ✅ `render_hollow_candles()` function in `renderer.py`
2. ✅ Test function in `tests/test_renderer_hollow.py`
3. ✅ Sample chart with mix of hollow/filled candles

## Complexity: LOW-MEDIUM
Estimated time: 30-45 minutes (very similar to existing candlestick)

## Notes
- This is the easiest of the 5 tasks because it's just a minor modification of existing `render_ohlcv_chart()`
- Can literally copy 90% of the code and change only the body drawing logic (lines ~680-720)

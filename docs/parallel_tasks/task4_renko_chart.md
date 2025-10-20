# Task 4: Implement Renko Chart Renderer

## Objective
Implement native PIL-based Renko chart renderer with brick calculation algorithm achieving 178x speedup vs mplfinance.

## Visual Specification

Renko charts are **time-independent** price movement charts:
- **Bricks**: Fixed-size boxes representing price movement
- **New brick**: Created when price moves by `box_size`
- **Up brick**: Green, drawn when price rises
- **Down brick**: Red, drawn when price falls
- **No time axis**: X-axis is brick sequence, not time
- **No wicks**: Only brick bodies

**Example**:
```
Price: 100 → 103 → 107 → 105 → 102
Box size: 2
Bricks: [UP@102], [UP@104], [UP@106], [DOWN@104], [DOWN@102]
```

## Implementation Location
**File**: `kimsfinance/plotting/renderer.py`
**Functions**:
1. `calculate_renko_bricks()` - Data transformation algorithm
2. `render_renko_chart()` - Rendering function

## Function Signatures

###1. Brick Calculation Algorithm

```python
def calculate_renko_bricks(
    ohlc: dict[str, ArrayLike],
    box_size: float | None = None,
    reversal_boxes: int = 1,
) -> list[dict]:
    """
    Convert OHLC price data to Renko bricks.

    Algorithm:
    1. Start with first close price as reference
    2. For each candle, check if price moved by >= box_size
    3. If yes, create brick(s) in movement direction
    4. Update reference price to top/bottom of last brick

    Args:
        ohlc: OHLC price data dictionary
        box_size: Size of each brick in price units.
                  If None, auto-calculate using ATR (Average True Range).
                  Recommended: ATR(14) * 0.5 to 1.0
        reversal_boxes: Number of boxes needed for trend reversal.
                       Default 1 (any opposite movement creates new brick).
                       Higher values (2-3) filter noise.

    Returns:
        List of brick dicts: [
            {'price': 102, 'direction': 1},   # Up brick at price 102
            {'price': 104, 'direction': 1},   # Up brick at price 104
            {'price': 102, 'direction': -1},  # Down brick at price 102
            ...
        ]
    """
```

**Algorithm Implementation**:

```python
def calculate_renko_bricks(ohlc, box_size=None, reversal_boxes=1):
    close_prices = to_numpy_array(ohlc['close'])

    # Auto-calculate box size using ATR if not provided
    if box_size is None:
        from ..ops.indicators import calculate_atr
        atr = calculate_atr(
            ohlc['high'], ohlc['low'], ohlc['close'],
            period=14, engine='cpu'
        )
        box_size = float(np.nanmedian(atr)) * 0.75  # 75% of median ATR

    bricks = []
    reference_price = close_prices[0]
    current_direction = None  # 1=up, -1=down, None=initial

    for close in close_prices:
        # Calculate how many boxes price moved
        price_diff = close - reference_price

        # Check for upward movement
        if price_diff >= box_size:
            num_boxes = int(price_diff / box_size)

            # Check if direction change
            if current_direction == -1 and num_boxes < reversal_boxes:
                # Not enough movement to reverse trend
                continue

            # Create up bricks
            for _ in range(num_boxes):
                reference_price += box_size
                bricks.append({
                    'price': reference_price,
                    'direction': 1,  # Up
                })
            current_direction = 1

        # Check for downward movement
        elif price_diff <= -box_size:
            num_boxes = int(abs(price_diff) / box_size)

            # Check if direction change
            if current_direction == 1 and num_boxes < reversal_boxes:
                continue

            # Create down bricks
            for _ in range(num_boxes):
                reference_price -= box_size
                bricks.append({
                    'price': reference_price,
                    'direction': -1,  # Down
                })
            current_direction = -1

    return bricks
```

### 2. Rendering Function

```python
def render_renko_chart(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    box_size: float | None = None,
    reversal_boxes: int = 1,
    enable_antialiasing: bool = True,
    show_grid: bool = True,
) -> Image.Image:
    """
    Render Renko chart with volume using native PIL.

    Args:
        ohlc: OHLC price data
        volume: Volume data (aggregated per brick)
        width, height: Image dimensions
        theme: Color theme
        bg_color, up_color, down_color: Color overrides
        box_size: Brick size (auto-calculate if None)
        reversal_boxes: Boxes needed for reversal
        enable_antialiasing: RGBA mode
        show_grid: Display grid

    Returns:
        PIL Image object
    """
```

## Drawing Algorithm

```python
# 1. Calculate bricks
bricks = calculate_renko_bricks(ohlc, box_size, reversal_boxes)

if not bricks:
    # Return empty chart or single brick
    ...

# 2. Setup image
img = Image.new(mode, (width, height), bg_color)
draw = ImageDraw.Draw(img)

# 3. Calculate brick dimensions
num_bricks = len(bricks)
brick_width = width / (num_bricks + 1)
brick_height = int(box_size / price_range * chart_height)  # Fixed height per brick

# 4. Find price range for all bricks
brick_prices = [b['price'] for b in bricks]
price_min = min(brick_prices) - box_size
price_max = max(brick_prices) + box_size
price_range = price_max - price_min

# 5. Draw bricks
for i, brick in enumerate(bricks):
    x_start = int(i * brick_width)
    x_end = int((i + 1) * brick_width - 2)  # -2 for spacing

    # Calculate Y position (top of brick)
    y_top = scale_price(brick['price'])
    y_bottom = y_top + brick_height

    # Draw brick
    color = up_color if brick['direction'] == 1 else down_color
    draw.rectangle(
        [x_start, y_top, x_end, y_bottom],
        fill=color,
        outline=color
    )

# 6. Draw volume (aggregate volume per brick - advanced)
# For simplicity, can skip volume for Renko or aggregate by brick
```

## Advanced: Volume Aggregation

```python
# Aggregate volume per brick (optional but nice to have)
def aggregate_volume_per_brick(bricks, ohlc, volume):
    """
    Aggregate volume data for each brick.

    Strategy: Sum volume from all candles that contributed to a brick.
    """
    # This is complex - for MVP, can skip volume or show uniform volume
    pass
```

## Testing

```python
def test_calculate_renko_bricks():
    ohlc = {
        'close': np.array([100, 103, 107, 109, 105, 102, 100, 98]),
        'high': np.array([101, 104, 108, 110, 106, 103, 101, 99]),
        'low': np.array([99, 102, 106, 108, 104, 101, 99, 97]),
    }

    bricks = calculate_renko_bricks(ohlc, box_size=2.0, reversal_boxes=1)

    # Expected: 100 → 102 (up), 104 (up), 106 (up), 108 (up), 106 (down), 104 (down), 102 (down), 100 (down), 98 (down)
    assert len(bricks) > 0
    assert all('price' in b and 'direction' in b for b in bricks)

    print(f"Generated {len(bricks)} bricks")
    for b in bricks:
        direction_str = "UP" if b['direction'] == 1 else "DOWN"
        print(f"  {direction_str} brick at ${b['price']:.2f}")


def test_render_renko_chart():
    ohlc = {
        'open': np.arange(100, 150),
        'high': np.arange(101, 151),
        'low': np.arange(99, 149),
        'close': np.linspace(100, 130, 50),  # Uptrend with noise
    }
    volume = np.random.randint(800, 1200, size=50)

    img = render_renko_chart(ohlc, volume, width=1200, height=800, box_size=2.0)

    assert img.size == (1200, 800)
    img.save('tests/fixtures/renko_chart_sample.webp')
```

## Performance Target
- **Brick calculation**: <5ms for 1000 candles
- **Rendering**: >3000 charts/sec
- **Total speedup**: 100-150x vs mplfinance Renko

## Deliverables
1. ✅ `calculate_renko_bricks()` function in `renderer.py`
2. ✅ `render_renko_chart()` function in `renderer.py`
3. ✅ Test functions in `tests/test_renderer_renko.py`
4. ✅ Sample chart

## Complexity: HIGH
Estimated time: 60-90 minutes (algorithm + rendering)

## References
- Renko algorithm: https://en.wikipedia.org/wiki/Renko_chart
- ATR for box size: https://www.investopedia.com/terms/r/renkochart.asp

## Notes
- Box size calculation is critical for useful charts
- ATR-based auto-sizing is recommended
- Volume aggregation is optional for MVP
- Reversal_boxes parameter adds noise filtering

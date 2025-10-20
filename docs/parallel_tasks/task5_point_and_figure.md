# Task 5: Implement Point and Figure (PNF) Chart Renderer

## Objective
Implement native PIL-based Point and Figure chart renderer with column calculation algorithm achieving 178x speedup vs mplfinance.

## Visual Specification

Point and Figure (PNF) charts show pure price movement without time:
- **X columns**: Rising price (each X = box_size increase)
- **O columns**: Falling price (each O = box_size decrease)
- **New column**: Created when price reverses by reversal_boxes * box_size
- **No time axis**: X-axis is column sequence
- **No volume**: PNF focuses purely on price

**Example**:
```
Price: 100 → 104 → 108 → 103 → 98 → 95
Box size: 2, Reversal: 3 boxes (3 * 2 = 6 price units)

Column 1 (X): 102, 104, 106, 108 (rising)
Column 2 (O): 106, 104, 102, 100, 98, 96 (falling - reversal of 6+ units)
```

## Implementation Location
**File**: `kimsfinance/plotting/renderer.py`
**Functions**:
1. `calculate_pnf_columns()` - Data transformation algorithm
2. `render_pnf_chart()` - Rendering function

## Function Signatures

### 1. Column Calculation Algorithm

```python
def calculate_pnf_columns(
    ohlc: dict[str, ArrayLike],
    box_size: float | None = None,
    reversal_boxes: int = 3,
) -> list[dict]:
    """
    Convert OHLC price data to Point and Figure columns.

    Algorithm:
    1. Start with first close price, determine direction from second candle
    2. For each candle, check high for X boxes, low for O boxes
    3. If current column continues: add boxes
    4. If reversal detected (opposite direction >= reversal_boxes): start new column
    5. Ignore moves that don't meet box size threshold

    Args:
        ohlc: OHLC price data dictionary
        box_size: Price per box. If None, auto-calculate using ATR.
                  Typical: ATR(14) * 0.5 to 2.0
        reversal_boxes: Number of boxes needed for trend reversal.
                       Default 3 (traditional PNF standard).
                       Higher = fewer columns, smoother trend.

    Returns:
        List of column dicts: [
            {
                'type': 'X',  # or 'O'
                'boxes': [102, 104, 106, 108],  # Prices for each box
                'start_idx': 0,  # Index in original data where column started
            },
            {
                'type': 'O',
                'boxes': [106, 104, 102, 100],
                'start_idx': 10,
            },
            ...
        ]

    Notes:
        - Uses high/low prices, not just close
        - More accurate than close-based algorithms
        - Returns empty list if insufficient price movement
    """
```

**Algorithm Implementation** (Complex!):

```python
def calculate_pnf_columns(ohlc, box_size=None, reversal_boxes=3):
    high_prices = to_numpy_array(ohlc['high'])
    low_prices = to_numpy_array(ohlc['low'])
    close_prices = to_numpy_array(ohlc['close'])

    # Auto-calculate box size using ATR
    if box_size is None:
        from ..ops.indicators import calculate_atr
        atr = calculate_atr(
            ohlc['high'], ohlc['low'], ohlc['close'],
            period=14, engine='cpu'
        )
        box_size = float(np.nanmedian(atr))  # Use median ATR

    columns = []
    current_column = None
    reference_price = close_prices[0]

    # Round reference price to nearest box
    reference_price = round(reference_price / box_size) * box_size

    for i in range(len(close_prices)):
        high = high_prices[i]
        low = low_prices[i]

        # How many boxes can we go up from reference?
        boxes_up = int((high - reference_price) / box_size)

        # How many boxes can we go down from reference?
        boxes_down = int((reference_price - low) / box_size)

        # Current column is rising (X column)
        if current_column is None or current_column['type'] == 'X':
            # Try to add X boxes
            if boxes_up > 0:
                if current_column is None:
                    current_column = {'type': 'X', 'boxes': [], 'start_idx': i}

                # Add X boxes
                for j in range(boxes_up):
                    reference_price += box_size
                    current_column['boxes'].append(reference_price)

            # Check for reversal to O
            elif boxes_down >= reversal_boxes:
                # Save current X column
                if current_column and current_column['boxes']:
                    columns.append(current_column)

                # Start new O column
                current_column = {'type': 'O', 'boxes': [], 'start_idx': i}

                # Add O boxes (going down from previous high)
                # Go back to top of last X column
                if columns:
                    reference_price = columns[-1]['boxes'][-1] if columns[-1]['boxes'] else reference_price

                for j in range(boxes_down):
                    reference_price -= box_size
                    current_column['boxes'].append(reference_price)

        # Current column is falling (O column)
        elif current_column['type'] == 'O':
            # Try to add O boxes
            if boxes_down > 0:
                for j in range(boxes_down):
                    reference_price -= box_size
                    current_column['boxes'].append(reference_price)

            # Check for reversal to X
            elif boxes_up >= reversal_boxes:
                # Save current O column
                if current_column['boxes']:
                    columns.append(current_column)

                # Start new X column
                current_column = {'type': 'X', 'boxes': [], 'start_idx': i}

                # Go back to bottom of last O column
                if columns:
                    reference_price = columns[-1]['boxes'][-1] if columns[-1]['boxes'] else reference_price

                for j in range(boxes_up):
                    reference_price += box_size
                    current_column['boxes'].append(reference_price)

    # Add final column
    if current_column and current_column['boxes']:
        columns.append(current_column)

    return columns
```

### 2. Rendering Function

```python
def render_pnf_chart(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    box_size: float | None = None,
    reversal_boxes: int = 3,
    enable_antialiasing: bool = True,
    show_grid: bool = True,
) -> Image.Image:
    """
    Render Point and Figure chart using native PIL.

    Args:
        ohlc: OHLC price data
        volume: Volume data (not used in PNF - only price matters)
        width, height: Image dimensions
        theme: Color theme
        bg_color, up_color, down_color: Color overrides
        box_size: Price per box (auto-calculate if None)
        reversal_boxes: Boxes needed for reversal (default 3)
        enable_antialiasing: RGBA mode for smoother X and O
        show_grid: Display grid

    Returns:
        PIL Image object
    """
```

## Drawing Algorithm

```python
# 1. Calculate PNF columns
columns = calculate_pnf_columns(ohlc, box_size, reversal_boxes)

if not columns:
    # Return empty chart
    ...

# 2. Setup image
img = Image.new(mode, (width, height), bg_color)
draw = ImageDraw.Draw(img)

# 3. Calculate column dimensions
num_columns = len(columns)
column_width = width / (num_columns + 1)
box_width = int(column_width * 0.8)  # 80% of column width
box_height = int(box_size / price_range * chart_height)

# 4. Find price range
all_prices = []
for col in columns:
    all_prices.extend(col['boxes'])
price_min = min(all_prices) - box_size
price_max = max(all_prices) + box_size
price_range = price_max - price_min

# 5. Draw columns
for col_idx, column in enumerate(columns):
    x_start = int(col_idx * column_width)
    x_center = x_start + column_width // 2

    for box_price in column['boxes']:
        y_center = scale_price(box_price)

        if column['type'] == 'X':
            # Draw X (two diagonal lines)
            half_box = box_width // 2
            # Top-left to bottom-right
            draw.line(
                [(x_center - half_box, y_center - half_box),
                 (x_center + half_box, y_center + half_box)],
                fill=up_color, width=2
            )
            # Bottom-left to top-right
            draw.line(
                [(x_center - half_box, y_center + half_box),
                 (x_center + half_box, y_center - half_box)],
                fill=up_color, width=2
            )

        else:  # 'O'
            # Draw O (ellipse/circle)
            half_box = box_width // 2
            draw.ellipse(
                [x_center - half_box, y_center - half_box,
                 x_center + half_box, y_center + half_box],
                outline=down_color, width=2
            )

# Note: PNF charts typically don't show volume
# Can optionally add volume at bottom, but not standard
```

## Testing

```python
def test_calculate_pnf_columns():
    # Trending up then reversal down
    ohlc = {
        'high': np.array([101, 104, 107, 110, 109, 106, 103, 100]),
        'low': np.array([99, 102, 105, 108, 106, 103, 100, 97]),
        'close': np.array([100, 103, 106, 109, 107, 104, 101, 98]),
    }

    columns = calculate_pnf_columns(ohlc, box_size=2.0, reversal_boxes=3)

    # Should get at least 2 columns (X then O)
    assert len(columns) >= 2

    # First column should be X (rising)
    assert columns[0]['type'] == 'X'
    assert len(columns[0]['boxes']) > 0

    # Check structure
    for col in columns:
        assert col['type'] in ['X', 'O']
        assert 'boxes' in col
        assert 'start_idx' in col

    print(f"Generated {len(columns)} columns:")
    for i, col in enumerate(columns):
        print(f"  Column {i+1} ({col['type']}): {len(col['boxes'])} boxes")


def test_render_pnf_chart():
    # Generate uptrend then downtrend data
    ohlc = {
        'open': np.linspace(100, 130, 50),
        'high': np.linspace(102, 135, 50),
        'low': np.linspace(98, 128, 50),
        'close': np.concatenate([
            np.linspace(100, 130, 30),  # Uptrend
            np.linspace(130, 110, 20),  # Downtrend
        ]),
    }
    volume = np.random.randint(800, 1200, size=50)

    img = render_pnf_chart(ohlc, volume, width=1200, height=800,
                          box_size=2.0, reversal_boxes=3)

    assert img.size == (1200, 800)
    img.save('tests/fixtures/pnf_chart_sample.webp')

    # Should see X columns (uptrend) then O columns (downtrend)
```

## Performance Target
- **Column calculation**: <10ms for 1000 candles
- **Rendering**: >2000 charts/sec (complex X/O drawing)
- **Total speedup**: 100-150x vs mplfinance PNF

## Deliverables
1. ✅ `calculate_pnf_columns()` function in `renderer.py`
2. ✅ `render_pnf_chart()` function in `renderer.py`
3. ✅ Test functions in `tests/test_renderer_pnf.py`
4. ✅ Sample chart showing X and O columns

## Complexity: VERY HIGH
Estimated time: 90-120 minutes (complex algorithm + rendering)

## References
- Point and Figure: https://en.wikipedia.org/wiki/Point_and_figure_chart
- Traditional PNF: https://www.investopedia.com/terms/p/pointandfigurechart.asp
- Algorithm details: https://school.stockcharts.com/doku.php?id=chart_analysis:pnf_charts

## Notes
- PNF algorithm is stateful and complex - requires careful testing
- Traditional PNF uses 3-box reversal (standard)
- Box size critical - ATR-based auto-sizing recommended
- High/low prices give more accurate columns than close-only
- Volume not traditionally shown on PNF (pure price action)
- X and O drawing must be clear - use adequate line width (2px+)

## Edge Cases to Handle
1. Insufficient price movement (no columns generated)
2. Very small box size (too many boxes)
3. Very large box size (no boxes)
4. Single column (no reversals)
5. Empty data

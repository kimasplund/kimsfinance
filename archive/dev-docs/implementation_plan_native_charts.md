# Native Chart Type Implementation Plan

## Objective
Implement 5 missing chart types natively in PIL to achieve 178x speedup vs mplfinance for ALL chart types, not just candlesticks.

## Current State

### Bug Identified
**File**: `kimsfinance/api/plot.py` line 89
- **Problem**: Delegates to `mpf.plot()` instead of native renderer
- **Impact**: Only 7-10x speedup (indicator acceleration) instead of 178x (full native rendering)
- **Fix Required**: Route to native PIL renderers

### Existing Implementation
**File**: `kimsfinance/plotting/renderer.py`
- ✅ `render_ohlcv_chart()` - Candlestick with volume (932 lines total)
- Architecture: PIL-based, vectorized NumPy coordinate computation, optional Numba JIT
- Performance: 178x faster than mplfinance

### Missing Chart Types
1. ❌ OHLC bars
2. ❌ Line charts
3. ❌ Renko charts
4. ❌ Hollow candles
5. ❌ Point and Figure (PNF)

---

## Architecture Design

### Shared Components (Reusable)

From existing `render_ohlcv_chart()`:
- Image creation and color management
- Price/volume scaling functions
- Grid drawing (`_draw_grid()`)
- Theme system (THEMES_RGB, THEMES_RGBA)
- Coordinate calculation helpers
- Save utilities (`save_chart()`)

### Chart-Specific Rendering

Each chart type needs its own rendering function:
- `render_ohlc_bars()` - OHLC bars
- `render_line_chart()` - Line chart
- `render_renko_chart()` - Renko bricks
- `render_hollow_candles()` - Hollow candles
- `render_pnf_chart()` - Point and Figure

### Unified API

```python
def render_chart(
    ohlc: dict,
    volume: ArrayLike,
    chart_type: str = "candle",  # NEW PARAMETER
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    **kwargs
) -> Image.Image:
    """Route to appropriate renderer based on chart_type"""

    if chart_type == "candle":
        return render_ohlcv_chart(ohlc, volume, width, height, theme, **kwargs)
    elif chart_type == "ohlc":
        return render_ohlc_bars(ohlc, volume, width, height, theme, **kwargs)
    elif chart_type == "line":
        return render_line_chart(ohlc, volume, width, height, theme, **kwargs)
    elif chart_type == "renko":
        return render_renko_chart(ohlc, volume, width, height, theme, **kwargs)
    elif chart_type == "hollow_and_filled":
        return render_hollow_candles(ohlc, volume, width, height, theme, **kwargs)
    elif chart_type == "pnf":
        return render_pnf_chart(ohlc, volume, width, height, theme, **kwargs)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")
```

---

## Implementation Details

### 1. OHLC Bars (Simplest)

**Visual Description**: Vertical lines with left tick (open) and right tick (close)

**Drawing Logic**:
```python
# For each bar:
# 1. Draw vertical line from high to low
draw.line([(x_center, y_high), (x_center, y_low)], fill=color, width=1)

# 2. Draw left tick for open
tick_length = bar_width // 2
draw.line([(x_center - tick_length, y_open), (x_center, y_open)], fill=color, width=1)

# 3. Draw right tick for close
draw.line([(x_center, y_close), (x_center + tick_length, y_close)], fill=color, width=1)
```

**Color Logic**: Up bars (close > open) = green, Down bars = red

**Complexity**: LOW (very similar to candlestick wicks)

---

### 2. Line Chart (Simple)

**Visual Description**: Connect close prices with continuous line

**Drawing Logic**:
```python
# Calculate all (x, y) points
points = [(x_coords[i], y_close[i]) for i in range(num_candles)]

# Draw polyline
draw.line(points, fill=line_color, width=2)

# Optional: Add volume bars below
```

**Color Logic**: Single color (theme-based), optional gradient

**Complexity**: LOW (just polyline drawing)

---

### 3. Hollow Candles (Medium)

**Visual Description**:
- Close > Open: Hollow (outline only), fill=background
- Close < Open: Filled, fill=down_color

**Drawing Logic**:
```python
if close >= open:  # Bullish
    # Draw outline rectangle only
    draw.rectangle([x_start, body_top, x_end, body_bottom],
                   outline=up_color, width=1)
else:  # Bearish
    # Draw filled rectangle
    draw.rectangle([x_start, body_top, x_end, body_bottom],
                   fill=down_color, outline=down_color)

# Wicks same as candlestick
```

**Color Logic**: Hollow=outline only, Filled=solid

**Complexity**: LOW (minor modification of candlestick)

---

### 4. Renko Bricks (Complex)

**Visual Description**: Time-independent boxes, new brick when price moves by box_size

**Algorithm**:
```python
def calculate_renko_bricks(close_prices, box_size):
    """Convert OHLC to Renko bricks"""
    bricks = []
    reference_price = close_prices[0]

    for price in close_prices:
        # Calculate how many boxes moved
        price_diff = price - reference_price
        num_boxes = int(abs(price_diff) / box_size)

        if num_boxes > 0:
            direction = 1 if price_diff > 0 else -1
            for _ in range(num_boxes):
                brick_price = reference_price + (direction * box_size)
                bricks.append({
                    'price': brick_price,
                    'direction': direction,  # 1=up, -1=down
                })
                reference_price = brick_price

    return bricks

# Drawing: Each brick is a square box
for i, brick in enumerate(bricks):
    x = i * brick_width
    y = scale_price(brick['price'])
    color = up_color if brick['direction'] == 1 else down_color
    draw.rectangle([x, y, x + brick_width, y + box_height], fill=color)
```

**Parameters**:
- `box_size`: Price movement per brick (auto-calculate from ATR or manual)
- `reversal_boxes`: Number of boxes for reversal (typically 3)

**Complexity**: HIGH (requires brick calculation algorithm)

---

### 5. Point and Figure (Most Complex)

**Visual Description**: X columns (rising) and O columns (falling), no time axis

**Algorithm**:
```python
def calculate_pnf_columns(high, low, close, box_size=None, reversal_boxes=3):
    """
    Convert OHLC to Point and Figure columns

    Returns:
        List of columns, each column is list of boxes with type 'X' or 'O'
    """
    if box_size is None:
        box_size = calculate_atr_box_size(high, low, close)

    columns = []
    current_column = {'type': None, 'boxes': []}
    reference_price = close[0]

    for i in range(len(close)):
        high_price = high[i]
        low_price = low[i]

        # Check for new X boxes (rising)
        if high_price >= reference_price + box_size:
            num_boxes = int((high_price - reference_price) / box_size)
            # Add X boxes or start new X column
            if current_column['type'] == 'X' or current_column['type'] is None:
                # Continue X column
                for _ in range(num_boxes):
                    current_column['boxes'].append(reference_price + box_size)
                    reference_price += box_size
                current_column['type'] = 'X'
            else:
                # Reversal check: need reversal_boxes * box_size movement
                if num_boxes >= reversal_boxes:
                    columns.append(current_column)
                    current_column = {'type': 'X', 'boxes': []}
                    for _ in range(num_boxes):
                        current_column['boxes'].append(reference_price + box_size)
                        reference_price += box_size

        # Check for new O boxes (falling)
        elif low_price <= reference_price - box_size:
            num_boxes = int((reference_price - low_price) / box_size)
            if current_column['type'] == 'O' or current_column['type'] is None:
                # Continue O column
                for _ in range(num_boxes):
                    current_column['boxes'].append(reference_price - box_size)
                    reference_price -= box_size
                current_column['type'] = 'O'
            else:
                # Reversal check
                if num_boxes >= reversal_boxes:
                    columns.append(current_column)
                    current_column = {'type': 'O', 'boxes': []}
                    for _ in range(num_boxes):
                        current_column['boxes'].append(reference_price - box_size)
                        reference_price -= box_size

    if current_column['boxes']:
        columns.append(current_column)

    return columns

# Drawing
for col_idx, column in enumerate(columns):
    x = col_idx * box_width
    for box_idx, price in enumerate(column['boxes']):
        y = scale_price(price)
        if column['type'] == 'X':
            # Draw X
            draw.line([(x, y), (x + box_width, y + box_height)], fill=up_color, width=2)
            draw.line([(x, y + box_height), (x + box_width, y)], fill=up_color, width=2)
        else:  # 'O'
            # Draw O (circle)
            draw.ellipse([x, y, x + box_width, y + box_height], outline=down_color, width=2)
```

**Parameters**:
- `box_size`: Price per box (ATR-based or manual)
- `reversal_boxes`: Boxes needed for reversal (default 3)

**Complexity**: VERY HIGH (complex algorithm, different data structure)

---

## Implementation Strategy (Parallel Execution)

### Phase 1: Shared Infrastructure (Sequential)
1. Extract shared functions from `render_ohlcv_chart()`
2. Create `_ChartRenderer` base class or shared utilities module
3. Update API routing in `plot.py`

### Phase 2: Chart Type Implementation (PARALLEL - 5 agents)

**Task 1: OHLC Bars** (parallel-task-executor-v2)
- Implement `render_ohlc_bars()`
- Test with sample data
- Benchmark performance

**Task 2: Line Charts** (parallel-task-executor-v2)
- Implement `render_line_chart()`
- Test with sample data
- Benchmark performance

**Task 3: Renko Charts** (parallel-task-executor-v2)
- Implement brick calculation algorithm
- Implement `render_renko_chart()`
- Test with sample data
- Benchmark performance

**Task 4: Hollow Candles** (parallel-task-executor-v2)
- Implement `render_hollow_candles()`
- Test with sample data
- Benchmark performance

**Task 5: Point and Figure** (parallel-task-executor-v2)
- Implement PNF column calculation
- Implement `render_pnf_chart()`
- Test with sample data
- Benchmark performance

### Phase 3: Integration (Sequential)
1. Update `api/plot.py` to route to native renderers
2. Add chart type validation
3. Update tests
4. Update documentation

### Phase 4: Validation (Parallel)
1. Benchmark all chart types vs mplfinance
2. Generate sample charts for all types
3. Visual quality verification

---

## API Changes

### Before (BUGGY)
```python
# kimsfinance/api/plot.py
def plot(data, type='ohlc', ...):
    # BUG: Always delegates to mplfinance!
    return mpf.plot(data, type=type, ...)
```

### After (FIXED)
```python
# kimsfinance/api/plot.py
def plot(data, type='candle', savefig=None, returnfig=False, ...):
    # Convert data to OHLC dict
    ohlc = {
        'open': data['Open'].to_numpy(),
        'high': data['High'].to_numpy(),
        'low': data['Low'].to_numpy(),
        'close': data['Close'].to_numpy(),
    }
    volume = data['Volume'].to_numpy()

    # Use native renderer!
    from kimsfinance.plotting.renderer import render_chart
    img = render_chart(ohlc, volume, chart_type=type, ...)

    if savefig:
        img.save(savefig)

    if returnfig:
        # Convert PIL Image to matplotlib figure for compatibility
        fig, ax = plt.subplots()
        ax.imshow(img)
        return fig, ax

    return img
```

---

## Performance Targets

All chart types should achieve:
- **Rendering**: >5000 charts/sec (similar to candlestick)
- **Encoding (WebP fast)**: <25 ms/chart
- **Total**: 178x faster than mplfinance equivalent

---

## File Structure

```
kimsfinance/plotting/
  renderer.py          # Main renderer (932 lines currently)
  chart_types.py       # NEW: Chart-specific renderers
  algorithms.py        # NEW: Renko/PNF calculation algorithms
  shared.py            # NEW: Shared utilities (scaling, grid, etc.)
```

OR keep everything in `renderer.py` (simpler, ~2000 lines total)

---

## Testing Strategy

For each chart type:
```python
def test_render_[chart_type]():
    ohlc = generate_test_ohlc(100)
    volume = generate_test_volume(100)

    img = render_[chart_type](ohlc, volume)

    assert img.size == (1920, 1080)
    assert img.mode in ['RGB', 'RGBA']

    # Visual regression test
    img.save(f'tests/fixtures/{chart_type}_expected.webp')
```

---

## Dependencies

No new dependencies needed! All chart types can be implemented with:
- ✅ PIL (already used)
- ✅ NumPy (already used)
- ✅ Numba (optional, already supported)

---

## Timeline Estimate

**Phase 1**: Shared infrastructure (1-2 hours)
**Phase 2**: Chart implementations (parallel, 2-3 hours total)
  - OHLC: 30 min
  - Line: 30 min
  - Hollow: 30 min
  - Renko: 1 hour
  - PNF: 1.5 hours
**Phase 3**: Integration (1 hour)
**Phase 4**: Validation (1 hour)

**Total**: 5-7 hours with parallel execution

---

## Success Criteria

1. ✅ All 5 chart types render correctly
2. ✅ Performance: 150-200x faster than mplfinance
3. ✅ API fixed: `kf.plot()` uses native renderers
4. ✅ All tests pass
5. ✅ Sample charts generated for all types
6. ✅ Documentation updated

---

## Notes

- Renko and PNF require data transformation (not just visual rendering)
- These algorithms are stateful and complex
- May want to cache brick/column calculations for performance
- Consider adding to `kimsfinance.ops.aggregations` module

---

**Status**: Ready for parallel implementation
**Next Step**: Launch 5 parallel-task-executor-v2 agents for Phase 2

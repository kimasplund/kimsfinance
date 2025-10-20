# Task 2: Implement Line Chart Renderer

## Objective
Implement native PIL-based line chart renderer for close prices achieving 178x speedup vs mplfinance.

## Visual Specification

Line chart connects close prices with continuous line:
- **Polyline**: Connects all close prices in order
- **Line width**: 2px for visibility
- **Color**: Single color based on theme
- **Optional**: Fill area under line

## Implementation Location
**File**: `kimsfinance/plotting/renderer.py`
**Function**: `render_line_chart(ohlc, volume, width, height, theme, **kwargs)`

## Function Signature
```python
def render_line_chart(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    line_color: str | None = None,
    line_width: int = 2,
    fill_area: bool = False,
    enable_antialiasing: bool = True,
    show_grid: bool = True,
) -> Image.Image:
    """
    Render line chart connecting close prices using native PIL.

    Args:
        ohlc: Dict with 'open', 'high', 'low', 'close' arrays
        volume: Volume data array
        width: Image width in pixels
        height: Image height in pixels
        theme: Color theme ('classic', 'modern', 'tradingview', 'light')
        bg_color: Override background color (hex)
        line_color: Line color (hex), defaults to theme's up_color
        line_width: Width of line in pixels
        fill_area: Fill area under line with semi-transparent color
        enable_antialiasing: Use RGBA mode for smoother rendering
        show_grid: Display price/time grid lines

    Returns:
        PIL Image object
    """
```

## Drawing Algorithm

```python
# Calculate all line points
num_points = len(ohlc['close'])
point_spacing = width / (num_points + 1)

points = []
for i in range(num_points):
    x = int((i + 0.5) * point_spacing)
    y = scale_price(ohlc['close'][i])
    points.append((x, y))

# Draw line connecting all points
draw.line(points, fill=line_color, width=line_width, joint='curve')

# Optional: Fill area under line
if fill_area:
    # Create polygon from points + bottom edge
    polygon_points = points.copy()
    # Add bottom-right corner
    polygon_points.append((points[-1][0], chart_height))
    # Add bottom-left corner
    polygon_points.append((points[0][0], chart_height))

    # Fill with semi-transparent color
    fill_color_alpha = _hex_to_rgba(line_color, alpha=50)  # 20% opacity
    draw.polygon(polygon_points, fill=fill_color_alpha)
```

## Vectorized Optimization

```python
# Vectorize coordinate calculation
indices = np.arange(num_points)
x_coords = ((indices + 0.5) * point_spacing).astype(np.int32)
y_coords = scale_prices_vectorized(ohlc['close'])

# Create point list
points = list(zip(x_coords.tolist(), y_coords.tolist()))

# Draw in one PIL call
draw.line(points, fill=line_color, width=line_width, joint='curve')
```

## Reusable Components

From existing `render_ohlcv_chart()`:
- Theme color setup
- Price scaling
- Grid drawing (`_draw_grid()`)
- Volume bars (reuse)

## Advanced Features (Optional)

1. **Multi-line**: Draw multiple series (e.g., high/low/close)
   ```python
   for series_name, series_data in multi_series.items():
       points = calculate_points(series_data)
       draw.line(points, fill=series_colors[series_name], width=2)
   ```

2. **Smooth curves**: Use PIL's curve smoothing
   ```python
   draw.line(points, fill=line_color, width=line_width, joint='curve')
   ```

3. **Markers**: Add dots at each data point
   ```python
   for x, y in points:
       draw.ellipse([x-2, y-2, x+2, y+2], fill=line_color)
   ```

## Testing

```python
def test_render_line_chart():
    ohlc = {
        'open': np.array([100, 102, 101, 103, 105]),
        'high': np.array([105, 106, 105, 108, 110]),
        'low': np.array([98, 100, 99, 102, 103]),
        'close': np.array([103, 101, 104, 106, 108]),
    }
    volume = np.array([1000, 1200, 900, 1500, 1100])

    # Test basic line chart
    img = render_line_chart(ohlc, volume, width=800, height=600)
    assert img.size == (800, 600)
    img.save('tests/fixtures/line_chart_basic.webp')

    # Test with fill area
    img_filled = render_line_chart(ohlc, volume, width=800, height=600, fill_area=True)
    img_filled.save('tests/fixtures/line_chart_filled.webp')
```

## Performance Target
- **Rendering**: >8000 charts/sec (simpler than candlesticks)
- **Speedup**: 200-300x vs mplfinance line chart

## Deliverables
1. ✅ `render_line_chart()` function in `renderer.py`
2. ✅ Test function in `tests/test_renderer_line.py`
3. ✅ Sample charts (basic + filled area)

## Complexity: LOW
Estimated time: 30-45 minutes

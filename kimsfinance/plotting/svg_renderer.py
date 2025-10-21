from __future__ import annotations

import gzip
import numpy as np

try:
    import svgwrite
    SVGWRITE_AVAILABLE = True
except ImportError:
    SVGWRITE_AVAILABLE = False

from ..core.types import ArrayLike
from ..utils.array_utils import to_numpy_array
from ..config.themes import THEMES
from ..data.pnf import calculate_pnf_columns
from ..data.renko import calculate_renko_bricks


def _save_svg_or_svgz(dwg: "svgwrite.Drawing", output_path: str) -> None:
    """
    Save SVG drawing to file, with automatic SVGZ compression if path ends with .svgz.

    Args:
        dwg: svgwrite.Drawing instance
        output_path: Path to save file (*.svg or *.svgz)

    Examples:
        >>> _save_svg_or_svgz(dwg, 'chart.svg')   # Saves uncompressed SVG
        >>> _save_svg_or_svgz(dwg, 'chart.svgz')  # Saves gzipped SVGZ (75-85% smaller)
    """
    if output_path.endswith(".svgz"):
        # Save as compressed SVGZ (gzipped SVG)
        svg_string = dwg.tostring()
        with open(output_path, "wb") as f:
            f.write(gzip.compress(svg_string.encode("utf-8"), compresslevel=9))
    else:
        # Save as regular SVG
        dwg.saveas(output_path)


def render_candlestick_svg(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike | None = None,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    show_grid: bool = True,
    output_path: str | None = None,
) -> str:
    """
    Render candlestick chart as true vector SVG.

    This function creates infinitely scalable vector graphics using SVG,
    as opposed to the raster-based PIL rendering. SVG charts can be scaled
    to any size without losing quality and typically have smaller file sizes
    for charts with moderate numbers of candles.

    Args:
        ohlc: A dictionary containing 'open', 'high', 'low', 'close' arrays.
        volume: An array of volume data. If None, no volume panel is shown.
        width: The width of the output SVG in pixels.
        height: The height of the output SVG in pixels.
        theme: Color theme to use. Options: 'classic', 'modern', 'tradingview', 'light'.
               Defaults to 'classic'.
        bg_color: Override background color (hex string). If None, uses theme color.
        up_color: Override color for bullish candles (hex string). If None, uses theme color.
        down_color: Override color for bearish candles (hex string). If None, uses theme color.
        show_grid: Display grid lines for price levels and time markers. Defaults to True.
        output_path: Path to save the SVG file. If None, returns SVG as string without saving.

    Returns:
        SVG content as XML string. If output_path is provided, also saves to file.

    Raises:
        ImportError: If svgwrite is not installed.

    Examples:
        >>> ohlc_dict = {'open': [100, 102], 'high': [103, 105], 'low': [99, 101], 'close': [102, 104]}
        >>> volume_array = np.array([1000, 1500])
        >>> svg_str = render_candlestick_svg(ohlc_dict, volume_array, output_path='chart.svg')
        >>> # Chart saved to chart.svg and SVG string returned

    Notes:
        - SVG files are infinitely scalable (vector graphics)
        - File sizes are typically smaller than raster formats for charts with <1000 candles
        - For very large datasets (10K+ candles), raster formats may be more efficient
        - SVG can be opened in browsers, edited in Inkscape/Illustrator, embedded in web pages
    """
    if not SVGWRITE_AVAILABLE:
        raise ImportError(
            "svgwrite is required for SVG export. " "Install with: pip install svgwrite"
        )

    # Get theme colors
    theme_colors = THEMES.get(theme, THEMES["classic"])
    bg_color_final = bg_color or theme_colors["bg"]
    up_color_final = up_color or theme_colors["up"]
    down_color_final = down_color or theme_colors["down"]
    grid_color_final = theme_colors["grid"]

    # Convert to numpy arrays
    open_prices = np.ascontiguousarray(to_numpy_array(ohlc["open"]))
    high_prices = np.ascontiguousarray(to_numpy_array(ohlc["high"]))
    low_prices = np.ascontiguousarray(to_numpy_array(ohlc["low"]))
    close_prices = np.ascontiguousarray(to_numpy_array(ohlc["close"]))

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(width, height))

    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=bg_color_final))

    # Define chart areas (70% for candlestick, 30% for volume if volume data provided)
    has_volume = volume is not None
    if has_volume:
        volume_data = np.ascontiguousarray(to_numpy_array(volume))
        chart_height = int(height * 0.7)
        volume_height = int(height * 0.3)
    else:
        chart_height = height
        volume_height = 0

    # Price and volume scaling
    price_min = float(np.min(low_prices))
    price_max = float(np.max(high_prices))
    price_range = price_max - price_min

    if has_volume:
        volume_max = float(np.max(volume_data))
        volume_range = volume_max

    # Candlestick width calculation
    num_candles = len(open_prices)
    candle_width = width / (num_candles + 1)
    spacing = candle_width * 0.2
    bar_width = candle_width - spacing

    # Wick width (scaled for SVG, minimum 1px)
    wick_width = max(1.0, bar_width * 0.1)

    # Draw grid lines (background layer)
    if show_grid:
        # Horizontal price grid lines (10 divisions)
        grid_group = dwg.add(
            dwg.g(id="grid", stroke=grid_color_final, stroke_width=1, opacity=0.25)
        )
        for i in range(1, 10):
            y = int(i * chart_height / 10)
            grid_group.add(dwg.line(start=(0, y), end=(width, y)))

        # Vertical time grid lines (max 20 lines)
        num_vertical_lines = min(20, num_candles // 10 + 1)
        if num_vertical_lines > 1:
            interval = num_candles / num_vertical_lines
            for i in range(num_vertical_lines):
                x = int(i * interval * candle_width)
                grid_group.add(dwg.line(start=(x, 0), end=(x, height)))

    # Create groups for organized SVG structure
    candles_group = dwg.add(dwg.g(id="candles"))
    if has_volume:
        volume_group = dwg.add(dwg.g(id="volume"))

    # Draw candlesticks
    for i in range(num_candles):
        o = float(open_prices[i])
        h = float(high_prices[i])
        l = float(low_prices[i])
        c = float(close_prices[i])

        # Determine color
        is_bullish = c >= o
        color = up_color_final if is_bullish else down_color_final

        # Calculate positions
        x = i * candle_width + spacing / 2
        x_center = x + bar_width / 2

        # Y coordinates (inverted: 0 is top of chart)
        y_high = chart_height - ((h - price_min) / price_range) * chart_height
        y_low = chart_height - ((l - price_min) / price_range) * chart_height
        y_open = chart_height - ((o - price_min) / price_range) * chart_height
        y_close = chart_height - ((c - price_min) / price_range) * chart_height

        # Body top and bottom
        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close)
        body_height = body_bottom - body_top

        # Ensure minimum body height for visibility (doji candles)
        if body_height < 1:
            body_height = 1

        # Draw wick (vertical line from low to high)
        candles_group.add(
            dwg.line(
                start=(x_center, y_high),
                end=(x_center, y_low),
                stroke=color,
                stroke_width=wick_width,
            )
        )

        # Draw body (rectangle)
        candles_group.add(dwg.rect(insert=(x, body_top), size=(bar_width, body_height), fill=color))

        # Draw volume bar if volume data provided
        if has_volume:
            vol = float(volume_data[i])
            vol_height = (vol / volume_range) * volume_height
            vol_y = height - vol_height

            volume_group.add(
                dwg.rect(insert=(x, vol_y), size=(bar_width, vol_height), fill=color, opacity=0.5)
            )

    # Save or return SVG
    if output_path:
        _save_svg_or_svgz(dwg, output_path)

    return dwg.tostring()


def render_ohlc_bars_svg(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike | None = None,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    show_grid: bool = True,
    output_path: str | None = None,
) -> str:
    """
    Render OHLC bar chart as true vector SVG.

    OHLC (Open-High-Low-Close) bars are rendered as vector graphics with:
    - Vertical line: From high to low price
    - Left tick: Open price (horizontal line extending left from vertical)
    - Right tick: Close price (horizontal line extending right from vertical)

    Colors are determined by close vs open (bullish/bearish).

    Args:
        ohlc: A dictionary containing 'open', 'high', 'low', 'close' arrays.
        volume: An array of volume data. If None, no volume panel is shown.
        width: The width of the output SVG in pixels.
        height: The height of the output SVG in pixels.
        theme: Color theme to use. Options: 'classic', 'modern', 'tradingview', 'light'.
               Defaults to 'classic'.
        bg_color: Override background color (hex string). If None, uses theme color.
        up_color: Override color for bullish bars (hex string). If None, uses theme color.
        down_color: Override color for bearish bars (hex string). If None, uses theme color.
        show_grid: Display grid lines for price levels and time markers. Defaults to True.
        output_path: Path to save the SVG file. If None, returns SVG as string without saving.

    Returns:
        SVG content as XML string. If output_path is provided, also saves to file.

    Raises:
        ImportError: If svgwrite is not installed.

    Examples:
        >>> ohlc_dict = {'open': [100, 102], 'high': [103, 105], 'low': [99, 101], 'close': [102, 104]}
        >>> volume_array = np.array([1000, 1500])
        >>> svg_str = render_ohlc_bars_svg(ohlc_dict, volume_array, output_path='ohlc.svg')
        >>> # Chart saved to ohlc.svg and SVG string returned

    Notes:
        - SVG files are infinitely scalable (vector graphics)
        - Tick length is 40% of bar width for visual balance
        - File sizes are typically smaller than raster formats for moderate datasets
        - SVG can be opened in browsers, edited in Inkscape/Illustrator, embedded in web pages
    """
    if not SVGWRITE_AVAILABLE:
        raise ImportError(
            "svgwrite is required for SVG export. " "Install with: pip install svgwrite"
        )

    # Get theme colors
    theme_colors = THEMES.get(theme, THEMES["classic"])
    bg_color_final = bg_color or theme_colors["bg"]
    up_color_final = up_color or theme_colors["up"]
    down_color_final = down_color or theme_colors["down"]
    grid_color_final = theme_colors["grid"]

    # Convert to numpy arrays
    open_prices = np.ascontiguousarray(to_numpy_array(ohlc["open"]))
    high_prices = np.ascontiguousarray(to_numpy_array(ohlc["high"]))
    low_prices = np.ascontiguousarray(to_numpy_array(ohlc["low"]))
    close_prices = np.ascontiguousarray(to_numpy_array(ohlc["close"]))

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(width, height))

    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=bg_color_final))

    # Define chart areas (70% for OHLC bars, 30% for volume if volume data provided)
    has_volume = volume is not None
    if has_volume:
        volume_data = np.ascontiguousarray(to_numpy_array(volume))
        chart_height = int(height * 0.7)
        volume_height = int(height * 0.3)
    else:
        chart_height = height
        volume_height = 0

    # Price and volume scaling
    price_min = float(np.min(low_prices))
    price_max = float(np.max(high_prices))
    price_range = price_max - price_min

    if has_volume:
        volume_max = float(np.max(volume_data))
        volume_range = volume_max

    # Bar width calculations
    num_bars = len(open_prices)
    bar_width = width / (num_bars + 1)
    tick_length = bar_width * 0.4  # 40% of bar width for ticks

    # Draw grid lines (background layer)
    if show_grid:
        # Horizontal price grid lines (10 divisions)
        grid_group = dwg.add(
            dwg.g(id="grid", stroke=grid_color_final, stroke_width=1, opacity=0.25)
        )
        for i in range(1, 10):
            y = int(i * chart_height / 10)
            grid_group.add(dwg.line(start=(0, y), end=(width, y)))

        # Vertical time grid lines (max 20 lines)
        num_vertical_lines = min(20, num_bars // 10 + 1)
        if num_vertical_lines > 1:
            interval = num_bars / num_vertical_lines
            for i in range(num_vertical_lines):
                x = int(i * interval * bar_width)
                grid_group.add(dwg.line(start=(x, 0), end=(x, height)))

    # Create groups for organized SVG structure
    bars_group = dwg.add(dwg.g(id="ohlc_bars"))
    if has_volume:
        volume_group = dwg.add(dwg.g(id="volume"))

    # Draw OHLC bars
    for i in range(num_bars):
        o = float(open_prices[i])
        h = float(high_prices[i])
        l = float(low_prices[i])
        c = float(close_prices[i])

        # Determine color (bullish if close >= open)
        is_bullish = c >= o
        color = up_color_final if is_bullish else down_color_final

        # Calculate positions
        x_center = (i + 0.5) * bar_width
        x_left = x_center - tick_length
        x_right = x_center + tick_length

        # Y coordinates (inverted: 0 is top of chart)
        y_high = chart_height - ((h - price_min) / price_range) * chart_height
        y_low = chart_height - ((l - price_min) / price_range) * chart_height
        y_open = chart_height - ((o - price_min) / price_range) * chart_height
        y_close = chart_height - ((c - price_min) / price_range) * chart_height

        # 1. Draw vertical line (high to low)
        bars_group.add(
            dwg.line(start=(x_center, y_high), end=(x_center, y_low), stroke=color, stroke_width=1)
        )

        # 2. Draw left tick (open)
        bars_group.add(
            dwg.line(start=(x_left, y_open), end=(x_center, y_open), stroke=color, stroke_width=1)
        )

        # 3. Draw right tick (close)
        bars_group.add(
            dwg.line(
                start=(x_center, y_close), end=(x_right, y_close), stroke=color, stroke_width=1
            )
        )

        # 4. Draw volume bar if volume data provided
        if has_volume:
            vol = float(volume_data[i])
            vol_height = (vol / volume_range) * volume_height
            vol_y = height - vol_height
            vol_x = (i + 0.25) * bar_width
            vol_width = bar_width * 0.5

            volume_group.add(
                dwg.rect(
                    insert=(vol_x, vol_y), size=(vol_width, vol_height), fill=color, opacity=0.5
                )
            )

    # Save or return SVG
    if output_path:
        _save_svg_or_svgz(dwg, output_path)

    return dwg.tostring()


def render_line_chart_svg(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike | None = None,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    line_color: str | None = None,
    line_width: int = 2,
    fill_area: bool = False,
    show_grid: bool = True,
    output_path: str | None = None,
) -> str:
    """
    Render line chart as true vector SVG.

    This function creates infinitely scalable vector graphics using SVG,
    as opposed to the raster-based PIL rendering. SVG charts can be scaled
    to any size without losing quality and typically have smaller file sizes.

    Args:
        ohlc: A dictionary containing 'open', 'high', 'low', 'close' arrays.
        volume: An array of volume data. If None, no volume panel is shown.
        width: The width of the output SVG in pixels.
        height: The height of the output SVG in pixels.
        theme: Color theme to use. Options: 'classic', 'modern', 'tradingview', 'light'.
               Defaults to 'classic'.
        bg_color: Override background color (hex string). If None, uses theme color.
        line_color: Override line color (hex string). If None, uses theme's up_color.
        line_width: Width of the line in pixels. Defaults to 2.
        fill_area: Fill area under the line with semi-transparent color. Defaults to False.
        show_grid: Display grid lines for price levels and time markers. Defaults to True.
        output_path: Path to save the SVG file. If None, returns SVG as string without saving.

    Returns:
        SVG content as XML string. If output_path is provided, also saves to file.

    Raises:
        ImportError: If svgwrite is not installed.

    Examples:
        >>> ohlc_dict = {'open': [100, 102], 'high': [103, 105], 'low': [99, 101], 'close': [102, 104]}
        >>> volume_array = np.array([1000, 1500])
        >>> svg_str = render_line_chart_svg(ohlc_dict, volume_array, output_path='chart.svg')
        >>> # Chart saved to chart.svg and SVG string returned

        >>> # Line chart with filled area
        >>> svg_str = render_line_chart_svg(ohlc_dict, volume_array, fill_area=True, output_path='filled.svg')

    Notes:
        - SVG files are infinitely scalable (vector graphics)
        - File sizes are typically smaller than raster formats for charts with <1000 candles
        - For very large datasets (10K+ candles), raster formats may be more efficient
        - SVG can be opened in browsers, edited in Inkscape/Illustrator, embedded in web pages
        - Uses polyline or path for smooth line rendering
    """
    if not SVGWRITE_AVAILABLE:
        raise ImportError(
            "svgwrite is required for SVG export. " "Install with: pip install svgwrite"
        )

    # Get theme colors
    theme_colors = THEMES.get(theme, THEMES["classic"])
    bg_color_final = bg_color or theme_colors["bg"]
    # Line color defaults to theme's up_color
    line_color_final = line_color or theme_colors["up"]
    grid_color_final = theme_colors["grid"]

    # Convert to numpy arrays
    close_prices = np.ascontiguousarray(to_numpy_array(ohlc["close"]))
    high_prices = np.ascontiguousarray(to_numpy_array(ohlc["high"]))
    low_prices = np.ascontiguousarray(to_numpy_array(ohlc["low"]))

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(width, height))

    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=bg_color_final))

    # Define chart areas (70% for price chart, 30% for volume if volume data provided)
    has_volume = volume is not None
    if has_volume:
        volume_data = np.ascontiguousarray(to_numpy_array(volume))
        chart_height = int(height * 0.7)
        volume_height = int(height * 0.3)
    else:
        chart_height = height
        volume_height = 0

    # Price and volume scaling
    price_min = float(np.min(low_prices))
    price_max = float(np.max(high_prices))
    price_range = price_max - price_min

    if has_volume:
        volume_max = float(np.max(volume_data))
        volume_range = volume_max

    # Point spacing calculation
    num_points = len(close_prices)
    point_spacing = width / (num_points + 1)

    # Draw grid lines (background layer)
    if show_grid:
        # Horizontal price grid lines (10 divisions)
        grid_group = dwg.add(
            dwg.g(id="grid", stroke=grid_color_final, stroke_width=1, opacity=0.25)
        )
        for i in range(1, 10):
            y = int(i * chart_height / 10)
            grid_group.add(dwg.line(start=(0, y), end=(width, y)))

        # Vertical time grid lines (max 20 lines)
        num_vertical_lines = min(20, num_points // 10 + 1)
        if num_vertical_lines > 1:
            interval = num_points / num_vertical_lines
            for i in range(num_vertical_lines):
                x = int(i * interval * point_spacing)
                grid_group.add(dwg.line(start=(x, 0), end=(x, height)))

    # Create groups for organized SVG structure
    line_group = dwg.add(dwg.g(id="line"))
    if has_volume:
        volume_group = dwg.add(dwg.g(id="volume"))

    # Calculate line points
    points = []
    for i in range(num_points):
        c = float(close_prices[i])

        # X coordinate
        x = (i + 0.5) * point_spacing

        # Y coordinate (inverted: 0 is top of chart)
        y = chart_height - ((c - price_min) / price_range) * chart_height

        points.append((x, y))

    # Draw filled area if requested
    if fill_area and len(points) > 0:
        # Create polygon from points + bottom edge
        polygon_points = points.copy()
        # Add bottom-right corner
        polygon_points.append((points[-1][0], chart_height))
        # Add bottom-left corner
        polygon_points.append((points[0][0], chart_height))

        # Fill with semi-transparent color (20% opacity)
        line_group.add(dwg.polygon(points=polygon_points, fill=line_color_final, opacity=0.2))

    # Draw line connecting all points
    if len(points) > 1:
        line_group.add(
            dwg.polyline(
                points=points,
                stroke=line_color_final,
                stroke_width=line_width,
                fill="none",
                stroke_linejoin="round",
                stroke_linecap="round",
            )
        )

    # Draw volume bars if volume data provided
    if has_volume:
        bar_spacing = point_spacing * 0.2
        bar_width_val = point_spacing - bar_spacing

        for i in range(num_points):
            vol = float(volume_data[i])
            vol_height = (vol / volume_range) * volume_height
            vol_y = height - vol_height

            x_start = i * point_spacing + bar_spacing / 2

            volume_group.add(
                dwg.rect(
                    insert=(x_start, vol_y),
                    size=(bar_width_val, vol_height),
                    fill=line_color_final,
                    opacity=0.5,
                )
            )

    # Save or return SVG
    if output_path:
        _save_svg_or_svgz(dwg, output_path)

    return dwg.tostring()


def render_renko_chart_svg(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike | None = None,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    box_size: float | None = None,
    reversal_boxes: int = 1,
    show_grid: bool = True,
    output_path: str | None = None,
) -> str:
    """
    Render Renko chart as true vector SVG.

    This function creates infinitely scalable vector graphics for Renko charts,
    which are time-independent price charts showing only fixed-size price movements.
    SVG charts can be scaled to any size without losing quality.

    Args:
        ohlc: A dictionary containing 'open', 'high', 'low', 'close' arrays.
        volume: An array of volume data. If None, no volume panel is shown.
        width: The width of the output SVG in pixels.
        height: The height of the output SVG in pixels.
        theme: Color theme to use. Options: 'classic', 'modern', 'tradingview', 'light'.
               Defaults to 'classic'.
        bg_color: Override background color (hex string). If None, uses theme color.
        up_color: Override color for up bricks (hex string). If None, uses theme color.
        down_color: Override color for down bricks (hex string). If None, uses theme color.
        box_size: Brick size in price units. If None, auto-calculated using ATR.
        reversal_boxes: Boxes needed for trend reversal. Higher values filter more noise.
        show_grid: Display grid lines for price levels and brick markers. Defaults to True.
        output_path: Path to save the SVG file. If None, returns SVG as string without saving.

    Returns:
        SVG content as XML string. If output_path is provided, also saves to file.

    Raises:
        ImportError: If svgwrite is not installed.

    Examples:
        >>> ohlc_dict = {'open': [100, 102, 105], 'high': [101, 104, 106],
        ...              'low': [99, 101, 104], 'close': [100, 103, 105]}
        >>> volume_array = np.array([1000, 1500, 1200])
        >>> svg_str = render_renko_chart_svg(ohlc_dict, volume_array,
        ...                                   box_size=2.0, output_path='renko.svg')
        >>> # Chart saved to renko.svg and SVG string returned

    Notes:
        - SVG files are infinitely scalable (vector graphics)
        - Renko charts filter out minor price fluctuations
        - X-axis represents brick sequence, not time
        - Each brick represents fixed price movement (box_size)
        - Volume displayed as uniform bars (advanced aggregation optional)
    """
    if not SVGWRITE_AVAILABLE:
        raise ImportError(
            "svgwrite is required for SVG export. " "Install with: pip install svgwrite"
        )

    # Calculate Renko bricks from OHLC data
    bricks = calculate_renko_bricks(ohlc, box_size, reversal_boxes)

    # Handle edge case: no bricks generated
    if not bricks:
        # Create empty chart with background
        dwg = svgwrite.Drawing(size=(width, height))
        theme_colors = THEMES.get(theme, THEMES["classic"])
        bg_color_final = bg_color or theme_colors["bg"]
        dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=bg_color_final))

        if output_path:
            _save_svg_or_svgz(dwg, output_path)
        return dwg.tostring()

    # Get actual box_size used for brick calculation
    if box_size is None:
        from ..ops.indicators import calculate_atr

        high_prices = to_numpy_array(ohlc["high"])
        low_prices = to_numpy_array(ohlc["low"])
        close_prices = to_numpy_array(ohlc["close"])
        atr = calculate_atr(high_prices, low_prices, close_prices, period=14, engine="cpu")
        box_size = float(np.nanmedian(atr)) * 0.75

    # Get theme colors
    theme_colors = THEMES.get(theme, THEMES["classic"])
    bg_color_final = bg_color or theme_colors["bg"]
    up_color_final = up_color or theme_colors["up"]
    down_color_final = down_color or theme_colors["down"]
    grid_color_final = theme_colors["grid"]

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(width, height))

    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=bg_color_final))

    # Define chart areas (70% for bricks, 30% for volume)
    has_volume = volume is not None
    chart_height = int(height * 0.7)
    volume_height = int(height * 0.3)

    # Calculate price range for all bricks
    brick_prices = np.array([b["price"] for b in bricks])
    price_min = float(np.min(brick_prices) - box_size)
    price_max = float(np.max(brick_prices) + box_size)
    price_range = price_max - price_min

    if price_range == 0:
        price_range = 1.0  # Avoid division by zero

    # Calculate brick dimensions
    num_bricks = len(bricks)
    brick_width = width / (num_bricks + 1)
    spacing = brick_width * 0.1  # 10% spacing between bricks
    bar_width = brick_width - spacing

    # Calculate brick height (fixed for all bricks)
    brick_height = (box_size / price_range) * chart_height
    if brick_height < 1:
        brick_height = 1  # Minimum 1 pixel

    def scale_price(price: float) -> float:
        """Scale price to chart Y coordinate."""
        return chart_height - ((price - price_min) / price_range) * chart_height

    # Draw grid lines (background layer)
    if show_grid:
        grid_group = dwg.add(
            dwg.g(id="grid", stroke=grid_color_final, stroke_width=1, opacity=0.25)
        )

        # Horizontal price grid lines (10 divisions)
        for i in range(1, 10):
            y = int(i * chart_height / 10)
            grid_group.add(dwg.line(start=(0, y), end=(width, y)))

        # Vertical time grid lines (max 20 lines)
        num_vertical_lines = min(20, num_bricks // 10 + 1)
        if num_vertical_lines > 1:
            interval = num_bricks / num_vertical_lines
            for i in range(num_vertical_lines):
                x = int(i * interval * brick_width)
                grid_group.add(dwg.line(start=(x, 0), end=(x, height)))

    # Create groups for organized SVG structure
    bricks_group = dwg.add(dwg.g(id="bricks"))
    if has_volume:
        volume_group = dwg.add(dwg.g(id="volume"))
        volume_data = np.ascontiguousarray(to_numpy_array(volume))

    # Draw bricks
    for i, brick in enumerate(bricks):
        x_start = i * brick_width + spacing / 2
        x_end = x_start + bar_width

        # Calculate Y position
        # For up bricks: price is the top
        # For down bricks: price is the bottom
        if brick["direction"] == 1:
            # Up brick: draw from price (top) downward by brick_height
            y_top = scale_price(brick["price"])
            y_bottom = y_top + brick_height
            color = up_color_final
        else:
            # Down brick: draw from price (bottom) upward by brick_height
            y_bottom = scale_price(brick["price"])
            y_top = y_bottom - brick_height
            color = down_color_final

        # Draw brick rectangle
        bricks_group.add(
            dwg.rect(
                insert=(x_start, y_top), size=(bar_width, brick_height), fill=color, stroke=color
            )
        )

        # Draw uniform volume bars (simplified - not aggregated per brick)
        if has_volume:
            # For MVP, show uniform volume bars as placeholder
            volume_bar_height = volume_height // 2  # Uniform height
            volume_group.add(
                dwg.rect(
                    insert=(x_start, height - volume_bar_height),
                    size=(bar_width, volume_bar_height),
                    fill=color,
                    opacity=0.5,
                )
            )

    # Save or return SVG
    if output_path:
        _save_svg_or_svgz(dwg, output_path)

    return dwg.tostring()


def render_pnf_chart_svg(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike | None = None,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    box_size: float | None = None,
    reversal_boxes: int = 3,
    show_grid: bool = True,
    output_path: str | None = None,
) -> str:
    """
    Render Point & Figure chart as true vector SVG.

    Point & Figure charts are time-independent price charts showing columns of X's
    (rising prices) and O's (falling prices). This function creates infinitely scalable
    vector graphics using SVG, ideal for high-quality charts and web embedding.

    Args:
        ohlc: A dictionary containing 'open', 'high', 'low', 'close' arrays.
        volume: Volume data (not used in P&F - included for API compatibility).
        width: The width of the output SVG in pixels.
        height: The height of the output SVG in pixels.
        theme: Color theme to use. Options: 'classic', 'modern', 'tradingview', 'light'.
               Defaults to 'classic'.
        bg_color: Override background color (hex string). If None, uses theme color.
        up_color: Override color for X symbols (hex string). If None, uses theme color.
        down_color: Override color for O symbols (hex string). If None, uses theme color.
        box_size: Price per box. If None, auto-calculated using ATR.
        reversal_boxes: Number of boxes needed for trend reversal. Default 3.
        show_grid: Display grid lines for price levels and columns. Defaults to True.
        output_path: Path to save the SVG file. If None, returns SVG as string without saving.

    Returns:
        SVG content as XML string. If output_path is provided, also saves to file.

    Raises:
        ImportError: If svgwrite is not installed.

    Examples:
        >>> ohlc_dict = {'open': [100, 102], 'high': [103, 105], 'low': [99, 101], 'close': [102, 104]}
        >>> svg_str = render_pnf_chart_svg(ohlc_dict, output_path='pnf.svg')
        >>> # Chart saved to pnf.svg and SVG string returned

    Notes:
        - P&F charts are time-independent - only price movement matters
        - X symbols indicate rising prices (bullish)
        - O symbols indicate falling prices (bearish)
        - Volume data is ignored (P&F focuses purely on price)
        - SVG files are infinitely scalable and ideal for web use
    """
    if not SVGWRITE_AVAILABLE:
        raise ImportError(
            "svgwrite is required for SVG export. " "Install with: pip install svgwrite"
        )

    # Get theme colors
    theme_colors = THEMES.get(theme, THEMES["classic"])
    bg_color_final = bg_color or theme_colors["bg"]
    up_color_final = up_color or theme_colors["up"]
    down_color_final = down_color or theme_colors["down"]
    grid_color_final = theme_colors["grid"]

    # Calculate PNF columns using existing algorithm
    columns = calculate_pnf_columns(ohlc, box_size, reversal_boxes)

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(width, height))

    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=bg_color_final))

    # If no columns, return empty chart
    if not columns:
        if output_path:
            _save_svg_or_svgz(dwg, output_path)
        return dwg.tostring()

    # Find price range from all boxes
    all_prices = []
    for col in columns:
        all_prices.extend(col["boxes"])

    # Calculate box_size if it was auto-calculated
    if box_size is None:
        high_prices = to_numpy_array(ohlc["high"])
        low_prices = to_numpy_array(ohlc["low"])
        close_prices = to_numpy_array(ohlc["close"])

        if len(close_prices) >= 14:
            from ..ops.indicators import calculate_atr

            atr = calculate_atr(ohlc["high"], ohlc["low"], ohlc["close"], period=14, engine="cpu")
            box_size = float(np.nanmedian(atr))
        else:
            data_price_range = float(np.max(high_prices) - np.min(low_prices))
            box_size = data_price_range * 0.01 if data_price_range > 0 else 1.0

    price_min = min(all_prices) - box_size
    price_max = max(all_prices) + box_size
    price_range = price_max - price_min

    # Avoid division by zero
    if price_range == 0:
        price_range = 1.0

    def scale_price(price: float) -> float:
        """Convert price to Y coordinate"""
        return height - ((price - price_min) / price_range * height)

    # Calculate column dimensions
    num_columns = len(columns)
    column_width = width / (num_columns + 1)
    box_width = column_width * 0.8

    # Calculate box height based on box_size
    box_height = (box_size / price_range) * height
    box_height = max(box_height, 10.0)

    # Draw grid lines (background layer)
    if show_grid:
        grid_group = dwg.add(
            dwg.g(id="grid", stroke=grid_color_final, stroke_width=1, opacity=0.25)
        )

        # Horizontal price lines (10 divisions)
        for i in range(1, 10):
            y = int(i * height / 10)
            grid_group.add(dwg.line(start=(0, y), end=(width, y)))

        # Vertical column lines
        for col_idx in range(num_columns + 1):
            x = int(col_idx * column_width)
            grid_group.add(dwg.line(start=(x, 0), end=(x, height)))

    # Create groups for organized SVG structure
    symbols_group = dwg.add(dwg.g(id="pnf_symbols"))

    # Draw columns
    for col_idx, column in enumerate(columns):
        x_start = col_idx * column_width
        x_center = x_start + column_width / 2

        for box_price in column["boxes"]:
            y_center = scale_price(box_price)
            half_box = box_width / 2
            half_height = box_height / 2

            if column["type"] == "X":
                # Draw X as two diagonal lines forming an X shape
                # Create a path element for the X
                x_path = dwg.path(
                    d=f"M {x_center - half_box},{y_center - half_height} "
                    f"L {x_center + half_box},{y_center + half_height} "
                    f"M {x_center - half_box},{y_center + half_height} "
                    f"L {x_center + half_box},{y_center - half_height}",
                    stroke=up_color_final,
                    stroke_width=2,
                    fill="none",
                )
                symbols_group.add(x_path)

            else:  # 'O'
                # Draw O as a circle
                o_circle = dwg.circle(
                    center=(x_center, y_center),
                    r=half_box,
                    stroke=down_color_final,
                    stroke_width=2,
                    fill="none",
                )
                symbols_group.add(o_circle)

    # Save or return SVG
    if output_path:
        _save_svg_or_svgz(dwg, output_path)

    return dwg.tostring()


def render_hollow_candles_svg(
    ohlc: dict[str, ArrayLike],
    volume: ArrayLike | None = None,
    width: int = 1920,
    height: int = 1080,
    theme: str = "classic",
    bg_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    show_grid: bool = True,
    output_path: str | None = None,
) -> str:
    """
    Render hollow candlestick chart as true vector SVG.

    Bullish candles (close >= open) are drawn hollow (outline only).
    Bearish candles (close < open) are drawn solid (filled).

    This function creates infinitely scalable vector graphics using SVG,
    as opposed to the raster-based PIL rendering. SVG charts can be scaled
    to any size without losing quality and typically have smaller file sizes
    for charts with moderate numbers of candles.

    Args:
        ohlc: A dictionary containing 'open', 'high', 'low', 'close' arrays.
        volume: An array of volume data. If None, no volume panel is shown.
        width: The width of the output SVG in pixels.
        height: The height of the output SVG in pixels.
        theme: Color theme to use. Options: 'classic', 'modern', 'tradingview', 'light'.
               Defaults to 'classic'.
        bg_color: Override background color (hex string). If None, uses theme color.
        up_color: Override color for bullish candles (hex string). If None, uses theme color.
        down_color: Override color for bearish candles (hex string). If None, uses theme color.
        show_grid: Display grid lines for price levels and time markers. Defaults to True.
        output_path: Path to save the SVG file. If None, returns SVG as string without saving.

    Returns:
        SVG content as XML string. If output_path is provided, also saves to file.

    Raises:
        ImportError: If svgwrite is not installed.

    Examples:
        >>> ohlc_dict = {'open': [100, 102], 'high': [103, 105], 'low': [99, 101], 'close': [102, 104]}
        >>> volume_array = np.array([1000, 1500])
        >>> svg_str = render_hollow_candles_svg(ohlc_dict, volume_array, output_path='chart.svg')
        >>> # Chart saved to chart.svg and SVG string returned

    Notes:
        - Hollow candles provide better visual distinction of trend direction
        - Bullish candles show only the outline (hollow), bearish are filled
        - SVG files are infinitely scalable (vector graphics)
        - File sizes are typically smaller than raster formats for charts with <1000 candles
        - For very large datasets (10K+ candles), raster formats may be more efficient
        - SVG can be opened in browsers, edited in Inkscape/Illustrator, embedded in web pages
    """
    if not SVGWRITE_AVAILABLE:
        raise ImportError(
            "svgwrite is required for SVG export. " "Install with: pip install svgwrite"
        )

    # Get theme colors
    theme_colors = THEMES.get(theme, THEMES["classic"])
    bg_color_final = bg_color or theme_colors["bg"]
    up_color_final = up_color or theme_colors["up"]
    down_color_final = down_color or theme_colors["down"]
    grid_color_final = theme_colors["grid"]

    # Convert to numpy arrays
    open_prices = np.ascontiguousarray(to_numpy_array(ohlc["open"]))
    high_prices = np.ascontiguousarray(to_numpy_array(ohlc["high"]))
    low_prices = np.ascontiguousarray(to_numpy_array(ohlc["low"]))
    close_prices = np.ascontiguousarray(to_numpy_array(ohlc["close"]))

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(width, height))

    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=bg_color_final))

    # Define chart areas (70% for candlestick, 30% for volume if volume data provided)
    has_volume = volume is not None
    if has_volume:
        volume_data = np.ascontiguousarray(to_numpy_array(volume))
        chart_height = int(height * 0.7)
        volume_height = int(height * 0.3)
    else:
        chart_height = height
        volume_height = 0

    # Price and volume scaling
    price_min = float(np.min(low_prices))
    price_max = float(np.max(high_prices))
    price_range = price_max - price_min

    if has_volume:
        volume_max = float(np.max(volume_data))
        volume_range = volume_max

    # Candlestick width calculation
    num_candles = len(open_prices)
    candle_width = width / (num_candles + 1)
    spacing = candle_width * 0.2
    bar_width = candle_width - spacing

    # Wick width (scaled for SVG, minimum 1px)
    wick_width = max(1.0, bar_width * 0.1)

    # Draw grid lines (background layer)
    if show_grid:
        # Horizontal price grid lines (10 divisions)
        grid_group = dwg.add(
            dwg.g(id="grid", stroke=grid_color_final, stroke_width=1, opacity=0.25)
        )
        for i in range(1, 10):
            y = int(i * chart_height / 10)
            grid_group.add(dwg.line(start=(0, y), end=(width, y)))

        # Vertical time grid lines (max 20 lines)
        num_vertical_lines = min(20, num_candles // 10 + 1)
        if num_vertical_lines > 1:
            interval = num_candles / num_vertical_lines
            for i in range(num_vertical_lines):
                x = int(i * interval * candle_width)
                grid_group.add(dwg.line(start=(x, 0), end=(x, height)))

    # Create groups for organized SVG structure
    candles_group = dwg.add(dwg.g(id="candles"))
    if has_volume:
        volume_group = dwg.add(dwg.g(id="volume"))

    # Draw hollow candlesticks
    for i in range(num_candles):
        o = float(open_prices[i])
        h = float(high_prices[i])
        l = float(low_prices[i])
        c = float(close_prices[i])

        # Determine if bullish or bearish
        is_bullish = c >= o
        color = up_color_final if is_bullish else down_color_final

        # Calculate positions
        x = i * candle_width + spacing / 2
        x_center = x + bar_width / 2

        # Y coordinates (inverted: 0 is top of chart)
        y_high = chart_height - ((h - price_min) / price_range) * chart_height
        y_low = chart_height - ((l - price_min) / price_range) * chart_height
        y_open = chart_height - ((o - price_min) / price_range) * chart_height
        y_close = chart_height - ((c - price_min) / price_range) * chart_height

        # Body top and bottom
        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close)
        body_height = body_bottom - body_top

        # Ensure minimum body height for visibility (doji candles)
        if body_height < 1:
            body_height = 1

        # Draw wick (vertical line from low to high)
        candles_group.add(
            dwg.line(
                start=(x_center, y_high),
                end=(x_center, y_low),
                stroke=color,
                stroke_width=wick_width,
            )
        )

        # Draw body - HOLLOW vs FILLED based on direction
        if is_bullish:
            # Bullish: HOLLOW (outline only, no fill)
            candles_group.add(
                dwg.rect(
                    insert=(x, body_top),
                    size=(bar_width, body_height),
                    fill="none",
                    stroke=color,
                    stroke_width=1,
                )
            )
        else:
            # Bearish: FILLED (solid rectangle)
            candles_group.add(
                dwg.rect(insert=(x, body_top), size=(bar_width, body_height), fill=color)
            )

        # Draw volume bar if volume data provided
        if has_volume:
            vol = float(volume_data[i])
            vol_height = (vol / volume_range) * volume_height
            vol_y = height - vol_height

            volume_group.add(
                dwg.rect(insert=(x, vol_y), size=(bar_width, vol_height), fill=color, opacity=0.5)
            )

    # Save or return SVG
    if output_path:
        _save_svg_or_svgz(dwg, output_path)

    return dwg.tostring()

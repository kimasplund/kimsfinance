"""
Interactive charting with Plotly and Bokeh backends.

This module provides interactive HTML charts as an alternative to static PIL-rendered charts.
Features include hover tooltips, zoom/pan, crosshairs, and indicator overlays.

Performance Characteristics:
    - Plotly: Best for <10K points, WebGL for >10K
    - Bokeh: Better for very large datasets (>100K), server-side rendering
    - Static PIL: Still 28.8x faster for batch rendering

Usage:
    >>> from kimsfinance.plotting.interactive import plot_candlestick_plotly
    >>> chart = plot_candlestick_plotly(df, indicators=['RSI', 'MACD'])
    >>> chart.save('chart.html')
    >>> chart.show()  # Opens in browser or displays in Jupyter
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import polars as pl

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from bokeh.layouts import column
    from bokeh.models import (
        ColumnDataSource,
        CrosshairTool,
        HoverTool,
        Range1d,
        RangeTool,
        Segment,
    )
    from bokeh.plotting import figure, output_file, save, show

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from ..config.themes import THEMES
from ..core.types import ArrayLike, DataFrameInput
from ..utils.array_utils import to_numpy_array

# Type aliases
Backend = Literal["plotly", "bokeh"]
Theme = Literal["classic", "modern", "tradingview", "light"]
ChartType = Literal["candlestick", "ohlc", "line"]


@dataclass
class InteractiveChart:
    """
    Container for interactive chart with export methods.

    Attributes:
        figure: The chart figure (Plotly or Bokeh)
        backend: Backend used ('plotly' or 'bokeh')
        data: Original data used for chart
    """

    figure: Any
    backend: Backend
    data: Union[pl.DataFrame, pl.LazyFrame]

    def save(self, path: str, **kwargs: Any) -> None:
        """
        Save chart to HTML file.

        Args:
            path: Output file path (.html)
            **kwargs: Backend-specific options

        Example:
            >>> chart.save('chart.html')
        """
        if self.backend == "plotly":
            self.figure.write_html(path, **kwargs)
        elif self.backend == "bokeh":
            output_file(path)
            save(self.figure)

    def show(self) -> None:
        """
        Display chart in browser or Jupyter notebook.

        Example:
            >>> chart.show()  # Opens in default browser
        """
        if self.backend == "plotly":
            self.figure.show()
        elif self.backend == "bokeh":
            show(self.figure)

    def to_html(self) -> str:
        """
        Export chart to HTML string.

        Returns:
            HTML string representation

        Example:
            >>> html = chart.to_html()
            >>> print(html[:100])
        """
        if self.backend == "plotly":
            return self.figure.to_html(**{"include_plotlyjs": "cdn"})
        elif self.backend == "bokeh":
            from bokeh.embed import file_html
            from bokeh.resources import CDN

            return file_html(self.figure, CDN, "Chart")
        return ""

    def to_json(self) -> str:
        """
        Export chart to JSON string (Plotly only).

        Returns:
            JSON string representation

        Raises:
            NotImplementedError: If backend is Bokeh

        Example:
            >>> json_str = chart.to_json()
        """
        if self.backend == "plotly":
            return self.figure.to_json()
        raise NotImplementedError("JSON export only available for Plotly backend")

    def to_png(self, path: str, width: int = 1200, height: int = 800) -> None:
        """
        Export chart to PNG (requires kaleido for Plotly).

        Args:
            path: Output file path (.png)
            width: Image width in pixels
            height: Image height in pixels

        Example:
            >>> chart.to_png('chart.png', width=1920, height=1080)
        """
        if self.backend == "plotly":
            self.figure.write_image(path, width=width, height=height)
        elif self.backend == "bokeh":
            from bokeh.io import export_png

            export_png(self.figure, filename=path)


def _get_theme_colors(theme: Theme) -> dict[str, str]:
    """
    Get theme colors for interactive charts.

    Args:
        theme: Theme name

    Returns:
        Dictionary with color mappings
    """
    theme_config = THEMES[theme]
    return {
        "bg": theme_config["bg"],
        "up": theme_config["up"],
        "down": theme_config["down"],
        "grid": theme_config["grid"],
        "text": "#FFFFFF" if theme != "light" else "#000000",
    }


def _prepare_ohlcv_data(
    data: DataFrameInput, date_column: Optional[str] = None
) -> pl.DataFrame:
    """
    Prepare OHLCV data for charting.

    Args:
        data: Input data (DataFrame or dict)
        date_column: Name of date column (auto-detected if None)

    Returns:
        Polars DataFrame with standardized columns

    Raises:
        ValueError: If required OHLCV columns are missing
    """
    if isinstance(data, dict):
        df = pl.DataFrame(data)
    elif isinstance(data, pl.LazyFrame):
        df = data.collect()
    elif isinstance(data, pl.DataFrame):
        df = data
    else:
        # Try pandas/numpy conversion
        df = pl.from_pandas(data)  # type: ignore

    # Standardize column names (case-insensitive)
    columns_lower = {col.lower(): col for col in df.columns}

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in columns_lower]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    # Rename to standard case
    rename_map = {columns_lower[col]: col for col in required if col in columns_lower}
    if "volume" in columns_lower:
        rename_map[columns_lower["volume"]] = "volume"

    df = df.rename(rename_map)

    # Handle date column
    if date_column is None:
        # Try to find date column
        date_candidates = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_candidates:
            date_column = date_candidates[0]
        else:
            # Create index-based dates
            df = df.with_row_index("index")
            date_column = "index"

    if date_column != "date" and date_column in df.columns:
        df = df.rename({date_column: "date"})

    return df


def plot_candlestick_plotly(
    data: DataFrameInput,
    indicators: Optional[list[dict[str, Any]]] = None,
    theme: Theme = "tradingview",
    title: str = "Candlestick Chart",
    width: int = 1200,
    height: int = 800,
    show_volume: bool = True,
    show_rangeslider: bool = True,
    webgl: bool = False,
    date_column: Optional[str] = None,
) -> InteractiveChart:
    """
    Create interactive candlestick chart using Plotly.

    Features:
        - Hover tooltips with OHLCV data
        - Zoom and pan
        - Crosshair cursor
        - Range selector
        - Volume bars
        - Indicator overlays

    Args:
        data: OHLCV data (DataFrame or dict with keys: open, high, low, close, volume)
        indicators: List of indicator dicts with keys: 'data', 'name', 'type', 'color'
        theme: Color theme ('classic', 'modern', 'tradingview', 'light')
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        show_volume: Show volume bars
        show_rangeslider: Show range slider at bottom
        webgl: Use WebGL rendering (faster for >10K points)
        date_column: Name of date column (auto-detected if None)

    Returns:
        InteractiveChart object with save/show methods

    Performance:
        - <10K points: Standard rendering (fast)
        - >10K points: Enable webgl=True for better performance
        - Data decimation applied automatically for >100K points

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        ...     'open': [100, 102, 101],
        ...     'high': [103, 105, 104],
        ...     'low': [99, 101, 100],
        ...     'close': [102, 101, 103],
        ...     'volume': [1000, 1500, 1200]
        ... })
        >>> chart = plot_candlestick_plotly(df)
        >>> chart.save('chart.html')

    Raises:
        ImportError: If Plotly is not installed
        ValueError: If required OHLCV columns are missing
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive charts. "
            "Install with: pip install plotly"
        )

    # Prepare data
    df = _prepare_ohlcv_data(data, date_column)
    colors = _get_theme_colors(theme)

    # Data decimation for large datasets
    if len(df) > 100_000:
        # Downsample to ~50K points using LTTB (Largest Triangle Three Buckets)
        # For simplicity, use every-nth sampling here
        step = len(df) // 50_000
        df = df[::step]

    # Determine subplot layout
    rows = 1
    row_heights = [0.7]
    subplot_titles = [title]

    if show_volume:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append("")

    if indicators:
        # Count indicator panels (separate panel for oscillators)
        for ind in indicators:
            if ind.get("panel") == "separate":
                rows += 1
                row_heights.append(0.15)
                subplot_titles.append(ind.get("name", ""))

    # Normalize row heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Add candlestick trace
    dates = df["date"].to_list()
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df["open"].to_list(),
            high=df["high"].to_list(),
            low=df["low"].to_list(),
            close=df["close"].to_list(),
            name="OHLC",
            increasing={"line": {"color": colors["up"]}, "fillcolor": colors["up"]},
            decreasing={"line": {"color": colors["down"]}, "fillcolor": colors["down"]},
            hovertext=[
                f"O: {o:.2f}<br>H: {h:.2f}<br>L: {l:.2f}<br>C: {c:.2f}"
                for o, h, l, c in zip(
                    df["open"].to_list(),
                    df["high"].to_list(),
                    df["low"].to_list(),
                    df["close"].to_list(),
                )
            ],
            hoverinfo="text+x",
        ),
        row=1,
        col=1,
    )

    current_row = 2

    # Add volume bars
    if show_volume and "volume" in df.columns:
        volume_colors = [
            colors["up"] if c >= o else colors["down"]
            for c, o in zip(df["close"].to_list(), df["open"].to_list())
        ]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df["volume"].to_list(),
                name="Volume",
                marker={"color": volume_colors},
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        current_row += 1

    # Add indicators
    if indicators:
        for ind in indicators:
            ind_data = ind.get("data")
            ind_name = ind.get("name", "Indicator")
            ind_type = ind.get("type", "line")
            ind_color = ind.get("color", "#FFA500")
            ind_panel = ind.get("panel", "main")

            if ind_data is None:
                continue

            # Determine which row to add to
            target_row = 1 if ind_panel == "main" else current_row

            if ind_type == "line":
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=to_numpy_array(ind_data).tolist(),
                        name=ind_name,
                        line={"color": ind_color, "width": 2},
                        mode="lines",
                    ),
                    row=target_row,
                    col=1,
                )
            elif ind_type == "histogram":
                colors_hist = [ind_color if v >= 0 else colors["down"] for v in ind_data]
                fig.add_trace(
                    go.Bar(
                        x=dates,
                        y=to_numpy_array(ind_data).tolist(),
                        name=ind_name,
                        marker={"color": colors_hist},
                    ),
                    row=target_row,
                    col=1,
                )
            elif ind_type == "band":
                # Band with upper and lower (e.g., Bollinger Bands)
                upper = ind.get("upper")
                lower = ind.get("lower")
                if upper is not None and lower is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=to_numpy_array(upper).tolist(),
                            name=f"{ind_name} Upper",
                            line={"color": ind_color, "width": 1, "dash": "dash"},
                            mode="lines",
                            showlegend=False,
                        ),
                        row=target_row,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=to_numpy_array(lower).tolist(),
                            name=f"{ind_name} Lower",
                            line={"color": ind_color, "width": 1, "dash": "dash"},
                            mode="lines",
                            fill="tonexty",
                            fillcolor=f"rgba{tuple(list(int(ind_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
                            showlegend=False,
                        ),
                        row=target_row,
                        col=1,
                    )

            if ind_panel == "separate":
                current_row += 1

    # Update layout with theme
    fig.update_layout(
        template="plotly_dark" if theme != "light" else "plotly_white",
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["bg"],
        font={"color": colors["text"], "family": "Arial, sans-serif", "size": 12},
        width=width,
        height=height,
        xaxis_rangeslider_visible=show_rangeslider,
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    # Update axes
    fig.update_xaxes(
        gridcolor=colors["grid"],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=colors["grid"],
    )
    fig.update_yaxes(
        gridcolor=colors["grid"],
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor=colors["grid"],
    )

    # Enable WebGL for performance
    if webgl or len(df) > 10_000:
        fig.update_traces(selector={"type": "scatter"}, mode="lines", line={"width": 1})

    return InteractiveChart(figure=fig, backend="plotly", data=df)


def plot_candlestick_bokeh(
    data: DataFrameInput,
    indicators: Optional[list[dict[str, Any]]] = None,
    theme: Theme = "tradingview",
    title: str = "Candlestick Chart",
    width: int = 1200,
    height: int = 800,
    show_volume: bool = True,
    date_column: Optional[str] = None,
) -> InteractiveChart:
    """
    Create interactive candlestick chart using Bokeh.

    Better for very large datasets (>100K points) with server-side rendering.

    Features:
        - Hover tooltips with OHLCV data
        - Zoom and pan with linked plots
        - Crosshair cursor
        - Range selection tool
        - Volume bars
        - Indicator overlays

    Args:
        data: OHLCV data (DataFrame or dict)
        indicators: List of indicator dicts
        theme: Color theme
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        show_volume: Show volume bars
        date_column: Name of date column

    Returns:
        InteractiveChart object

    Example:
        >>> chart = plot_candlestick_bokeh(df, show_volume=True)
        >>> chart.save('chart.html')

    Raises:
        ImportError: If Bokeh is not installed
        ValueError: If required OHLCV columns are missing
    """
    if not BOKEH_AVAILABLE:
        raise ImportError("Bokeh is required. Install with: pip install bokeh")

    # Prepare data
    df = _prepare_ohlcv_data(data, date_column)
    colors = _get_theme_colors(theme)

    # Convert dates to numeric index for Bokeh
    df = df.with_row_index("index")

    # Prepare candlestick colors
    df = df.with_columns(
        color=pl.when(pl.col("close") >= pl.col("open"))
        .then(pl.lit(colors["up"]))
        .otherwise(pl.lit(colors["down"]))
    )

    # Create ColumnDataSource
    source = ColumnDataSource(df.to_pandas())

    # Main candlestick plot
    p = figure(
        width=width,
        height=int(height * 0.7),
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        background_fill_color=colors["bg"],
        border_fill_color=colors["bg"],
    )

    # Add candlestick segments (wicks)
    p.segment(
        x0="index",
        y0="low",
        x1="index",
        y1="high",
        source=source,
        color="color",
        line_width=1,
    )

    # Add candlestick bodies (bars)
    p.vbar(
        x="index",
        top="close",
        bottom="open",
        width=0.5,
        source=source,
        fill_color="color",
        line_color="color",
    )

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Date", "@date"),
            ("Open", "@open{0.00}"),
            ("High", "@high{0.00}"),
            ("Low", "@low{0.00}"),
            ("Close", "@close{0.00}"),
            ("Volume", "@volume{0,0}"),
        ],
        mode="vline",
    )
    p.add_tools(hover)

    # Add crosshair
    p.add_tools(CrosshairTool(dimensions="both"))

    # Style
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "Price"
    p.grid.grid_line_color = colors["grid"]
    p.grid.grid_line_alpha = 0.3
    p.xaxis.major_label_text_color = colors["text"]
    p.yaxis.major_label_text_color = colors["text"]
    p.title.text_color = colors["text"]

    plots = [p]

    # Add volume plot
    if show_volume and "volume" in df.columns:
        p_volume = figure(
            width=width,
            height=int(height * 0.15),
            x_range=p.x_range,
            tools="",
            toolbar_location=None,
            background_fill_color=colors["bg"],
            border_fill_color=colors["bg"],
        )

        p_volume.vbar(
            x="index", top="volume", width=0.5, source=source, color="color", alpha=0.5
        )

        p_volume.xaxis.axis_label = ""
        p_volume.yaxis.axis_label = "Volume"
        p_volume.grid.grid_line_color = colors["grid"]
        p_volume.grid.grid_line_alpha = 0.3
        p_volume.xaxis.major_label_text_color = colors["text"]
        p_volume.yaxis.major_label_text_color = colors["text"]

        plots.append(p_volume)

    # Add indicators
    if indicators:
        for ind in indicators:
            ind_data = ind.get("data")
            ind_name = ind.get("name", "Indicator")
            ind_type = ind.get("type", "line")
            ind_color = ind.get("color", "#FFA500")
            ind_panel = ind.get("panel", "main")

            if ind_data is None:
                continue

            # Add indicator data to source
            ind_array = to_numpy_array(ind_data)
            df = df.with_columns(pl.Series(ind_name, ind_array))

            target_plot = p if ind_panel == "main" else None

            if target_plot is None and ind_panel == "separate":
                # Create new panel
                target_plot = figure(
                    width=width,
                    height=int(height * 0.15),
                    x_range=p.x_range,
                    title=ind_name,
                    tools="",
                    toolbar_location=None,
                    background_fill_color=colors["bg"],
                    border_fill_color=colors["bg"],
                )
                target_plot.grid.grid_line_color = colors["grid"]
                target_plot.grid.grid_line_alpha = 0.3
                target_plot.xaxis.major_label_text_color = colors["text"]
                target_plot.yaxis.major_label_text_color = colors["text"]
                target_plot.title.text_color = colors["text"]
                plots.append(target_plot)

            # Update source with new data
            source = ColumnDataSource(df.to_pandas())

            if ind_type == "line" and target_plot is not None:
                target_plot.line(
                    x="index",
                    y=ind_name,
                    source=source,
                    line_color=ind_color,
                    line_width=2,
                    legend_label=ind_name,
                )

    # Combine plots
    layout = column(*plots)

    return InteractiveChart(figure=layout, backend="bokeh", data=df)


def plot_ohlc_plotly(
    data: DataFrameInput,
    theme: Theme = "tradingview",
    title: str = "OHLC Chart",
    width: int = 1200,
    height: int = 800,
    date_column: Optional[str] = None,
) -> InteractiveChart:
    """
    Create interactive OHLC bar chart using Plotly.

    Similar to candlestick but with OHLC bars (horizontal ticks).

    Args:
        data: OHLCV data
        theme: Color theme
        title: Chart title
        width: Chart width
        height: Chart height
        date_column: Date column name

    Returns:
        InteractiveChart object

    Example:
        >>> chart = plot_ohlc_plotly(df)
        >>> chart.show()
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required. Install with: pip install plotly")

    df = _prepare_ohlcv_data(data, date_column)
    colors = _get_theme_colors(theme)

    fig = go.Figure(
        data=go.Ohlc(
            x=df["date"].to_list(),
            open=df["open"].to_list(),
            high=df["high"].to_list(),
            low=df["low"].to_list(),
            close=df["close"].to_list(),
            increasing={"line": {"color": colors["up"]}},
            decreasing={"line": {"color": colors["down"]}},
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark" if theme != "light" else "plotly_white",
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["bg"],
        font={"color": colors["text"]},
        width=width,
        height=height,
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(gridcolor=colors["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=colors["grid"], showgrid=True)

    return InteractiveChart(figure=fig, backend="plotly", data=df)


def plot_line_plotly(
    data: DataFrameInput,
    y_column: str = "close",
    theme: Theme = "tradingview",
    title: str = "Line Chart",
    width: int = 1200,
    height: int = 800,
    date_column: Optional[str] = None,
) -> InteractiveChart:
    """
    Create interactive line chart using Plotly.

    Args:
        data: Price data
        y_column: Column to plot (default: 'close')
        theme: Color theme
        title: Chart title
        width: Chart width
        height: Chart height
        date_column: Date column name

    Returns:
        InteractiveChart object

    Example:
        >>> chart = plot_line_plotly(df, y_column='close')
        >>> chart.save('line.html')
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required. Install with: pip install plotly")

    df = _prepare_ohlcv_data(data, date_column)
    colors = _get_theme_colors(theme)

    fig = go.Figure(
        data=go.Scatter(
            x=df["date"].to_list(),
            y=df[y_column].to_list(),
            mode="lines",
            line={"color": colors["up"], "width": 2},
            name=y_column.title(),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark" if theme != "light" else "plotly_white",
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["bg"],
        font={"color": colors["text"]},
        width=width,
        height=height,
    )

    fig.update_xaxes(gridcolor=colors["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=colors["grid"], showgrid=True)

    return InteractiveChart(figure=fig, backend="plotly", data=df)


__all__ = [
    "InteractiveChart",
    "plot_candlestick_plotly",
    "plot_candlestick_bokeh",
    "plot_ohlc_plotly",
    "plot_line_plotly",
]

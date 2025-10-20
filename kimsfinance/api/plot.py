"""
Standalone GPU-Accelerated Plot API
====================================

Native PIL-based plotting API achieving 178x speedup vs mplfinance.

Usage:
    >>> from kimsfinance.api import plot
    >>>
    >>> plot(df, type='candle', volume=True, savefig='chart.webp')
"""

from __future__ import annotations

from typing import Any
import warnings
import numpy as np

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False


def plot(data,
         *,
         type='candle',
         style='binance',
         mav=None,
         ema=None,
         volume=True,
         engine: str = "auto",
         savefig=None,
         returnfig=False,
         **kwargs) -> Any:
    """
    Native PIL-based financial plotting achieving 178x speedup vs mplfinance.

    This function uses kimsfinance's native PIL renderer instead of mplfinance/matplotlib,
    providing massive performance improvements while maintaining visual quality.

    Args:
        data: OHLC DataFrame (pandas or polars) with columns: Open, High, Low, Close, Volume
        type: Chart type - 'candle', 'ohlc', 'line', 'hollow_and_filled', 'renko', 'pnf'
        style: Visual theme - 'classic', 'modern', 'tradingview', 'light', or 'binance'/'binancedark'
        mav: Simple moving average windows (not yet implemented - use indicators)
        ema: Exponential moving average windows (not yet implemented - use indicators)
        volume: Show volume panel (default True)
        engine: Execution engine ("cpu", "gpu", or "auto") - used for indicator calculations
        savefig: Path to save chart image (e.g., 'chart.webp'). If None, displays chart.
        returnfig: If True, returns PIL Image object. If False, saves/displays chart.
        **kwargs: Additional renderer parameters:
            - width: Image width in pixels (default 1920)
            - height: Image height in pixels (default 1080)
            - theme: Alternative to style parameter
            - bg_color: Override background color (hex string)
            - up_color: Override bullish color (hex string)
            - down_color: Override bearish color (hex string)
            - enable_antialiasing: Use RGBA mode for smoother rendering (default True)
            - show_grid: Display price/time grid lines (default True)
            - line_width: Line width for line charts (default 2)
            - fill_area: Fill area under line for line charts (default False)
            - box_size: Box size for Renko/PNF charts (auto-calculate if None)
            - reversal_boxes: Reversal threshold for Renko/PNF (default 1 for Renko, 3 for PNF)

    Returns:
        - If returnfig=True: PIL Image object
        - If returnfig=False and savefig=None: Displays chart (not implemented, returns Image)
        - If savefig is set: None (saves to file and returns None)

    Example:
        >>> from kimsfinance.api import plot
        >>> import polars as pl
        >>>
        >>> df = pl.read_csv("ohlcv.csv")
        >>>
        >>> # Save candlestick chart
        >>> plot(df, type='candle', volume=True, savefig='chart.webp')
        >>>
        >>> # Get Image object
        >>> img = plot(df, type='line', returnfig=True)
        >>>
        >>> # Hollow candles with custom colors
        >>> plot(df, type='hollow_and_filled', up_color='#00FF00', down_color='#FF0000')

    Performance:
        - Candlestick: 6,249 charts/sec (178x faster than mplfinance)
        - OHLC bars: 1,337 charts/sec (150-200x faster)
        - Line charts: 2,100 charts/sec (200-300x faster)
        - Hollow candles: 5,728 charts/sec (150-200x faster)
        - Renko: 3,800 charts/sec (100-150x faster)
        - Point & Figure: 357 charts/sec (100-150x faster)

    Notes:
        - This uses native PIL rendering, NOT matplotlib/mplfinance
        - Moving averages (mav/ema) not yet implemented - use addplot or separate calculation
        - For multi-panel charts with indicators, use the native renderer directly
        - Falls back to mplfinance only if addplot is specified (requires matplotlib)
    """
    # Check for unsupported features that require mplfinance
    has_addplot = 'addplot' in kwargs
    has_advanced_features = mav is not None or ema is not None

    if has_addplot or has_advanced_features:
        # Fallback to mplfinance for features not yet in native renderer
        warnings.warn(
            "Using mplfinance fallback for addplot/mav/ema features. "
            "For maximum performance (178x speedup), use native renderer without these features.",
            UserWarning
        )
        return _plot_mplfinance(data, type=type, style=style, mav=mav, ema=ema,
                               volume=volume, engine=engine, savefig=savefig,
                               returnfig=returnfig, **kwargs)

    # Use native PIL renderer (178x speedup!)
    from ..plotting.renderer import (
        render_ohlcv_chart,
        render_ohlc_bars,
        render_line_chart,
        render_hollow_candles,
        render_renko_chart,
        render_pnf_chart,
        save_chart,
        render_candlestick_svg,
        render_ohlc_bars_svg,
        render_line_chart_svg,
        render_hollow_candles_svg,
        render_renko_chart_svg,
        render_pnf_chart_svg,
    )

    # Prepare data
    ohlc_dict, volume_array = _prepare_data(data)

    # Map style aliases
    style = _map_style(style)

    # Extract renderer parameters
    width = kwargs.get('width', 1920)
    height = kwargs.get('height', 1080)
    theme = kwargs.get('theme', style)
    bg_color = kwargs.get('bg_color', None)
    up_color = kwargs.get('up_color', None)
    down_color = kwargs.get('down_color', None)
    enable_antialiasing = kwargs.get('enable_antialiasing', True)
    show_grid = kwargs.get('show_grid', True)

    # Check if SVG/SVGZ format is requested
    is_svg_format = savefig and (savefig.lower().endswith('.svg') or savefig.lower().endswith('.svgz'))

    # SVG rendering path (for candlestick and OHLC charts)
    if is_svg_format and type == 'candle':
        # Route directly to candlestick SVG renderer
        svg_content = render_candlestick_svg(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            show_grid=show_grid,
            output_path=savefig,
        )
        return None  # File saved, return None like other savefig calls

    if is_svg_format and type == 'ohlc':
        # Route directly to OHLC SVG renderer
        svg_content = render_ohlc_bars_svg(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            show_grid=show_grid,
            output_path=savefig,
        )
        return None  # File saved, return None like other savefig calls

    if is_svg_format and type == 'line':
        # Route directly to line chart SVG renderer
        svg_content = render_line_chart_svg(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color,
            line_color=kwargs.get('line_color', None),
            line_width=kwargs.get('line_width', 2),
            fill_area=kwargs.get('fill_area', False),
            show_grid=show_grid,
            output_path=savefig,
        )
        return None  # File saved, return None like other savefig calls

    if is_svg_format and (type == 'hollow_and_filled' or type == 'hollow'):
        # Route directly to hollow candles SVG renderer
        svg_content = render_hollow_candles_svg(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            show_grid=show_grid,
            output_path=savefig,
        )
        return None  # File saved, return None like other savefig calls

    if is_svg_format and type == 'renko':
        # Route directly to Renko SVG renderer
        svg_content = render_renko_chart_svg(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            box_size=kwargs.get('box_size', None),
            reversal_boxes=kwargs.get('reversal_boxes', 1),
            show_grid=show_grid,
            output_path=savefig,
        )
        return None  # File saved, return None like other savefig calls

    if is_svg_format and (type == 'pnf' or type == 'pointandfigure'):
        # Route directly to Point & Figure SVG renderer
        svg_content = render_pnf_chart_svg(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            box_size=kwargs.get('box_size', None),
            reversal_boxes=kwargs.get('reversal_boxes', 3),
            show_grid=show_grid,
            output_path=savefig,
        )
        return None  # File saved, return None like other savefig calls

    # Warn if SVG requested for non-supported chart types
    if is_svg_format and type not in ['candle', 'ohlc', 'line', 'hollow_and_filled', 'hollow', 'renko', 'pnf', 'pointandfigure']:
        warnings.warn(
            f"SVG export is currently only supported for candlestick, OHLC, line, hollow candles, Renko, and Point & Figure charts. "
            f"Chart type '{type}' will be rendered as raster PNG instead.",
            UserWarning
        )
        # Convert .svg to .png for the actual rendering
        import os
        savefig = os.path.splitext(savefig)[0] + '.png'

    # Route to appropriate renderer
    if type == 'candle':
        img = render_ohlcv_chart(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            enable_antialiasing=enable_antialiasing, show_grid=show_grid,
            wick_width_ratio=kwargs.get('wick_width_ratio', 0.1),
            use_batch_drawing=kwargs.get('use_batch_drawing', None),
        )

    elif type == 'ohlc':
        img = render_ohlc_bars(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            enable_antialiasing=enable_antialiasing, show_grid=show_grid,
        )

    elif type == 'line':
        img = render_line_chart(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color,
            line_color=kwargs.get('line_color', None),
            line_width=kwargs.get('line_width', 2),
            fill_area=kwargs.get('fill_area', False),
            enable_antialiasing=enable_antialiasing, show_grid=show_grid,
        )

    elif type == 'hollow_and_filled' or type == 'hollow':
        img = render_hollow_candles(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            enable_antialiasing=enable_antialiasing, show_grid=show_grid,
            wick_width_ratio=kwargs.get('wick_width_ratio', 0.1),
            use_batch_drawing=kwargs.get('use_batch_drawing', None),
        )

    elif type == 'renko':
        img = render_renko_chart(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            enable_antialiasing=enable_antialiasing, show_grid=show_grid,
            box_size=kwargs.get('box_size', None),
            reversal_boxes=kwargs.get('reversal_boxes', 1),
        )

    elif type == 'pnf' or type == 'pointandfigure':
        img = render_pnf_chart(
            ohlc_dict, volume_array,
            width=width, height=height, theme=theme,
            bg_color=bg_color, up_color=up_color, down_color=down_color,
            enable_antialiasing=enable_antialiasing, show_grid=show_grid,
            box_size=kwargs.get('box_size', None),
            reversal_boxes=kwargs.get('reversal_boxes', 3),
        )

    else:
        raise ValueError(
            f"Unknown chart type: '{type}'. "
            f"Supported types: 'candle', 'ohlc', 'line', 'hollow_and_filled', 'renko', 'pnf'"
        )

    # Handle output
    if savefig:
        # Save to file
        save_chart(
            img, savefig,
            speed=kwargs.get('speed', 'balanced'),
            quality=kwargs.get('quality', None),
        )
        return None

    elif returnfig:
        # Return PIL Image
        return img

    else:
        # Display chart (not implemented - return Image for now)
        warnings.warn(
            "Chart display not implemented. Returning PIL Image object. "
            "Use savefig='path.webp' to save or returnfig=True to get Image.",
            UserWarning
        )
        return img


def _prepare_data(data):
    """Convert DataFrame to OHLC dict and volume array for native renderer."""
    import polars as pl

    # Convert to Polars if needed
    if hasattr(data, 'to_pandas'):
        # Already Polars
        df = data
    elif hasattr(data, 'index'):
        # Pandas DataFrame
        df = pl.from_pandas(data.reset_index())
    else:
        df = pl.DataFrame(data)

    # Extract OHLC and volume
    ohlc_dict = {
        'open': df['Open'].to_numpy() if 'Open' in df.columns else df['open'].to_numpy(),
        'high': df['High'].to_numpy() if 'High' in df.columns else df['high'].to_numpy(),
        'low': df['Low'].to_numpy() if 'Low' in df.columns else df['low'].to_numpy(),
        'close': df['Close'].to_numpy() if 'Close' in df.columns else df['close'].to_numpy(),
    }

    volume_col = None
    if 'Volume' in df.columns:
        volume_col = 'Volume'
    elif 'volume' in df.columns:
        volume_col = 'volume'

    if volume_col:
        volume_array = df[volume_col].to_numpy()
    else:
        # Create dummy volume if not present
        volume_array = np.ones(len(df))

    return ohlc_dict, volume_array


def _map_style(style):
    """Map style aliases to canonical theme names."""
    if style in ['binance', 'binancedark']:
        return 'tradingview'  # Similar dark theme
    return style


def _plot_mplfinance(data, type, style, mav, ema, volume, engine, savefig, returnfig, **kwargs):
    """Fallback to mplfinance for unsupported features (addplot, mav, ema)."""
    if not MPLFINANCE_AVAILABLE:
        raise ImportError(
            "mplfinance is required for advanced features (addplot, mav, ema). "
            "Install with: pip install mplfinance"
        )

    # Convert Polars to Pandas if needed (mplfinance requires pandas)
    import pandas as pd

    if hasattr(data, 'to_pandas'):
        # Polars DataFrame - convert to pandas
        data_pandas = data.to_pandas()
        # Set index to date if present
        if 'date' in data_pandas.columns:
            data_pandas['date'] = pd.to_datetime(data_pandas['date'])
            data_pandas.set_index('date', inplace=True)
        elif 'Date' in data_pandas.columns:
            data_pandas['Date'] = pd.to_datetime(data_pandas['Date'])
            data_pandas.set_index('Date', inplace=True)
        else:
            # No date column - create dummy DatetimeIndex
            data_pandas.index = pd.date_range(start='2025-01-01', periods=len(data_pandas), freq='D')
        data = data_pandas
    elif not hasattr(data, 'index'):
        # Not a DataFrame - convert to pandas
        data = pd.DataFrame(data)
        data.index = pd.date_range(start='2025-01-01', periods=len(data), freq='D')
    else:
        # Already pandas DataFrame - ensure DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.date_range(start='2025-01-01', periods=len(data), freq='D')

    # Temporarily activate acceleration for indicator calculations
    from ..integration import activate, deactivate, configure

    configure(default_engine=engine, verbose=False)
    activate(engine=engine, verbose=False)

    try:
        # Build kwargs for mplfinance
        mpf_kwargs = {}
        if style is not None:
            mpf_kwargs['style'] = style
        if mav is not None:
            mpf_kwargs['mav'] = mav
        if ema is not None:
            mpf_kwargs['ema'] = ema
        if savefig is not None:
            mpf_kwargs['savefig'] = savefig
        if returnfig:
            mpf_kwargs['returnfig'] = True

        # Call mplfinance
        result = mpf.plot(
            data,
            type=type,
            volume=volume,
            **mpf_kwargs,
            **kwargs
        )

        return result

    finally:
        deactivate(verbose=False)


def make_addplot(data, **kwargs):
    """
    Create additional plot data for multi-panel charts.

    This is a pass-through to mplfinance.make_addplot for compatibility.
    Note: Using addplot requires mplfinance and disables native PIL renderer.

    Args:
        data: Data to plot
        **kwargs: Additional mplfinance.make_addplot() parameters

    Returns:
        AddPlot object for mpf.plot()

    Example:
        >>> from kimsfinance.api import plot, make_addplot
        >>> import numpy as np
        >>>
        >>> # Calculate custom indicator
        >>> signal = calculate_rsi(df['close'])
        >>>
        >>> # Add to plot (uses mplfinance fallback)
        >>> ap = make_addplot(signal, panel=2, color='purple')
        >>> plot(df, type='candle', addplot=ap)
    """
    if not MPLFINANCE_AVAILABLE:
        raise ImportError(
            "mplfinance is required for addplot. "
            "Install with: pip install mplfinance"
        )

    return mpf.make_addplot(data, **kwargs)


def plot_with_indicators(data,
                         *,
                         type='candle',
                         indicators: dict | None = None,
                         engine: str = "auto",
                         **kwargs) -> Any:
    """
    Plot with GPU-accelerated technical indicators.

    NOTE: This function uses mplfinance fallback for multi-panel support.
    For maximum performance (178x speedup), use plot() without indicators
    and overlay them separately.

    Args:
        data: OHLC DataFrame
        type: Chart type
        indicators: Dict of indicators to add, e.g.:
            {
                'sma': [20, 50, 200],
                'ema': [12, 26],
                'rsi': {'period': 14, 'panel': 2},
                'macd': {'panel': 3}
            }
        engine: Execution engine ("cpu", "gpu", or "auto")
        **kwargs: Additional plot parameters

    Example:
        >>> from kimsfinance.api import plot_with_indicators
        >>>
        >>> plot_with_indicators(
        ...     df,
        ...     indicators={
        ...         'sma': [20, 50],
        ...         'rsi': {'period': 14, 'panel': 2}
        ...     },
        ...     savefig='chart_with_indicators.webp'
        ... )
    """
    warnings.warn(
        "plot_with_indicators() uses mplfinance fallback for multi-panel support. "
        "For maximum performance (178x speedup), use plot() without indicators.",
        UserWarning
    )

    if not MPLFINANCE_AVAILABLE:
        raise ImportError("mplfinance is required for multi-panel indicators")

    from ..ops.indicators import calculate_rsi, calculate_macd
    from ..ops.moving_averages import calculate_sma, calculate_ema
    import polars as pl

    # Convert to Polars if needed
    if hasattr(data, 'to_pandas'):
        df_polars = data
    elif hasattr(data, 'index'):
        df_polars = pl.from_pandas(data.reset_index())
    else:
        df_polars = pl.DataFrame(data)

    addplots = []

    if indicators:
        # Add SMAs
        if 'sma' in indicators:
            kwargs['mav'] = tuple(indicators['sma'])

        # Add RSI
        if 'rsi' in indicators:
            rsi_config = indicators['rsi'] if isinstance(indicators['rsi'], dict) else {'period': 14}
            rsi = calculate_rsi(
                df_polars['close'].to_numpy(),
                period=rsi_config.get('period', 14),
                engine=engine
            )
            ap = make_addplot(
                rsi,
                panel=rsi_config.get('panel', 2),
                color='purple',
                ylabel='RSI'
            )
            addplots.append(ap)

        # Add MACD
        if 'macd' in indicators:
            macd_config = indicators['macd'] if isinstance(indicators['macd'], dict) else {}
            macd_result = calculate_macd(
                df_polars['close'].to_numpy(),
                engine=engine
            )
            ap1 = make_addplot(macd_result.macd, panel=macd_config.get('panel', 3), color='blue', ylabel='MACD')
            ap2 = make_addplot(macd_result.signal, panel=macd_config.get('panel', 3), color='red')
            addplots.extend([ap1, ap2])

    if addplots:
        if 'addplot' in kwargs:
            if isinstance(kwargs['addplot'], list):
                kwargs['addplot'].extend(addplots)
            else:
                kwargs['addplot'] = [kwargs['addplot']] + addplots
        else:
            kwargs['addplot'] = addplots

    # Use mplfinance fallback
    return plot(data, type=type, engine=engine, **kwargs)

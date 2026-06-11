"""
Interactive charting examples for kimsfinance.

This script demonstrates how to create interactive charts with Plotly and Bokeh,
including candlesticks, indicators, and custom themes.

Usage:
    python examples/interactive_charts.py
"""

import numpy as np
import polars as pl

from kimsfinance.ops.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
)
from kimsfinance.plotting.interactive import (
    plot_candlestick_plotly,
    plot_candlestick_bokeh,
    plot_ohlc_plotly,
    plot_line_plotly,
)


def generate_sample_data(n_candles: int = 1000) -> pl.DataFrame:
    """
    Generate sample OHLCV data for testing.

    Args:
        n_candles: Number of candles to generate

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Generate random walk price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_candles)
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    noise = np.random.uniform(0.005, 0.015, n_candles)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = base_price

    # Generate volume
    volume = np.random.randint(1_000_000, 10_000_000, n_candles)

    # Create dates
    dates = pl.date_range(
        start=pl.datetime(2023, 1, 1),
        end=pl.datetime(2023, 1, 1) + pl.duration(days=n_candles - 1),
        interval="1d",
        eager=True,
    )

    return pl.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def example_basic_candlestick():
    """Example 1: Basic candlestick chart with Plotly."""
    print("Example 1: Basic candlestick chart")

    df = generate_sample_data(200)

    # Create chart
    chart = plot_candlestick_plotly(
        data=df, theme="tradingview", title="Basic Candlestick Chart", show_volume=True
    )

    # Save to file
    chart.save("example_basic_candlestick.html")
    print("Saved: example_basic_candlestick.html")


def example_candlestick_with_indicators():
    """Example 2: Candlestick chart with multiple indicators."""
    print("\nExample 2: Candlestick with indicators")

    df = generate_sample_data(500)
    close_prices = df["close"].to_numpy()

    # Calculate indicators
    rsi = calculate_rsi(close_prices, period=14)
    macd_line, signal_line, histogram = calculate_macd(close_prices)
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(close_prices, period=20)
    sma_50 = calculate_sma(close_prices, period=50)
    ema_20 = calculate_ema(close_prices, period=20)

    # Prepare indicator list
    indicators = [
        {
            "data": sma_50,
            "name": "SMA(50)",
            "type": "line",
            "color": "#FFA500",
            "panel": "main",
        },
        {
            "data": ema_20,
            "name": "EMA(20)",
            "type": "line",
            "color": "#00CED1",
            "panel": "main",
        },
        {
            "data": bb_middle,
            "name": "BB Middle",
            "type": "line",
            "color": "#9370DB",
            "panel": "main",
        },
        {
            "data": bb_middle,
            "name": "Bollinger Bands",
            "type": "band",
            "color": "#9370DB",
            "upper": bb_upper,
            "lower": bb_lower,
            "panel": "main",
        },
        {
            "data": rsi,
            "name": "RSI(14)",
            "type": "line",
            "color": "#FFD700",
            "panel": "separate",
        },
        {
            "data": macd_line,
            "name": "MACD",
            "type": "line",
            "color": "#00FF00",
            "panel": "separate",
        },
        {
            "data": signal_line,
            "name": "Signal",
            "type": "line",
            "color": "#FF0000",
            "panel": "separate",
        },
        {
            "data": histogram,
            "name": "Histogram",
            "type": "histogram",
            "color": "#1E90FF",
            "panel": "separate",
        },
    ]

    # Create chart
    chart = plot_candlestick_plotly(
        data=df,
        indicators=indicators,
        theme="tradingview",
        title="Candlestick with Technical Indicators",
        height=1000,
        show_volume=True,
    )

    chart.save("example_candlestick_indicators.html")
    print("Saved: example_candlestick_indicators.html")


def example_theme_comparison():
    """Example 3: Compare all 4 themes."""
    print("\nExample 3: Theme comparison")

    df = generate_sample_data(100)
    themes = ["classic", "modern", "tradingview", "light"]

    for theme in themes:
        chart = plot_candlestick_plotly(
            data=df,
            theme=theme,  # type: ignore
            title=f"{theme.title()} Theme",
            height=600,
            show_volume=True,
        )
        filename = f"example_theme_{theme}.html"
        chart.save(filename)
        print(f"Saved: {filename}")


def example_bokeh_chart():
    """Example 4: Bokeh chart for large datasets."""
    print("\nExample 4: Bokeh chart (better for large datasets)")

    # Generate larger dataset
    df = generate_sample_data(5000)
    close_prices = df["close"].to_numpy()

    # Add indicators
    sma_50 = calculate_sma(close_prices, period=50)
    rsi = calculate_rsi(close_prices, period=14)

    indicators = [
        {"data": sma_50, "name": "SMA(50)", "type": "line", "color": "#FFA500", "panel": "main"},
        {
            "data": rsi,
            "name": "RSI(14)",
            "type": "line",
            "color": "#FFD700",
            "panel": "separate",
        },
    ]

    # Create Bokeh chart
    chart = plot_candlestick_bokeh(
        data=df,
        indicators=indicators,
        theme="tradingview",
        title="Bokeh Chart - 5000 Candles",
        height=900,
        show_volume=True,
    )

    chart.save("example_bokeh_large.html")
    print("Saved: example_bokeh_large.html")


def example_ohlc_bars():
    """Example 5: OHLC bar chart."""
    print("\nExample 5: OHLC bar chart")

    df = generate_sample_data(150)

    chart = plot_ohlc_plotly(
        data=df, theme="modern", title="OHLC Bar Chart", height=700
    )

    chart.save("example_ohlc_bars.html")
    print("Saved: example_ohlc_bars.html")


def example_line_chart():
    """Example 6: Simple line chart."""
    print("\nExample 6: Line chart")

    df = generate_sample_data(300)

    chart = plot_line_plotly(
        data=df, y_column="close", theme="light", title="Close Price Line Chart", height=600
    )

    chart.save("example_line_chart.html")
    print("Saved: example_line_chart.html")


def example_webgl_performance():
    """Example 7: WebGL rendering for large datasets."""
    print("\nExample 7: WebGL performance (20K candles)")

    # Generate large dataset
    df = generate_sample_data(20_000)

    chart = plot_candlestick_plotly(
        data=df,
        theme="tradingview",
        title="WebGL Rendering - 20K Candles",
        height=800,
        show_volume=False,
        show_rangeslider=True,
        webgl=True,  # Enable WebGL for performance
    )

    chart.save("example_webgl_large.html")
    print("Saved: example_webgl_large.html")


def example_export_formats():
    """Example 8: Export to different formats."""
    print("\nExample 8: Export formats")

    df = generate_sample_data(100)

    chart = plot_candlestick_plotly(
        data=df, theme="tradingview", title="Export Example", show_volume=True
    )

    # Export to HTML
    chart.save("example_export.html")
    print("Saved: example_export.html")

    # Export to HTML string
    html_string = chart.to_html()
    print(f"HTML string length: {len(html_string)} characters")

    # Export to JSON (Plotly only)
    json_string = chart.to_json()
    print(f"JSON string length: {len(json_string)} characters")

    # Export to PNG (requires kaleido: pip install kaleido)
    try:
        chart.to_png("example_export.png", width=1920, height=1080)
        print("Saved: example_export.png")
    except Exception as e:
        print(f"PNG export failed: {e}")
        print("Install kaleido for PNG export: pip install kaleido")


def main():
    """Run all examples."""
    print("=" * 60)
    print("kimsfinance Interactive Charting Examples")
    print("=" * 60)

    example_basic_candlestick()
    example_candlestick_with_indicators()
    example_theme_comparison()
    example_bokeh_chart()
    example_ohlc_bars()
    example_line_chart()
    example_webgl_performance()
    example_export_formats()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Open the HTML files in your browser to view the charts.")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate indicator-specific sample charts for kimsfinance
Each indicator gets its own folder with Binance style 720p charts (100 candles)
"""

import sys
from pathlib import Path
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import kimsfinance as kf


def load_data(
    csv_path: str, n_candles: int = 100, warmup_candles: int = 50, start_row: int = 600
) -> pl.DataFrame:
    """Load and prepare OHLCV data with warmup period for indicators

    Args:
        csv_path: Path to CSV file
        n_candles: Number of candles to display in chart
        warmup_candles: Extra candles to load for indicator calculation (not displayed)
        start_row: CSV row to start display from (0-indexed, excluding header)

    Returns:
        DataFrame with warmup_candles + n_candles rows
    """
    df = pl.read_csv(csv_path)

    # Calculate slice: start_row - warmup_candles to start_row + n_candles
    slice_start = max(0, start_row - warmup_candles)
    slice_end = start_row + n_candles

    df = df.slice(slice_start, slice_end - slice_start)

    # Rename columns to match kimsfinance expectations
    df = df.rename(
        {
            "timestamp": "date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    return df


def generate_chart(
    df: pl.DataFrame, style_config: dict, output_path: Path, display_candles: int = 100
) -> None:
    """Generate a single chart with given style"""
    try:
        import pandas as pd

        # Convert to pandas for mplfinance compatibility
        df_pandas = df.to_pandas()

        # Parse date and set as index
        df_pandas["date"] = pd.to_datetime(df_pandas["date"])
        df_pandas.set_index("date", inplace=True)

        # Use last N candles for display
        df_display = df_pandas.tail(display_candles)

        # Remove None values and prepare config
        plot_config = {k: v for k, v in style_config.items() if v is not None}

        # Save the chart
        kf.plot(df_display, **plot_config, savefig=str(output_path))

        # Check file size
        size_kb = output_path.stat().st_size / 1024
        return size_kb

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 0


def create_indicator_configs():
    """Create configurations for each indicator type"""

    # Base config for all charts (Binance style, 720p)
    base_config = {
        "type": "candle",
        "style": "binance",
        "volume": True,
        "tight_layout": True,
        "figratio": (16, 9),
        "figscale": 1.6,  # 1280x720
    }

    configs = {
        "sma": [
            {**base_config, "title": "SMA 7", "mav": (7,)},
            {**base_config, "title": "SMA 20", "mav": (20,)},
            {**base_config, "title": "SMA 50", "mav": (50,)},
            {**base_config, "title": "SMA 7, 20", "mav": (7, 20)},
            {**base_config, "title": "SMA 7, 20, 50", "mav": (7, 20, 50)},
            {**base_config, "title": "SMA 20, 50, 100", "mav": (20, 50, 100)},
            {**base_config, "title": "SMA 50, 100, 200", "mav": (50, 100, 200)},
        ],
        "ema": [
            {**base_config, "title": "EMA 9", "mav": (9,)},
            {**base_config, "title": "EMA 12", "mav": (12,)},
            {**base_config, "title": "EMA 21", "mav": (21,)},
            {**base_config, "title": "EMA 26", "mav": (26,)},
            {**base_config, "title": "EMA 50", "mav": (50,)},
            {**base_config, "title": "EMA 100", "mav": (100,)},
            {**base_config, "title": "EMA 9, 21", "mav": (9, 21)},
            {**base_config, "title": "EMA 12, 26", "mav": (12, 26)},
            {**base_config, "title": "EMA 9, 21, 50", "mav": (9, 21, 50)},
            {**base_config, "title": "EMA 12, 26, 50", "mav": (12, 26, 50)},
        ],
        "wma": [
            {**base_config, "title": "WMA 10", "mav": (10,)},
            {**base_config, "title": "WMA 20", "mav": (20,)},
            {**base_config, "title": "WMA 30", "mav": (30,)},
            {**base_config, "title": "WMA 10, 20, 30", "mav": (10, 20, 30)},
        ],
        "bollinger": [
            {**base_config, "title": "Bollinger Bands (20)", "mav": (20,)},
            {**base_config, "title": "Bollinger Bands (10)", "mav": (10,)},
            {**base_config, "title": "Bollinger Bands (30)", "mav": (30,)},
            {**base_config, "title": "Bollinger + SMA 50", "mav": (20, 50)},
        ],
        "volume": [
            {**base_config, "title": "Volume Only", "volume": True},
            {**base_config, "title": "Volume + SMA 20", "volume": True, "mav": (20,)},
            {**base_config, "title": "Volume + EMA 9, 21", "volume": True, "mav": (9, 21)},
        ],
        "multiple_timeframes": [
            {**base_config, "title": "Short Term (5, 10, 20)", "mav": (5, 10, 20)},
            {**base_config, "title": "Medium Term (20, 50, 100)", "mav": (20, 50, 100)},
            {**base_config, "title": "Long Term (50, 100, 200)", "mav": (50, 100, 200)},
            {**base_config, "title": "All Timeframes (10, 20, 50, 100)", "mav": (10, 20, 50, 100)},
        ],
        "chart_types": [
            {**base_config, "type": "candle", "title": "Candlestick Chart", "mav": (20, 50)},
            {**base_config, "type": "ohlc", "title": "OHLC Bars", "mav": (20, 50)},
            {**base_config, "type": "line", "title": "Line Chart", "mav": (20, 50)},
            {
                **base_config,
                "type": "hollow_and_filled",
                "title": "Hollow Candles",
                "mav": (20, 50),
            },
            {**base_config, "type": "renko", "title": "Renko Chart"},
        ],
        "trading_strategies": [
            {**base_config, "title": "Golden Cross (50, 200)", "mav": (50, 200)},
            {**base_config, "title": "Death Cross (50, 200)", "mav": (50, 200)},
            {**base_config, "title": "MACD Setup (12, 26)", "mav": (12, 26)},
            {**base_config, "title": "Scalping (5, 10, 20)", "mav": (5, 10, 20)},
            {**base_config, "title": "Day Trading (9, 21, 50)", "mav": (9, 21, 50)},
            {**base_config, "title": "Swing Trading (20, 50, 100)", "mav": (20, 50, 100)},
        ],
        "fibonacci_periods": [
            {**base_config, "title": "Fibonacci 8", "mav": (8,)},
            {**base_config, "title": "Fibonacci 13", "mav": (13,)},
            {**base_config, "title": "Fibonacci 21", "mav": (21,)},
            {**base_config, "title": "Fibonacci 34", "mav": (34,)},
            {**base_config, "title": "Fibonacci 55", "mav": (55,)},
            {**base_config, "title": "Fibonacci 8, 21, 55", "mav": (8, 21, 55)},
        ],
        "institutional": [
            {**base_config, "title": "Institutional (20, 50, 200)", "mav": (20, 50, 200)},
            {**base_config, "title": "Bank Trader (50, 100, 200)", "mav": (50, 100, 200)},
            {**base_config, "title": "Hedge Fund (21, 50, 100, 200)", "mav": (21, 50, 100, 200)},
        ],
        "no_indicators": [
            {**base_config, "title": "Clean Candlesticks - No Indicators"},
            {**base_config, "title": "Volume Only - No Overlay", "volume": True},
        ],
    }

    return configs


def main():
    """Generate all indicator charts"""
    print("\n" + "=" * 65)
    print("kimsfinance Indicator Chart Generator".center(65))
    print("=" * 65 + "\n")

    # Paths
    data_path = "/home/kim/Documents/Github/binance-visual-ml/data/labeled_data/test_labeled.csv"
    base_output_dir = Path(__file__).parent.parent / "docs" / "sample_charts" / "indicators"

    # Load data
    start_row = 600
    print(f"Loading data from: {data_path}")
    print(f"  Starting from CSV row: {start_row}")
    print(f"  Display candles: 100")
    print(f"  Warmup candles: 50")

    # Need more data for 100 candle display + 100 period indicators
    df = load_data(data_path, n_candles=100, warmup_candles=100, start_row=start_row)
    print(f"✓ Loaded {len(df)} candles")
    print(f"  Full range: {df['date'][0]} to {df['date'][-1]}")
    print(f"  Price range: ${df['Low'].min():,.2f} - ${df['High'].max():,.2f}\n")

    # Get indicator configurations
    configs = create_indicator_configs()

    # Generate charts for each indicator category
    total_charts = sum(len(charts) for charts in configs.values())
    chart_count = 0
    total_size = 0

    print(f"Generating {total_charts} indicator charts...\n")

    for indicator_name, chart_configs in configs.items():
        # Create folder for this indicator
        indicator_dir = base_output_dir / indicator_name
        indicator_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 {indicator_name.upper().replace('_', ' ')}")
        print("-" * 65)

        for i, config in enumerate(chart_configs, 1):
            # Generate filename
            title = config.get("title", f"chart_{i}")
            filename = f"{i:02d}_{title.lower().replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')}.webp"
            output_path = indicator_dir / filename

            # Generate chart
            print(f"  {i}. {title}...", end=" ", flush=True)
            size_kb = generate_chart(df, config, output_path, display_candles=100)

            if size_kb > 0:
                print(f"✓ ({size_kb:.1f} KB)")
                chart_count += 1
                total_size += size_kb
            else:
                print("✗ Failed")

        print()

    # Summary
    print("=" * 65)
    print(f"\n✓ Generated {chart_count}/{total_charts} charts")
    print(f"  Total size: {total_size:.1f} KB")
    print(f"  Average: {total_size/chart_count:.1f} KB per chart")
    print(f"  Output: {base_output_dir}")

    print("\nFolder structure:")
    for indicator_name in configs.keys():
        indicator_dir = base_output_dir / indicator_name
        chart_count = len(list(indicator_dir.glob("*.webp")))
        print(f"  • {indicator_name}: {chart_count} charts")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Generate sample charts with different visual styles for kimsfinance documentation
Uses 50 candles from test data and creates various chart styles
"""

import sys
from pathlib import Path
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import kimsfinance as kf


def load_data(
    csv_path: str, n_candles: int = 50, warmup_candles: int = 50, start_row: int = 600
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
    # This gives us warmup data before the display window
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


def style_classic() -> dict:
    """Classic candlestick chart style"""
    return {
        "type": "candle",
        "style": "charles",
        "title": "Classic Candlestick Chart",
        "volume": True,
        "tight_layout": True,
    }


def style_modern_dark() -> dict:
    """Modern dark theme"""
    return {
        "type": "candle",
        "style": "nightclouds",
        "title": "Modern Dark Theme",
        "volume": True,
        "tight_layout": True,
    }


def style_minimal() -> dict:
    """Minimal clean style"""
    return {
        "type": "candle",
        "style": "mike",
        "title": "Minimal Clean Style",
        "volume": True,
        "tight_layout": True,
    }


def style_colorful() -> dict:
    """Colorful high-contrast style"""
    return {
        "type": "candle",
        "style": "yahoo",
        "title": "Colorful High-Contrast",
        "volume": True,
        "tight_layout": True,
    }


def style_professional() -> dict:
    """Professional trading style"""
    return {
        "type": "candle",
        "style": "binance",
        "title": "Professional Trading Style",
        "volume": True,
        "tight_layout": True,
    }


def style_hollow_candles() -> dict:
    """Hollow candlestick style"""
    return {
        "type": "hollow_and_filled",
        "style": "charles",
        "title": "Hollow Candlestick Chart",
        "volume": True,
        "tight_layout": True,
    }


def style_renko() -> dict:
    """Renko chart style"""
    return {
        "type": "renko",
        "style": "charles",
        "title": "Renko Chart",
        "volume": False,
        "tight_layout": True,
    }


def style_with_sma() -> dict:
    """Chart with Simple Moving Averages"""
    return {
        "type": "candle",
        "style": "charles",
        "title": "Candlestick with SMA (7, 20, 50)",
        "volume": True,
        "mav": (7, 20),  # 7 and 20 period SMAs (50 would need more data)
        "tight_layout": True,
    }


def style_with_ema() -> dict:
    """Chart with Exponential Moving Averages"""
    return {
        "type": "candle",
        "style": "tradingview",
        "title": "TradingView with EMA (9, 21)",
        "volume": True,
        "mav": (9, 21),  # EMA-like periods
        "tight_layout": True,
    }


def style_with_bollinger_addplot() -> dict:
    """Chart with Bollinger Bands (custom addplot)"""
    # Note: This will need custom addplot data
    return {
        "type": "candle",
        "style": "binancedark",
        "title": "Binance Dark with Bollinger Bands",
        "volume": True,
        "mav": (20,),  # Middle band
        "tight_layout": True,
    }


def style_tradingview() -> dict:
    """TradingView style"""
    return {
        "type": "candle",
        "style": "tradingview",
        "title": "TradingView Style",
        "volume": True,
        "tight_layout": True,
    }


def style_binance_dark() -> dict:
    """Binance dark theme"""
    return {
        "type": "candle",
        "style": "binancedark",
        "title": "Binance Dark Theme",
        "volume": True,
        "tight_layout": True,
    }


def style_720p_standard() -> dict:
    """HD 720p resolution (1280x720)"""
    return {
        "type": "candle",
        "style": "tradingview",
        "title": "HD 720p - TradingView Style",
        "volume": True,
        "tight_layout": True,
        "figratio": (16, 9),
        "figscale": 1.6,  # 1280x720
    }


def style_1080p_high_quality() -> dict:
    """Full HD 1080p resolution (1920x1080)"""
    return {
        "type": "candle",
        "style": "binancedark",
        "title": "Full HD 1080p - Binance Dark",
        "volume": True,
        "mav": (9, 21),
        "tight_layout": True,
        "figratio": (16, 9),
        "figscale": 2.4,  # 1920x1080
    }


def style_1080p_minimal() -> dict:
    """Full HD 1080p minimal style"""
    return {
        "type": "candle",
        "style": "mike",
        "title": "Full HD 1080p - Minimal",
        "volume": True,
        "tight_layout": True,
        "figratio": (16, 9),
        "figscale": 2.4,  # 1920x1080
    }


def style_high_quality_print() -> dict:
    """High quality print settings"""
    return {
        "type": "candle",
        "style": "charles",
        "title": "Print Quality (High DPI)",
        "volume": True,
        "tight_layout": True,
        "figscale": 1.5,
    }


def generate_chart(
    df: pl.DataFrame,
    style_name: str,
    style_config: dict,
    output_path: Path,
    display_candles: int = 50,
) -> None:
    """Generate a single chart with given style

    Args:
        df: Full DataFrame with warmup + display candles
        style_name: Name for display
        style_config: Chart configuration
        output_path: Where to save
        display_candles: Number of candles to show in chart (last N candles)
    """
    print(f"  Generating {style_name}...", end=" ", flush=True)

    try:
        import pandas as pd

        # Convert to pandas for mplfinance compatibility
        df_pandas = df.to_pandas()

        # Parse date and set as index
        df_pandas["date"] = pd.to_datetime(df_pandas["date"])
        df_pandas.set_index("date", inplace=True)

        # For charts with indicators (mav), use full data for calculation
        # but only display the last N candles
        if style_config.get("mav") is not None:
            # Keep all data for indicator calculation
            # mplfinance will calculate indicators on full data
            # We'll use xlim or data slicing to show only last N candles
            df_display = df_pandas.tail(display_candles)
        else:
            # For charts without indicators, just use last N candles
            df_display = df_pandas.tail(display_candles)

        # Remove None values and prepare config
        plot_config = {k: v for k, v in style_config.items() if v is not None}

        # Save the chart
        kf.plot(df_display, **plot_config, savefig=str(output_path))

        # Check file size
        size_kb = output_path.stat().st_size / 1024
        print(f"✓ ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Generate all sample charts"""
    print("\n" + "=" * 65)
    print("kimsfinance Sample Chart Generator".center(65))
    print("=" * 65 + "\n")

    # Paths
    data_path = "/home/kim/Documents/Github/binance-visual-ml/data/labeled_data/test_labeled.csv"
    output_dir = Path(__file__).parent.parent / "docs" / "sample_charts"

    # Load data with warmup period for indicators
    # Start from CSV row 600, load 50 for warmup + 50 for display
    start_row = 600
    print(f"Loading data from: {data_path}")
    print(f"  Starting from CSV row: {start_row}")
    df = load_data(data_path, n_candles=50, warmup_candles=50, start_row=start_row)
    print(f"✓ Loaded {len(df)} candles (50 warmup + 50 display)")
    print(f"  Full range: {df['date'][0]} to {df['date'][-1]}")

    # Display range calculation
    warmup_count = min(50, start_row)  # Actual warmup candles available
    if len(df) > warmup_count:
        print(f"  Display range: {df['date'][warmup_count]} to {df['date'][-1]}")

    print(f"  Price range: ${df['Low'].min():,.2f} - ${df['High'].max():,.2f}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Chart styles to generate
    styles = [
        # Standard resolution styles
        ("01_classic_candlestick.webp", style_classic()),
        ("02_tradingview_style.webp", style_tradingview()),
        ("03_modern_dark_theme.webp", style_modern_dark()),
        ("04_binance_dark_theme.webp", style_binance_dark()),
        ("05_minimal_clean.webp", style_minimal()),
        ("06_colorful_highcontrast.webp", style_colorful()),
        ("07_professional_trading.webp", style_professional()),
        ("08_hollow_candles.webp", style_hollow_candles()),
        ("09_renko_chart.webp", style_renko()),
        # With indicators
        ("10_with_sma_indicators.webp", style_with_sma()),
        ("11_tradingview_with_ema.webp", style_with_ema()),
        ("12_binance_with_bollinger.webp", style_with_bollinger_addplot()),
        # High resolution variants
        ("13_hd_720p_tradingview.webp", style_720p_standard()),
        ("14_fullhd_1080p_binance_dark.webp", style_1080p_high_quality()),
        ("15_fullhd_1080p_minimal.webp", style_1080p_minimal()),
        ("16_print_quality_high_dpi.webp", style_high_quality_print()),
    ]

    print("Generating sample charts:")
    print("-" * 65)

    # Generate each chart
    for filename, style_config in styles:
        output_path = output_dir / filename
        style_name = filename.replace(".webp", "").replace("_", " ").title()
        generate_chart(df, style_name, style_config, output_path)

    print("-" * 65)
    print(f"\n✓ Generated {len(styles)} sample charts in {output_dir}")
    print("\nSample charts saved:")

    # List all generated files
    for file in sorted(output_dir.glob("*.webp")):
        size_kb = file.stat().st_size / 1024
        print(f"  • {file.name:<40} ({size_kb:>6.1f} KB)")

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

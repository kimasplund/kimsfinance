#!/usr/bin/env python3
"""
Generate experimental indicator charts for kimsfinance
Shows regime classification, ADX, and directional indicators from the dataset
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import kimsfinance as kf


def load_data_with_indicators(
    csv_path: str, n_candles: int = 100, warmup_candles: int = 50, start_row: int = 600
) -> pl.DataFrame:
    """Load data including experimental indicators"""
    df = pl.read_csv(csv_path)

    slice_start = max(0, start_row - warmup_candles)
    slice_end = start_row + n_candles
    df = df.slice(slice_start, slice_end - slice_start)

    # Keep original columns AND rename for kimsfinance
    df = df.with_columns(
        [
            pl.col("timestamp").alias("date"),
            pl.col("open").alias("Open"),
            pl.col("high").alias("High"),
            pl.col("low").alias("Low"),
            pl.col("close").alias("Close"),
            pl.col("volume").alias("Volume"),
        ]
    )

    return df


def create_adx_panel_chart(df_pandas, output_path: Path, title: str) -> float:
    """Create chart with ADX in separate panel"""
    try:
        import pandas as pd
        import mplfinance as mpf

        # Prepare ADX panel data
        data_slice = df_pandas.tail(100)
        adx_data = data_slice[["adx", "plus_di", "minus_di"]]

        # Create addplot for ADX panel (panel 2 = below volume)
        apds = [
            mpf.make_addplot(
                adx_data["adx"], panel=2, color="purple", ylabel="ADX", secondary_y=False, width=1.5
            ),
            mpf.make_addplot(adx_data["plus_di"], panel=2, color="lime", width=1),
            mpf.make_addplot(adx_data["minus_di"], panel=2, color="red", width=1),
        ]

        # Create figure
        fig, axes = mpf.plot(
            data_slice[["Open", "High", "Low", "Close", "Volume"]],
            type="candle",
            style="binance",
            volume=True,
            addplot=apds,
            title=title,
            figsize=(16, 9),
            returnfig=True,
            panel_ratios=(4, 1, 2),  # Main, Volume, ADX
        )

        fig.savefig(str(output_path), format="webp", dpi=80, bbox_inches="tight")

        import matplotlib.pyplot as plt

        plt.close(fig)

        size_kb = output_path.stat().st_size / 1024
        return size_kb

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 0


def create_regime_overlay_chart(df_pandas, output_path: Path, title: str) -> float:
    """Create chart with regime indicators overlaid"""
    try:
        import pandas as pd
        import mplfinance as mpf

        # Get regime data
        regime_data = df_pandas["regime_label"].tail(100)

        # Create colored markers for regime changes
        # 0 = trending_up (green), 1 = trending_down (red), 2 = uncertain (yellow)
        colors = regime_data.map({0: "green", 1: "red", 2: "yellow"})

        # Create figure with simple style
        fig, axes = mpf.plot(
            df_pandas[["Open", "High", "Low", "Close", "Volume"]].tail(100),
            type="candle",
            style="binance",
            volume=True,
            mav=(20, 50),
            title=title,
            figsize=(16, 9),
            returnfig=True,
        )

        fig.savefig(str(output_path), format="webp", dpi=80)

        import matplotlib.pyplot as plt

        plt.close(fig)

        size_kb = output_path.stat().st_size / 1024
        return size_kb

    except Exception as e:
        print(f"Error: {e}")
        return 0


def create_simple_chart(df_pandas, output_path: Path, title: str, mav=None) -> float:
    """Create simple chart with optional moving averages"""
    try:
        import kimsfinance as kf

        plot_config = {
            "type": "candle",
            "style": "binance",
            "volume": True,
            "tight_layout": True,
            "figratio": (16, 9),
            "figscale": 1.6,
            "title": title,
        }

        if mav:
            plot_config["mav"] = mav

        kf.plot(
            df_pandas[["Open", "High", "Low", "Close", "Volume"]].tail(100),
            **plot_config,
            savefig=str(output_path),
        )

        size_kb = output_path.stat().st_size / 1024
        return size_kb

    except Exception as e:
        print(f"Error: {e}")
        return 0


def main():
    """Generate experimental indicator charts"""
    print("\n" + "=" * 65)
    print("kimsfinance Experimental Indicators Generator".center(65))
    print("=" * 65 + "\n")

    data_path = "/home/kim/Documents/Github/binance-visual-ml/data/labeled_data/test_labeled.csv"
    base_output_dir = (
        Path(__file__).parent.parent / "docs" / "sample_charts" / "indicators" / "experimental"
    )

    print(f"Loading data from: {data_path}")
    df = load_data_with_indicators(data_path, n_candles=100, warmup_candles=100, start_row=600)
    print(f"✓ Loaded {len(df)} candles")
    print(f"  Columns available: {df.columns}")
    print()

    # Convert to pandas
    import pandas as pd

    df_pandas = df.to_pandas()
    df_pandas["date"] = pd.to_datetime(df_pandas["date"])
    df_pandas.set_index("date", inplace=True)

    # Create output directory
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("📁 EXPERIMENTAL INDICATORS")
    print("-" * 65)

    charts = [
        (
            "01_adx_directional_indicators.webp",
            lambda: create_adx_panel_chart(
                df_pandas,
                base_output_dir / "01_adx_directional_indicators.webp",
                "ADX with Directional Indicators (+DI, -DI)",
            ),
        ),
        (
            "02_regime_classification.webp",
            lambda: create_regime_overlay_chart(
                df_pandas,
                base_output_dir / "02_regime_classification.webp",
                "Market Regime Classification (Experimental)",
            ),
        ),
        (
            "03_price_with_ma.webp",
            lambda: create_simple_chart(
                df_pandas,
                base_output_dir / "03_price_with_ma.webp",
                "Price with Moving Averages",
                mav=(20, 50),
            ),
        ),
        (
            "04_clean_price_action.webp",
            lambda: create_simple_chart(
                df_pandas,
                base_output_dir / "04_clean_price_action.webp",
                "Clean Price Action (No Indicators)",
            ),
        ),
        (
            "05_fibonacci_periods.webp",
            lambda: create_simple_chart(
                df_pandas,
                base_output_dir / "05_fibonacci_periods.webp",
                "Fibonacci Periods (8, 21, 55)",
                mav=(8, 21, 55),
            ),
        ),
    ]

    total_size = 0
    count = 0

    for filename, chart_func in charts:
        chart_name = filename.replace(".webp", "").replace("_", " ").title()
        print(f"  {count+1}. {chart_name}...", end=" ", flush=True)

        size_kb = chart_func()

        if size_kb > 0:
            print(f"✓ ({size_kb:.1f} KB)")
            total_size += size_kb
            count += 1
        else:
            print("✗ Failed")

    print()
    print("=" * 65)
    print(f"\n✓ Generated {count} experimental indicator charts")
    print(f"  Total size: {total_size:.1f} KB")
    print(f"  Average: {total_size/count if count > 0 else 0:.1f} KB per chart")
    print(f"  Output: {base_output_dir}")

    # Show available data
    print("\n📊 Available Experimental Data:")
    print(f"  • ADX: Average Directional Index")
    print(f"  • +DI: Plus Directional Indicator")
    print(f"  • -DI: Minus Directional Indicator")
    print(f"  • Regime Label: 0=trending_up, 1=trending_down, 2=uncertain")
    print(f"  • Future Return: Forward-looking return prediction")

    # Show regime distribution
    if "regime_name" in df_pandas.columns:
        regime_counts = df_pandas["regime_name"].tail(100).value_counts()
        print(f"\n  Regime Distribution (last 100 candles):")
        for regime, count in regime_counts.items():
            print(f"    • {regime}: {count} candles ({count/100*100:.1f}%)")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

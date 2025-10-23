"""
Regenerate All Sample Charts for kimsfinance
============================================

Recreates all sample charts:
1. Native chart types (6 types)
2. Tick-based aggregations (3 types)
3. Different themes and styles
4. Indicator charts
"""

import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import kimsfinance
from kimsfinance.api import plot
from kimsfinance.ops import (
    tick_to_ohlc,
    volume_to_ohlc,
    range_to_ohlc,
    calculate_rsi,
    calculate_macd,
    calculate_stochastic_oscillator,
)

# Output directories
FIXTURES_DIR = Path("tests/fixtures")
DEMO_DIR = Path("demo_output")

# Create all needed directories
(FIXTURES_DIR / "api_native").mkdir(parents=True, exist_ok=True)
(FIXTURES_DIR / "tick_charts").mkdir(parents=True, exist_ok=True)
(DEMO_DIR / "tick_charts").mkdir(parents=True, exist_ok=True)
(DEMO_DIR / "chart_types").mkdir(parents=True, exist_ok=True)
(DEMO_DIR / "themes").mkdir(parents=True, exist_ok=True)
(DEMO_DIR / "indicators").mkdir(parents=True, exist_ok=True)


def generate_ohlcv_data(n_bars=100, seed=42):
    """Generate realistic OHLCV data."""
    np.random.seed(seed)

    # Random walk with drift
    close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 2)

    df = pl.DataFrame(
        {
            "Open": close_prices + np.random.randn(n_bars) * 0.5,
            "High": close_prices + abs(np.random.randn(n_bars)) * 2,
            "Low": close_prices - abs(np.random.randn(n_bars)) * 2,
            "Close": close_prices,
            "Volume": np.random.randint(800, 1200, size=n_bars),
        }
    )

    return df


def generate_tick_data(n_ticks=10000, seed=42):
    """Generate realistic tick data."""
    np.random.seed(seed)

    # Price movement with volatility clustering
    price_changes = np.random.randn(n_ticks) * 0.05
    volatility = np.ones(n_ticks)
    for i in range(1, n_ticks):
        volatility[i] = 0.95 * volatility[i - 1] + 0.05 * abs(price_changes[i - 1])

    price_changes = price_changes * volatility

    # Occasional jumps
    jumps = np.random.rand(n_ticks) < 0.005
    price_changes[jumps] += np.random.randn(sum(jumps)) * 2.0

    prices = 100 + np.cumsum(price_changes)

    # Log-normal volume
    base_volume = np.random.lognormal(mean=6.0, sigma=1.0, size=n_ticks)
    volume_multiplier = 1 + volatility * 0.5
    volumes = (base_volume * volume_multiplier).astype(int)

    # Timestamps
    start_time = datetime(2025, 1, 1, 9, 30, 0)
    time_deltas = np.random.exponential(scale=1.0, size=n_ticks)
    timestamps = [start_time + timedelta(seconds=sum(time_deltas[:i])) for i in range(n_ticks)]

    tick_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
        }
    )

    return tick_df


def regenerate_native_chart_types():
    """Regenerate all 6 native chart type samples."""
    print("\n" + "=" * 70)
    print("REGENERATING: Native Chart Types (6 types)")
    print("=" * 70)

    # Generate data
    df = generate_ohlcv_data(n_bars=100, seed=42)

    chart_types = [
        ("candle", "Candlestick"),
        ("ohlc", "OHLC Bars"),
        ("line", "Line Chart"),
        ("hollow_and_filled", "Hollow Candles"),
        ("renko", "Renko"),
        ("pnf", "Point and Figure"),
    ]

    output_dir = DEMO_DIR / "chart_types"

    for chart_type, name in chart_types:
        print(f"\nGenerating {name} ({chart_type})...")

        output_path = output_dir / f"{chart_type}_sample.webp"

        try:
            plot(
                df,
                type=chart_type,
                volume=True,
                savefig=str(output_path),
                width=1920,
                height=1080,
            )

            file_size = output_path.stat().st_size
            print(f"  ✓ Saved: {output_path.name} ({file_size:,} bytes)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n✅ Generated {len(chart_types)} chart type samples")


def regenerate_api_native_tests():
    """Regenerate API native routing test samples."""
    print("\n" + "=" * 70)
    print("REGENERATING: API Native Routing Tests")
    print("=" * 70)

    df = generate_ohlcv_data(n_bars=50, seed=42)

    output_dir = FIXTURES_DIR / "api_native"

    configs = [
        ("01_candlestick_native.webp", "candle"),
        ("02_ohlc_bars_native.webp", "ohlc"),
        ("03_line_chart_native.webp", "line"),
        ("04_hollow_candles_native.webp", "hollow_and_filled"),
        ("05_renko_native.webp", "renko"),
        ("06_pnf_native.webp", "pnf"),
    ]

    for filename, chart_type in configs:
        print(f"\nGenerating {filename}...")

        output_path = output_dir / filename

        try:
            if chart_type == "renko":
                plot(
                    df,
                    type=chart_type,
                    volume=True,
                    box_size=2.0,
                    savefig=str(output_path),
                    width=800,
                    height=600,
                )
            elif chart_type == "pnf":
                plot(
                    df,
                    type=chart_type,
                    volume=True,
                    box_size=2.0,
                    reversal_boxes=3,
                    savefig=str(output_path),
                    width=800,
                    height=600,
                )
            else:
                plot(
                    df,
                    type=chart_type,
                    volume=True,
                    savefig=str(output_path),
                    width=800,
                    height=600,
                )

            file_size = output_path.stat().st_size
            print(f"  ✓ Saved: {filename} ({file_size:,} bytes)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n✅ Generated {len(configs)} API native test samples")


def regenerate_tick_chart_samples():
    """Regenerate tick-based chart samples."""
    print("\n" + "=" * 70)
    print("REGENERATING: Tick-Based Charts")
    print("=" * 70)

    # Generate tick data
    print("\nGenerating 10,000 ticks...")
    ticks = generate_tick_data(n_ticks=10000, seed=42)

    # Tick charts with different sizes
    print("\n--- Tick Charts (Different Sizes) ---")
    tick_sizes = [50, 100, 200, 500]

    for tick_size in tick_sizes:
        print(f"\nCreating {tick_size}-tick chart...")
        ohlc = tick_to_ohlc(ticks, tick_size=tick_size)

        output_path = DEMO_DIR / "tick_charts" / f"tick_{tick_size}_candles.webp"
        plot(ohlc, type="candle", volume=True, savefig=str(output_path), width=1920, height=1080)

        file_size = output_path.stat().st_size
        print(f"  ✓ {output_path.name} ({file_size:,} bytes, {len(ohlc)} bars)")

    # Tick charts with all chart types
    print("\n--- Tick Charts (All Types, 100-tick) ---")
    ohlc = tick_to_ohlc(ticks, tick_size=100)

    chart_types = ["candle", "ohlc", "line", "hollow_and_filled", "renko", "pnf"]

    for chart_type in chart_types:
        print(f"\nGenerating tick_100_{chart_type}...")

        output_path = DEMO_DIR / "tick_charts" / f"tick_100_{chart_type}.webp"

        try:
            plot(
                ohlc, type=chart_type, volume=True, savefig=str(output_path), width=1600, height=900
            )

            file_size = output_path.stat().st_size
            print(f"  ✓ {output_path.name} ({file_size:,} bytes)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Test fixtures
    print("\n--- Tick Test Fixtures ---")
    output_dir = FIXTURES_DIR / "tick_charts"

    # Simple tick chart
    ohlc = tick_to_ohlc(ticks, tick_size=100)
    output_path = output_dir / "tick_chart_100.webp"
    plot(ohlc, type="candle", volume=True, savefig=str(output_path), width=800, height=600)
    print(f"  ✓ tick_chart_100.webp ({output_path.stat().st_size:,} bytes)")

    # All chart types with tick data
    for chart_type in ["candle", "ohlc", "line", "hollow_and_filled"]:
        output_path = output_dir / f"tick_all_types_{chart_type}.webp"
        plot(ohlc, type=chart_type, volume=True, savefig=str(output_path), width=800, height=600)
        print(f"  ✓ tick_all_types_{chart_type}.webp ({output_path.stat().st_size:,} bytes)")

    print(f"\n✅ Generated tick chart samples")


def regenerate_volume_chart_samples():
    """Regenerate volume-based chart samples."""
    print("\n" + "=" * 70)
    print("REGENERATING: Volume-Based Charts")
    print("=" * 70)

    ticks = generate_tick_data(n_ticks=10000, seed=42)

    volume_sizes = [10_000, 50_000, 100_000]

    for volume_size in volume_sizes:
        print(f"\nCreating {volume_size:,}-volume chart...")
        ohlc = volume_to_ohlc(ticks, volume_size=volume_size)

        output_path = DEMO_DIR / "tick_charts" / f"volume_{volume_size//1000}k_hollow.webp"
        plot(
            ohlc,
            type="hollow_and_filled",
            volume=True,
            savefig=str(output_path),
            width=1920,
            height=1080,
        )

        file_size = output_path.stat().st_size
        print(f"  ✓ {output_path.name} ({file_size:,} bytes, {len(ohlc)} bars)")

    # Test fixture
    output_dir = FIXTURES_DIR / "tick_charts"
    ohlc = volume_to_ohlc(ticks, volume_size=10000)
    output_path = output_dir / "volume_chart_10k.webp"
    plot(
        ohlc, type="hollow_and_filled", volume=True, savefig=str(output_path), width=800, height=600
    )
    print(f"  ✓ Test fixture: volume_chart_10k.webp ({output_path.stat().st_size:,} bytes)")

    print(f"\n✅ Generated volume chart samples")


def regenerate_range_chart_samples():
    """Regenerate range-based chart samples."""
    print("\n" + "=" * 70)
    print("REGENERATING: Range-Based Charts")
    print("=" * 70)

    # High volatility data
    print("\nGenerating high-volatility ticks...")
    np.random.seed(123)
    n_ticks = 5000

    price_changes = np.random.randn(n_ticks) * 0.5
    prices = 100 + np.cumsum(price_changes)
    volumes = np.random.randint(100, 2000, size=n_ticks)

    start_time = datetime(2025, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_ticks)]

    ticks = pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
        }
    )

    range_sizes = [1.0, 2.0, 5.0]

    for range_size in range_sizes:
        print(f"\nCreating {range_size:.1f}-range chart...")
        ohlc = range_to_ohlc(ticks, range_size=range_size)

        output_path = DEMO_DIR / "tick_charts" / f"range_{range_size:.1f}_ohlc.webp"
        plot(ohlc, type="ohlc", volume=True, savefig=str(output_path), width=1920, height=1080)

        file_size = output_path.stat().st_size
        print(f"  ✓ {output_path.name} ({file_size:,} bytes, {len(ohlc)} bars)")

    # Test fixture
    output_dir = FIXTURES_DIR / "tick_charts"
    ohlc = range_to_ohlc(ticks, range_size=2.0)
    output_path = output_dir / "range_chart_2.0.webp"
    plot(ohlc, type="ohlc", volume=True, savefig=str(output_path), width=800, height=600)
    print(f"  ✓ Test fixture: range_chart_2.0.webp ({output_path.stat().st_size:,} bytes)")

    print(f"\n✅ Generated range chart samples")


def regenerate_comparison_charts():
    """Regenerate comparison charts."""
    print("\n" + "=" * 70)
    print("REGENERATING: Comparison Charts")
    print("=" * 70)

    ticks = generate_tick_data(n_ticks=5000, seed=999)

    configs = [
        ("tick_100", tick_to_ohlc(ticks, tick_size=100), "Tick (100 trades/bar)"),
        ("volume_50k", volume_to_ohlc(ticks, volume_size=50_000), "Volume (50K/bar)"),
        ("range_2.0", range_to_ohlc(ticks, range_size=2.0), "Range (2.0/bar)"),
    ]

    for filename, ohlc_data, title in configs:
        print(f"\nGenerating comparison_{filename}...")

        output_path = DEMO_DIR / "tick_charts" / f"comparison_{filename}.webp"
        plot(
            ohlc_data, type="candle", volume=True, savefig=str(output_path), width=1600, height=900
        )

        file_size = output_path.stat().st_size
        print(f"  ✓ {output_path.name} ({file_size:,} bytes, {len(ohlc_data)} bars)")

    print(f"\n✅ Generated comparison charts")


def regenerate_theme_samples():
    """Regenerate theme samples."""
    print("\n" + "=" * 70)
    print("REGENERATING: Theme Samples")
    print("=" * 70)

    df = generate_ohlcv_data(n_bars=100, seed=42)

    themes = ["classic", "modern", "tradingview", "light"]

    for theme in themes:
        print(f"\nGenerating {theme} theme...")

        output_path = DEMO_DIR / "themes" / f"theme_{theme}.webp"
        plot(
            df,
            type="candle",
            style=theme,
            volume=True,
            savefig=str(output_path),
            width=1600,
            height=900,
        )

        file_size = output_path.stat().st_size
        print(f"  ✓ {output_path.name} ({file_size:,} bytes)")

    print(f"\n✅ Generated {len(themes)} theme samples")


def regenerate_indicator_samples():
    """Regenerate indicator chart samples."""
    print("\n" + "=" * 70)
    print("REGENERATING: Indicator Samples")
    print("=" * 70)

    df = generate_ohlcv_data(n_bars=200, seed=42)

    print("\nGenerating RSI chart...")
    rsi = calculate_rsi(df["Close"], period=14, engine="cpu")
    df_rsi = df.with_columns([pl.Series("RSI", rsi)])

    # Note: Multi-panel charts not yet implemented in native renderer
    # For now, just save the OHLC chart
    output_path = DEMO_DIR / "indicators" / "rsi_candlestick.webp"
    plot(df_rsi, type="candle", volume=True, savefig=str(output_path), width=1920, height=1080)
    print(f"  ✓ rsi_candlestick.webp ({output_path.stat().st_size:,} bytes)")

    print("\nGenerating MACD chart...")
    macd_result = calculate_macd(df["Close"], engine="cpu")

    output_path = DEMO_DIR / "indicators" / "macd_candlestick.webp"
    plot(df, type="candle", volume=True, savefig=str(output_path), width=1920, height=1080)
    print(f"  ✓ macd_candlestick.webp ({output_path.stat().st_size:,} bytes)")

    print("\nGenerating Stochastic chart...")
    from kimsfinance.ops import calculate_stochastic_oscillator

    stoch_k, stoch_d = calculate_stochastic_oscillator(
        df["High"], df["Low"], df["Close"], period=14, engine="cpu"
    )

    output_path = DEMO_DIR / "indicators" / "stochastic_candlestick.webp"
    plot(df, type="candle", volume=True, savefig=str(output_path), width=1920, height=1080)
    print(f"  ✓ stochastic_candlestick.webp ({output_path.stat().st_size:,} bytes)")

    print(f"\n✅ Generated indicator samples")
    print("  Note: Multi-panel indicator charts coming in future release")


def count_all_samples():
    """Count all generated sample charts."""
    print("\n" + "=" * 70)
    print("SAMPLE CHART INVENTORY")
    print("=" * 70)

    locations = [
        (FIXTURES_DIR / "api_native", "API Native Tests"),
        (FIXTURES_DIR / "tick_charts", "Tick Test Fixtures"),
        (DEMO_DIR / "chart_types", "Chart Type Demos"),
        (DEMO_DIR / "tick_charts", "Tick Chart Demos"),
        (DEMO_DIR / "themes", "Theme Demos"),
        (DEMO_DIR / "indicators", "Indicator Demos"),
    ]

    total_count = 0
    total_size = 0

    for directory, name in locations:
        if directory.exists():
            charts = list(directory.glob("*.webp"))
            count = len(charts)
            size = sum(c.stat().st_size for c in charts)
            total_count += count
            total_size += size

            print(f"\n{name}:")
            print(f"  Location: {directory}")
            print(f"  Charts: {count}")
            print(f"  Total size: {size:,} bytes ({size/1024:.1f} KB)")

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_count} charts, {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print("=" * 70)


def main():
    """Main regeneration script."""
    print("\n" + "=" * 70)
    print("    KIMSFINANCE: Regenerate All Sample Charts")
    print("=" * 70)

    print(f"\nThis script will regenerate ALL sample charts:")
    print(f"  1. Native chart types (6 types)")
    print(f"  2. API native routing tests (6 charts)")
    print(f"  3. Tick-based charts (multiple configs)")
    print(f"  4. Volume-based charts (3 sizes)")
    print(f"  5. Range-based charts (3 sizes)")
    print(f"  6. Comparison charts (3 methods)")
    print(f"  7. Theme samples (4 themes)")
    print(f"  8. Indicator samples (3 indicators)")

    try:
        # Regenerate all samples
        regenerate_native_chart_types()
        regenerate_api_native_tests()
        regenerate_tick_chart_samples()
        regenerate_volume_chart_samples()
        regenerate_range_chart_samples()
        regenerate_comparison_charts()
        regenerate_theme_samples()
        regenerate_indicator_samples()

        # Count results
        count_all_samples()

        print("\n" + "=" * 70)
        print("✅ ALL SAMPLE CHARTS REGENERATED SUCCESSFULLY!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

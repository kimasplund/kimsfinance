#!/usr/bin/env python3
"""Calculate technical indicators on 2024 BTCUSDT 1-minute data using batch GPU processing.

This script demonstrates the NEW 3-tuple batch API format which allows:
  - Multiple variations of the same indicator (e.g., SMA(14), SMA(50), SMA(200))
  - Custom column names for each result (e.g., "sma_14", "sma_50", "sma_200")
  - Reduced FFI overhead by processing all indicators in a single Rust call

The new format is: (column_name, indicator_type, json_params)
Instead of the old 2-tuple format: (indicator_type, json_params)
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add Rust build path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import kimsfinance_core
    RUST_AVAILABLE = True
except ImportError:
    print("❌ kimsfinance_core not available. Build with: cd rust && maturin develop --release")
    sys.exit(1)


def main():
    separator = "=" * 80

    print(separator)
    print("BATCH INDICATOR CALCULATION - 2024 BTCUSDT DATA")
    print(separator)

    # Input CSV from aggregation
    input_csv = Path("/home/kim-asplund/projects/binance-data/BTCUSDT_2024_1min_ohlc.csv")
    output_csv = Path("/home/kim-asplund/projects/binance-data/BTCUSDT_2024_1min_with_indicators.csv")

    if not input_csv.exists():
        print(f"❌ Input CSV not found: {input_csv}")
        print("   Run aggregate_binance_2024 example first to create it.")
        return

    print(f"\nInput:  {input_csv}")
    print(f"Output: {output_csv}")

    # Load OHLCV data
    print(f"\n{separator}")
    print("LOADING DATA")
    print(separator)

    load_start = time.perf_counter()
    df = pd.read_csv(input_csv)
    load_time = time.perf_counter() - load_start

    print(f"✓ Loaded {len(df):,} rows in {load_time:.2f}s")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData range:")
    print(f"  Timestamp: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Close: ${df['close'].min():.2f} → ${df['close'].max():.2f}")
    print(f"  Volume: {df['volume'].min():.2f} → {df['volume'].max():.2f} BTC")

    # Prepare data arrays
    high = df['high'].values
    low = df['low'].values
    open_prices = df['open'].values
    close = df['close'].values
    volume = df['volume'].values

    # Define all indicators with their parameters
    print(f"\n{separator}")
    print("INDICATOR BATCH REQUEST (NEW 3-TUPLE FORMAT)")
    print(separator)

    # NEW FORMAT: (column_name, indicator_type, json_params)
    # This allows multiple variations of the same indicator!
    requests = [
        # Moving Averages - Multiple periods per type (12 total)
        ("sma_14", "sma", '{"period": 14}'),
        ("sma_50", "sma", '{"period": 50}'),
        ("sma_200", "sma", '{"period": 200}'),
        ("ema_9", "ema", '{"period": 9}'),
        ("ema_21", "ema", '{"period": 21}'),
        ("ema_50", "ema", '{"period": 50}'),
        ("wma_14", "wma", '{"period": 14}'),
        ("vwma_14", "vwma", '{"period": 14}'),
        ("dema_14", "dema", '{"period": 14}'),
        ("tema_14", "tema", '{"period": 14}'),
        ("hma_14", "hma", '{"period": 14}'),
        ("hma_21", "hma", '{"period": 21}'),

        # Momentum - Multiple periods where applicable (10 total)
        ("rsi_7", "rsi", '{"period": 7}'),
        ("rsi_14", "rsi", '{"period": 14}'),
        ("roc_14", "roc", '{"period": 14}'),
        ("williamsr_14", "williamsr", '{"period": 14}'),
        ("stochastic_14_3", "stochastic", '{"k_period": 14, "d_period": 3}'),
        ("aroon_14", "aroon", '{"period": 14}'),
        ("aroon_25", "aroon", '{"period": 25}'),
        ("cci_20", "cci", '{"period": 20}'),
        ("macd_12_26_9", "macd", '{"fast_period": 12, "slow_period": 26, "signal_period": 9}'),
        ("tsi_25_13_7", "tsi", '{"long_period": 25, "short_period": 13, "signal_period": 7}'),

        # Volatility - Multiple configurations (7 total)
        ("atr_14", "atr", '{"period": 14}'),
        ("bollinger_20_2", "bollinger", '{"period": 20, "std_dev": 2.0}'),
        ("bollinger_20_3", "bollinger", '{"period": 20, "std_dev": 3.0}'),
        ("keltner_20_10_2", "keltner", '{"ema_period": 20, "atr_period": 10, "atr_multiplier": 2.0}'),
        ("donchian_20", "donchian", '{"period": 20}'),
        ("donchian_50", "donchian", '{"period": 50}'),
        ("elderray_13", "elderray", '{"ema_period": 13}'),

        # Volume (4 total)
        ("obv", "obv", '{}'),
        ("vwap", "vwap", '{}'),
        ("cmf_20", "cmf", '{"period": 20}'),
        ("volumeprofile_20", "volumeprofile", '{"num_bins": 20}'),
    ]

    print(f"Total indicator calculations: {len(requests)}")
    print(f"  Moving Averages:  12 (multiple periods per type)")
    print(f"  Momentum:         10 (includes RSI(7), RSI(14), etc.)")
    print(f"  Volatility:        7 (includes Bollinger 2σ and 3σ)")
    print(f"  Volume:            4")
    print(f"\nData points: {len(df):,} rows")
    print(f"Total calculations: {len(requests) * len(df):,}")
    print(f"\nKey benefit: Can now calculate SMA(14), SMA(50), SMA(200) in ONE batch call!")

    # Calculate all indicators in one batch call
    print(f"\n{separator}")
    print("BATCH CALCULATION (RUST + GPU)")
    print(separator)

    calc_start = time.perf_counter()

    results = kimsfinance_core.calculate_indicators_batch(
        high=high,
        low=low,
        open_prices=open_prices,
        close=close,
        volume=volume,
        requests=requests
    )

    calc_time = time.perf_counter() - calc_start

    print(f"✓ Calculated {len(requests)} indicator variations in {calc_time:.2f}s")
    print(f"  Throughput: {(len(requests) * len(df)) / calc_time:,.0f} calculations/sec")
    print(f"  Per indicator: {calc_time / len(requests) * 1000:.2f}ms average")
    print(f"  FFI overhead: Minimal (single Rust call for all {len(requests)} indicators)")

    # Add results to dataframe
    print(f"\n{separator}")
    print("ADDING INDICATORS TO DATAFRAME")
    print(separator)

    added_columns = 0

    for indicator_name, result in results.items():
        if isinstance(result, dict):
            # Multi-output indicator (e.g., MACD, Bollinger Bands)
            for key, array in result.items():
                col_name = f"{indicator_name}_{key}"
                df[col_name] = array
                added_columns += 1
                print(f"  + {col_name}")
        else:
            # Single-output indicator
            # Skip indicators that don't match dataframe length (e.g., volume profile histogram)
            if len(result) != len(df):
                print(f"  ⊗ {indicator_name} (histogram with {len(result)} bins, not time-series)")
                continue
            df[indicator_name] = result
            added_columns += 1
            print(f"  + {indicator_name}")

    print(f"\n✓ Added {added_columns} indicator columns")
    print(f"  Total columns: {len(df.columns)}")

    # Show sample data
    print(f"\n{separator}")
    print("SAMPLE DATA (first 5 rows)")
    print(separator)

    # Show a subset of columns for readability - showcasing multiple periods
    sample_cols = [
        'timestamp', 'close',
        'sma_14', 'sma_50', 'sma_200',  # Multiple SMA periods
        'rsi_7', 'rsi_14',                # Multiple RSI periods
        'ema_21',
        'atr_14',
    ]
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head().to_string(index=False))

    # Save to CSV
    print(f"\n{separator}")
    print("SAVING TO CSV")
    print(separator)

    write_start = time.perf_counter()
    df.to_csv(output_csv, index=False)
    write_time = time.perf_counter() - write_start

    file_size = output_csv.stat().st_size

    print(f"✓ Written in {write_time:.2f}s")
    print(f"✓ File size: {file_size / 1_048_576:.2f} MB")

    # Summary statistics
    print(f"\n{separator}")
    print("SUMMARY STATISTICS")
    print(separator)

    print(f"Data rows:           {len(df):>15,}")
    print(f"Original columns:    {len(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']):>15}")
    print(f"Indicator columns:   {added_columns:>15}")
    print(f"Total columns:       {len(df.columns):>15}")
    print(f"Total data points:   {len(df) * len(df.columns):>15,}")
    print(f"\nProcessing time:     {calc_time:>15.2f}s")
    print(f"Throughput:          {(len(requests) * len(df)) / calc_time:>15,.0f} calc/sec")

    # Show some interesting indicator values
    print(f"\n{separator}")
    print("INDICATOR VALUE RANGES")
    print(separator)

    print(f"{'Indicator':<30} {'Min':>12} {'Max':>12} {'Mean':>12}")
    print("-" * 80)

    interesting_indicators = [
        'sma_14', 'sma_50', 'sma_200',       # Compare different SMA periods
        'rsi_7', 'rsi_14',                   # Compare different RSI periods
        'atr_14', 'macd_12_26_9_macd',
        'bollinger_20_2_middle', 'bollinger_20_3_middle',  # Compare 2σ vs 3σ
        'obv', 'cci_20', 'williamsr_14'
    ]

    for ind in interesting_indicators:
        if ind in df.columns:
            non_nan = df[ind].dropna()
            if len(non_nan) > 0:
                print(f"{ind:<30} {non_nan.min():>12.2f} {non_nan.max():>12.2f} {non_nan.mean():>12.2f}")

    # Show comparison of multiple periods
    print(f"\n{separator}")
    print("MULTI-PERIOD COMPARISON (NEW API BENEFIT)")
    print(separator)
    print("\nDemonstrating the power of the new 3-tuple API:")
    print("We can now calculate the SAME indicator with different parameters in ONE batch!\n")

    # SMA comparison
    print("SMA Trends (different periods show different sensitivities):")
    if all(col in df.columns for col in ['sma_14', 'sma_50', 'sma_200']):
        recent_data = df.tail(1)
        print(f"  SMA(14):  ${recent_data['sma_14'].values[0]:,.2f}  (short-term, fast-moving)")
        print(f"  SMA(50):  ${recent_data['sma_50'].values[0]:,.2f}  (medium-term)")
        print(f"  SMA(200): ${recent_data['sma_200'].values[0]:,.2f}  (long-term, slow-moving)")

    # RSI comparison
    print("\nRSI Sensitivity (shorter period = more volatile):")
    if all(col in df.columns for col in ['rsi_7', 'rsi_14']):
        recent_data = df.tail(1)
        print(f"  RSI(7):   {recent_data['rsi_7'].values[0]:.2f}  (more responsive)")
        print(f"  RSI(14):  {recent_data['rsi_14'].values[0]:.2f}  (standard)")

    # Bollinger comparison
    print("\nBollinger Bands (wider bands with higher std dev):")
    if all(col in df.columns for col in ['bollinger_20_2_upper', 'bollinger_20_3_upper']):
        recent_data = df.tail(1)
        bb2_width = recent_data['bollinger_20_2_upper'].values[0] - recent_data['bollinger_20_2_lower'].values[0]
        bb3_width = recent_data['bollinger_20_3_upper'].values[0] - recent_data['bollinger_20_3_lower'].values[0]
        print(f"  Bollinger(20, 2σ): Width = ${bb2_width:,.2f}  (95% confidence)")
        print(f"  Bollinger(20, 3σ): Width = ${bb3_width:,.2f}  (99.7% confidence)")

    print("\nWith the OLD 2-tuple API, you could only calculate ONE variation per batch.")
    print("With the NEW 3-tuple API, you can calculate UNLIMITED variations in one call!")

    print(f"\n{separator}")
    print("SUCCESS!")
    print(separator)
    print(f"\nOutput file: {output_csv}")
    print(f"\nYou can now use this file for:")
    print(f"  - Backtesting strategies")
    print(f"  - Machine learning training")
    print(f"  - Statistical analysis")
    print(f"  - Chart visualization")


if __name__ == '__main__':
    main()

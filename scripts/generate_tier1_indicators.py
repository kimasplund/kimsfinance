#!/usr/bin/env python3
"""
Generate Tier 1 Indicator Charts
RSI, MACD, Stochastic, Multi-panel layouts
"""

import polars as pl
import pandas as pd
import numpy as np
import mplfinance as mpf
import os

print("=" * 80)
print("TIER 1 INDICATOR CHART GENERATION")
print("=" * 80)

# Load data
csv_path = "/home/kim/Documents/Github/binance-visual-ml/data/labeled_data/test_labeled.csv"
df = pl.read_csv(csv_path)

# Configuration
start_row = 1000
n_candles = 100
warmup = 50

# Slice with warmup
df = df.slice(start_row - warmup, n_candles + warmup)

# Rename columns
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

# Convert to pandas
df_pd = df.to_pandas()
df_pd["date"] = pd.to_datetime(df_pd["date"])
df_pd.set_index("date", inplace=True)

# Remove warmup for display
df_display = df_pd.iloc[warmup:]

# Base configuration
base_config = {
    "type": "candle",
    "style": "binance",
    "volume": True,
    "tight_layout": True,
    "figratio": (16, 9),
    "figscale": 1.6,  # 720p
}

# Create directories
os.makedirs("docs/sample_charts/indicators/rsi", exist_ok=True)
os.makedirs("docs/sample_charts/indicators/macd", exist_ok=True)
os.makedirs("docs/sample_charts/indicators/stochastic", exist_ok=True)
os.makedirs("docs/sample_charts/indicators/multi_panel", exist_ok=True)

print("\n" + "=" * 80)
print("RSI INDICATORS (4 charts)")
print("=" * 80)


# Calculate RSI
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# RSI Chart 1: Standard RSI 14
print("\n1. RSI 14 period...")
df_display["RSI_14"] = calculate_rsi(df_display["Close"], 14)

apds = [
    mpf.make_addplot(df_display["RSI_14"], panel=2, color="cyan", ylabel="RSI"),
    mpf.make_addplot([70] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.5),
    mpf.make_addplot([30] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.5),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/rsi/01_rsi_14.webp",
)
print("   ✅ Saved: indicators/rsi/01_rsi_14.webp")

# RSI Chart 2: Multiple RSI periods
print("2. Multiple RSI periods (7, 14, 21)...")
df_display["RSI_7"] = calculate_rsi(df_display["Close"], 7)
df_display["RSI_21"] = calculate_rsi(df_display["Close"], 21)

apds = [
    mpf.make_addplot(df_display["RSI_7"], panel=2, color="yellow", ylabel="RSI", label="RSI 7"),
    mpf.make_addplot(df_display["RSI_14"], panel=2, color="cyan", label="RSI 14"),
    mpf.make_addplot(df_display["RSI_21"], panel=2, color="magenta", label="RSI 21"),
    mpf.make_addplot([70] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.3),
    mpf.make_addplot([30] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.3),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/rsi/02_rsi_7_14_21.webp",
)
print("   ✅ Saved: indicators/rsi/02_rsi_7_14_21.webp")

# RSI Chart 3: Oversold example
print("3. RSI oversold/overbought zones...")
apds = [
    mpf.make_addplot(df_display["RSI_14"], panel=2, color="cyan", ylabel="RSI"),
    mpf.make_addplot([80] * len(df_display), panel=2, color="darkred", linestyle="--", alpha=0.5),
    mpf.make_addplot([70] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.5),
    mpf.make_addplot([30] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.5),
    mpf.make_addplot([20] * len(df_display), panel=2, color="darkgreen", linestyle="--", alpha=0.5),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/rsi/03_rsi_zones.webp",
)
print("   ✅ Saved: indicators/rsi/03_rsi_zones.webp")

# RSI Chart 4: With SMA on RSI
print("4. RSI with moving average...")
df_display["RSI_SMA"] = df_display["RSI_14"].rolling(9).mean()

apds = [
    mpf.make_addplot(df_display["RSI_14"], panel=2, color="cyan", ylabel="RSI"),
    mpf.make_addplot(df_display["RSI_SMA"], panel=2, color="orange", linestyle="--"),
    mpf.make_addplot([70] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.3),
    mpf.make_addplot([30] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.3),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/rsi/04_rsi_with_sma.webp",
)
print("   ✅ Saved: indicators/rsi/04_rsi_with_sma.webp")

print("\n" + "=" * 80)
print("MACD INDICATORS (4 charts)")
print("=" * 80)


# Calculate MACD
def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# MACD Chart 1: Standard MACD
print("\n1. Standard MACD (12, 26, 9)...")
macd, signal, hist = calculate_macd(df_display["Close"])

colors = ["red" if h < 0 else "green" for h in hist]

apds = [
    mpf.make_addplot(macd, panel=2, color="cyan", ylabel="MACD"),
    mpf.make_addplot(signal, panel=2, color="orange"),
    mpf.make_addplot(hist, panel=2, type="bar", color=colors, alpha=0.3),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1.2),
    savefig="docs/sample_charts/indicators/macd/01_macd_standard.webp",
)
print("   ✅ Saved: indicators/macd/01_macd_standard.webp")

# MACD Chart 2: Fast MACD
print("2. Fast MACD (5, 35, 5)...")
macd_fast, signal_fast, hist_fast = calculate_macd(df_display["Close"], 5, 35, 5)

colors = ["red" if h < 0 else "green" for h in hist_fast]

apds = [
    mpf.make_addplot(macd_fast, panel=2, color="cyan", ylabel="MACD"),
    mpf.make_addplot(signal_fast, panel=2, color="orange"),
    mpf.make_addplot(hist_fast, panel=2, type="bar", color=colors, alpha=0.3),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1.2),
    savefig="docs/sample_charts/indicators/macd/02_macd_fast.webp",
)
print("   ✅ Saved: indicators/macd/02_macd_fast.webp")

# MACD Chart 3: MACD with zero line
print("3. MACD with zero line emphasis...")
colors = ["red" if h < 0 else "green" for h in hist]

apds = [
    mpf.make_addplot(macd, panel=2, color="cyan", ylabel="MACD"),
    mpf.make_addplot(signal, panel=2, color="orange"),
    mpf.make_addplot(hist, panel=2, type="bar", color=colors, alpha=0.4),
    mpf.make_addplot([0] * len(df_display), panel=2, color="white", linestyle="--", alpha=0.5),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1.2),
    savefig="docs/sample_charts/indicators/macd/03_macd_zero_line.webp",
)
print("   ✅ Saved: indicators/macd/03_macd_zero_line.webp")

# MACD Chart 4: MACD histogram only
print("4. MACD histogram only...")
colors = ["red" if h < 0 else "green" for h in hist]

apds = [
    mpf.make_addplot(hist, panel=2, type="bar", color=colors, ylabel="MACD Hist"),
    mpf.make_addplot([0] * len(df_display), panel=2, color="white", linestyle="-", alpha=0.5),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/macd/04_macd_histogram.webp",
)
print("   ✅ Saved: indicators/macd/04_macd_histogram.webp")

print("\n" + "=" * 80)
print("STOCHASTIC INDICATORS (3 charts)")
print("=" * 80)


# Calculate Stochastic
def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


# Stochastic Chart 1: Standard 14,3
print("\n1. Stochastic (14, 3)...")
k, d = calculate_stochastic(df_display["High"], df_display["Low"], df_display["Close"])

apds = [
    mpf.make_addplot(k, panel=2, color="cyan", ylabel="Stochastic"),
    mpf.make_addplot(d, panel=2, color="orange"),
    mpf.make_addplot([80] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.5),
    mpf.make_addplot([20] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.5),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/stochastic/01_stochastic_14_3.webp",
)
print("   ✅ Saved: indicators/stochastic/01_stochastic_14_3.webp")

# Stochastic Chart 2: Fast stochastic 5,3
print("2. Fast Stochastic (5, 3)...")
k_fast, d_fast = calculate_stochastic(
    df_display["High"], df_display["Low"], df_display["Close"], 5, 3
)

apds = [
    mpf.make_addplot(k_fast, panel=2, color="cyan", ylabel="Stochastic"),
    mpf.make_addplot(d_fast, panel=2, color="orange"),
    mpf.make_addplot([80] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.5),
    mpf.make_addplot([20] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.5),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/stochastic/02_stochastic_fast.webp",
)
print("   ✅ Saved: indicators/stochastic/02_stochastic_fast.webp")

# Stochastic Chart 3: With zones
print("3. Stochastic with overbought/oversold zones...")
apds = [
    mpf.make_addplot(k, panel=2, color="cyan", ylabel="Stochastic"),
    mpf.make_addplot(d, panel=2, color="orange"),
    mpf.make_addplot([80] * len(df_display), panel=2, color="darkred", linestyle="--", alpha=0.5),
    mpf.make_addplot([70] * len(df_display), panel=2, color="red", linestyle=":", alpha=0.3),
    mpf.make_addplot([30] * len(df_display), panel=2, color="green", linestyle=":", alpha=0.3),
    mpf.make_addplot([20] * len(df_display), panel=2, color="darkgreen", linestyle="--", alpha=0.5),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/stochastic/03_stochastic_zones.webp",
)
print("   ✅ Saved: indicators/stochastic/03_stochastic_zones.webp")

print("\n" + "=" * 80)
print("MULTI-PANEL DASHBOARDS (4 charts)")
print("=" * 80)

# Multi-panel Chart 1: Price + Volume + RSI
print("\n1. Price + Volume + RSI...")
apds = [
    mpf.make_addplot(df_display["RSI_14"], panel=2, color="cyan", ylabel="RSI"),
    mpf.make_addplot([70] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.3),
    mpf.make_addplot([30] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.3),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/multi_panel/01_price_volume_rsi.webp",
)
print("   ✅ Saved: indicators/multi_panel/01_price_volume_rsi.webp")

# Multi-panel Chart 2: Price + Volume + MACD
print("2. Price + Volume + MACD...")
colors = ["red" if h < 0 else "green" for h in hist]

apds = [
    mpf.make_addplot(macd, panel=2, color="cyan", ylabel="MACD"),
    mpf.make_addplot(signal, panel=2, color="orange"),
    mpf.make_addplot(hist, panel=2, type="bar", color=colors, alpha=0.3),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1.2),
    savefig="docs/sample_charts/indicators/multi_panel/02_price_volume_macd.webp",
)
print("   ✅ Saved: indicators/multi_panel/02_price_volume_macd.webp")

# Multi-panel Chart 3: Price + Volume + Stochastic
print("3. Price + Volume + Stochastic...")
apds = [
    mpf.make_addplot(k, panel=2, color="cyan", ylabel="Stochastic"),
    mpf.make_addplot(d, panel=2, color="orange"),
    mpf.make_addplot([80] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.3),
    mpf.make_addplot([20] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.3),
]

mpf.plot(
    df_display,
    **base_config,
    addplot=apds,
    panel_ratios=(3, 1, 1),
    savefig="docs/sample_charts/indicators/multi_panel/03_price_volume_stochastic.webp",
)
print("   ✅ Saved: indicators/multi_panel/03_price_volume_stochastic.webp")

# Multi-panel Chart 4: Full dashboard (Price + Volume + RSI + MACD)
print("4. Full dashboard (Price + Volume + RSI + MACD)...")

colors = ["red" if h < 0 else "green" for h in hist]

apds = [
    mpf.make_addplot(df_display["RSI_14"], panel=2, color="cyan", ylabel="RSI"),
    mpf.make_addplot([70] * len(df_display), panel=2, color="red", linestyle="--", alpha=0.3),
    mpf.make_addplot([30] * len(df_display), panel=2, color="green", linestyle="--", alpha=0.3),
    mpf.make_addplot(macd, panel=3, color="cyan", ylabel="MACD"),
    mpf.make_addplot(signal, panel=3, color="orange"),
    mpf.make_addplot(hist, panel=3, type="bar", color=colors, alpha=0.3),
]

# Create custom config for larger dashboard
dashboard_config = base_config.copy()
dashboard_config["figscale"] = 2.0

mpf.plot(
    df_display,
    **dashboard_config,
    addplot=apds,
    panel_ratios=(3, 0.8, 1, 1.2),
    savefig="docs/sample_charts/indicators/multi_panel/04_full_dashboard.webp",
)
print("   ✅ Saved: indicators/multi_panel/04_full_dashboard.webp")

print("\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)

# Summary
total_size = 0
for root, dirs, files in os.walk("docs/sample_charts/indicators"):
    for f in files:
        if f.endswith(".webp"):
            total_size += os.path.getsize(os.path.join(root, f))

print(f"\nTotal charts generated: 15")
print(f"Total size: {total_size / 1024:.1f} KB")
print(f"Average size per chart: {total_size / 1024 / 15:.1f} KB")
print("\nCategories:")
print("  ✅ RSI: 4 charts")
print("  ✅ MACD: 4 charts")
print("  ✅ Stochastic: 3 charts")
print("  ✅ Multi-panel: 4 charts")

print("\n" + "=" * 80)

#!/usr/bin/env python3
"""
Demonstration of the Aroon indicator implementation.

This script shows how to use the calculate_aroon function and
demonstrates its ability to identify trends.
"""

import numpy as np
from kimsfinance.ops import calculate_aroon

# Generate sample data with uptrend, consolidation, and downtrend
n = 100
np.random.seed(42)

# Create trending data
uptrend = 100 + np.arange(40) * 0.8  # Strong uptrend
consolidation = uptrend[-1] + np.random.randn(20) * 0.2  # Consolidation
downtrend = consolidation[-1] - np.arange(40) * 0.6  # Downtrend

prices = np.concatenate([uptrend, consolidation, downtrend])
highs = prices + np.abs(np.random.randn(n) * 0.3)
lows = prices - np.abs(np.random.randn(n) * 0.3)

# Calculate Aroon indicator
print("Calculating Aroon indicator...")
aroon_up, aroon_down = calculate_aroon(highs, lows, period=25, engine="cpu")

print(f"\nData points: {len(aroon_up)}")
print(f"Valid Aroon values: {np.sum(~np.isnan(aroon_up))}")

# Analyze different phases
phases = [
    ("Uptrend (bars 30-40)", 30, 40),
    ("Consolidation (bars 45-55)", 45, 55),
    ("Downtrend (bars 70-80)", 70, 80),
]

for phase_name, start, end in phases:
    avg_up = np.mean(aroon_up[start:end])
    avg_down = np.mean(aroon_down[start:end])

    print(f"\n{phase_name}:")
    print(f"  Aroon Up:   {avg_up:.2f}")
    print(f"  Aroon Down: {avg_down:.2f}")

    if avg_up > 70 and avg_down < 30:
        print(f"  Signal: STRONG UPTREND ↑")
    elif avg_down > 70 and avg_up < 30:
        print(f"  Signal: STRONG DOWNTREND ↓")
    elif abs(avg_up - avg_down) < 20:
        print(f"  Signal: CONSOLIDATION ↔")
    elif avg_up > avg_down:
        print(f"  Signal: Weak uptrend")
    else:
        print(f"  Signal: Weak downtrend")

# Detect crossovers
valid_idx = ~(np.isnan(aroon_up) | np.isnan(aroon_down))
bullish_cross = (
    (aroon_up[1:] > aroon_down[1:]) &
    (aroon_up[:-1] <= aroon_down[:-1]) &
    valid_idx[1:]
)
bearish_cross = (
    (aroon_down[1:] > aroon_up[1:]) &
    (aroon_down[:-1] <= aroon_up[:-1]) &
    valid_idx[1:]
)

bullish_signals = np.where(bullish_cross)[0] + 1
bearish_signals = np.where(bearish_cross)[0] + 1

print(f"\nCrossover Signals:")
print(f"  Bullish crossovers (Aroon Up crosses above Down): {len(bullish_signals)}")
if len(bullish_signals) > 0:
    print(f"    At bars: {bullish_signals[:5]}")  # Show first 5

print(f"  Bearish crossovers (Aroon Down crosses above Up): {len(bearish_signals)}")
if len(bearish_signals) > 0:
    print(f"    At bars: {bearish_signals[:5]}")  # Show first 5

print("\n" + "="*60)
print("Aroon Indicator Summary")
print("="*60)
print("Aroon measures time since highest high (Up) and lowest low (Down)")
print("Values range from 0-100:")
print("  - Aroon Up = 100: Just made new high (strong uptrend)")
print("  - Aroon Down = 100: Just made new low (strong downtrend)")
print("  - Both near 50: No clear trend (consolidation)")
print("\nKey Signals:")
print("  - Aroon Up > 70 & Down < 30: Strong uptrend")
print("  - Aroon Down > 70 & Up < 30: Strong downtrend")
print("  - Up crosses above Down: Bullish signal")
print("  - Down crosses above Up: Bearish signal")
print("="*60)

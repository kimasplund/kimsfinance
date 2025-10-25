#!/usr/bin/env python3
"""Test script to verify all 24 technical indicator Python bindings work."""

import numpy as np
import sys

try:
    import kimsfinance_core
except ImportError:
    print("ERROR: kimsfinance_core not installed. Install with:")
    print("  cd rust && maturin develop --release")
    sys.exit(1)

print(f"kimsfinance_core version: {kimsfinance_core.__version__}")
print(f"Module doc: {kimsfinance_core.__doc__}")
print()

# Generate test data
np.random.seed(42)
n = 100
prices = np.cumsum(np.random.randn(n) * 0.5) + 100
high = prices + np.abs(np.random.randn(n) * 2)
low = prices - np.abs(np.random.randn(n) * 2)
open_prices = prices + np.random.randn(n) * 1
close = prices + np.random.randn(n) * 1
volume = np.abs(np.random.randn(n) * 1000) + 1000

print("Testing all 24 technical indicators...")
print("=" * 60)

# Track successes and failures
successes = []
failures = []

def test_indicator(name, func, *args, **kwargs):
    """Test a single indicator function."""
    try:
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            # Multi-output indicator
            print(f"✓ {name}: {len(result)} outputs", end="")
            details = []
            for key, arr in result.items():
                details.append(f"{key}({np.sum(~np.isnan(arr))} values)")
            print(f" - {', '.join(details)}")
        else:
            # Single output indicator
            non_nan = np.sum(~np.isnan(result))
            print(f"✓ {name}: shape {result.shape}, non-NaN: {non_nan}")
        successes.append(name)
        return True
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {e}")
        failures.append((name, str(e)))
        return False

# Moving Averages (7 indicators)
print("\n--- Moving Averages (7) ---")
test_indicator("SMA", kimsfinance_core.calculate_sma, prices)
test_indicator("EMA", kimsfinance_core.calculate_ema, prices)
test_indicator("WMA", kimsfinance_core.calculate_wma, prices)
test_indicator("VWMA", kimsfinance_core.calculate_vwma, prices, volume)
test_indicator("DEMA", kimsfinance_core.calculate_dema, prices)
test_indicator("TEMA", kimsfinance_core.calculate_tema, prices)
test_indicator("HMA", kimsfinance_core.calculate_hma, prices)

# Momentum Indicators (8 indicators)
print("\n--- Momentum Indicators (8) ---")
test_indicator("RSI", kimsfinance_core.calculate_rsi, prices)
test_indicator("ROC", kimsfinance_core.calculate_roc, prices)
test_indicator("Williams %R", kimsfinance_core.calculate_williams_r, high, low, close)
test_indicator("Stochastic", kimsfinance_core.calculate_stochastic, high, low, close)
test_indicator("Aroon", kimsfinance_core.calculate_aroon, high, low)
test_indicator("CCI", kimsfinance_core.calculate_cci, high, low, close)
test_indicator("MACD", kimsfinance_core.calculate_macd, prices)
test_indicator("TSI", kimsfinance_core.calculate_tsi, prices)

# Volatility Indicators (5 indicators)
print("\n--- Volatility Indicators (5) ---")
test_indicator("ATR", kimsfinance_core.calculate_atr, high, low, close)
test_indicator("Bollinger Bands", kimsfinance_core.calculate_bollinger_bands, prices)
test_indicator("Keltner Channels", kimsfinance_core.calculate_keltner_channels, high, low, close)
test_indicator("Donchian Channels", kimsfinance_core.calculate_donchian_channels, high, low)
test_indicator("Elder Ray", kimsfinance_core.calculate_elder_ray, high, low, close)

# Volume Indicators (4 indicators)
print("\n--- Volume Indicators (4) ---")
test_indicator("OBV", kimsfinance_core.calculate_obv, close, volume)
test_indicator("VWAP", kimsfinance_core.calculate_vwap, high, low, close, volume)
test_indicator("CMF", kimsfinance_core.calculate_cmf, high, low, close, volume)
test_indicator("Volume Profile", kimsfinance_core.calculate_volume_profile, high, low, close, volume)

# Summary
print("\n" + "=" * 60)
print(f"SUMMARY: {len(successes)}/24 indicators passed")
print(f"Successes: {len(successes)}")
print(f"Failures: {len(failures)}")

if failures:
    print("\nFailed indicators:")
    for name, error in failures:
        print(f"  - {name}: {error}")
    sys.exit(1)
else:
    print("\n✓ All 24 technical indicators working correctly!")
    sys.exit(0)

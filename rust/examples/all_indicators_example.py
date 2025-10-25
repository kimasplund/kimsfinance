#!/usr/bin/env python3
"""
Comprehensive example showing all 24 technical indicators from kimsfinance_core.

This demonstrates:
- All 7 moving averages
- All 8 momentum indicators
- All 5 volatility indicators
- All 4 volume indicators
- Proper handling of multi-output indicators
- NaN warmup period handling
"""

import numpy as np
import kimsfinance_core

print(f"kimsfinance_core v{kimsfinance_core.__version__}")
print("=" * 70)

# Generate realistic-looking OHLCV data
np.random.seed(42)
n = 200  # Enough data for all indicators to have valid outputs

# Simulate price movement with trend + noise
trend = np.linspace(100, 120, n)
noise = np.cumsum(np.random.randn(n) * 0.3)
close = trend + noise

# Generate OHLCV from close
high = close + np.abs(np.random.randn(n) * 1.5)
low = close - np.abs(np.random.randn(n) * 1.5)
open_prices = close + np.random.randn(n) * 0.5
volume = np.abs(np.random.randn(n) * 1000) + 5000

print(f"\nData: {n} candles")
print(f"Price range: {low.min():.2f} - {high.max():.2f}")
print(f"Average volume: {volume.mean():.0f}")
print()

# ============================================================================
# MOVING AVERAGES (7)
# ============================================================================
print("MOVING AVERAGES")
print("-" * 70)

sma = kimsfinance_core.calculate_sma(close, period=20)
print(f"SMA(20):  Latest = {sma[-1]:.2f}, Valid values = {np.sum(~np.isnan(sma))}")

ema = kimsfinance_core.calculate_ema(close, period=20)
print(f"EMA(20):  Latest = {ema[-1]:.2f}, Valid values = {np.sum(~np.isnan(ema))}")

wma = kimsfinance_core.calculate_wma(close, period=20)
print(f"WMA(20):  Latest = {wma[-1]:.2f}, Valid values = {np.sum(~np.isnan(wma))}")

vwma = kimsfinance_core.calculate_vwma(close, volume, period=20)
print(f"VWMA(20): Latest = {vwma[-1]:.2f}, Valid values = {np.sum(~np.isnan(vwma))}")

dema = kimsfinance_core.calculate_dema(close, period=20)
print(f"DEMA(20): Latest = {dema[-1]:.2f}, Valid values = {np.sum(~np.isnan(dema))}")

tema = kimsfinance_core.calculate_tema(close, period=20)
print(f"TEMA(20): Latest = {tema[-1]:.2f}, Valid values = {np.sum(~np.isnan(tema))}")

hma = kimsfinance_core.calculate_hma(close, period=20)
print(f"HMA(20):  Latest = {hma[-1]:.2f}, Valid values = {np.sum(~np.isnan(hma))}")

# ============================================================================
# MOMENTUM INDICATORS (8)
# ============================================================================
print("\n\nMOMENTUM INDICATORS")
print("-" * 70)

rsi = kimsfinance_core.calculate_rsi(close, period=14)
print(f"RSI(14):       Latest = {rsi[-1]:.2f}, Valid values = {np.sum(~np.isnan(rsi))}")

roc = kimsfinance_core.calculate_roc(close, period=14)
print(f"ROC(14):       Latest = {roc[-1]:.2f}%, Valid values = {np.sum(~np.isnan(roc))}")

williams_r = kimsfinance_core.calculate_williams_r(high, low, close, period=14)
print(f"Williams %R:   Latest = {williams_r[-1]:.2f}, Valid values = {np.sum(~np.isnan(williams_r))}")

stoch = kimsfinance_core.calculate_stochastic(high, low, close, k_period=14, d_period=3)
print(f"Stochastic:    %K = {stoch['k'][-1]:.2f}, %D = {stoch['d'][-1]:.2f}")
print(f"               Valid K = {np.sum(~np.isnan(stoch['k']))}, Valid D = {np.sum(~np.isnan(stoch['d']))}")

aroon = kimsfinance_core.calculate_aroon(high, low, period=14)
aroon_osc = aroon['aroon_up'] - aroon['aroon_down']  # Calculate oscillator
print(f"Aroon:         Up = {aroon['aroon_up'][-1]:.2f}, Down = {aroon['aroon_down'][-1]:.2f}")
print(f"               Oscillator = {aroon_osc[-1]:.2f}")

cci = kimsfinance_core.calculate_cci(high, low, close, period=20)
print(f"CCI(20):       Latest = {cci[-1]:.2f}, Valid values = {np.sum(~np.isnan(cci))}")

macd = kimsfinance_core.calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
print(f"MACD:          Line = {macd['macd'][-1]:.2f}, Signal = {macd['signal'][-1]:.2f}")
print(f"               Histogram = {macd['histogram'][-1]:.2f}")
print(f"               Valid values = {np.sum(~np.isnan(macd['macd']))}")

tsi = kimsfinance_core.calculate_tsi(close, long_period=25, short_period=13, signal_period=7)
print(f"TSI:           TSI = {tsi['tsi'][-1]:.2f}, Signal = {tsi['signal'][-1]:.2f}")
print(f"               Valid TSI = {np.sum(~np.isnan(tsi['tsi']))}, Valid Signal = {np.sum(~np.isnan(tsi['signal']))}")

# ============================================================================
# VOLATILITY INDICATORS (5)
# ============================================================================
print("\n\nVOLATILITY INDICATORS")
print("-" * 70)

atr = kimsfinance_core.calculate_atr(high, low, close, period=14)
print(f"ATR(14):       Latest = {atr[-1]:.2f}, Valid values = {np.sum(~np.isnan(atr))}")

bb = kimsfinance_core.calculate_bollinger_bands(close, period=20, std_dev=2.0)
print(f"Bollinger:     Upper = {bb['upper'][-1]:.2f}, Middle = {bb['middle'][-1]:.2f}")
print(f"               Lower = {bb['lower'][-1]:.2f}, Width = {bb['upper'][-1] - bb['lower'][-1]:.2f}")
print(f"               Valid values = {np.sum(~np.isnan(bb['middle']))}")

kc = kimsfinance_core.calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)
print(f"Keltner:       Upper = {kc['upper'][-1]:.2f}, Middle = {kc['middle'][-1]:.2f}")
print(f"               Lower = {kc['lower'][-1]:.2f}, Width = {kc['upper'][-1] - kc['lower'][-1]:.2f}")

dc = kimsfinance_core.calculate_donchian_channels(high, low, period=20)
print(f"Donchian:      Upper = {dc['upper'][-1]:.2f}, Middle = {dc['middle'][-1]:.2f}")
print(f"               Lower = {dc['lower'][-1]:.2f}, Width = {dc['upper'][-1] - dc['lower'][-1]:.2f}")

elder = kimsfinance_core.calculate_elder_ray(high, low, close, ema_period=13)
print(f"Elder Ray:     Bull Power = {elder['bull_power'][-1]:.2f}")
print(f"               Bear Power = {elder['bear_power'][-1]:.2f}")
print(f"               Net = {elder['bull_power'][-1] + elder['bear_power'][-1]:.2f}")

# ============================================================================
# VOLUME INDICATORS (4)
# ============================================================================
print("\n\nVOLUME INDICATORS")
print("-" * 70)

obv = kimsfinance_core.calculate_obv(close, volume)
print(f"OBV:           Latest = {obv[-1]:.0f}, All values valid = {np.sum(~np.isnan(obv)) == n}")

vwap = kimsfinance_core.calculate_vwap(high, low, close, volume)
print(f"VWAP:          Latest = {vwap[-1]:.2f}, All values valid = {np.sum(~np.isnan(vwap)) == n}")

cmf = kimsfinance_core.calculate_cmf(high, low, close, volume, period=20)
print(f"CMF(20):       Latest = {cmf[-1]:.4f}, Valid values = {np.sum(~np.isnan(cmf))}")

vp = kimsfinance_core.calculate_volume_profile(high, low, close, volume, num_bins=20)
max_volume_idx = np.argmax(vp)
price_range = high.max() - low.min()
poc_price = low.min() + (max_volume_idx + 0.5) * (price_range / len(vp))
print(f"Volume Profile: {len(vp)} bins, POC at ~{poc_price:.2f}")
print(f"                Total volume distribution = {vp.sum():.0f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"All 24 technical indicators calculated successfully!")
print(f"Data size: {n} candles")
print(f"Total indicators: 24 (7 MA + 8 momentum + 5 volatility + 4 volume)")
print("\nPerformance notes:")
print("- Indicators use SIMD-optimized Rust implementations")
print("- 3-8x faster than equivalent pandas/NumPy code")
print("- Zero-copy FFI for minimal overhead")
print("- Parallel processing for datasets >500-5000 rows")
print("\nNext steps:")
print("- Use these indicators in trading strategies")
print("- Combine with coordinate calculations for chart rendering")
print("- Benchmark against Python implementations")

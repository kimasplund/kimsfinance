# kimsfinance_core Quick Reference

**Version**: 0.1.0 | **Language**: Python (Rust-accelerated) | **Performance**: 3-8x faster than pandas

## Installation

```bash
cd rust && maturin develop --release
```

## Import

```python
import kimsfinance_core
import numpy as np
```

## Single Output Indicators

```python
# Moving Averages
sma = kimsfinance_core.calculate_sma(prices, period=14)
ema = kimsfinance_core.calculate_ema(prices, period=14)
wma = kimsfinance_core.calculate_wma(prices, period=14)
vwma = kimsfinance_core.calculate_vwma(prices, volume, period=14)
dema = kimsfinance_core.calculate_dema(prices, period=14)
tema = kimsfinance_core.calculate_tema(prices, period=14)
hma = kimsfinance_core.calculate_hma(prices, period=14)

# Momentum
rsi = kimsfinance_core.calculate_rsi(prices, period=14)  # 0-100 range
roc = kimsfinance_core.calculate_roc(prices, period=14)  # Percentage
williams_r = kimsfinance_core.calculate_williams_r(high, low, close, period=14)  # -100 to 0
cci = kimsfinance_core.calculate_cci(high, low, close, period=20)

# Volatility
atr = kimsfinance_core.calculate_atr(high, low, close, period=14)

# Volume
obv = kimsfinance_core.calculate_obv(close, volume)
vwap = kimsfinance_core.calculate_vwap(high, low, close, volume)
cmf = kimsfinance_core.calculate_cmf(high, low, close, volume, period=20)  # -1 to 1 range
vp = kimsfinance_core.calculate_volume_profile(high, low, close, volume, num_bins=20)
```

## Multi-Output Indicators (Return Dictionaries)

```python
# Stochastic Oscillator
stoch = kimsfinance_core.calculate_stochastic(high, low, close, k_period=14, d_period=3)
k = stoch['k']  # Fast %K
d = stoch['d']  # Slow %D

# Aroon Indicator
aroon = kimsfinance_core.calculate_aroon(high, low, period=14)
up = aroon['aroon_up']
down = aroon['aroon_down']
oscillator = up - down  # Calculate oscillator manually

# MACD
macd = kimsfinance_core.calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)
macd_line = macd['macd']
signal = macd['signal']
histogram = macd['histogram']

# TSI
tsi = kimsfinance_core.calculate_tsi(prices, long_period=25, short_period=13, signal_period=7)
tsi_line = tsi['tsi']
signal = tsi['signal']

# Bollinger Bands
bb = kimsfinance_core.calculate_bollinger_bands(prices, period=20, std_dev=2.0)
middle = bb['middle']
upper = bb['upper']
lower = bb['lower']

# Keltner Channels
kc = kimsfinance_core.calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)
middle = kc['middle']
upper = kc['upper']
lower = kc['lower']

# Donchian Channels
dc = kimsfinance_core.calculate_donchian_channels(high, low, period=20)
middle = dc['middle']
upper = dc['upper']
lower = dc['lower']

# Elder Ray
elder = kimsfinance_core.calculate_elder_ray(high, low, close, ema_period=13)
bull_power = elder['bull_power']
bear_power = elder['bear_power']
```

## Error Handling

```python
try:
    rsi = kimsfinance_core.calculate_rsi(prices, period=0)  # Invalid
except ValueError as e:
    print(f"Invalid parameter: {e}")

try:
    rsi = kimsfinance_core.calculate_rsi(prices[:5], period=14)  # Insufficient data
except RuntimeError as e:
    print(f"Computation error: {e}")
```

## NaN Handling

First `N` values are NaN (warmup period):

```python
rsi = kimsfinance_core.calculate_rsi(prices, period=14)
# First 15 values are NaN

# Get valid values only
valid_rsi = rsi[~np.isnan(rsi)]

# Or use from a specific index
rsi_from_20 = rsi[20:]  # Skip warmup
```

## Complete Example

```python
import numpy as np
import kimsfinance_core

# Generate OHLCV data
n = 200
close = np.cumsum(np.random.randn(n) * 0.5) + 100
high = close + np.abs(np.random.randn(n) * 2)
low = close - np.abs(np.random.randn(n) * 2)
volume = np.abs(np.random.randn(n) * 1000) + 5000

# Calculate indicators
rsi = kimsfinance_core.calculate_rsi(close, period=14)
macd = kimsfinance_core.calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
bb = kimsfinance_core.calculate_bollinger_bands(close, period=20, std_dev=2.0)
atr = kimsfinance_core.calculate_atr(high, low, close, period=14)
vwap = kimsfinance_core.calculate_vwap(high, low, close, volume)

# Use results
print(f"Latest RSI: {rsi[-1]:.2f}")
print(f"MACD: {macd['macd'][-1]:.2f}, Signal: {macd['signal'][-1]:.2f}")
print(f"BB Width: {bb['upper'][-1] - bb['lower'][-1]:.2f}")
print(f"ATR: {atr[-1]:.2f}")
print(f"VWAP: {vwap[-1]:.2f}")

# Trading signal example
if rsi[-1] > 70:
    print("Overbought")
elif rsi[-1] < 30:
    print("Oversold")

if macd['histogram'][-1] > 0:
    print("Bullish MACD")
else:
    print("Bearish MACD")
```

## Performance Tips

1. **Use enough data**: 200+ candles for all indicators to have valid values
2. **Batch processing**: Calculate all needed indicators in one script (reduces overhead)
3. **Parallel datasets**: Rust uses Rayon for parallel processing automatically
4. **Zero-copy**: NumPy arrays are passed by reference (no copying)

## Common Patterns

### Check for valid values
```python
rsi = kimsfinance_core.calculate_rsi(prices, period=14)
valid_count = np.sum(~np.isnan(rsi))
print(f"Valid RSI values: {valid_count}/{len(rsi)}")
```

### Crossover detection
```python
macd = kimsfinance_core.calculate_macd(prices)
crossover = (macd['macd'][-2] < macd['signal'][-2]) and (macd['macd'][-1] > macd['signal'][-1])
```

### Bollinger Bands squeeze
```python
bb = kimsfinance_core.calculate_bollinger_bands(prices, period=20)
width = bb['upper'] - bb['lower']
squeeze = width[-1] < np.percentile(width[~np.isnan(width)], 10)
```

### Volume confirmation
```python
obv = kimsfinance_core.calculate_obv(close, volume)
price_up = close[-1] > close[-2]
volume_confirm = obv[-1] > obv[-2]
strong_signal = price_up and volume_confirm
```

## All 24 Indicators at a Glance

| Category | Count | Indicators |
|----------|-------|-----------|
| **Moving Averages** | 7 | SMA, EMA, WMA, VWMA, DEMA, TEMA, HMA |
| **Momentum** | 8 | RSI, ROC, Williams %R, Stochastic, Aroon, CCI, MACD, TSI |
| **Volatility** | 5 | ATR, Bollinger Bands, Keltner Channels, Donchian Channels, Elder Ray |
| **Volume** | 4 | OBV, VWAP, CMF, Volume Profile |
| **Total** | **24** | All production-ready |

## Documentation

- Full API docs: `rust/PYTHON_BINDINGS.md`
- Example script: `rust/examples/all_indicators_example.py`
- Test suite: `test_rust_bindings.py`
- Completion report: `RUST_BINDINGS_COMPLETE.md`

## Support

```python
# Version check
print(kimsfinance_core.__version__)  # 0.1.0

# Module documentation
print(kimsfinance_core.__doc__)
```

---

**Performance**: 3-8x faster than pandas | **Technology**: Rust + PyO3 | **Status**: Production-ready

# Batch Indicator API

## Overview

The batch indicator API allows calculating multiple technical indicators in a single Python→Rust FFI call, reducing overhead for multi-indicator workflows.

## Implementation Details

### Files Modified
- `/home/kim-asplund/projects/kimsfinance/rust/Cargo.toml` - Added `serde_json = "1.0"`
- `/home/kim-asplund/projects/kimsfinance/rust/src/lib.rs` - Added batch module import and `calculate_indicators_batch()` function
- `/home/kim-asplund/projects/kimsfinance/rust/src/batch.rs` - Added clippy allow for acronyms, `#[allow(dead_code)]` for `open` field

### Python API

```python
import kimsfinance_core
import numpy as np

# Sample data
high = np.array([110.0, 115.0, 120.0, ...])
low = np.array([105.0, 110.0, 115.0, ...])
open_prices = np.array([107.0, 111.0, 116.0, ...])
close = np.array([108.0, 112.0, 118.0, ...])
volume = np.array([1000.0, 1500.0, 2000.0, ...])

# Single FFI call for all indicators
results = kimsfinance_core.calculate_indicators_batch(
    high, low, open_prices, close, volume,
    requests=[
        ("rsi", '{"period": 14}'),
        ("macd", '{"fast_period": 12, "slow_period": 26, "signal_period": 9}'),
        ("atr", '{"period": 14}'),
        ("bollinger", '{"period": 20, "std_dev": 2.0}'),
        ("sma", '{"period": 20}'),
        ("ema", '{"period": 14}'),
    ]
)

# Access results
rsi = results['rsi']  # NumPy array
macd_line = results['macd']['line']  # Multi-output indicators return nested dicts
macd_signal = results['macd']['signal']
macd_histogram = results['macd']['histogram']
```

## Supported Indicators

### Moving Averages
- `sma` - Simple Moving Average: `{"period": 14}`
- `ema` - Exponential Moving Average: `{"period": 14}`
- `wma` - Weighted Moving Average: `{"period": 14}`
- `vwma` - Volume Weighted Moving Average: `{"period": 14}`
- `dema` - Double Exponential Moving Average: `{"period": 14}`
- `tema` - Triple Exponential Moving Average: `{"period": 14}`
- `hma` - Hull Moving Average: `{"period": 14}`

### Momentum Indicators
- `rsi` - Relative Strength Index: `{"period": 14}`
- `roc` - Rate of Change: `{"period": 14}`
- `williamsr` or `williams_r` - Williams %R: `{"period": 14}`
- `stochastic` - Stochastic Oscillator: `{"k_period": 14, "d_period": 3}`
- `aroon` - Aroon Indicator: `{"period": 14}`
- `cci` - Commodity Channel Index: `{"period": 20}`
- `macd` - MACD: `{"fast_period": 12, "slow_period": 26, "signal_period": 9}`
- `tsi` - True Strength Index: `{"long_period": 25, "short_period": 13, "signal_period": 7}`

### Volatility Indicators
- `atr` - Average True Range: `{"period": 14}`
- `bollinger`, `bollingerbands`, `bb` - Bollinger Bands: `{"period": 20, "std_dev": 2.0}`
- `keltner`, `keltnerchannels`, `kc` - Keltner Channels: `{"ema_period": 20, "atr_period": 10, "atr_multiplier": 2.0}`
- `donchian`, `donchianchannels`, `dc` - Donchian Channels: `{"period": 20}`
- `elderray`, `elder_ray` - Elder Ray: `{"ema_period": 13}`

### Volume Indicators
- `obv` - On-Balance Volume: `{}`
- `vwap` - Volume Weighted Average Price: `{}`
- `cmf` - Chaikin Money Flow: `{"period": 20}`
- `volumeprofile`, `volume_profile` - Volume Profile: `{"num_bins": 20}`

### Trend Indicators
- `parabolicsar`, `psar`, `sar` - Parabolic SAR: `{"af_start": 0.02, "af_increment": 0.02, "af_max": 0.2}`
- `pivotpoints`, `pivot` - Pivot Points: `{}`

## Output Structure

### Single-Output Indicators
Returns NumPy array directly:
```python
rsi = results['rsi']  # np.ndarray
```

### Multi-Output Indicators
Returns nested dictionary:
```python
macd_dict = results['macd']  # dict
macd_line = macd_dict['line']  # np.ndarray
macd_signal = macd_dict['signal']  # np.ndarray
macd_histogram = macd_dict['histogram']  # np.ndarray
```

#### Key Mappings

**MACD**: `{'line', 'signal', 'histogram'}`
**Stochastic**: `{'k', 'd'}`
**Aroon**: `{'up', 'down'}`
**Bollinger Bands**: `{'middle', 'upper', 'lower'}`
**Keltner Channels**: `{'middle', 'upper', 'lower'}`
**Donchian Channels**: `{'middle', 'upper', 'lower'}`
**Elder Ray**: `{'bull_power', 'bear_power'}`
**TSI**: `{'tsi', 'signal'}`
**Pivot Points**: `{'pp', 'r1', 'r2', 'r3', 's1', 's2', 's3'}`

## Performance Characteristics

### When to Use Batch API
- **Multiple indicators needed**: 5+ indicators in same workflow
- **Large datasets**: >10,000 rows where computation dominates
- **Production workflows**: Where consistent API and error handling matters

### When to Use Individual APIs
- **Single indicator**: No benefit from batching
- **Small datasets**: <1,000 rows where FFI overhead is negligible
- **Different data sources**: Each indicator uses different OHLCV data

### Benchmark Results (1000 candles, 10 indicators)
- **Batch API**: ~3-24ms (varies with system)
- **Individual APIs**: ~1-17ms (varies with system)
- **Speedup**: 0.7-1.5x (dataset size dependent)

**Note**: FFI overhead reduction is most significant for:
1. Very large datasets (100K+ rows)
2. Many indicators (20+)
3. High-frequency workflows (1000s of calculations/sec)

## Error Handling

All errors are raised as Python exceptions:

```python
try:
    results = kimsfinance_core.calculate_indicators_batch(...)
except ValueError as e:
    # Invalid indicator name or malformed JSON params
    print(f"Parameter error: {e}")
except RuntimeError as e:
    # Indicator calculation error (e.g., insufficient data)
    print(f"Calculation error: {e}")
```

## Implementation Details

### Architecture
1. **Python Layer**: Accepts OHLCV arrays + list of (name, json_params) tuples
2. **JSON Parsing**: Parses JSON to `IndicatorRequest` enum using indicator name
3. **Batch Calculation**: Calls existing `batch::calculate_batch()` function
4. **Result Conversion**: Converts Rust HashMap to Python dict with NumPy arrays

### Zero-Copy Data Flow
- OHLCV data: PyReadonlyArray → ArrayView (zero-copy)
- Results: Vec<f64> → NumPy array (single copy, unavoidable)

### Memory Layout
```
Python                  Rust                    Python
------                  ----                    ------
NumPy arrays     →     ArrayView<f64>    →     NumPy arrays
(high, low, ...)       (zero-copy view)        (single copy)
```

## Testing

Run the test suite:
```bash
cd rust
cargo build --release
python3 test_batch_api.py
```

Expected output:
```
✓ Batch API exposed successfully
✓ All 10 indicators calculated in single call
✓ Results validated against individual calls
```

## Version Information

- **Rust**: 1.90.0+ (Edition 2024)
- **PyO3**: 0.27.1
- **serde_json**: 1.0
- **Implementation**: 2025-01-26

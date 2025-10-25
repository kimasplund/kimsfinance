# Python Bindings for kimsfinance_core

## Overview

Complete Python bindings for all 24 technical indicators plus coordinate calculations.

**Status**: ✅ All 24 indicators exported and tested (2025-10-25)

**Rust Version**: 1.90.0, Edition 2024
**PyO3 Version**: 0.27.1
**Build Tool**: maturin

## Installation

```bash
cd rust
maturin develop --release
```

## Exported Functions

### Coordinate Calculations (1 function)

- `calculate_coordinates_py()` - High-performance candlestick coordinate calculation

### Moving Averages (7 indicators)

1. `calculate_sma(prices, period=14)` - Simple Moving Average
2. `calculate_ema(prices, period=14)` - Exponential Moving Average
3. `calculate_wma(prices, period=14)` - Weighted Moving Average
4. `calculate_vwma(prices, volume, period=14)` - Volume Weighted Moving Average
5. `calculate_dema(prices, period=14)` - Double Exponential Moving Average
6. `calculate_tema(prices, period=14)` - Triple Exponential Moving Average
7. `calculate_hma(prices, period=14)` - Hull Moving Average

### Momentum Indicators (8 indicators)

1. `calculate_rsi(prices, period=14)` - Relative Strength Index (0-100 range)
2. `calculate_roc(prices, period=14)` - Rate of Change (percentage)
3. `calculate_williams_r(high, low, close, period=14)` - Williams %R (-100 to 0)
4. `calculate_stochastic(high, low, close, k_period=14, d_period=3)` - Returns dict: `{k, d}`
5. `calculate_aroon(high, low, period=14)` - Returns dict: `{aroon_up, aroon_down}`
6. `calculate_cci(high, low, close, period=20)` - Commodity Channel Index
7. `calculate_macd(prices, fast=12, slow=26, signal=9)` - Returns dict: `{macd, signal, histogram}`
8. `calculate_tsi(prices, long=25, short=13, signal=7)` - Returns dict: `{tsi, signal}`

### Volatility Indicators (5 indicators)

1. `calculate_atr(high, low, close, period=14)` - Average True Range
2. `calculate_bollinger_bands(prices, period=20, std_dev=2.0)` - Returns dict: `{middle, upper, lower}`
3. `calculate_keltner_channels(high, low, close, ema=20, atr=10, mult=2.0)` - Returns dict: `{middle, upper, lower}`
4. `calculate_donchian_channels(high, low, period=20)` - Returns dict: `{middle, upper, lower}`
5. `calculate_elder_ray(high, low, close, ema_period=13)` - Returns dict: `{bull_power, bear_power}`

### Volume Indicators (4 indicators)

1. `calculate_obv(close, volume)` - On-Balance Volume
2. `calculate_vwap(high, low, close, volume)` - Volume Weighted Average Price
3. `calculate_cmf(high, low, close, volume, period=20)` - Chaikin Money Flow (-1 to 1 range)
4. `calculate_volume_profile(high, low, close, volume, num_bins=20)` - Volume distribution histogram

## Python Usage Examples

### Single Output Indicator

```python
import numpy as np
import kimsfinance_core

prices = np.array([100.0, 102.0, 101.0, 105.0, 103.0, 107.0])
rsi = kimsfinance_core.calculate_rsi(prices, period=14)
print(rsi)  # NumPy array with NaN for warmup period
```

### Multi-Output Indicator (Dictionary)

```python
import numpy as np
import kimsfinance_core

prices = np.array([...])  # Your price data
result = kimsfinance_core.calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)

macd_line = result['macd']
signal_line = result['signal']
histogram = result['histogram']
```

### OHLCV Indicators

```python
import numpy as np
import kimsfinance_core

high = np.array([...])
low = np.array([...])
close = np.array([...])
volume = np.array([...])

atr = kimsfinance_core.calculate_atr(high, low, close, period=14)
vwap = kimsfinance_core.calculate_vwap(high, low, close, volume)
```

## Performance Characteristics

- **3-8x faster** than pandas/NumPy implementations
- **SIMD-optimized** for x86_64 AVX2
- **Parallel processing** for datasets >500-5000 rows (varies by indicator)
- **Zero-allocation** hot paths for maximum performance
- **NaN handling** for warmup periods (first N values based on period)

## Return Value Patterns

### Single Output
- Returns: `np.ndarray` of float64
- Length: Same as input
- Warmup: First `period` values are NaN

### Multi-Output (Dictionary)
- Returns: `dict` with NumPy arrays
- Keys vary by indicator (see list above)
- All arrays have same length as input

## Error Handling

Python exceptions are raised for:
- `ValueError` - Invalid parameters (e.g., period=0, negative values)
- `RuntimeError` - Computation errors (e.g., insufficient data, array length mismatch)

Example:
```python
try:
    rsi = kimsfinance_core.calculate_rsi(prices, period=0)
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

## Building from Source

### Development Build (Fast iteration)
```bash
cd rust
maturin develop
```

### Release Build (Optimized)
```bash
cd rust
maturin develop --release
```

### Production Wheel
```bash
cd rust
maturin build --release
# Wheel created in rust/target/wheels/
pip install target/wheels/kimsfinance_core-0.1.0-*.whl
```

## Testing

### Run Rust unit tests
```bash
cd rust
cargo test
```

### Test Python bindings
```bash
python test_rust_bindings.py
```

Expected output:
```
SUMMARY: 24/24 indicators passed
✓ All 24 technical indicators working correctly!
```

## Module Metadata

```python
import kimsfinance_core

print(kimsfinance_core.__version__)  # "0.1.0"
print(kimsfinance_core.__doc__)      # Module documentation
```

## Implementation Details

### Pattern Used: PyO3 0.27.1

- **Input**: `PyReadonlyArray1<'_, f64>` for zero-copy NumPy array access
- **Output**: `PyArray1<f64>` or `PyDict` for multi-output
- **Signatures**: `#[pyo3(signature = (...))]` for default parameters
- **Error Mapping**: Rust `IndicatorError` → Python exceptions

### Zero-Copy FFI

Input arrays are passed as zero-copy views using `as_array()`:
```rust
let prices_view = prices.as_array();  // No allocation
```

Output arrays use `into_pyarray(py)`:
```rust
Ok(result.into_pyarray(py))  // Transfers ownership to Python
```

### Multi-Output Pattern

Dictionary creation for indicators with multiple outputs:
```rust
let dict = PyDict::new(py);
dict.set_item("macd", output.primary.into_pyarray(py))?;
dict.set_item("signal", output.secondary[0].clone().into_pyarray(py))?;
dict.set_item("histogram", output.secondary[1].clone().into_pyarray(py))?;
Ok(dict)
```

## Crate Structure

```
rust/
├── src/
│   ├── lib.rs              # Python bindings (THIS FILE - 24 indicators)
│   ├── coordinates.rs      # Coordinate calculation
│   ├── types.rs           # Common types
│   ├── indicators/
│   │   ├── mod.rs
│   │   ├── core.rs        # Traits (Indicator, MultiOutputIndicator)
│   │   ├── utils.rs       # Shared utilities (SMA, EMA, etc.)
│   │   ├── moving_averages.rs  # 7 indicators
│   │   ├── momentum.rs    # 8 indicators
│   │   ├── volatility.rs  # 5 indicators
│   │   ├── volume.rs      # 4 indicators
│   │   └── trend.rs       # Additional indicators
│   └── batch.rs           # Batch processing
├── Cargo.toml
└── benches/               # Criterion benchmarks

```

## Known Limitations

1. **Warmup Period**: First `N` values are NaN based on indicator period
   - RSI(14) → First 15 values NaN
   - MACD(12,26,9) → First 35 values NaN

2. **Multi-Smoothing**: Indicators with double/triple smoothing need more data
   - DEMA, TEMA require ~2-3x the period for valid results
   - TSI requires long_period + short_period + signal_period warmup

3. **Aroon Oscillator**: Not included in Rust output
   - Calculate manually: `aroon_up - aroon_down`

4. **Clippy Warnings**: 4 warnings about missing `Default` implementations
   - Does not affect functionality
   - Can be fixed by adding `#[derive(Default)]` or impl blocks

## Future Enhancements

- [ ] Batch API for processing multiple indicators in one FFI call
- [ ] Async support for parallel indicator calculation
- [ ] Custom allocators for large datasets
- [ ] SIMD optimizations for ARM NEON
- [ ] GPU acceleration via CuPy/CUDA

## Version History

- **v0.1.0** (2025-10-25): Initial release
  - 24 technical indicators
  - Coordinate calculations
  - PyO3 0.27.1 bindings
  - Edition 2024 support
  - Rust 1.90.0 compatibility

## License

Same as parent project (kimsfinance)

## Contributing

When adding new indicators:

1. Implement in `rust/src/indicators/`
2. Add Python binding in `rust/src/lib.rs`:
   ```rust
   #[pyfunction]
   #[pyo3(signature = (prices, period = 14))]
   fn calculate_my_indicator<'py>(
       py: Python<'py>,
       prices: PyReadonlyArray1<'_, f64>,
       period: usize,
   ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
       // Implementation
   }
   ```
3. Register in `kimsfinance_core` module
4. Add test case to `test_rust_bindings.py`
5. Update this documentation

## Support

For issues, feature requests, or questions:
- Check existing tests in `rust/src/indicators/*/tests`
- Run benchmarks: `cd rust && cargo bench`
- Profile with: `py-spy record python your_script.py`

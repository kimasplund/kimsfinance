# Rust vs Python/NumPy Indicator Benchmark

## Overview

This benchmark comprehensively compares Rust-accelerated indicators against Python/NumPy implementations across **24 individual indicators + 1 batch API** to determine optimal usage patterns.

**File**: `benchmark_indicators_rust.py`

## Prerequisites

### 1. Build Rust Extension

```bash
cd /home/kim-asplund/projects/kimsfinance/rust
maturin develop --release
```

This compiles the Rust extension with full optimizations and installs it as `kimsfinance_core`.

### 2. Install Dependencies

```bash
pip install numpy scipy
```

## Running the Benchmark

```bash
cd /home/kim-asplund/projects/kimsfinance
python benchmarks/benchmark_indicators_rust.py
```

**Duration**: ~10-20 minutes (depending on CPU)

## What It Tests

### Indicator Categories (24 indicators)

1. **Moving Averages (7)**
   - SMA, EMA, WMA, VWMA
   - DEMA, TEMA, HMA

2. **Momentum (8)**
   - RSI, ROC, MACD
   - Williams %R, Stochastic, Aroon
   - CCI, TSI

3. **Volatility (5)**
   - ATR, Bollinger Bands
   - Keltner Channels, Donchian Channels
   - Elder Ray

4. **Volume (3)**
   - OBV, VWAP, CMF

5. **Batch API (1)**
   - 10 indicators calculated in a single FFI call

### Dataset Sizes

- **100 candles**: Rust should dominate (3-5x speedup)
- **1,000 candles**: Rust still faster (2-3x speedup)
- **10,000 candles**: Crossover point (FFI overhead emerges)
- **100,000 candles**: Python/NumPy wins (FFI overhead dominates)

### Statistical Methodology

- **Iterations**: 500 (small), 100 (medium), 50 (large)
- **Warmup**: 10 iterations for Rust, 5 for Python
- **Metric**: Median timing (robust to outliers)
- **Significance**: Mann-Whitney U test (non-parametric)
- **Confidence**: 95% threshold for "significant difference"

## Expected Results

### Hypothesis

```
Dataset Size    Rust Speedup    Winner        Reason
------------    ------------    ------        ------
100 candles     3-5x            Rust ✅       FFI overhead negligible
1,000 candles   2-3x            Rust ✅       FFI overhead low
10,000 candles  0.8-1.2x        Mixed ⚖️      Crossover threshold
100,000 candles 0.67-0.93x      Python ❌     FFI overhead dominates
```

### Batch API Advantage

Batch API minimizes FFI overhead by:
1. Crossing FFI boundary once (not 10 times)
2. Reusing OHLCV data in Rust (no repeated copies)
3. Parallel indicator calculation

**Expected**: Batch API extends Rust viability to ~10K-50K candles.

## Output Format

```
INDICATOR BENCHMARK RESULTS (Rust vs NumPy)
============================================

Dataset: 100 candles
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MOVING AVERAGES (7 indicators)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SMA                       | NumPy:   0.045ms | Rust:   0.012ms | Speedup: ✅ 3.75x    | Confidence: 🟢 98.5%
  EMA                       | NumPy:   0.052ms | Rust:   0.014ms | Speedup: ✅ 3.71x    | Confidence: 🟢 97.2%
  WMA                       | NumPy:   0.068ms | Rust:   0.019ms | Speedup: ✅ 3.58x    | Confidence: 🟢 96.8%
  ...

Dataset: 10,000 candles
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RSI                       | NumPy:   2.450ms | Rust:   2.890ms | Speedup: ❌ 0.85x    | Confidence: 🟢 95.1%
  ...

COMPREHENSIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📏 Dataset Size: 100 candles
   Rust wins: 24/25 indicators
   Python wins: 1/25 indicators
   Average speedup: 3.42x
   Median speedup: 3.38x
   ⚡ Fastest Rust gain: SMA (4.2x)
   🐌 Slowest Rust gain: MACD (2.1x)

RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 Crossover Threshold: ~8,500 candles
   ✅ Use Rust for datasets < 8,500 candles
   ✅ Use Python/NumPy for datasets >= 8,500 candles

💡 Consider Batch API for >1,000 candles to minimize FFI overhead
   (see benchmark results above)
```

## Interpreting Results

### Speedup Indicators

- ✅ **Green (>= 1.0x)**: Rust is faster
- ❌ **Red (< 1.0x)**: Python/NumPy is faster

### Confidence Indicators

- 🟢 **High (>= 95%)**: Statistically significant difference
- 🟡 **Medium (80-95%)**: Likely significant
- 🔴 **Low (< 80%)**: Difference may be noise

### Key Metrics

1. **Median Time**: Robust to outliers (better than mean)
2. **Speedup**: Python time / Rust time (>1.0 = Rust wins)
3. **Confidence**: 1 - p_value (from Mann-Whitney U test)

## Troubleshooting

### Rust Extension Not Found

```bash
❌ ERROR: Rust extension 'kimsfinance_core' not available
Build with: cd rust && maturin develop --release
```

**Solution**:
```bash
cd /home/kim-asplund/projects/kimsfinance/rust
maturin develop --release
```

### Import Errors

```python
ImportError: cannot import name 'calculate_xxx' from 'kimsfinance.ops.indicators'
```

**Solution**: Check that indicator is implemented in both Python and Rust. Not all indicators may be available in both implementations yet.

### Performance Doesn't Match Expected

**Possible causes**:
1. **Cold cache**: Run benchmark multiple times
2. **CPU throttling**: Check `cpupower frequency-info`
3. **Background processes**: Close other applications
4. **Debug build**: Ensure `--release` flag was used

## Advanced Usage

### Custom Dataset Sizes

Edit `benchmark_indicators_rust.py`:

```python
# Line ~768
sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
```

### Custom Iterations

Edit `benchmark_indicators_rust.py`:

```python
# Line ~596-600
if size <= 1000:
    n_iterations = 1000  # Increase for more precision
elif size <= 10000:
    n_iterations = 500
else:
    n_iterations = 100
```

### Filtering Indicators

Comment out unwanted benchmarks in `run_benchmarks_for_size()`:

```python
# results.append(benchmark_sma(data, size))  # Skip SMA
# results.append(benchmark_ema(data, size))  # Skip EMA
results.append(benchmark_rsi(data, size))    # Keep RSI
```

## Integration with kimsfinance

### Automatic Engine Selection

Based on benchmark results, update `kimsfinance/core/engine.py`:

```python
RUST_CROSSOVER_THRESHOLDS = {
    "sma": 8500,
    "ema": 9000,
    "rsi": 7500,
    "atr": 8000,
    # ... based on benchmark results
}

def select_engine_for_indicator(indicator_name: str, data_size: int) -> str:
    threshold = RUST_CROSSOVER_THRESHOLDS.get(indicator_name, 10000)
    return "rust" if data_size < threshold else "python"
```

### Batch API Integration

For calculating multiple indicators:

```python
from kimsfinance.ops.batch import calculate_indicators_batch

# Automatically uses Rust batch API if available and beneficial
results = calculate_indicators_batch(
    high, low, open, close, volume,
    indicators=["rsi", "macd", "atr", "bollinger"]
)
```

## Performance Targets

### Moving Averages

| Indicator | 100 candles | 1K candles | 10K candles |
|-----------|-------------|------------|-------------|
| SMA       | 3.5-4.5x    | 2.5-3.5x   | 0.8-1.2x    |
| EMA       | 3.0-4.0x    | 2.0-3.0x   | 0.7-1.1x    |
| WMA       | 3.5-4.5x    | 2.5-3.5x   | 0.8-1.2x    |

### Momentum

| Indicator | 100 candles | 1K candles | 10K candles |
|-----------|-------------|------------|-------------|
| RSI       | 3.0-4.0x    | 2.0-3.0x   | 0.7-1.0x    |
| MACD      | 2.5-3.5x    | 1.8-2.5x   | 0.6-0.9x    |
| Stochastic| 3.5-4.5x    | 2.5-3.5x   | 0.8-1.1x    |

### Batch API

| Dataset Size | Individual Calls | Batch API | Speedup |
|--------------|------------------|-----------|---------|
| 100 candles  | 1.2ms total      | 0.4ms     | 3.0x    |
| 1K candles   | 8.5ms total      | 2.8ms     | 3.0x    |
| 10K candles  | 65ms total       | 25ms      | 2.6x    |

## Files

- `benchmark_indicators_rust.py` - Main benchmark script
- `README_RUST_BENCHMARK.md` - This file
- `BENCHMARK_RESULTS_RUST.md` - Actual results (generated after run)

## Contributing

To add new indicator benchmarks:

1. Implement indicator in Rust (`rust/src/indicators/`)
2. Add Python binding in `rust/src/lib.rs`
3. Add benchmark function in `benchmark_indicators_rust.py`
4. Add to main runner in `run_benchmarks_for_size()`
5. Run and validate results

## References

- **Rust Implementation**: `/home/kim-asplund/projects/kimsfinance/rust/`
- **Python Implementation**: `/home/kim-asplund/projects/kimsfinance/kimsfinance/ops/indicators/`
- **Engine Manager**: `/home/kim-asplund/projects/kimsfinance/kimsfinance/core/engine.py`
- **kimsfinance Docs**: `/home/kim-asplund/projects/kimsfinance/docs/`

---

**Last Updated**: 2025-10-25
**Benchmark Version**: v1.0
**Python Version**: 3.13+
**Rust Version**: 1.90+

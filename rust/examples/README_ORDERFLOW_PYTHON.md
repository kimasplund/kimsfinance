# Python Orderflow API Examples

This directory contains comprehensive examples for using the GPU-accelerated orderflow Python bindings in `kimsfinance_core`.

## Overview

The orderflow module provides:

- **GPU-Accelerated Processing**: 500M-1B features/sec, 3-4B signals/sec
- **6 Orderflow Features**: Imbalance, volume delta, trade intensity, price velocity, volume velocity, cumulative delta
- **Multi-Strategy Execution**: Process 10-20 strategies in parallel with a single GPU kernel
- **INT8 Quantization**: 8x memory compression for features
- **Fused Kernel Architecture**: Eliminates 48-60MB of intermediate memory transfers

## Quick Start

### Prerequisites

1. **Build the Rust extension with GPU support:**
   ```bash
   cargo build --release --features gpu,python
   ```

2. **Set up Python path:**
   ```bash
   export PYTHONPATH=$(pwd)/target/release:$PYTHONPATH
   ```

3. **Test the import:**
   ```bash
   python3 -c "import kimsfinance_core; print('✅ Import successful')"
   ```

### Running the Examples

```bash
cd examples/
./python_orderflow_example.py
```

This will run all 6 examples demonstrating different use cases.

## Example Breakdown

### Example 1: Basic Usage - Single Strategy

Demonstrates the fundamental workflow:
1. Check GPU availability
2. Initialize orderflow processor
3. Generate synthetic tick data
4. Configure a single strategy (momentum)
5. Process the batch
6. Analyze signals (buy/sell/hold)

**Key Concepts:**
- `OrderflowProcessor()` - Main GPU processing class
- `StrategyConfig.momentum()` - Predefined strategy
- `process_batch()` - Core processing method
- Signal interpretation: 1=Buy, -1=Sell, 0=Hold

### Example 2: Multiple Strategies in Parallel

Shows how to process multiple strategies simultaneously:
- 5 different strategies (momentum, mean reversion, breakout, scalping, trend following)
- Parallel GPU execution with single data transfer
- Performance comparison between strategies
- Signal distribution analysis

**Performance:**
- 100K ticks × 5 strategies: ~50-100ms
- Throughput: 3-5B signals/sec

### Example 3: Feature Calibration

Demonstrates dynamic range calibration for custom strategies:
1. Generate calibration data sample
2. Run `calibrate_ranges()` to determine feature min/max
3. Create custom strategy with calibrated ranges
4. Process with optimized quantization

**6 Features Calibrated:**
1. Buy/Sell Imbalance (0.0-1.0)
2. Volume Delta (buy - sell)
3. Trade Intensity (trades per window)
4. Price Velocity (rate of price change)
5. Volume Velocity (rate of volume change)
6. Cumulative Volume Delta (running sum)

### Example 4: CPU Fallback

Shows graceful degradation when GPU is unavailable:
- Check GPU availability with `orderflow_gpu_available()`
- Handle `RuntimeError` on processor initialization
- Suggest CPU-based alternatives
- Demonstrate error handling best practices

### Example 5: Integration with Backtesting

Complete workflow from signal generation to backtest execution:
1. Generate realistic market data
2. Create orderflow signals with multiple strategies
3. Implement simple backtest logic
4. Calculate performance metrics (return, drawdown, trades)

**Backtest Metrics:**
- Total Return
- Max Drawdown
- Number of Trades
- Equity Curve

### Example 6: Performance Benchmark

Comprehensive performance testing across different data sizes:
- Tests: 1K, 10K, 100K, 1M ticks
- 10 strategies in parallel
- Measures: Latency, throughput, signals/sec
- Includes warmup runs for accurate timing

**Expected Performance (RTX 3500 Ada):**
- 1M ticks × 10 strategies: ~150-200ms
- Throughput: 5-7M ticks/sec
- Signal generation: 3-4B signals/sec

## API Reference

### Core Classes

#### `OrderflowProcessor`

GPU-accelerated orderflow processor.

```python
processor = kimsfinance_core.OrderflowProcessor()
```

**Methods:**

- `is_gpu_available() -> bool` - Check if GPU is operational
- `calibrate_ranges(...) -> [f32; 12]` - Calibrate feature ranges
- `process_batch(...) -> OrderflowResult` - Process tick data with strategies

#### `StrategyConfig`

Configuration for an orderflow strategy.

**Predefined Strategies:**

```python
# Momentum: Buy when buy/sell imbalance > 0.6 and volume delta > 1000
momentum = StrategyConfig.momentum()

# Mean Reversion: Buy when imbalance < 0.4 and volume delta < -1000
mean_rev = StrategyConfig.mean_reversion()

# Breakout: Buy when trade intensity > 100 and price velocity > 0.001
breakout = StrategyConfig.breakout()

# Scalping: Buy when imbalance > 0.55 and abs(volume_delta) < 500
scalping = StrategyConfig.scalping()

# Trend Following: Buy when volume delta > 5000 and price velocity > 0.002
trend = StrategyConfig.trend_following()
```

**Custom Strategy:**

```python
custom = StrategyConfig(
    strategy_type="momentum",
    feature_mins=[0.0, -10000.0, 0.0, -1.0, -1.0, 0.0],
    feature_maxs=[1.0, 10000.0, 1000.0, 1.0, 1.0, 10000.0]
)
```

**Properties:**
- `.strategy_type` - Strategy type string
- `.feature_mins` - Minimum values for quantization
- `.feature_maxs` - Maximum values for quantization

#### `OrderflowResult`

Result from batch processing.

```python
result = processor.process_batch(...)
```

**Properties:**
- `.signals` - NumPy array `[num_strategies, num_ticks]` (int8: -1=Sell, 0=Hold, 1=Buy)
- `.features` - NumPy array `[num_strategies, num_ticks * 6]` (int8: 0-255 quantized)
- `.num_strategies` - Number of strategies processed
- `.num_ticks` - Number of ticks processed

**Methods:**
- `.to_dict()` - Convert to Python dictionary

### Helper Functions

```python
# Check GPU availability
if kimsfinance_core.orderflow_gpu_available():
    processor = kimsfinance_core.OrderflowProcessor()
```

## Data Format

### Input Data (NumPy Arrays)

All arrays must be the same length `N`:

```python
timestamps: np.ndarray[np.int64]     # Unix timestamps in milliseconds
close_prices: np.ndarray[np.float32] # Close prices
volumes: np.ndarray[np.float32]      # Total trade volumes
buy_volumes: np.ndarray[np.float32]  # Buy-side volumes (taker bought)
sell_volumes: np.ndarray[np.float32] # Sell-side volumes (taker sold)
```

### Output Data

#### Signals Array

Shape: `[num_strategies, num_ticks]`
Type: `int8`

Values:
- `1` - Buy signal
- `0` - Hold signal
- `-1` - Sell signal

#### Features Array

Shape: `[num_strategies, num_ticks * 6]`
Type: `int8` (quantized to 0-255)

Features per tick (in order):
1. Buy/Sell Imbalance (0.0-1.0)
2. Volume Delta (buy - sell)
3. Trade Intensity (trades/window)
4. Price Velocity (Δprice/Δtime)
5. Volume Velocity (Δvolume/Δtime)
6. Cumulative Volume Delta (running sum)

To reshape for easier access:
```python
features_reshaped = result.features[strategy_idx].reshape(-1, 6)
# Now features_reshaped[tick_idx, feature_idx]
```

## Performance Optimization Tips

### 1. Batch Size Selection

- **Optimal**: 100K-1M ticks per batch
- **Too small** (<10K): GPU underutilized
- **Too large** (>10M): May exceed GPU memory

### 2. Strategy Count

- **Optimal**: 10-20 strategies
- **Sweet spot**: Maximizes GPU occupancy
- **More strategies**: Minimal additional cost (parallel execution)

### 3. Calibration

- Use `calibrate_ranges()` for optimal quantization
- Run on representative data sample (50K-100K ticks)
- Reuse calibration results for similar market conditions

### 4. Memory Management

- GPU memory usage: ~1MB per 100K ticks
- Features are INT8 quantized: 8x compression vs FP64
- Fused kernel: Avoids 48-60MB intermediate transfers

## Error Handling

### Common Issues

#### GPU Not Available

```python
if not kimsfinance_core.orderflow_gpu_available():
    print("⚠️ GPU not available")
    # Fallback to CPU implementation or exit
    sys.exit(1)
```

#### Mismatched Array Lengths

```python
try:
    result = processor.process_batch(timestamps, prices, volumes, buy_vols, sell_vols, strategies)
except RuntimeError as e:
    if "same length" in str(e):
        print("❌ Input arrays must have same length")
```

#### Empty Input

```python
if len(timestamps) == 0:
    raise ValueError("Input arrays cannot be empty")
```

#### No Strategies

```python
if len(strategies) == 0:
    raise ValueError("Must provide at least one strategy")
```

## Integration Examples

### With Tick Aggregation

```python
from kimsfinance_core import GpuTickAggregator, OrderflowProcessor

# 1. Aggregate tick data
aggregator = GpuTickAggregator()
candles = aggregator.aggregate(timestamps, prices, volumes, sides, timeframe_ms=300000)

# 2. Process orderflow
processor = OrderflowProcessor()
result = processor.process_batch(
    candles.timestamps,
    candles.close,
    candles.volume,
    buy_volumes,  # Compute from sides
    sell_volumes,
    strategies
)
```

### With Backtesting

```python
from kimsfinance_core import TickBacktestEngine, TickBacktestConfig, OrderflowProcessor

# 1. Generate signals
processor = OrderflowProcessor()
result = processor.process_batch(timestamps, prices, volumes, buy_vols, sell_vols, strategies)

# 2. Run backtest
config = TickBacktestConfig(initial_capital=10000.0)
engine = TickBacktestEngine(config)

for strategy_idx in range(result.num_strategies):
    signals = result.signals[strategy_idx]
    backtest_result = engine.run(timestamps, prices, volumes, is_buyer_maker, signals, timeframe_ms=300000)
    print(f"Strategy {strategy_idx}: Return={backtest_result.total_return:.2f}%")
```

### With Pandas

```python
import pandas as pd
import numpy as np
from kimsfinance_core import OrderflowProcessor

# Load data from DataFrame
df = pd.read_csv("tick_data.csv")

# Convert to NumPy arrays
timestamps = df['timestamp'].values.astype(np.int64)
prices = df['price'].values.astype(np.float32)
volumes = df['volume'].values.astype(np.float32)

# Compute buy/sell volumes from side
buy_volumes = np.where(df['side'] == 'buy', volumes, 0).astype(np.float32)
sell_volumes = np.where(df['side'] == 'sell', volumes, 0).astype(np.float32)

# Process
processor = OrderflowProcessor()
result = processor.process_batch(timestamps, prices, volumes, buy_volumes, sell_volumes, strategies)

# Add signals back to DataFrame
df['signal_momentum'] = result.signals[0]
```

## Troubleshooting

### Import Error

```
ImportError: No module named 'kimsfinance_core'
```

**Solution:**
```bash
export PYTHONPATH=/path/to/kimsfinance/rust/target/release:$PYTHONPATH
```

### GPU Initialization Failed

```
RuntimeError: GPU initialization failed: ...
```

**Possible causes:**
1. No CUDA-capable GPU
2. CUDA driver not installed
3. Incompatible CUDA version
4. GPU already in use

**Check:**
```bash
nvidia-smi  # Verify GPU is visible
```

### Performance Lower Than Expected

**Checklist:**
1. GPU is not in power-saving mode
2. Data is pre-allocated (avoid Python list to NumPy conversions in hot loop)
3. Batch size is adequate (>10K ticks)
4. Multiple strategies used (better GPU utilization)
5. No other GPU processes running

## Benchmarking

Run the benchmark example:

```bash
cd examples/
python3 python_orderflow_example.py
# Look for "EXAMPLE 6: Performance Benchmark"
```

Profile with Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx --output=orderflow_profile python3 python_orderflow_example.py
nsys-ui orderflow_profile.nsys-rep
```

## Further Reading

- **Rust Source**: `src/orderflow_py.rs` - Python bindings implementation
- **Type Stubs**: `kimsfinance_core.pyi` - Full API type signatures
- **GPU Kernels**: `src/gpu/orderflow_batch.rs` - Rust GPU implementation
- **CUDA Kernels**: `src/gpu/kernels/orderflow_signals_batch.cu` - Low-level CUDA code
- **Documentation**: `docs/ORDERFLOW_SIGNALS_GPU.md` - Architecture and design

## Contributing

When adding new examples:

1. Follow the existing structure (numbered examples)
2. Include comprehensive comments
3. Generate synthetic data (don't require external files)
4. Handle GPU unavailability gracefully
5. Show realistic use cases
6. Measure and report performance

## License

This example code is part of the kimsfinance project and follows the same license as the parent project.

---

**Last Updated**: November 2025
**Author**: kimsfinance development team
**Version**: 1.0.0

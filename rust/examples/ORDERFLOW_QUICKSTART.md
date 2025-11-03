# Orderflow Python API - Quick Start Guide

## 30-Second Setup

```bash
# 1. Build with GPU support
cargo build --release --features gpu,python

# 2. Set Python path
export PYTHONPATH=$(pwd)/target/release:$PYTHONPATH

# 3. Run example
python3 examples/python_orderflow_example.py
```

## Minimal Working Example

```python
import numpy as np
import kimsfinance_core

# Initialize processor
processor = kimsfinance_core.OrderflowProcessor()

# Prepare data (10K ticks)
n = 10_000
timestamps = np.arange(n, dtype=np.int64) * 100
close_prices = (50000.0 + np.random.randn(n).cumsum() * 10).astype(np.float32)
volumes = np.random.exponential(100.0, n).astype(np.float32)
buy_volumes = (volumes * 0.5).astype(np.float32)
sell_volumes = (volumes - buy_volumes).astype(np.float32)

# Configure strategy
strategies = [kimsfinance_core.StrategyConfig.momentum()]

# Process
result = processor.process_batch(
    timestamps, close_prices, volumes, buy_volumes, sell_volumes, strategies
)

# Extract signals
signals = result.signals[0]  # -1=Sell, 0=Hold, 1=Buy
print(f"Buy signals: {np.sum(signals == 1)}")
print(f"Sell signals: {np.sum(signals == -1)}")
```

## API Cheat Sheet

### Classes

```python
# Processor
processor = kimsfinance_core.OrderflowProcessor()
processor.is_gpu_available()  # Check GPU
processor.calibrate_ranges(...)  # Calibrate feature ranges
result = processor.process_batch(...)  # Main processing

# Strategy Config
momentum = kimsfinance_core.StrategyConfig.momentum()
mean_rev = kimsfinance_core.StrategyConfig.mean_reversion()
breakout = kimsfinance_core.StrategyConfig.breakout()
scalping = kimsfinance_core.StrategyConfig.scalping()
trend = kimsfinance_core.StrategyConfig.trend_following()

# Custom strategy
custom = kimsfinance_core.StrategyConfig(
    "momentum",
    feature_mins=[0.0, -10000, 0.0, -1.0, -1.0, 0.0],
    feature_maxs=[1.0, 10000, 1000, 1.0, 1.0, 10000]
)

# Result
result.signals  # [num_strategies, num_ticks] int8
result.features  # [num_strategies, num_ticks*6] int8
result.num_strategies  # int
result.num_ticks  # int
result.to_dict()  # dict
```

### Input Data Format

All arrays must have same length:

```python
timestamps: np.ndarray[np.int64]     # Milliseconds since epoch
close_prices: np.ndarray[np.float32] # Prices
volumes: np.ndarray[np.float32]      # Total volumes
buy_volumes: np.ndarray[np.float32]  # Buy side
sell_volumes: np.ndarray[np.float32] # Sell side
```

### Signal Values

```python
1   # Buy signal
0   # Hold signal
-1  # Sell signal
```

### Features (6 per tick)

1. Buy/Sell Imbalance (0.0-1.0)
2. Volume Delta (buy - sell)
3. Trade Intensity (trades/window)
4. Price Velocity (Δprice/Δtime)
5. Volume Velocity (Δvolume/Δtime)
6. Cumulative Volume Delta (running sum)

## Common Patterns

### Multiple Strategies

```python
strategies = [
    kimsfinance_core.StrategyConfig.momentum(),
    kimsfinance_core.StrategyConfig.mean_reversion(),
    kimsfinance_core.StrategyConfig.breakout(),
]

result = processor.process_batch(ts, prices, vols, buy_vols, sell_vols, strategies)

# Access each strategy's signals
for i, strategy in enumerate(strategies):
    signals = result.signals[i]
    print(f"Strategy {i}: {np.sum(signals == 1)} buys, {np.sum(signals == -1)} sells")
```

### Error Handling

```python
# Check GPU
if not kimsfinance_core.orderflow_gpu_available():
    print("GPU not available!")
    sys.exit(1)

# Handle errors
try:
    processor = kimsfinance_core.OrderflowProcessor()
except RuntimeError as e:
    print(f"GPU init failed: {e}")
    # Fallback to CPU
```

### With Pandas

```python
import pandas as pd

df = pd.read_csv("ticks.csv")
timestamps = df['timestamp'].values.astype(np.int64)
prices = df['price'].values.astype(np.float32)
# ... process ...
df['signal'] = result.signals[0]
```

### Simple Backtest

```python
signals = result.signals[0]
position = 0
cash = 10000.0

for i in range(len(signals)):
    if signals[i] == 1 and position == 0:  # Buy
        position = cash / prices[i]
        cash = 0
    elif signals[i] == -1 and position > 0:  # Sell
        cash = position * prices[i]
        position = 0

final_equity = cash + position * prices[-1]
print(f"Return: {(final_equity / 10000 - 1) * 100:.2f}%")
```

## Performance Tips

✅ **DO:**
- Use 10-20 strategies (better GPU utilization)
- Batch 100K-1M ticks per call
- Calibrate ranges for optimal quantization
- Warmup GPU before benchmarking

❌ **DON'T:**
- Process <10K ticks (GPU underutilized)
- Convert Python lists to NumPy in hot loop
- Create processor in tight loop
- Mix int32/int64 timestamps

## Troubleshooting

### Import Error
```bash
export PYTHONPATH=/path/to/rust/target/release:$PYTHONPATH
```

### GPU Not Found
```bash
nvidia-smi  # Check GPU
```

### Wrong Array Type
```python
# ❌ Wrong
timestamps = np.arange(n)  # defaults to int32

# ✅ Correct
timestamps = np.arange(n, dtype=np.int64)
```

### Array Length Mismatch
```python
# All must be same length
assert len(timestamps) == len(prices) == len(volumes) == len(buy_vols) == len(sell_vols)
```

## Examples

| Example | Purpose | Runtime |
|---------|---------|---------|
| `example_1_basic_usage()` | Single strategy | ~10ms |
| `example_2_multiple_strategies()` | 5 strategies × 100K ticks | ~50ms |
| `example_3_calibration()` | Feature range calibration | ~30ms |
| `example_4_cpu_fallback()` | Error handling | N/A |
| `example_5_backtesting_integration()` | Full backtest workflow | ~50ms |
| `example_6_performance_benchmark()` | Scaling test | Varies |

## Next Steps

1. ✅ Run: `python3 examples/python_orderflow_example.py`
2. 📖 Read: `examples/README_ORDERFLOW_PYTHON.md`
3. 🔍 Study: `src/orderflow_py.rs` (source)
4. 📊 Profile: `nsys profile python3 examples/python_orderflow_example.py`
5. 🚀 Integrate: Add to your trading pipeline

## Support

- **Documentation**: `docs/ORDERFLOW_SIGNALS_GPU.md`
- **Source Code**: `src/orderflow_py.rs`, `src/gpu/orderflow_batch.rs`
- **Type Stubs**: `kimsfinance_core.pyi`
- **Examples**: `examples/python_orderflow_example.py`

---

**Version**: 1.0.0
**Last Updated**: November 2025
**GPU Required**: NVIDIA CUDA 11.0+

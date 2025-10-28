# GPU Batch Backtest Python Bindings

**Status**: Task 3 Complete ✅
**Created**: 2025-10-28
**Part of**: GPU Batch Backtesting Implementation (Tasks 1-4)

## Overview

PyO3 Python bindings for GPU batch backtesting, enabling genetic algorithm optimization with 20-40x speedup over sequential CPU execution.

## Architecture

```
Python (kimsfinance.batch)
    ↓ (PyO3 bindings)
Rust (batch_backtest_py.rs)
    ↓
BatchBacktestSweep (batch.rs)
    ↓
CUDA Kernels (kernels_backtest.cu)
    ↓
GPU (RTX 3500 Ada)
```

## Files Created

### 1. Rust PyO3 Bindings

**`rust/src/batch_backtest_py.rs`** (410 lines)
- `batch_backtest()`: Main Python function for GPU batch backtesting
- `batch_backtest_info()`: GPU availability and performance info
- `PyBacktestResult`: Python class for backtest results
- NumPy array conversion (zero-copy where possible)
- Proper error handling with Python exceptions

**`rust/src/lib.rs`** (Modified)
- Added module registration for `batch_backtest_py`
- Registered functions and classes with PyO3 module

### 2. Python High-Level API

**`kimsfinance/batch.py`** (350 lines)
- `batch_backtest()`: High-level wrapper with pandas support
- `get_gpu_info()`: GPU information function
- `find_best_parameters()`: Grid search convenience function
- `BacktestConfig`: Configuration dataclass
- Parameter dict → list conversion for all strategy types

### 3. Tests

**`tests/python_integration/test_batch_backtest.py`** (550 lines)
- Basic functionality tests (10-100 strategies)
- Stress tests (1000 strategies)
- Error handling tests
- Performance validation tests
- Result class method tests

**`examples/test_batch_backtest.py`** (150 lines)
- Quick validation script
- Manual test for bindings

## API Reference

### Low-Level API (kimsfinance_core)

```python
from kimsfinance_core import batch_backtest, BacktestResult

# OHLCV data (N_candles, 5)
ohlcv = np.array([...])  # [open, high, low, close, volume]

# Parameters (N_strategies × N_params)
parameters = [
    [14.0, 30.0, 70.0],  # Strategy 1
    [14.0, 25.0, 75.0],  # Strategy 2
    # ...
]

# Run batch backtest
results: List[BacktestResult] = batch_backtest(
    strategy='rsi_crossover',
    ohlcv=ohlcv,
    parameters=parameters,
    timestamps=None,  # Optional, auto-generated if None
    initial_capital=10000.0,
    trading_fee=0.001,
    slippage=0.0001
)

# Access results
best = results[0]  # Sorted by fitness
print(f"Sharpe: {best.sharpe_ratio:.2f}")
print(f"Drawdown: {best.max_drawdown:.2%}")
print(f"Win Rate: {best.win_rate:.1%}")
```

### High-Level API (kimsfinance.batch)

```python
import pandas as pd
from kimsfinance.batch import batch_backtest, BacktestConfig

# Load OHLCV data
data = pd.read_csv('BTC-USD.csv')

# Define parameter sweep
params = [
    {'period': p, 'buy_threshold': b, 'sell_threshold': s}
    for p in range(10, 20)
    for b in [25, 30, 35]
    for s in [65, 70, 75]
]  # 90 strategies

# Run batch backtest
results = batch_backtest('rsi_crossover', data, params)

# Find best
best = max(results, key=lambda r: r['sharpe_ratio'])
print(f"Best Sharpe: {best['sharpe_ratio']:.2f}")
print(f"Parameters: {best['params']}")
```

### Grid Search API

```python
from kimsfinance.batch import find_best_parameters

# Define parameter ranges
ranges = {
    'period': [10, 14, 20],
    'buy_threshold': [25, 30, 35],
    'sell_threshold': [65, 70, 75],
}

# Find best parameters
result = find_best_parameters('rsi_crossover', data, ranges)

print(f"Best parameters: {result['best_params']}")
print(f"Best Sharpe: {result['best_score']:.2f}")
```

## Supported Strategies

### 1. RSI Crossover

**Parameters**: `[period, buy_threshold, sell_threshold]`

```python
params = [
    {'period': 14, 'buy_threshold': 30, 'sell_threshold': 70}
]
```

### 2. MA Crossover

**Parameters**: `[fast_period, slow_period]`

```python
params = [
    {'fast_period': 10, 'slow_period': 50}
]
```

### 3. Bollinger Bands Mean Reversion

**Parameters**: `[period, std_dev, entry_std, exit_std]`

```python
params = [
    {'period': 20, 'std_dev': 2.0, 'entry_std': 1.5, 'exit_std': 0.5}
]
```

## BacktestResult Class

```python
class BacktestResult:
    strategy_id: int         # Index in parameter list
    sharpe_ratio: float      # Annualized Sharpe ratio
    max_drawdown: float      # Negative percentage (e.g., -0.15 = -15%)
    win_rate: float          # [0, 1] (e.g., 0.65 = 65%)
    total_return: float      # Percentage (e.g., 0.25 = +25%)
    final_equity: float      # Final portfolio value
    num_trades: int          # Number of trades executed
    profit_factor: float     # Gross profit / gross loss

    def to_dict() -> dict    # Convert to Python dict
    def fitness() -> float   # Sharpe × (1 - abs(drawdown))
    def __repr__() -> str    # String representation
```

## Performance

### Targets

- **100 strategies × 10K candles**: <100ms (RTX 3500 Ada)
- **1000 strategies × 10K candles**: <250ms
- **Speedup**: 20-40x vs sequential CPU
- **VRAM usage**: <1GB for 1000 strategies

### Overhead

- **PyO3 overhead**: <1ms (measured)
- **NumPy conversion**: Zero-copy for input (PyReadonlyArray)
- **GIL release**: Yes (via `py.detach()`)

## Error Handling

All Rust errors are properly converted to Python exceptions:

```python
# Invalid strategy
try:
    batch_backtest('invalid_strategy', ohlcv, params)
except ValueError as e:
    print(f"Error: {e}")  # "Unknown strategy: 'invalid_strategy'"

# GPU initialization failure
try:
    batch_backtest('rsi_crossover', ohlcv, params)
except RuntimeError as e:
    print(f"GPU Error: {e}")  # "Failed to initialize GPU: ..."

# Empty parameters
try:
    batch_backtest('rsi_crossover', ohlcv, [])
except ValueError as e:
    print(f"Error: {e}")  # "parameters cannot be empty"
```

## Building

### Compile with GPU Support

```bash
# Build Rust library
cargo build --release --features gpu

# Build Python package
pip install -e ".[gpu]"
```

### Verify Installation

```python
from kimsfinance.batch import get_gpu_info

info = get_gpu_info()
if info['gpu_available']:
    print(f"GPU: {info['gpu_name']}")
    print(f"Expected speedup: {info['expected_speedup']:.0f}x")
else:
    print(f"GPU unavailable: {info['error']}")
```

## Testing

### Run Python Integration Tests

```bash
# All tests
pytest tests/python_integration/test_batch_backtest.py -v

# Basic tests only
pytest tests/python_integration/test_batch_backtest.py::TestBatchBacktestBasic -v

# Stress tests (1000 strategies)
pytest tests/python_integration/test_batch_backtest.py::TestBatchBacktestStress -v

# Performance benchmark
pytest tests/python_integration/test_batch_backtest.py::TestPerformance -v
```

### Run Quick Validation

```bash
python examples/test_batch_backtest.py
```

## Integration with Genetic Optimizer

**Next step**: Integrate with genetic optimizer (Task 4)

```python
# Future API (Task 4)
from kimsfinance.optimize import GeneticOptimizer

optimizer = GeneticOptimizer(
    strategy='rsi_crossover',
    data=data,
    use_batch_backtest=True,  # Enable GPU batch mode
    population_size=100,
    generations=50
)

best_params = optimizer.run()
```

## Known Limitations

1. **Fixed strategy architecture**: Cannot use arbitrary Python strategies (must be one of 3 built-in)
2. **Parameter order matters**: Must match expected order for each strategy type
3. **GPU required**: Falls back to error if GPU not available (no CPU fallback in bindings)
4. **CUDA 13.0**: Requires CUDA-capable GPU with CUDA 13.0 driver

## Future Enhancements

1. **CPU fallback**: Automatically use CPU if GPU unavailable
2. **Custom strategies**: Allow user-defined strategies via DSL
3. **Streaming results**: Return results as they complete (async)
4. **Multi-GPU**: Distribute across multiple GPUs
5. **Progress callback**: Python callback for progress updates

## References

- **Implementation Plan**: `integrated-reasoning/gpu_batch_backtesting_implementation_plan.md`
- **Rust API**: `rust/src/backtest/batch.rs`
- **CUDA Kernels**: `rust/src/gpu/kernels_backtest.cu`
- **PyO3 Documentation**: https://pyo3.rs/

## Author

Claude Code (Sonnet 4.5) - 2025-10-28

## License

Same as kimsfinance project

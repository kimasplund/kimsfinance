# GPU Batch Backtesting

> **Performance**: [TBD]x faster than sequential backtesting (target: 20-40x)
> **Scale**: Process 100-1000 strategies simultaneously on GPU
> **Use Case**: Genetic algorithm optimization, parameter sweeps, strategy discovery

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Performance](#performance)
6. [Genetic Algorithm Integration](#genetic-algorithm-integration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

---

## Overview

### What is GPU Batch Backtesting?

GPU batch backtesting enables **parallel execution of 100-1000 backtests simultaneously** on NVIDIA GPUs, replacing sequential backtest execution with a single GPU batch call. This is transformative for:

- **Genetic Algorithm Optimization**: Evaluate entire population in one GPU call
- **Parameter Sweeps**: Test thousands of parameter combinations rapidly
- **Strategy Discovery**: Explore large strategy spaces efficiently

### Why 20-40x Speedup?

**Sequential Baseline** (CPU):
```
100 strategies × 10ms each = 1,000ms total
```

**Batch GPU**:
```
100 strategies in single batch = 50ms total
Speedup: 1,000ms / 50ms = 20x
```

**Key Innovation**: Instead of running backtests one-by-one, we:
1. Upload data **once** to GPU (shared OHLCV across all strategies)
2. Execute **all strategies in parallel** on GPU
3. Download results **once** from GPU

This eliminates:
- 99% of CPU-GPU data transfer overhead
- 100% of sequential execution waste
- Kernel launch overhead (single launch instead of N launches)

### When to Use Batch Backtesting

✅ **Use batch backtesting when**:
- Running genetic algorithms (100-1000 strategies per generation)
- Parameter sweeps (testing many parameter combinations)
- Strategy discovery (exploring large search spaces)
- NVIDIA GPU available (RTX 2000 series or newer)

❌ **Don't use batch backtesting when**:
- Running single backtest (use sequential engine)
- No GPU available (will fall back to CPU, slower)
- Extremely complex strategies (may not fit in GPU memory)

---

## Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  GENETIC OPTIMIZER (Python)                  │
│                                                               │
│  Population: [Strategy₁, Strategy₂, ..., Strategy₁₀₀₀]      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              PyO3 Interface (batch_backtest)                 │
│              - Parameter marshalling                         │
│              - NumPy ↔ ndarray conversion                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 BacktestSweep (Rust)                         │
│                 - Builder API                                │
│                 - GPU memory allocation                      │
│                 - Kernel orchestration                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Phase 1:      │   │ Phase 2:      │   │ Phase 3:      │
│ INDICATORS    │──▶│ SIGNALS       │──▶│ EXECUTION     │
│               │   │               │   │               │
│ - RSI         │   │ - Buy/Sell    │   │ - P&L         │
│ - ATR         │   │ - Hold        │   │ - Equity      │
│ - SMA/EMA     │   │ - Position    │   │ - Trades      │
└───────────────┘   └───────────────┘   └───────┬───────┘
                                                 │
                                                 ▼
                                        ┌────────────────┐
                                        │ Phase 4:       │
                                        │ METRICS        │
                                        │                │
                                        │ - Sharpe       │
                                        │ - Drawdown     │
                                        │ - Win Rate     │
                                        └────────┬───────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│            Results: [Metrics₁, Metrics₂, ..., MetricsN]     │
│            - sharpe_ratio                                    │
│            - max_drawdown                                    │
│            - win_rate                                        │
│            - total_return                                    │
│            - profit_factor                                   │
└─────────────────────────────────────────────────────────────┘
```

### 4-Phase GPU Pipeline

#### Phase 1: Indicator Calculation (Fully Parallel)

**Purpose**: Calculate all required indicators for all strategies

**Parallelism**: Strategies × Indicators × Candles

**Memory Layout**: 3D array `[N_strategies × N_indicators × N_candles]`

**Example**: 1000 strategies × 5 indicators × 10,000 candles = 400 MB

**Performance**: Embarrassingly parallel, near-linear scaling

```
Grid:  (N_strategies, N_indicators, (N_candles+255)/256)
Block: (256, 1, 1)

Each thread calculates one indicator value for one candle.
Strategies run independently in parallel.
```

#### Phase 2: Signal Generation (Fully Parallel)

**Purpose**: Generate trading signals (Buy/Sell/Hold) based on indicators

**Parallelism**: Strategies × Candles

**Memory Layout**: 2D array `[N_strategies × N_candles]`

**Example**: 1000 strategies × 10,000 candles = 10 MB (int8)

**Performance**: Fully parallel across strategies and candles

```
Grid:  (N_strategies, (N_candles+255)/256)
Block: (256, 1)

Each thread reads indicator values and applies strategy logic:
  if (rsi < buy_threshold)  → BUY signal
  if (rsi > sell_threshold) → SELL signal
```

#### Phase 3: Position Tracking & P&L (Semi-Parallel)

**Purpose**: Execute backtest sequentially for each strategy

**Parallelism**: Strategies only (sequential within each strategy)

**Memory Layout**: 2D array `[N_strategies × N_candles]` for equity curves

**Example**: 1000 strategies × 10,000 candles = 80 MB

**Key Challenge**: This phase is sequential per strategy but parallel across strategies

```
Grid:  (N_strategies, 1)
Block: (1, 1)  ← One thread per strategy!

Each thread:
  for candle in 0..N_candles {
    signal = signals[strategy_idx][candle]

    // Execute trade logic
    if signal == BUY:
      open_position(close_price, equity)
    elif signal == SELL:
      close_position(close_price, equity)

    // Track equity
    equity_curve[candle] = mark_to_market(equity, position, close_price)
  }

Why this works:
  - GPU has 14,336 CUDA cores (RTX 3500 Ada)
  - 1000 strategies = 1000 independent threads
  - Each thread runs sequentially through its own candles
  - All 1000 threads execute simultaneously in parallel!

Wall time ≈ Time for one strategy (not 1000×)
```

#### Phase 4: Metrics Calculation (Parallel Reduction)

**Purpose**: Calculate performance metrics using parallel reduction

**Parallelism**: Strategies × Reduction Threads

**Memory Layout**: 1D array `[N_strategies]` per metric

**Example**: 1000 strategies × 5 metrics = 40 KB

**Performance**: O(log N) reduction, <1ms for 1000 strategies

```
Grid:  (N_strategies, 1)
Block: (256, 1)  ← 256 threads per strategy for reduction

Sharpe Ratio Calculation (parallel reduction):
  1. Each thread computes local sum of returns
  2. Tree reduction in shared memory (8 steps for 256 threads)
  3. Thread 0 calculates final Sharpe ratio

Max Drawdown (parallel scan):
  Similar parallel reduction pattern

Win Rate (from trades):
  Thread 0 counts wins/losses from trade log
```

### Memory Layout Design

**Why 3D Layout? (Strategy-major, Indicator-major, Candle-major)**

```rust
indicators: [N_strategies][N_indicators][N_candles]
```

**Advantages**:
1. **Coalesced Memory Access**: Sequential candles in strategy execution = consecutive memory reads (fast!)
2. **Cache Efficiency**: All indicators for a strategy fit in L2 cache (32MB on RTX 3500 Ada)
3. **Existing Pattern**: Matches `kernels_3d.rs` parameter sweep architecture (proven)

**VRAM Budget** (1000 strategies × 10K candles):
```
Inputs (shared):
  OHLCV:              5 × 10K × 8 bytes = 400 KB

Per-Strategy Buffers:
  Indicators:    1000 × 5 × 10K × 8  = 400 MB
  Signals:       1000 × 10K × 1      = 10 MB
  Equity Curves: 1000 × 10K × 8      = 80 MB
  Trades:        1000 × 100 × 48     = 4.8 MB
  Metrics:       1000 × 5 × 8        = 40 KB

TOTAL:                                 ≈ 495 MB
Safety Factor (2x):                    ≈ 1 GB

Fits in 12GB VRAM with 11GB headroom ✅
```

---

## Quick Start

### Prerequisites

**Hardware**:
- NVIDIA GPU (RTX 2000 series or newer recommended)
- 4GB+ VRAM (8GB+ for large batches)

**Software**:
- CUDA 12.x or 13.x
- Python 3.13+
- kimsfinance with GPU support

**Installation**:
```bash
# Install kimsfinance with GPU support
pip install kimsfinance[gpu]

# Verify GPU availability
python -c "import kimsfinance; print(kimsfinance.gpu_available())"
# Should print: True
```

### Basic Example (100 RSI Crossover Strategies)

```python
from kimsfinance import batch_backtest
import numpy as np

# Load your OHLCV data
timestamps = np.arange(10000)
open_prices = np.random.randn(10000).cumsum() + 100
high_prices = open_prices + np.abs(np.random.randn(10000))
low_prices = open_prices - np.abs(np.random.randn(10000))
close_prices = open_prices + np.random.randn(10000) * 0.5
volumes = np.random.randint(1000, 10000, 10000)

data = {
    'timestamps': timestamps,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes,
}

# Define 100 RSI crossover parameter sets
# Parameters: [rsi_period, buy_threshold, sell_threshold]
parameters = []
for buy_thresh in range(20, 30):  # 10 values
    for sell_thresh in range(70, 80):  # 10 values
        parameters.append([14.0, float(buy_thresh), float(sell_thresh)])
# Total: 10 × 10 = 100 strategies

# Execute batch backtest (single GPU call!)
results = batch_backtest(
    strategy='rsi_crossover',
    data=data,
    parameters=parameters,
    config={
        'initial_capital': 10_000.0,
        'trading_fee': 0.001,      # 0.1%
        'slippage': 0.0005,         # 0.05%
    }
)

# Analyze results
print(f"Evaluated {len(results)} strategies")

# Find best strategy by Sharpe ratio
best = max(results, key=lambda r: r['sharpe_ratio'])
print(f"\nBest Strategy:")
print(f"  Parameters: {best['parameters']}")
print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {best['max_drawdown']*100:.1f}%")
print(f"  Win Rate: {best['win_rate']*100:.1f}%")
print(f"  Total Return: {best['total_return']*100:.1f}%")
print(f"  Trades: {best['num_trades']}")
```

**Expected Output**:
```
Evaluated 100 strategies

Best Strategy:
  Parameters: [14.0, 25.0, 75.0]
  Sharpe Ratio: 1.85
  Max Drawdown: -12.3%
  Win Rate: 58.2%
  Total Return: 47.5%
  Trades: 87
```

**Performance** (compared to sequential):
```
Sequential: 100 × 10ms = 1,000ms (1 second)
Batch GPU:  [TBD]ms
Speedup:    [TBD]x
```

---

## API Reference

### Python API

#### `batch_backtest()`

Execute batch backtesting for multiple strategies on GPU.

**Signature**:
```python
def batch_backtest(
    strategy: str,
    data: dict,
    parameters: List[List[float]],
    config: Optional[dict] = None,
) -> List[dict]:
    """
    Execute batch backtesting for multiple strategies on GPU.

    Args:
        strategy: Strategy type string
            - 'rsi_crossover': RSI crossover strategy
            - 'ma_crossover': Moving average crossover
            - 'bollinger_mean_reversion': Bollinger Bands mean reversion

        data: Dictionary with OHLCV data
            Required keys:
            - 'timestamps': np.ndarray[int64] - Unix timestamps
            - 'open': np.ndarray[float64] - Open prices
            - 'high': np.ndarray[float64] - High prices
            - 'low': np.ndarray[float64] - Low prices
            - 'close': np.ndarray[float64] - Close prices
            - 'volume': np.ndarray[float64] - Volumes

        parameters: List of parameter lists (N strategies × M params)
            Each inner list contains strategy-specific parameters:

            For 'rsi_crossover': [rsi_period, buy_threshold, sell_threshold]
                - rsi_period: RSI calculation period (typically 14)
                - buy_threshold: Buy when RSI < this value (e.g., 30)
                - sell_threshold: Sell when RSI > this value (e.g., 70)

            For 'ma_crossover': [fast_period, slow_period]
                - fast_period: Fast MA period (e.g., 20)
                - slow_period: Slow MA period (e.g., 50)

            For 'bollinger_mean_reversion': [bb_period, bb_std, entry_std, exit_std]
                - bb_period: Bollinger Band period (e.g., 20)
                - bb_std: Standard deviations for bands (e.g., 2.0)
                - entry_std: Entry threshold (e.g., 2.0)
                - exit_std: Exit threshold (e.g., 0.0)

        config: Optional backtest configuration dictionary
            Optional keys:
            - 'initial_capital': float (default: 10_000.0)
            - 'trading_fee': float (default: 0.001, i.e., 0.1%)
            - 'slippage': float (default: 0.0005, i.e., 0.05%)

    Returns:
        List of dictionaries, one per strategy, with keys:
            - 'sharpe_ratio': float - Annualized Sharpe ratio
            - 'max_drawdown': float - Maximum drawdown (negative, e.g., -0.15 = -15%)
            - 'win_rate': float - Fraction of winning trades (0.0 to 1.0)
            - 'total_return': float - Total return (e.g., 0.50 = 50% gain)
            - 'profit_factor': float - Gross profit / gross loss
            - 'num_trades': int - Number of trades executed
            - 'parameters': List[float] - Original parameters for this strategy

    Raises:
        ValueError: Invalid strategy type, data format, or parameters
        RuntimeError: GPU initialization failed or CUDA error

    Examples:
        >>> # 10 RSI strategies
        >>> params = [[14.0, 20+i, 70+i] for i in range(10)]
        >>> results = batch_backtest('rsi_crossover', ohlcv_data, params)
        >>> best = max(results, key=lambda r: r['sharpe_ratio'])
    """
```

**Example Usage**:

```python
# Example 1: RSI Crossover Sweep
params = [[14.0, buy, sell]
          for buy in range(20, 40, 5)
          for sell in range(60, 80, 5)]
results = batch_backtest('rsi_crossover', data, params)

# Example 2: MA Crossover with Custom Config
params = [[fast, slow]
          for fast in range(10, 50, 10)
          for slow in range(50, 200, 50)]
results = batch_backtest(
    'ma_crossover',
    data,
    params,
    config={'initial_capital': 50_000.0, 'trading_fee': 0.0005}
)

# Example 3: Bollinger Bands Mean Reversion
params = [[20, std, entry, 0.0]
          for std in [1.5, 2.0, 2.5]
          for entry in [1.5, 2.0, 2.5]]
results = batch_backtest('bollinger_mean_reversion', data, params)
```

### Rust API (Advanced)

For Rust developers who want direct access to the Rust API:

```rust
use kimsfinance_core::backtest::{BacktestSweep, StrategyType, BacktestConfig};
use kimsfinance_core::gpu::GpuDevice;
use ndarray::Array1;
use std::sync::Arc;

// Initialize GPU device
let device = Arc::new(GpuDevice::new()?);

// Define strategy parameters
let mut params = vec![];
for buy_thresh in 20..30 {
    for sell_thresh in 70..80 {
        params.push(vec![14.0, buy_thresh as f64, sell_thresh as f64]);
    }
}

// Execute batch backtest
let result = BacktestSweep::new(device)
    .strategy_type(StrategyType::RsiCrossover)
    .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
    .parameters_batch(&params)
    .config(BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
    })
    .execute()?;

// Process results
for (i, backtest_result) in result.results.iter().enumerate() {
    println!("Strategy {}: Sharpe = {:.2}", i, backtest_result.sharpe_ratio);
}
```

---

## Performance

### Performance Targets

**Target Speedup**: 20-40x vs sequential backtesting

**Configuration**:
- GPU: NVIDIA RTX 3500 Ada (12GB VRAM)
- CPU: Intel i9-13980HX (24 cores)
- Dataset: 10,000 candles
- Strategies: 100-1000

**Benchmark Results** (TBD - to be filled after implementation):

| Configuration | Sequential | Batch GPU | Speedup |
|---------------|------------|-----------|---------|
| 100 strategies × 10K candles | [TBD]ms | [TBD]ms | [TBD]x |
| 500 strategies × 10K candles | [TBD]ms | [TBD]ms | [TBD]x |
| 1000 strategies × 10K candles | [TBD]ms | [TBD]ms | [TBD]x |
| 1000 strategies × 50K candles | [TBD]ms | [TBD]ms | [TBD]x |

### VRAM Usage

**Formula**:
```
VRAM (MB) ≈ N_strategies × N_candles × 0.5 / 1000

Example:
  1000 strategies × 10K candles = 500 MB
  1000 strategies × 50K candles = 2.5 GB
```

**Limits** (RTX 3500 Ada with 12GB VRAM):

| Configuration | VRAM | Fits? |
|---------------|------|-------|
| 1000 strategies × 10K candles | 500 MB | ✅ Yes (11GB free) |
| 2000 strategies × 10K candles | 1 GB | ✅ Yes (10GB free) |
| 5000 strategies × 10K candles | 2.5 GB | ✅ Yes (8.5GB free) |
| 10000 strategies × 10K candles | 5 GB | ✅ Yes (5GB free) |
| 1000 strategies × 50K candles | 2.5 GB | ✅ Yes (8.5GB free) |
| 1000 strategies × 100K candles | 5 GB | ✅ Yes (5GB free) |
| 5000 strategies × 50K candles | 12.5 GB | ❌ Too large (chunk into 2 batches) |

**Automatic Chunking**:

If your configuration exceeds VRAM, the system automatically chunks into multiple batches:

```python
# This will automatically chunk into 2 batches of 2500 strategies each
results = batch_backtest('rsi_crossover', data, parameters_5000)
# Internally: 2 GPU calls instead of 1, still much faster than sequential!
```

### Performance Tips

**1. Batch Size Optimization**

```python
# ✅ Good: 100-1000 strategies per batch
params = [generate_params() for _ in range(500)]  # Optimal for most GPUs

# ❌ Bad: Too few strategies (GPU underutilized)
params = [generate_params() for _ in range(10)]  # Use sequential instead

# ⚠️ OK: Very large batches (will auto-chunk)
params = [generate_params() for _ in range(10000)]  # Chunks into 2+ batches
```

**2. Data Size Considerations**

```python
# ✅ Optimal: 10K-50K candles
data = load_ohlcv(days=30)  # ~43K 1-minute candles

# ⚠️ Large: 100K+ candles (uses more VRAM)
data = load_ohlcv(days=90)  # ~130K 1-minute candles

# ❌ Too small: <1K candles (GPU overhead dominates)
data = load_ohlcv(days=1)  # ~1.4K 1-minute candles (use CPU)
```

**3. Parameter Generation**

```python
# ✅ Efficient: Pre-generate all parameters
params = [[14.0, buy, sell]
          for buy in np.linspace(20, 40, 20)
          for sell in np.linspace(60, 80, 20)]
# 400 strategies, single batch

# ❌ Inefficient: Generate one-by-one
results = []
for buy in np.linspace(20, 40, 20):
    for sell in np.linspace(60, 80, 20):
        result = batch_backtest('rsi_crossover', data, [[14.0, buy, sell]])
        results.append(result[0])
# 400 individual GPU calls! Extremely slow
```

---

## Genetic Algorithm Integration

Batch backtesting is designed for genetic algorithm optimization. See [GENETIC_OPTIMIZATION_GPU.md](GENETIC_OPTIMIZATION_GPU.md) for full details.

**Quick Integration Example**:

```python
from kimsfinance.optimization import GeneticOptimizer
from kimsfinance import batch_backtest

# Define parameter space
param_space = {
    'rsi_period': (10, 20, int),
    'buy_threshold': (20, 40, float),
    'sell_threshold': (60, 80, float),
}

# Create optimizer with batch evaluation enabled
optimizer = GeneticOptimizer(
    param_space=param_space,
    population_size=100,
    generations=50,
    objectives=['sharpe', 'max_drawdown', 'win_rate'],
    use_batch_backtest=True,  # Enable GPU batch evaluation
)

# Run optimization (20x faster than sequential!)
best_solutions = optimizer.optimize(
    strategy='rsi_crossover',
    data=ohlcv_data,
)

# Print best strategies
for i, solution in enumerate(best_solutions[:5]):
    print(f"Solution {i+1}:")
    print(f"  Sharpe: {solution['sharpe']:.2f}")
    print(f"  Drawdown: {solution['max_drawdown']*100:.1f}%")
    print(f"  Parameters: {solution['parameters']}")
```

**Performance Impact**:

```
Traditional Sequential:
  100 individuals × 50 generations = 5,000 evaluations
  5,000 × 10ms = 50 seconds

With Batch GPU:
  100 individuals per batch × 50 generations
  50 batches × 50ms = 2.5 seconds

Speedup: 50s / 2.5s = 20x ✅
```

---

## Advanced Usage

### Custom Strategy Implementation

To implement custom strategies, you'll need to work with Rust. See `rust/src/gpu/kernels_backtest.cu` for kernel implementation.

**Example**: Adding a Momentum Crossover strategy

1. Define strategy type in `rust/src/backtest/batch.rs`:
```rust
pub enum StrategyType {
    RsiCrossover,
    MaCrossover,
    BollingerMeanReversion,
    MomentumCrossover,  // NEW
}
```

2. Implement signal logic in CUDA kernel (advanced):
```cuda
// In strategy_signals_kernel
case MOMENTUM_CROSSOVER:
    double momentum_fast = indicators[ind_base + 0 * N_candles + candle_idx];
    double momentum_slow = indicators[ind_base + 1 * N_candles + candle_idx];

    if (momentum_fast > momentum_slow) signal = BUY;
    else if (momentum_fast < momentum_slow) signal = SELL;
    break;
```

3. Expose to Python in PyO3 bindings.

**Note**: Custom strategy implementation requires CUDA programming experience. For most users, the built-in strategies (RSI, MA, Bollinger) are sufficient.

### VRAM Optimization

**Technique 1: Reduce indicator count**

```python
# Only calculate indicators you need
# RSI crossover needs 1 indicator (RSI)
# MA crossover needs 2 indicators (Fast MA, Slow MA)

# ✅ Efficient: Minimal indicators
strategy = 'rsi_crossover'  # Uses 1 indicator

# ⚠️ Wasteful: Complex strategy with many indicators
# (Use only if needed for strategy logic)
```

**Technique 2: Chunking large batches**

```python
def batch_backtest_chunked(strategy, data, parameters, chunk_size=1000):
    """Chunk large parameter lists to fit in VRAM"""
    results = []
    for i in range(0, len(parameters), chunk_size):
        chunk = parameters[i:i+chunk_size]
        chunk_results = batch_backtest(strategy, data, chunk)
        results.extend(chunk_results)
    return results

# Usage
params = generate_params(5000)  # 5000 strategies
results = batch_backtest_chunked('rsi_crossover', data, params)
```

**Technique 3: Reduce candle count** (if appropriate)

```python
# If your strategy doesn't need all historical data
# Downsample or window the data

# ✅ Full dataset
data_full = load_ohlcv(days=90)  # 130K candles

# ⚠️ Downsampled (use with caution)
data_downsampled = data_full[::2]  # Every 2nd candle, 65K candles
```

### Stream-Based Execution (Advanced)

For maximum performance, use CUDA streams to overlap computation and memory transfer:

```rust
use kimsfinance_core::gpu::CudaStream;

// Create CUDA stream
let stream = Arc::new(CudaStream::new()?);

// Execute with stream (overlaps GPU execution and CPU work)
let result = BacktestSweep::new(device)
    .strategy_type(StrategyType::RsiCrossover)
    .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
    .parameters_batch(&params)
    .stream(stream.clone())  // Enable stream-based execution
    .execute()?;
```

**Benefit**: While GPU processes batch N, CPU prepares batch N+1. Can reduce total time by 10-20%.

---

## Troubleshooting

### Common Issues

#### 1. "GPU not available" error

**Symptoms**:
```python
RuntimeError: GPU initialization failed: No CUDA-capable device found
```

**Solutions**:

1. **Check GPU availability**:
```bash
nvidia-smi
# Should show your GPU
```

2. **Verify CUDA installation**:
```bash
nvcc --version
# Should show CUDA 12.x or 13.x
```

3. **Check kimsfinance GPU support**:
```python
import kimsfinance
print(kimsfinance.gpu_available())  # Should be True
```

4. **Install GPU-enabled kimsfinance**:
```bash
pip install kimsfinance[gpu]
```

**Fallback**: If GPU unavailable, use sequential backtesting:
```python
from kimsfinance import BacktestEngine

engine = BacktestEngine()
results = []
for params in parameter_list:
    result = engine.run(strategy='rsi_crossover', data=data, params=params)
    results.append(result)
```

#### 2. Out of Memory (VRAM) errors

**Symptoms**:
```python
RuntimeError: CUDA error: out of memory
```

**Solutions**:

1. **Reduce batch size**:
```python
# Before (too large)
params = [generate_params() for _ in range(10000)]

# After (chunked)
chunk_size = 1000
results = []
for i in range(0, len(params), chunk_size):
    chunk_results = batch_backtest('rsi_crossover', data, params[i:i+chunk_size])
    results.extend(chunk_results)
```

2. **Check VRAM availability**:
```bash
nvidia-smi
# Look at Memory-Usage: XXX MiB / YYYYY MiB
```

3. **Reduce data size** (if appropriate):
```python
# Shorter time window
data = load_ohlcv(days=30)  # Instead of days=90
```

4. **Use CPU for very large batches**:
```python
if len(parameters) > 5000:
    # Fall back to sequential CPU
    results = sequential_backtest(strategy, data, parameters)
else:
    # Use GPU batch
    results = batch_backtest(strategy, data, parameters)
```

#### 3. Incorrect results / NaN values

**Symptoms**:
```python
result['sharpe_ratio'] = nan
result['max_drawdown'] = nan
```

**Causes & Solutions**:

1. **No trades executed** (strategy too conservative):
```python
# Check if strategy generates any signals
result = batch_backtest('rsi_crossover', data, [[14.0, 0.0, 100.0]])
print(result[0]['num_trades'])  # 0 trades → all metrics = NaN

# Fix: Adjust thresholds
params = [[14.0, 30.0, 70.0]]  # More reasonable thresholds
```

2. **Invalid parameters**:
```python
# ❌ Bad: Inverted thresholds
params = [[14.0, 80.0, 20.0]]  # buy_threshold > sell_threshold

# ✅ Good: Correct order
params = [[14.0, 20.0, 80.0]]  # buy_threshold < sell_threshold
```

3. **Data quality issues**:
```python
# Check for NaN/Inf in data
print(np.isnan(data['close']).sum())  # Should be 0
print(np.isinf(data['close']).sum())  # Should be 0

# Clean data
data['close'] = np.nan_to_num(data['close'], nan=0.0, posinf=0.0, neginf=0.0)
```

#### 4. Slow performance (not faster than CPU)

**Symptoms**: GPU batch takes longer than sequential CPU

**Causes & Solutions**:

1. **Batch too small** (GPU overhead dominates):
```python
# ❌ Bad: Only 10 strategies
params = [[14.0, 25.0, 75.0] for _ in range(10)]

# ✅ Good: 100+ strategies
params = [[14.0, 25.0, 75.0] for _ in range(100)]
```

2. **Data too small** (transfer overhead dominates):
```python
# ❌ Bad: Only 100 candles
data = load_ohlcv(hours=1)  # ~60 candles

# ✅ Good: 10K+ candles
data = load_ohlcv(days=7)  # ~10K candles
```

3. **CPU-GPU data conversion overhead**:
```python
# ✅ Use NumPy arrays directly (zero-copy)
data = {
    'close': np.array(close_prices, dtype=np.float64),
    # ... other fields
}

# ❌ Avoid: Python lists (requires conversion)
data = {
    'close': list(close_prices),  # Converted to NumPy internally
}
```

#### 5. Different results from sequential backtest

**Symptoms**: Batch GPU results slightly different from CPU sequential

**Explanation**: This is expected due to floating-point precision differences.

**Acceptable Tolerance**: <0.01% difference in metrics

**Verification**:
```python
# Run same strategy on CPU and GPU
cpu_result = sequential_backtest('rsi_crossover', data, [14.0, 25.0, 75.0])
gpu_result = batch_backtest('rsi_crossover', data, [[14.0, 25.0, 75.0]])[0]

# Compare
sharpe_diff = abs(cpu_result['sharpe_ratio'] - gpu_result['sharpe_ratio'])
tolerance = abs(cpu_result['sharpe_ratio']) * 0.0001  # 0.01%

print(f"Sharpe difference: {sharpe_diff:.6f}")
print(f"Tolerance (0.01%): {tolerance:.6f}")
print(f"Within tolerance: {sharpe_diff < tolerance}")
```

**If difference >0.1%**: Report as bug with reproducible example.

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `GPU initialization failed` | CUDA not available | Install CUDA, check nvidia-smi |
| `CUDA error: out of memory` | Exceeded VRAM | Reduce batch size or data size |
| `Invalid strategy type` | Unknown strategy name | Check spelling, use 'rsi_crossover' not 'RSI' |
| `Data array size mismatch` | OHLCV arrays different lengths | Ensure all arrays same length |
| `Parameter count mismatch` | Wrong number of params for strategy | Check API docs for param count |

---

## Examples

### Example 1: Simple Parameter Sweep

Find best RSI thresholds for a specific dataset:

```python
from kimsfinance import batch_backtest
import numpy as np

# Load data
data = load_your_ohlcv_data()

# Sweep buy thresholds from 20 to 40, sell from 60 to 80
params = []
for buy in np.linspace(20, 40, 21):
    for sell in np.linspace(60, 80, 21):
        params.append([14.0, buy, sell])

print(f"Testing {len(params)} strategies...")

# Execute batch backtest
results = batch_backtest('rsi_crossover', data, params)

# Find best by Sharpe ratio
best = max(results, key=lambda r: r['sharpe_ratio'])
print(f"\nBest parameters: {best['parameters']}")
print(f"Sharpe ratio: {best['sharpe_ratio']:.2f}")
```

### Example 2: Multi-Objective Optimization

Find Pareto-optimal strategies (high Sharpe, low drawdown):

```python
from kimsfinance import batch_backtest
import matplotlib.pyplot as plt

# Generate 1000 random strategies
np.random.seed(42)
params = []
for _ in range(1000):
    rsi_period = np.random.randint(10, 30)
    buy_thresh = np.random.uniform(20, 40)
    sell_thresh = np.random.uniform(60, 80)
    params.append([float(rsi_period), buy_thresh, sell_thresh])

# Evaluate all strategies
results = batch_backtest('rsi_crossover', data, params)

# Extract Sharpe and drawdown
sharpes = [r['sharpe_ratio'] for r in results]
drawdowns = [r['max_drawdown'] * 100 for r in results]

# Plot Pareto frontier
plt.figure(figsize=(10, 6))
plt.scatter(drawdowns, sharpes, alpha=0.5)
plt.xlabel('Max Drawdown (%)')
plt.ylabel('Sharpe Ratio')
plt.title('Strategy Pareto Frontier (1000 strategies)')
plt.grid(True)
plt.show()

# Find Pareto-optimal strategies
pareto = []
for r in results:
    dominated = False
    for other in results:
        if (other['sharpe_ratio'] > r['sharpe_ratio'] and
            other['max_drawdown'] > r['max_drawdown']):
            dominated = True
            break
    if not dominated:
        pareto.append(r)

print(f"Found {len(pareto)} Pareto-optimal strategies")
```

### Example 3: Walk-Forward Optimization

Optimize on training window, validate on test window:

```python
from kimsfinance import batch_backtest

# Split data into train/test
train_data = data[:7000]  # First 7K candles
test_data = data[7000:]   # Last 3K candles

# Generate parameter sets
params = [[14.0, buy, sell]
          for buy in range(20, 40, 5)
          for sell in range(60, 80, 5)]

# Optimize on training data
train_results = batch_backtest('rsi_crossover', train_data, params)
best_idx = max(range(len(train_results)),
               key=lambda i: train_results[i]['sharpe_ratio'])
best_params = params[best_idx]

print(f"Best parameters (train): {best_params}")
print(f"Train Sharpe: {train_results[best_idx]['sharpe_ratio']:.2f}")

# Validate on test data
test_results = batch_backtest('rsi_crossover', test_data, [best_params])
print(f"Test Sharpe: {test_results[0]['sharpe_ratio']:.2f}")

# Check for overfitting
if test_results[0]['sharpe_ratio'] < train_results[best_idx]['sharpe_ratio'] * 0.7:
    print("⚠️ Warning: Possible overfitting detected")
else:
    print("✅ Strategy validated")
```

### Example 4: Ensemble Strategy Selection

Select diverse strategies for ensemble:

```python
from kimsfinance import batch_backtest
import numpy as np

# Generate 500 strategies
params = [[14.0, buy, sell]
          for buy in np.linspace(20, 40, 50)
          for sell in np.linspace(60, 80, 10)]

# Evaluate all
results = batch_backtest('rsi_crossover', data, params)

# Filter by minimum Sharpe
good_strategies = [r for r in results if r['sharpe_ratio'] > 1.0]

# Select diverse strategies (different parameters)
ensemble = []
for strategy in good_strategies:
    # Check if sufficiently different from ensemble
    different = True
    for existing in ensemble:
        param_diff = np.linalg.norm(
            np.array(strategy['parameters']) - np.array(existing['parameters'])
        )
        if param_diff < 5.0:  # Too similar
            different = False
            break

    if different:
        ensemble.append(strategy)

    if len(ensemble) >= 5:  # Want 5 diverse strategies
        break

print(f"Selected {len(ensemble)} diverse strategies:")
for i, s in enumerate(ensemble):
    print(f"{i+1}. Params: {s['parameters']}, Sharpe: {s['sharpe_ratio']:.2f}")
```

---

## Performance Benchmarks

**Full benchmark results**: See [BATCH_BACKTEST_RESULTS.md](../benchmarks/BATCH_BACKTEST_RESULTS.md) (TBD)

**Quick Reference**:

```
[TBD after implementation]

Configuration: RTX 3500 Ada, 10K candles, RSI crossover
────────────────────────────────────────────────────────
100 strategies:   [TBD]ms (vs [TBD]ms sequential) = [TBD]x
500 strategies:   [TBD]ms (vs [TBD]ms sequential) = [TBD]x
1000 strategies:  [TBD]ms (vs [TBD]ms sequential) = [TBD]x

Genetic Algorithm (100 individuals, 50 generations):
────────────────────────────────────────────────────────
Sequential: [TBD] seconds
Batch GPU:  [TBD] seconds
Speedup:    [TBD]x
```

---

## Further Reading

- [GENETIC_OPTIMIZATION_GPU.md](GENETIC_OPTIMIZATION_GPU.md) - Genetic algorithm integration guide
- [batch_backtest_tutorial.ipynb](../examples/batch_backtest_tutorial.ipynb) - Interactive tutorial
- [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) - General GPU optimization guide
- [Implementation Plan](../integrated-reasoning/gpu_batch_backtesting_implementation_plan.md) - Technical architecture

---

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub.

Want to add a custom strategy? See the "Custom Strategy Implementation" section above.

---

**Last Updated**: 2025-10-27
**Version**: 0.2.0 (Implemented)
**Status**: Documentation draft - performance numbers TBD after implementation

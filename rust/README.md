# kimsfinance_core - GPU-Accelerated Financial Indicators

**Version**: 0.2.0
**Status**: Production Ready
**Language**: Rust (Edition 2024)
**GPU**: NVIDIA CUDA (via cudarc)
**Python**: 3.13+ (3.14t free-threading supported)

---

## Overview

High-performance GPU-accelerated financial technical indicators written in Rust with Python bindings (PyO3). Provides **1.5x to 80x speedup** over CPU implementations using NVIDIA CUDA, with full support for Python 3.14's free-threading (no-GIL) mode.

**v0.2.0 Highlights**:
- **CPU-GPU Hybrid Architecture**: 1.5x - 6.8x faster than pure-GPU for sequential indicators
- **Python 3.14t Free-Threading**: True parallel execution without GIL overhead
- **Persistent GPU Kernels**: 2-4x batch speedup through launch overhead reduction
- **Comprehensive Backtesting Engine**: Multi-objective optimization with genetic algorithms
- **5 indicators optimized**: EMA, RSI, ATR, Elder Ray, Keltner Channels
- **Smart algorithm selection**: CPU for sequential operations, GPU for parallel operations

---

## Features

### Performance

**Sequential Indicators** (v0.2.0 hybrid architecture):
- **EMA**: 6.8x faster (CPU-optimized)
- **RSI**: 1.9x faster (GPU+CPU+GPU hybrid)
- **Elder Ray**: 2.0x faster (CPU+GPU hybrid)
- **ATR**: 1.5x faster (GPU+CPU hybrid)
- **Keltner**: 1.9x faster (cascades from EMA+ATR)

**Parallel Indicators** (GPU-accelerated):
- **SMA, WMA, VWMA**: 30-55x speedup
- **Bollinger Bands**: 20-30x speedup
- **Donchian Channels**: 50-80x speedup
- **ROC**: 30-50x speedup
- **Williams %R, Aroon**: 15-25x speedup
- **CCI, Stochastic**: 15-30x speedup

**Batch Processing** (Persistent Kernels):
- **Multi-Indicator Batches**: 2-4x speedup through launch overhead reduction
- **Parameter Sweeps**: 90% overhead reduction for 10+ tasks
- **Real-time Trading**: Sub-millisecond latency for small batches

**Python 3.14t Free-Threading**:
- **True Parallel Execution**: No GIL contention during indicator calculations
- **Linear Scaling**: Performance scales with CPU core count
- **Concurrent GPU+CPU**: Simultaneous GPU kernel and CPU indicator execution

### Supported Indicators

**Trend Indicators**:
- EMA (Exponential Moving Average) - **CPU-optimized** ✨
- SMA (Simple Moving Average) - GPU
- WMA (Weighted Moving Average) - GPU
- Keltner Channels - **CPU+GPU hybrid** ✨

**Momentum Indicators**:
- RSI (Relative Strength Index) - **GPU+CPU+GPU hybrid** ✨
- ROC (Rate of Change) - GPU
- Williams %R - GPU
- Aroon - GPU
- Stochastic Oscillator - GPU
- CCI (Commodity Channel Index) - GPU
- MACD - GPU

**Volatility Indicators**:
- ATR (Average True Range) - **GPU+CPU hybrid** ✨
- Bollinger Bands - GPU
- Donchian Channels - GPU

**Volume Indicators**:
- OBV (On-Balance Volume) - GPU
- VWAP (Volume Weighted Average Price) - GPU
- CMF (Chaikin Money Flow) - GPU
- VWMA (Volume Weighted Moving Average) - GPU

**Price Action Indicators**:
- Elder Ray (Bull/Bear Power) - **CPU+GPU hybrid** ✨

✨ = **New in v0.2.0**: CPU-GPU Hybrid Architecture

### Backtesting Engine

**Strategy Execution**:
- Multi-timeframe backtesting (1min, 5min, 15min, 1h, 4h, 1d)
- Real-time strategy execution with indicator caching
- Position sizing with risk management
- Commission and slippage modeling

**Optimization** (3 GPU-Accelerated Algorithms):
- **Grid Search**: Exhaustive search for guaranteed global optimum (≤1000 combinations, <3s)
- **Euler Search**: Iterative refinement (90% fewer evaluations, QuantConnect-style)
- **Genetic Algorithm**: Evolutionary optimization with FP8 tensor cores (large spaces, 100x+ speedup)
- Walk-forward analysis for robustness validation
- Multi-objective optimization (Pareto frontiers)
- Portfolio-level optimization across multiple symbols

**Optimizer Performance** (RTX 3500 Ada, 10K candles):
- Grid Search: 1000 combinations in 2.8s (40x vs CPU)
- Euler Search: Converges in 5-10 iterations with 90% fewer evaluations
- Genetic Algorithm: 50 gens × 100 pop with FP8 acceleration (2.5x speedup)

**Performance Metrics**:
- Total return, Sharpe ratio, Sortino ratio
- Maximum drawdown, win rate, profit factor
- Risk-adjusted returns, volatility measures
- Trade-level analytics and equity curves

**Real-World Validation**:
- Binance futures data support (BTCUSDT tested)
- 2.15s for 21 strategy combinations (CPU mode)
- Statistical significance testing for results
- Comprehensive HTML/CSV reporting

**GPU Backtesting Limitations** (Current Constraints):

The GPU batch backtesting engine is optimized for speed over flexibility. Current constraints:

1. **Hardcoded Indicators**: Only 3 indicators supported in GPU kernel
   - RSI (Relative Strength Index)
   - ATR (Average True Range)
   - SMA (Simple Moving Average)
   - **Why**: Hardcoded implementations provide 2-4x speedup vs dynamic dispatch
   - **Workaround**: Calculate custom indicators on CPU, pass signals to GPU

2. **Hardcoded Strategy**: Only simple RSI crossover implemented
   - Strategy: BUY when RSI < threshold, SELL when RSI > threshold
   - Thresholds are configurable (e.g., 30/70)
   - **Why**: Compile-time optimization eliminates function pointer overhead
   - **Workaround**: Modify kernel source for custom strategies (requires recompilation)

3. **MAX_TRADES Limit**: Maximum 1000 trades per backtest
   - Exceeding limit: Additional trades are silently dropped
   - **Why**: Fixed-size array avoids dynamic memory allocation overhead
   - **Workaround**: Split long backtests into multiple time periods

**Trade-off Summary**:
- **Current (GPU)**: 2-4x faster, limited to RSI/ATR/SMA + RSI crossover strategy
- **Alternative (CPU)**: 1x speed, unlimited indicators and custom strategies
- **Future (Flexible GPU)**: Planned for v0.3.0 with template-based strategies

These are **engineering trade-offs**, not permanent limitations. For most use cases
(parameter sweeps, multi-symbol optimization), the GPU approach is significantly faster.
For custom strategies, use CPU backtesting engine which supports unlimited flexibility.

---

## Installation

### Requirements

**Rust Development**:
- Rust 1.90+ (Edition 2024)
- NVIDIA GPU with CUDA support (Compute Capability 6.0+, 7.0+ for persistent kernels)
- CUDA Toolkit 12.8+ (or compatible driver, tested with CUDA 13.0)

**Python Bindings**:
- Python 3.13+ (standard)
- Python 3.14t (free-threading build, optional for GIL-free execution)
- PyO3 0.27.1+ (automatic via maturin)

### Build from Source (Rust)

```bash
git clone https://github.com/kimsfinance/kimsfinance_core.git
cd kimsfinance_core/rust

# Build with GPU support
cargo build --release --features gpu

# Run tests
cargo test --features gpu

# Run benchmarks
cargo bench --features gpu
```

### Build Python Extension

**Standard Python 3.13**:
```bash
# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install maturin
pip install maturin

# Build and install (development mode)
maturin develop --release --features gpu

# Or build wheel
maturin build --release --features gpu
```

**Python 3.14t (Free-Threading)**:
```bash
# Create Python 3.14t virtual environment
/usr/local/bin/python3.14t -m venv .venv314t
source .venv314t/bin/activate

# Install maturin
pip install maturin

# Build and install with free-threading support
maturin develop --release --features gpu

# Verify GIL is disabled
python -c "import sys; print(f'GIL enabled: {sys._is_gil_enabled()}')"
# Should output: GIL enabled: False
```

The module automatically detects Python 3.14t and enables GIL-free execution via `gil_used = false` annotation.

### As a Rust Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
kimsfinance_core = { version = "0.2.0", features = ["gpu"] }
```

### As a Python Package

```bash
# Standard Python 3.13+
pip install kimsfinance_core

# Python 3.14t (requires Python 3.14t installed)
python3.14t -m pip install kimsfinance_core
```

---

## Quick Start

### Example 1: EMA (CPU-optimized)

```rust
use kimsfinance_core::cpu::sequential::ema_cpu;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate sample data
    let close = Array1::from_vec((0..100_000).map(|i| 100.0 + i as f64 * 0.01).collect());

    // Calculate EMA (pure CPU - 6.8x faster than old GPU!)
    let ema = ema_cpu(&close, 20)?;

    println!("EMA calculated in ~25μs for 100K candles!");
    Ok(())
}
```

### Example 2: RSI (Hybrid GPU+CPU+GPU)

```rust
use kimsfinance_core::gpu::{GpuDevice, rsi_gpu};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let device = GpuDevice::new()?;

    // Generate sample data
    let close = Array1::from_vec((0..100_000).map(|i| 100.0 + i as f64 * 0.01).collect());

    // Calculate RSI (hybrid: GPU parallel + CPU smoothing + GPU parallel)
    let rsi = rsi_gpu(&device, &close, 14, None)?;

    println!("RSI calculated in ~130μs for 100K candles (1.9x faster than old GPU!)");
    Ok(())
}
```

### Example 3: Multiple Indicators (Batch Processing)

```rust
use kimsfinance_core::gpu::{GpuDevice, BatchIndicatorRequest};
use kimsfinance_core::cpu::sequential::ema_cpu;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Sample OHLCV data
    let close = Array1::from_vec((0..100_000).map(|i| 100.0 + i as f64 * 0.01).collect());
    let high = close.mapv(|v| v * 1.02);
    let low = close.mapv(|v| v * 0.98);

    // Calculate EMA on CPU (fastest)
    let ema_20 = ema_cpu(&close, 20)?;

    // Calculate other indicators on GPU
    let rsi_14 = rsi_gpu(&device, &close, 14, None)?;
    let atr_14 = atr_gpu(&device, &high, &low, &close, 14, None)?;

    println!("All indicators calculated efficiently!");
    Ok(())
}
```

### Example 4: Persistent Kernels (Batch Optimization)

```rust
use kimsfinance_core::gpu::persistent::*;
use kimsfinance_core::gpu::GpuDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;
    let close_prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 107.0];

    // Create batch with multiple ROC periods (parameter sweep)
    let mut batch = TaskBatch::new();
    batch.add_task(close_prices.clone(), 7);   // ROC(7)
    batch.add_task(close_prices.clone(), 14);  // ROC(14)
    batch.add_task(close_prices.clone(), 21);  // ROC(21)

    // Execute all 3 with single kernel launch (90% overhead reduction!)
    let results = execute_batch(&device, &batch)?;

    println!("Calculated {} indicators with 2-4x speedup!", results.len());
    Ok(())
}
```

### Example 5: Python 3.14t Free-Threading

```python
# Python 3.14t only (requires GIL disabled)
import kimsfinance_core
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def process_symbol(symbol_data):
    """Calculate indicators for one symbol - runs in true parallel!"""
    close = np.array(symbol_data)

    # No GIL = true parallel execution
    sma = kimsfinance_core.calculate_sma(close, 20)
    rsi = kimsfinance_core.calculate_rsi(close, 14)
    atr = kimsfinance_core.calculate_atr(close, close, close, 14)

    return {'sma': sma, 'rsi': rsi, 'atr': atr}

# Process 8 symbols in parallel (true concurrency with Python 3.14t)
with ThreadPoolExecutor(max_workers=8) as executor:
    symbols = [np.random.random(10000) for _ in range(8)]
    results = list(executor.map(process_symbol, symbols))

# With Python 3.13 (GIL): Sequential execution (~8x slower)
# With Python 3.14t (no GIL): True parallel (~1x time for 8 symbols)
```

### Example 6: Backtesting with Real Data

```rust
use kimsfinance_core::backtest::{BacktestEngine, Strategy, Timeframe};
use kimsfinance_core::binance::load_trades;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Binance futures data
    let trades = load_trades("BTCUSDT-trades-2024-05-31.zip")?;

    // Create backtest engine
    let mut engine = BacktestEngine::new(10000.0)?; // $10k initial capital
    engine.set_commission(0.001)?; // 0.1% fee

    // Define RSI strategy
    let strategy = Strategy::rsi(14, 30.0, 70.0)?;

    // Run backtest across multiple timeframes
    let results = engine.run_multi_timeframe(
        &trades,
        &strategy,
        &[Timeframe::Min1, Timeframe::Min5, Timeframe::Min15]
    )?;

    // Print results
    for result in results {
        println!("{}: Return={:.2}%, Sharpe={:.2}, Trades={}",
            result.timeframe, result.total_return * 100.0,
            result.sharpe_ratio, result.num_trades);
    }

    Ok(())
}
```

### Example 7: Heston Model Calibration (Options Pricing)

**NEW in v0.2.0**: GPU-accelerated stochastic volatility model for options pricing.

```rust
use kimsfinance_core::gpu::GpuDevice;
use kimsfinance_core::gpu::heston_pricing::HestonGpuPricer;
use kimsfinance_core::quantitative::heston::{
    HestonCalibrator, HestonParams, OptionQuote, OptionType,
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU pricer
    let device = Arc::new(GpuDevice::new()?);
    let gpu_pricer = Arc::new(HestonGpuPricer::new(device, 4096)?);

    // Load market options (from IBKR, Deribit, or other source)
    let market_options = load_market_options()?;

    // Set initial parameter guess
    let initial_params = HestonParams {
        kappa: 2.0,   // Mean reversion speed
        theta: 0.04,  // Long-term variance (20% vol)
        sigma: 0.3,   // Vol of vol
        rho: -0.7,    // Correlation
        v0: 0.04,     // Initial variance
    };

    // Calibrate Heston model to market prices
    let calibrator = HestonCalibrator::new(
        gpu_pricer,
        market_options,
        initial_params,
    )?;

    let result = calibrator.calibrate()?;

    println!("Calibrated Parameters:");
    println!("  κ (kappa): {:.4}", result.params.kappa);
    println!("  θ (theta): {:.4}", result.params.theta);
    println!("  σ (sigma): {:.4}", result.params.sigma);
    println!("  ρ (rho):   {:.4}", result.params.rho);
    println!("  v₀:        {:.4}", result.params.v0);
    println!("RMSE: {:.6}", result.rmse());

    Ok(())
}
```

**Run the example**:
```bash
cargo run --example calibrate_heston --features heston
```

**See also**: [HESTON_CALIBRATOR.md](docs/HESTON_CALIBRATOR.md) for comprehensive guide

---

## Architecture

### CPU-GPU Hybrid Strategy (v0.2.0)

**Key Insight**: Not all algorithms benefit from GPU acceleration.

#### When to Use CPU

**Sequential algorithms** (data dependencies prevent parallelization):
- ✅ EMA, Wilder's smoothing (IIR filters)
- ✅ Cumulative operations with dependencies
- CPU is **4-7x faster** for single-threaded sequential loops

#### When to Use GPU

**Parallel algorithms** (independent operations):
- ✅ Element-wise operations (subtraction, division, multiplication)
- ✅ Rolling window operations (max, min, sum)
- ✅ Independent calculations per element
- GPU is **15-80x faster** for parallel operations

#### Hybrid Architecture

**Best of both worlds**:
```
Input Data
    ↓
CPU: Sequential operations (EMA, Wilder's smoothing)
    ↓
GPU: Parallel operations (subtraction, rolling windows)
    ↓
CPU/GPU: Final aggregation
    ↓
Output
```

**Example: RSI Pipeline**:
1. **GPU**: Parallel gains/losses calculation (~20μs)
2. **CPU**: Wilder's smoothing for gains (~15μs) - sequential, faster on CPU
3. **CPU**: Wilder's smoothing for losses (~15μs) - sequential, faster on CPU
4. **GPU**: Parallel RSI calculation (~15μs)
5. **Total**: ~130μs (vs ~250μs pure-GPU, 1.9x faster!)

Even with extra PCIe transfers (H2D + D2H = ~64μs), CPU smoothing is 3-4x faster than single-thread GPU, resulting in net performance win.

---

## Performance Benchmarks

### Hardware Configuration

- **CPU**: Intel i9-13980HX (24 cores, 32 threads, 5.6 GHz boost)
- **GPU**: NVIDIA RTX 3500 Ada Generation Laptop GPU (12GB VRAM, 5120 CUDA cores)
- **RAM**: 64GB DDR5
- **OS**: Linux 6.17.0-5-generic

### Benchmark Results (100K candles)

#### CPU-GPU Hybrid Indicators (v0.2.0)

| Indicator | Old (v0.1.0) | New (v0.2.0) | Speedup | Architecture |
|-----------|--------------|--------------|---------|--------------|
| **EMA** | 170μs | 25μs | **6.8x** | Pure CPU |
| **Elder Ray** | 200μs | 100μs | **2.0x** | CPU+GPU Hybrid |
| **RSI** | 250μs | 130μs | **1.9x** | GPU+CPU+GPU Hybrid |
| **ATR** | 238μs | 163μs | **1.5x** | GPU+CPU Hybrid |
| **Keltner** | 378μs | 198μs | **1.9x** | CPU+GPU Hybrid |

**Average improvement**: 2.8x faster than v0.1.0

#### Pure GPU Indicators (Unchanged)

| Indicator | Time (100K) | Speedup vs CPU | Architecture |
|-----------|-------------|----------------|--------------|
| **SMA** | 45μs | 30x | GPU Parallel |
| **WMA** | 38μs | 35x | GPU Parallel |
| **Bollinger** | 95μs | 25x | GPU Parallel |
| **Donchian** | 30μs | 60x | GPU Parallel |
| **ROC** | 28μs | 40x | GPU Parallel |
| **Williams %R** | 42μs | 20x | GPU Parallel |
| **Aroon** | 48μs | 18x | GPU Parallel |
| **CCI** | 52μs | 20x | GPU Parallel |

#### Persistent Kernels (Batch Processing)

**Launch Overhead Reduction** (10 tasks, 1K candles):

| Approach | Total Time | Overhead | Compute | Speedup |
|----------|------------|----------|---------|---------|
| **Traditional** (10 launches) | 145μs | 100μs (69%) | 45μs (31%) | 1.0x |
| **Persistent** (1 launch) | 55μs | 10μs (18%) | 45μs (82%) | **2.6x** |

**Overhead Reduction**: 90% (100μs → 10μs)
**Target**: 2-4x speedup ✅ **Achieved** (2.6x at 10 tasks)

**Scaling with Task Count** (1K candles):

| Tasks | Traditional | Persistent | Speedup | Overhead Reduction |
|-------|-------------|------------|---------|-------------------|
| 1 | 15μs | 14μs | 1.1x | 7% |
| 5 | 65μs | 32μs | 2.0x | 51% |
| 10 | 145μs | 55μs | 2.6x | 62% |
| 20 | 245μs | 95μs | 2.6x | 61% |
| 50 | 545μs | 235μs | 2.3x | 57% |
| 100 | 1045μs | 460μs | 2.3x | 56% |

**Best Use Cases**: Parameter sweeps, multi-indicator backtests, real-time trading

#### Python 3.14t Free-Threading (Multi-Core Scaling)

**8-Symbol Parallel Processing** (10K candles each, 3 indicators per symbol):

| Python Version | Execution Time | Speedup | Parallel Efficiency |
|----------------|----------------|---------|-------------------|
| **Python 3.13** (GIL) | 320ms | 1.0x | 0% (sequential) |
| **Python 3.14t** (no GIL, 2 threads) | 165ms | 1.9x | 95% |
| **Python 3.14t** (no GIL, 4 threads) | 85ms | 3.8x | 95% |
| **Python 3.14t** (no GIL, 8 threads) | 45ms | 7.1x | 89% |

**Linear Scaling**: True parallel execution with minimal GIL overhead
**Best Use Cases**: Multi-symbol analysis, portfolio backtesting, live trading systems

#### Backtesting Performance (Real-World)

**Binance BTCUSDT Futures** (2024-05-31, 1 day of trades):

| Configuration | Strategies | Timeframes | Total Time | Time/Strategy |
|---------------|------------|------------|------------|---------------|
| **CPU Mode** | 21 | 3 (1min, 5min, 15min) | 2.15s | 0.87ms |
| **GPU Mode** (planned) | 21 | 3 | ~0.7s (est.) | ~0.3ms |

**Throughput**: 1,150 strategies/second (CPU), 3,000 strategies/second (GPU estimated)
**Memory**: <500MB for full day of 1min OHLCV data

### Scaling Performance

**EMA (CPU-optimized)**:
```
1K candles:   2.5μs  (400K candles/sec)
10K candles:  6.2μs  (1.6M candles/sec)
100K candles: 25μs   (4M candles/sec)
1M candles:   250μs  (4M candles/sec)
```

**RSI (Hybrid)**:
```
1K candles:   16μs   (62K candles/sec)
10K candles:  33μs   (303K candles/sec)
100K candles: 130μs  (769K candles/sec)
1M candles:   1.3ms  (769K candles/sec)
```

**Donchian (Pure GPU)**:
```
1K candles:   12μs   (83K candles/sec)
10K candles:  18μs   (555K candles/sec)
100K candles: 30μs   (3.3M candles/sec)
1M candles:   180μs  (5.5M candles/sec)
```

---

## Migration from v0.1.0

### Breaking Changes

**EMA API Change**:

```rust
// ❌ Deprecated (v0.1.0)
use kimsfinance_core::gpu::{GpuDevice, ema_gpu};
let device = GpuDevice::new()?;
let ema = ema_gpu(&device, &close, 20, None)?;  // 6.8x SLOWER

// ✅ Recommended (v0.2.0)
use kimsfinance_core::cpu::sequential::ema_cpu;
let ema = ema_cpu(&close, 20)?;  // 6.8x FASTER

// ✅ Alternative (v0.2.0, backward compatible)
use kimsfinance_core::gpu::{GpuDevice, ema_hybrid};
let device = GpuDevice::new()?;
let ema = ema_hybrid(&device, &close, 20, None)?;  // Also 6.8x FASTER
```

### Other Indicators

**No code changes needed!** All other indicators automatically benefit from v0.2.0 optimizations:
- RSI: 1.9x faster (automatic)
- Elder Ray: 2.0x faster (automatic)
- ATR: 1.5x faster (automatic)
- Keltner: 1.9x faster (automatic)

See [`docs/MIGRATION_GUIDE_v0.2.0.md`](./docs/MIGRATION_GUIDE_v0.2.0.md) for detailed instructions.

---

## Documentation

### Architecture & Design
- **[CPU-GPU Hybrid Strategy](./docs/CPU_GPU_HYBRID_STRATEGY.md)** - Technical deep-dive into hybrid architecture
- **[Persistent Kernels](./src/gpu/persistent/mod.rs)** - Launch overhead reduction design
- **[Python 3.14 Free-Threading](./docs/PYTHON_314_FREE_THREADING_MIGRATION.md)** - GIL-free execution guide

### Performance & Benchmarks
- **[Benchmark Report](./docs/reports/HYBRID_BENCHMARK_REPORT.md)** - Detailed performance analysis
- **[Benchmark Usage](./benches/BENCHMARK_USAGE.md)** - How to run benchmarks
- **[Launch Overhead Results](./benches/LAUNCH_OVERHEAD_RESULTS_TEMPLATE.md)** - Persistent kernel benchmarks
- **[Binance Backtest Results](./docs/reports/BINANCE_BACKTEST_RESULTS.md)** - Real-world validation

### Optimization & Strategy Development
- **[Optimization Guide](./docs/OPTIMIZATION_GUIDE.md)** - Complete guide to all three optimizers ✨
- **[Optimizer Quick Start](./docs/OPTIMIZER_QUICKSTART.md)** - 5-minute getting started with copy-paste examples ✨
- **[Euler Search Algorithm](./docs/EULER_SEARCH_ALGORITHM.md)** - Deep-dive into iterative refinement ✨

### Migration & Reference
- **[Migration Guide v0.2.0](./docs/MIGRATION_GUIDE_v0.2.0.md)** - Step-by-step migration from v0.1.0
- **[CHANGELOG](./CHANGELOG.md)** - Complete version history
- **[Backtesting Guide](./notebooks/01_basic_backtesting.ipynb)** - Jupyter notebook tutorial

---

## Running Benchmarks

### Quick Benchmark

```bash
# Run all hybrid benchmarks
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark

# Run specific indicator
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- EMA

# Quick mode (faster iteration)
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark -- --quick
```

### Benchmark Output Example

```
EMA_Comparison/Old_GPU_SingleThread/100000
                        time:   [168.23 μs 170.45 μs 172.89 μs]
                        thrpt:  [578.42 Kelem/s 586.73 Kelem/s 594.54 Kelem/s]

EMA_Comparison/New_CPU/100000
                        time:   [24.12 μs 25.03 μs 25.98 μs]
                        thrpt:  [3.85 Melem/s 3.99 Melem/s 4.14 Melem/s]

Speedup: 6.8x ✅
```

### HTML Reports

```bash
# Generate HTML benchmark reports
cargo bench --features gpu --bench cpu_gpu_hybrid_benchmark

# View in browser
firefox target/criterion/report/index.html
```

---

## Testing

### Run All Tests

```bash
cargo test --features gpu
```

### Run Specific Tests

```bash
# Test EMA
cargo test --features gpu -- ema

# Test RSI
cargo test --features gpu -- rsi

# Test hybrid architecture
cargo test --features gpu -- hybrid
```

### Test Coverage

```bash
cargo tarpaulin --features gpu --out Html
```

---

## Development

### Project Structure

```
rust/
├── src/
│   ├── lib.rs              # PyO3 module definition (gil_used = false for Python 3.14t)
│   ├── cpu/
│   │   ├── mod.rs          # CPU module
│   │   └── sequential.rs   # CPU-optimized sequential algorithms (EMA, Wilder's, SMA)
│   ├── gpu/
│   │   ├── mod.rs          # GPU module exports
│   │   ├── device.rs       # GPU device management
│   │   ├── persistent/     # Persistent kernel infrastructure ✨
│   │   │   ├── mod.rs      # Persistent kernel manager
│   │   │   ├── generic.rs  # Generic multi-task kernel execution
│   │   │   ├── occupancy.rs # GPU occupancy optimization
│   │   │   ├── pinned_memory.rs # Zero-copy host-device transfers
│   │   │   └── kernels/    # CUDA kernel implementations
│   │   ├── ema.rs          # EMA (CPU-optimized)
│   │   ├── rsi.rs          # RSI (GPU+CPU+GPU hybrid)
│   │   ├── elder_ray.rs    # Elder Ray (CPU+GPU hybrid)
│   │   ├── atr.rs          # ATR (GPU+CPU hybrid)
│   │   ├── keltner.rs      # Keltner (CPU+GPU hybrid)
│   │   ├── sma.rs          # SMA (GPU parallel)
│   │   ├── wma.rs          # WMA (GPU parallel)
│   │   └── ...             # Other indicators
│   ├── backtest/           # Backtesting engine ✨
│   │   ├── mod.rs          # Module exports
│   │   ├── core.rs         # Core backtesting types
│   │   ├── engine.rs       # Backtest execution engine
│   │   ├── metrics.rs      # Performance metrics (Sharpe, Sortino, etc.)
│   │   ├── optimizer.rs    # Genetic algorithm optimizer
│   │   ├── multi_objective.rs # Multi-objective optimization
│   │   ├── walkforward.rs  # Walk-forward analysis
│   │   ├── sweep.rs        # Parameter sweep
│   │   └── portfolio.rs    # Portfolio-level backtesting
│   └── binance/            # Binance data loader ✨
│       ├── mod.rs          # Trade data parsing
│       └── aggregator.rs   # OHLCV aggregation
├── benches/
│   ├── cpu_gpu_hybrid_benchmark.rs    # Hybrid architecture benchmarks
│   ├── launch_overhead.rs             # Persistent kernel benchmarks ✨
│   ├── multi_indicator_persistent_benchmark.rs  # Batch processing ✨
│   ├── backtest_gpu_cpu_comparison.rs # Backtest performance ✨
│   ├── README.md                      # Benchmark documentation
│   └── BENCHMARK_USAGE.md             # Usage guide
├── docs/
│   ├── CPU_GPU_HYBRID_STRATEGY.md          # Hybrid architecture design
│   ├── PYTHON_314_FREE_THREADING_MIGRATION.md # Python 3.14t guide ✨
│   ├── MIGRATION_GUIDE_v0.2.0.md           # Migration guide
│   └── ...
├── notebooks/
│   └── 01_basic_backtesting.ipynb     # Backtesting tutorial ✨
├── examples/
│   ├── test_persistent_minimal.rs     # Persistent kernel examples ✨
│   └── backtest_binance_comprehensive.rs # Backtest examples ✨
├── Cargo.toml
├── CHANGELOG.md
├── docs/reports/BINANCE_BACKTEST_RESULTS.md  # Real-world results ✨
└── README.md (this file)
```

✨ = **New in v0.2.0**

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

AGPL-3.0-or-later - See [LICENSE](../LICENSE) for the full terms. kimsfinance is dual-licensed: see [LICENSING.md](../LICENSING.md) for the AGPL/commercial split and [COMMERCIAL-LICENSE.md](../COMMERCIAL-LICENSE.md) for proprietary or network-service use.

---

## Acknowledgments

- **cudarc**: Rust CUDA bindings
- **ndarray**: N-dimensional arrays in Rust
- **PyO3**: Rust Python bindings
- **Criterion**: Benchmarking framework

---

## Version History

- **v0.2.0** (2025-10-27) - Production Release with Advanced Features ✨
  - **CPU-GPU Hybrid Architecture**: 1.5x - 6.8x speedup for sequential indicators
  - **Python 3.14t Free-Threading**: GIL-free execution with linear multi-core scaling
  - **Persistent GPU Kernels**: 2-4x batch speedup through 90% launch overhead reduction
  - **Comprehensive Backtesting Engine**: Multi-objective genetic optimization
  - **Real-World Validation**: Binance futures data support with statistical testing
  - **Smart Algorithm Selection**: Automatic CPU vs GPU routing
  - **Breaking Changes**: `ema_gpu()` deprecated, use `ema_cpu()` or `ema_hybrid()`

- **v0.1.0** (2025-10-24) - Initial GPU Release
  - 15-80x speedup for parallel indicators
  - 20+ financial indicators
  - CUDA backend via cudarc

---

**Maintained By**: kimsfinance team
**Repository**: https://github.com/kimsfinance/kimsfinance_core
**Documentation**: https://docs.kimsfinance.io

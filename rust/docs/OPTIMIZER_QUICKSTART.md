# Optimizer Quick Start

**Get started with GPU-accelerated parameter optimization in 5 minutes**

## Table of Contents

- [Installation](#installation)
- [Quick Example: Grid Search](#quick-example-grid-search)
- [Quick Example: Euler Search](#quick-example-euler-search)
- [Quick Example: Genetic Algorithm](#quick-example-genetic-algorithm)
- [Common Recipes](#common-recipes)
- [Next Steps](#next-steps)

---

## Installation

### Rust (with GPU support)

```bash
# Clone repository
git clone https://github.com/kimsfinance/kimsfinance_core.git
cd kimsfinance_core/rust

# Build with GPU features
cargo build --release --features gpu

# Run examples
cargo run --example grid_search_demo --features gpu --release
```

### Python (via PyO3)

```bash
pip install kimsfinance-core
```

---

## Quick Example: Grid Search

**Use case**: Small parameter space, need exact optimum

### Rust

```rust
use kimsfinance_core::backtest::{
    GridSearchOptimizer, ParameterGrid, ParameterRange, BacktestConfig,
};
use kimsfinance_core::backtest::batch::StrategyType;
use kimsfinance_core::gpu::device::GpuDevice;
use std::sync::Arc;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize GPU
    let device = Arc::new(GpuDevice::new()?);

    // 2. Load OHLCV data (10K candles)
    let n = 10_000;
    let timestamps: Vec<i64> = (0..n).map(|i| i as i64 * 60).collect();
    let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 / 100.0)).collect());
    let open = close.clone();
    let high = &close + 1.0;
    let low = &close - 1.0;
    let volume = Array1::from_elem(n, 1000.0);

    // 3. Define parameter grid (150 combinations)
    let mut grid = ParameterGrid::new();
    grid.add_range("rsi_period", ParameterRange::Int { min: 10, max: 20, step: 2 });
    grid.add_range("buy_threshold", ParameterRange::Float { min: 20.0, max: 40.0, step: 5.0 });
    grid.add_range("sell_threshold", ParameterRange::Float { min: 60.0, max: 80.0, step: 5.0 });

    // 4. Create optimizer
    let optimizer = GridSearchOptimizer::new()
        .batch_size(500)
        .progress_interval(1);

    // 5. Run optimization
    let result = optimizer.optimize(
        device,
        StrategyType::RsiCrossover,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &grid,
        BacktestConfig::default(),
    )?;

    // 6. Print results
    println!("Best Parameters:");
    for (key, val) in &result.best_parameters {
        println!("  {}: {:.2}", key, val);
    }
    println!("Best Sharpe: {:.2}", result.best_result.sharpe_ratio);

    Ok(())
}
```

### Python

```python
import kimsfinance_core as kf
import numpy as np

# 1. Load OHLCV data
n = 10_000
timestamps = np.arange(n, dtype=np.int64) * 60
close = np.linspace(100, 200, n)
open_prices = close.copy()
high = close + 1.0
low = close - 1.0
volume = np.full(n, 1000.0)

# 2. Define parameter grid
param_grid = {
    "rsi_period": {"type": "int", "min": 10, "max": 20, "step": 2},
    "buy_threshold": {"type": "float", "min": 20.0, "max": 40.0, "step": 5.0},
    "sell_threshold": {"type": "float", "min": 60.0, "max": 80.0, "step": 5.0},
}

# 3. Run grid search
result = kf.grid_search_optimize(
    timestamps=timestamps,
    open=open_prices,
    high=high,
    low=low,
    close=close,
    volume=volume,
    param_grid=param_grid,
    strategy_type="rsi_crossover",
    batch_size=500,
)

# 4. Print results
print(f"Best Parameters: {result['best_parameters']}")
print(f"Best Sharpe: {result['best_fitness']:.2f}")
```

**Expected output**:
```
Best Parameters:
  rsi_period: 14.00
  buy_threshold: 30.00
  sell_threshold: 70.00
Best Sharpe: 1.85
Total time: 0.52s
```

---

## Quick Example: Euler Search

**Use case**: Medium parameter space, need fast convergence

### Rust

```rust
use kimsfinance_core::backtest::{EulerSearchOptimizer, StrategyType, BacktestConfig};
use kimsfinance_core::gpu::device::GpuDevice;
use std::sync::Arc;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize GPU
    let device = Arc::new(GpuDevice::new()?);

    // 2. Load OHLCV data (same as grid search)
    let n = 10_000;
    let timestamps: Vec<i64> = (0..n).map(|i| i as i64 * 60).collect();
    let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 / 100.0)).collect());
    let open = close.clone();
    let high = &close + 1.0;
    let low = &close - 1.0;
    let volume = Array1::from_elem(n, 1000.0);

    // 3. Create optimizer
    let mut optimizer = EulerSearchOptimizer::new(device.clone())
        .segment_amount(4)      // QuantConnect default
        .max_iterations(20)
        .batch_size(1000)
        .early_stopping_patience(Some(3));

    // 4. Define parameter search space
    optimizer.add_parameter("rsi_period", 5.0, 30.0, 5.0, 1.0);
    optimizer.add_parameter("buy_threshold", 20.0, 40.0, 5.0, 1.0);
    optimizer.add_parameter("sell_threshold", 60.0, 80.0, 5.0, 1.0);

    // 5. Run optimization
    let result = optimizer.optimize(
        StrategyType::RsiCrossover,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        BacktestConfig::default(),
    )?;

    // 6. Print results
    println!("Best Parameters:");
    for (key, val) in &result.best_parameters {
        println!("  {}: {:.2}", key, val);
    }
    println!("Best Sharpe: {:.2}", result.best_fitness);
    println!("Converged: {}", result.is_converged());
    println!("Iterations: {}", result.iterations);
    println!("Total Evaluations: {}", result.total_evaluations);

    Ok(())
}
```

### Python

```python
import kimsfinance_core as kf
import numpy as np

# 1. Load OHLCV data (same as grid search)
n = 10_000
timestamps = np.arange(n, dtype=np.int64) * 60
close = np.linspace(100, 200, n)
open_prices = close.copy()
high = close + 1.0
low = close - 1.0
volume = np.full(n, 1000.0)

# 2. Define parameter search space
parameters = [
    ("rsi_period", 5.0, 30.0, 5.0, 1.0),
    ("buy_threshold", 20.0, 40.0, 5.0, 1.0),
    ("sell_threshold", 60.0, 80.0, 5.0, 1.0),
]

# 3. Run Euler search
result = kf.euler_search_optimize(
    timestamps=timestamps,
    open=open_prices,
    high=high,
    low=low,
    close=close,
    volume=volume,
    parameters=parameters,
    strategy_type="rsi_crossover",
    segment_amount=4,
    max_iterations=20,
    batch_size=1000,
)

# 4. Print results
print(f"Best Parameters: {result['best_parameters']}")
print(f"Best Sharpe: {result['best_fitness']:.2f}")
print(f"Converged: {result['is_converged']}")
print(f"Iterations: {result['iterations']}")
```

**Expected output**:
```
🔍 Starting Euler Search optimization:
   Parameters: 3
   Segment amount: 4
   Max iterations: 15

Iteration 0: 125 evaluations, step sizes: [...]
   Best fitness: 1.250

Iteration 1: 125 evaluations, step sizes: [...]
   Best fitness: 1.680

✓ Early stopping: no improvement for 3 iterations

Best Parameters:
  rsi_period: 14.50
  buy_threshold: 32.50
  sell_threshold: 67.50
Best Sharpe: 1.85
Converged: True
Iterations: 6
Total Evaluations: 750
Total time: 2.1s
```

---

## Quick Example: Genetic Algorithm

**Use case**: Large parameter space, approximate solution

### Rust

```rust
use kimsfinance_core::backtest::{GeneticOptimizer, BacktestEngine, ParameterGrid, ParameterRange};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create backtest engine
    let engine = BacktestEngine::new(10_000.0)?;

    // 2. Load OHLCV data
    let n = 10_000;
    let timestamps: Vec<i64> = (0..n).map(|i| i as i64 * 60).collect();
    let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 / 100.0)).collect());
    let open = close.clone();
    let high = &close + 1.0;
    let low = &close - 1.0;
    let volume = Array1::from_elem(n, 1000.0);

    // 3. Define parameter grid
    let mut grid = ParameterGrid::new();
    grid.add_range("rsi_period", ParameterRange::Int { min: 10, max: 20, step: 1 });
    grid.add_range("buy_threshold", ParameterRange::Float { min: 20.0, max: 40.0, step: 1.0 });
    grid.add_range("sell_threshold", ParameterRange::Float { min: 60.0, max: 80.0, step: 1.0 });

    // 4. Create optimizer
    let optimizer = GeneticOptimizer::new()
        .population_size(100)
        .generations(50)
        .mutation_rate(0.1)
        .crossover_rate(0.8)
        .fp8_exploration_ratio(0.8);

    // 5. Define strategy (simplified for example)
    // In real code, implement Strategy trait
    let strategy = /* your strategy */;

    // 6. Run optimization
    let result = optimizer.optimize(
        &engine,
        &strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &grid,
    )?;

    // 7. Print results
    println!("Best Parameters:");
    for (key, val) in &result.best_parameters {
        println!("  {}: {:.2}", key, val);
    }
    println!("Best Fitness: {:.2}", result.best_fitness);
    println!("Converged at generation: {:?}", result.convergence_stats.generation_converged);

    Ok(())
}
```

### Python

```python
import kimsfinance_core as kf
import numpy as np

# 1. Load OHLCV data
n = 10_000
timestamps = np.arange(n, dtype=np.int64) * 60
close = np.linspace(100, 200, n)
open_prices = close.copy()
high = close + 1.0
low = close - 1.0
volume = np.full(n, 1000.0)

# 2. Define parameter ranges
param_grid = {
    "rsi_period": {"type": "int", "min": 10, "max": 20, "step": 1},
    "buy_threshold": {"type": "float", "min": 20.0, "max": 40.0, "step": 1.0},
    "sell_threshold": {"type": "float", "min": 60.0, "max": 80.0, "step": 1.0},
}

# 3. Run genetic optimization
result = kf.genetic_optimize(
    timestamps=timestamps,
    open=open_prices,
    high=high,
    low=low,
    close=close,
    volume=volume,
    param_grid=param_grid,
    strategy_type="rsi_crossover",
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    fp8_exploration_ratio=0.8,
)

# 4. Print results
print(f"Best Parameters: {result['best_parameters']}")
print(f"Best Fitness: {result['best_fitness']:.2f}")
print(f"Converged at generation: {result['converged_generation']}")
```

**Expected output**:
```
Genetic Optimizer: 100 individuals, 50 generations
  Adaptive mutation enabled (initial rate: 0.1000)
  GPU batch evaluation enabled (threshold: 50)

Gen 1/50 [FP8]: Fitness=1.234, Diversity=0.412, Mutation=0.1000
Gen 10/50 [FP8]: Fitness=1.678, Diversity=0.185, Mutation=0.1200 ↑
Gen 20/50 [FP8]: Fitness=1.803, Diversity=0.098, Mutation=0.1400 ↑
Gen 30/50 [FP8]: Fitness=1.842, Diversity=0.052, Mutation=0.1680 ↑
Gen 40/50 [FP64]: Fitness=1.851, Diversity=0.023, Mutation=0.1512 ↓
Gen 50/50 [FP64]: Fitness=1.856, Diversity=0.015, Mutation=0.1360 ↓

Best Parameters:
  rsi_period: 14.00
  buy_threshold: 31.50
  sell_threshold: 68.75
Best Fitness: 1.86
Converged at generation: 45
Total time: 18.5s
```

---

## Common Recipes

### Recipe 1: Load Real Data (Binance)

```rust
use kimsfinance_core::binance::{load_trades, aggregate_to_ohlcv, Timeframe};

fn load_binance_data() -> Result<OhlcvData, Box<dyn std::error::Error>> {
    // Load raw trades from parquet
    let trades = load_trades("BTCUSDT-trades-2024-05-31.zip")?;

    // Aggregate to OHLCV (5-minute bars)
    let ohlcv = aggregate_to_ohlcv(&trades, Timeframe::Min5)?;

    Ok(ohlcv)
}
```

```python
import kimsfinance_core as kf

# Load Binance trades
trades = kf.load_binance_trades("BTCUSDT-trades-2024-05-31.zip")

# Aggregate to OHLCV
ohlcv = kf.aggregate_to_ohlcv(trades, timeframe="5m")

timestamps = ohlcv["timestamps"]
open_prices = ohlcv["open"]
high = ohlcv["high"]
low = ohlcv["low"]
close = ohlcv["close"]
volume = ohlcv["volume"]
```

### Recipe 2: Compare Optimizers

```rust
// Grid Search (exhaustive)
let grid_result = GridSearchOptimizer::new()
    .batch_size(500)
    .optimize(/* ... */)?;

// Euler Search (iterative refinement)
let mut euler = EulerSearchOptimizer::new(device.clone());
euler.add_parameter("rsi_period", 5.0, 30.0, 5.0, 1.0);
let euler_result = euler.optimize(/* ... */)?;

// Compare results
println!("Grid Search: Sharpe={:.2}, Time={:.2}s",
    grid_result.best_fitness, grid_result.total_time_ms / 1000.0);
println!("Euler Search: Sharpe={:.2}, Time={:.2}s, Evals={}",
    euler_result.best_fitness, euler_result.total_time_ms / 1000.0,
    euler_result.total_evaluations);
```

### Recipe 3: Walk-Forward Validation

```rust
use kimsfinance_core::backtest::walkforward_optimize;

// Split data into 70% in-sample, 30% out-of-sample
let wf_result = walkforward_optimize(
    &optimizer,
    &data,
    in_sample_ratio: 0.7,
    num_folds: 5,
)?;

println!("In-Sample Sharpe: {:.2}", wf_result.in_sample_sharpe);
println!("Out-of-Sample Sharpe: {:.2}", wf_result.out_of_sample_sharpe);
println!("Overfitting Ratio: {:.2}", wf_result.overfitting_ratio());
```

### Recipe 4: Multi-Objective Optimization

```rust
use kimsfinance_core::backtest::{MultiObjectiveOptimizer, Objective};

let objectives = vec![
    Objective::MaximizeSharpe,
    Objective::MinimizeDrawdown,
    Objective::MaximizeProfitFactor,
];

let pareto_front = MultiObjectiveOptimizer::new()
    .objectives(objectives)
    .optimize(&optimizer, &data)?;

// Get best compromise solution
let best = pareto_front.best_compromise()?;
println!("Compromise: Sharpe={:.2}, Drawdown={:.2}%",
    best.sharpe, best.max_drawdown * 100.0);
```

### Recipe 5: Custom Fitness Function

```rust
// Instead of default Sharpe ratio, use custom fitness
fn custom_fitness(result: &BacktestResult) -> f64 {
    let sharpe = result.sharpe_ratio;
    let drawdown = result.max_drawdown;
    let win_rate = result.win_rate;

    // Weighted combination
    sharpe * 0.5 - drawdown * 20.0 + win_rate * 0.3
}

// Apply custom fitness (example - actual API may vary)
// See documentation for specific implementation
```

### Recipe 6: Parallel Multi-Symbol Optimization

```rust
use rayon::prelude::*;

let symbols = vec!["BTCUSDT", "ETHUSDT", "BNBUSDT"];

// Optimize each symbol in parallel
let results: Vec<OptimizerResult> = symbols.par_iter()
    .map(|symbol| {
        let data = load_symbol_data(symbol)?;
        let optimizer = EulerSearchOptimizer::new(device.clone());
        optimizer.optimize(/* ... */)
    })
    .collect::<Result<_, _>>()?;

// Aggregate best parameters across symbols
for (symbol, result) in symbols.iter().zip(results.iter()) {
    println!("{}: Sharpe={:.2}", symbol, result.best_fitness);
}
```

---

## Next Steps

### Comprehensive Guides

- **[Optimization Guide](./OPTIMIZATION_GUIDE.md)** - Complete guide to all three optimizers
  - When to use each optimizer
  - Performance characteristics
  - Troubleshooting

- **[Euler Search Algorithm](./EULER_SEARCH_ALGORITHM.md)** - Deep-dive into iterative refinement
  - Mathematical formulation
  - Convergence analysis
  - Comparison with other methods

### Examples

Run comprehensive examples:

```bash
# Grid Search demo
cargo run --example grid_search_demo --features gpu --release

# Euler Search demo
cargo run --example euler_search_demo --features gpu --release

# Genetic Algorithm demo
cargo run --example fp8_genetic_optimizer --features gpu --release

# Comprehensive backtest
cargo run --example comprehensive_backtest_demo --features gpu --release
```

### API Documentation

Generate and browse full API docs:

```bash
cargo doc --features gpu --open
```

### Benchmarks

Run performance benchmarks:

```bash
# Compare all three optimizers
cargo bench --features gpu --bench backtest_gpu_cpu_comparison

# Grid search scaling
cargo bench --features gpu grid_search

# Genetic algorithm precision
cargo bench --features gpu genetic_optimizer
```

### Community & Support

- **GitHub**: https://github.com/kimsfinance/kimsfinance_core
- **Documentation**: https://docs.kimsfinance.io
- **Issues**: Report bugs or request features
- **Discussions**: Ask questions, share strategies

---

**Pro Tips**:

1. **Start Small**: Test with 100-1000 candles first, then scale up
2. **Monitor GPU**: Use `nvidia-smi` to watch VRAM usage
3. **Validate Results**: Always use walk-forward or out-of-sample testing
4. **Save Parameters**: Store best parameters for production use
5. **Iterate**: Refine parameter ranges based on initial results

---

**Last Updated**: 2025-11-04
**Version**: 0.2.0
**Maintained By**: kimsfinance team

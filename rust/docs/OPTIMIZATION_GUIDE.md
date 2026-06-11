# Parameter Optimization Guide

**Complete guide to GPU-accelerated strategy parameter optimization in kimsfinance**

## Table of Contents

- [Overview](#overview)
- [When to Use Each Optimizer](#when-to-use-each-optimizer)
- [Grid Search](#grid-search)
- [Euler Search](#euler-search)
- [Genetic Algorithm](#genetic-algorithm)
- [Comparison Table](#comparison-table)
- [Python API](#python-api)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Benchmarks](#benchmarks)

---

## Overview

Parameter optimization is the process of finding the best parameter values for a trading strategy to maximize a fitness metric (e.g., Sharpe ratio). kimsfinance provides **three GPU-accelerated optimizers**, each designed for different use cases:

1. **Grid Search** - Exhaustive search for small parameter spaces
2. **Euler Search** - Iterative refinement for medium parameter spaces (QuantConnect-style)
3. **Genetic Algorithm** - Evolutionary optimization for large parameter spaces

### Why GPU Acceleration Matters

Traditional CPU backtesting evaluates strategies sequentially. GPU batch backtesting evaluates hundreds of strategies in parallel:

- **Grid Search**: 40x faster (1000 combos in <3s)
- **Euler Search**: 90% fewer evaluations than grid search
- **Genetic Algorithm**: 100x+ faster with FP8 tensor cores

All three optimizers use the same GPU batch backtesting infrastructure (`BatchBacktestSweep`) for parallel evaluation.

---

## When to Use Each Optimizer

### Grid Search
**Use when you need guaranteed global optimum for small parameter spaces**

✅ **Best for**:
- Small parameter spaces (≤1000 combinations)
- Need exact global optimum
- Quick validation of parameter ranges
- Final refinement after Euler/Genetic search

❌ **Avoid when**:
- Large parameter spaces (>10,000 combinations)
- Approximate solution acceptable
- Time-constrained optimization

**Example**: RSI strategy with 3 parameters (6×5×5 = 150 combinations)

### Euler Search
**Use when you need fast convergence for medium parameter spaces**

✅ **Best for**:
- Medium parameter spaces (100-10,000 combinations)
- Fast convergence (5-10 iterations typical)
- Iterative refinement around promising regions
- When grid search is too slow

❌ **Avoid when**:
- Need guaranteed global optimum (can get stuck in local optima)
- Parameter space has many local optima
- Extremely large parameter spaces

**Example**: MA crossover with 5 parameters (would be 100,000+ grid combinations, but Euler does ~1000 evaluations)

### Genetic Algorithm
**Use when you need flexible optimization for large parameter spaces**

✅ **Best for**:
- Large parameter spaces (>10,000 combinations)
- Approximate solutions acceptable
- Multi-modal optimization (many local optima)
- Continuous parameter ranges

❌ **Avoid when**:
- Need guaranteed global optimum
- Small parameter space (grid search is faster)
- Require fast convergence (<1 minute)

**Example**: Multi-indicator strategy with 8+ parameters (billions of combinations)

---

## Grid Search

### Algorithm

Grid Search performs **exhaustive evaluation** of all parameter combinations:

1. Generate Cartesian product of all parameter ranges
2. Split into batches (100-1000 per batch)
3. Evaluate each batch on GPU using `BatchBacktestSweep`
4. Find combination with highest fitness

**Complexity**: O(n^d) where n = steps per parameter, d = dimensions

### Usage (Rust)

```rust
use kimsfinance_core::backtest::{
    GridSearchOptimizer, ParameterGrid, ParameterRange, BacktestConfig,
};
use kimsfinance_core::backtest::batch::StrategyType;
use kimsfinance_core::gpu::device::GpuDevice;
use std::sync::Arc;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let device = Arc::new(GpuDevice::new()?);

    // Load or generate OHLCV data
    let timestamps: Vec<i64> = vec![/* ... */];
    let open = Array1::from_vec(vec![/* ... */]);
    let high = Array1::from_vec(vec![/* ... */]);
    let low = Array1::from_vec(vec![/* ... */]);
    let close = Array1::from_vec(vec![/* ... */]);
    let volume = Array1::from_vec(vec![/* ... */]);

    // Define parameter grid
    let mut grid = ParameterGrid::new();

    // RSI period: 10, 12, 14, 16, 18, 20 (6 values)
    grid.add_range("rsi_period", ParameterRange::Int {
        min: 10,
        max: 20,
        step: 2,
    });

    // Buy threshold: 20, 25, 30, 35, 40 (5 values)
    grid.add_range("buy_threshold", ParameterRange::Float {
        min: 20.0,
        max: 40.0,
        step: 5.0,
    });

    // Sell threshold: 60, 65, 70, 75, 80 (5 values)
    grid.add_range("sell_threshold", ParameterRange::Float {
        min: 60.0,
        max: 80.0,
        step: 5.0,
    });

    // Total: 6 × 5 × 5 = 150 combinations

    // Create optimizer
    let optimizer = GridSearchOptimizer::new()
        .batch_size(500)         // GPU batch size (100-1000)
        .progress_interval(1);   // Print progress every batch

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,      // 0.1%
        slippage: 0.0005,        // 0.05%
        execution_latency_ms: 10,
        use_gpu: true,
        force_cpu: false,
    };

    // Run optimization
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
        config,
    )?;

    // Print results
    println!("Best Parameters:");
    for (key, value) in &result.best_parameters {
        println!("  {}: {:.2}", key, value);
    }
    println!("Best Sharpe Ratio: {:.2}", result.best_result.sharpe_ratio);
    println!("Total Combinations: {}", grid.size());

    Ok(())
}
```

### Performance Characteristics

**Target**: 1000 combinations × 10K candles in <3 seconds (40x vs sequential)

**Scaling**:
```
100 combos × 10K candles:    ~0.3s
500 combos × 10K candles:    ~1.5s
1000 combos × 10K candles:   ~2.8s
5000 combos × 10K candles:   ~14s
```

**GPU Utilization**: >90% (batch processing)

### When Results Are Ready

Grid search returns:
- `best_parameters`: HashMap<String, f64> with optimal values
- `best_fitness`: Sharpe ratio with drawdown penalty
- `best_result`: Full backtest result (equity curve, trades, metrics)
- `convergence_history`: Best fitness per batch (for monitoring)

---

## Euler Search

### Algorithm

Euler Search uses **iterative grid refinement** (QuantConnect algorithm):

1. **Test Grid**: Evaluate N points across current search space
2. **Find Best**: Identify parameter set with highest fitness
3. **Refine**: Reduce step size and narrow boundaries around best
4. **Repeat**: Until step size falls below minimum threshold

**Refinement Formula** (each iteration):
```
new_step = max(min_step, current_step / segment_amount)
fractal = new_step × (segment_amount / 2)
new_range = [best_value ± fractal]
```

**Complexity**: O(iterations × batch_size) - typically 90% fewer evaluations than grid search

### Usage (Rust)

```rust
use kimsfinance_core::backtest::{
    EulerSearchOptimizer, StrategyType, BacktestConfig,
};
use kimsfinance_core::gpu::device::GpuDevice;
use std::sync::Arc;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let device = Arc::new(GpuDevice::new()?);

    // Load OHLCV data (same as Grid Search)
    let timestamps: Vec<i64> = vec![/* ... */];
    let open = Array1::from_vec(vec![/* ... */]);
    let high = Array1::from_vec(vec![/* ... */]);
    let low = Array1::from_vec(vec![/* ... */]);
    let close = Array1::from_vec(vec![/* ... */]);
    let volume = Array1::from_vec(vec![/* ... */]);

    // Create optimizer
    let mut optimizer = EulerSearchOptimizer::new(device.clone())
        .segment_amount(4)                // QuantConnect default
        .max_iterations(20)               // Stop after 20 iterations
        .batch_size(1000)                 // GPU batch size
        .early_stopping_patience(Some(3)); // Stop if no improvement for 3 iterations

    // Define parameter search space
    // (name, min, max, initial_step, min_step)
    optimizer.add_parameter("rsi_period", 5.0, 30.0, 5.0, 1.0);
    optimizer.add_parameter("buy_threshold", 20.0, 40.0, 5.0, 1.0);
    optimizer.add_parameter("sell_threshold", 60.0, 80.0, 5.0, 1.0);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        execution_latency_ms: 10,
        use_gpu: true,
        force_cpu: false,
    };

    // Run optimization
    let result = optimizer.optimize(
        StrategyType::RsiCrossover,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        config,
    )?;

    // Print results
    println!("Best Parameters:");
    for (key, value) in &result.best_parameters {
        println!("  {}: {:.2}", key, value);
    }
    println!("Best Sharpe Ratio: {:.2}", result.best_fitness);
    println!("Converged: {}", result.is_converged());
    println!("Iterations: {}", result.iterations);
    println!("Total Evaluations: {}", result.total_evaluations);

    // Calculate speedup vs grid search
    let grid_evaluations = 6 * 5 * 5; // Example grid size
    println!("Speedup vs Grid Search: {:.1}x",
        grid_evaluations as f64 / result.total_evaluations as f64);

    Ok(())
}
```

### Performance Characteristics

**Target**: <250ms per iteration for 1000 parameter sets

**Typical Convergence**: 5-10 iterations (90% fewer evaluations than grid search)

**Scaling**:
```
3 parameters × 10 iterations:  ~500-1000 evaluations total
5 parameters × 8 iterations:   ~800-1500 evaluations total
```

**GPU Utilization**: >90% (batch processing)

### Convergence Detection

Euler Search automatically stops when:
1. **All parameters converge**: step_size ≤ min_step
2. **Early stopping**: No fitness improvement for N iterations
3. **Max iterations**: Reaches max_iterations limit

Check convergence:
```rust
if result.is_converged() {
    println!("✓ Converged to local optimum");
} else {
    println!("✗ Stopped at max iterations (may not be optimal)");
}
```

---

## Genetic Algorithm

### Algorithm

Genetic Algorithm uses **evolutionary optimization**:

1. **Initialize**: Random population of parameter sets
2. **Evaluate**: Fitness for all individuals (GPU batch)
3. **Select**: Tournament selection for parents
4. **Crossover**: Combine parent parameters
5. **Mutate**: Add random variations
6. **Repeat**: Until convergence or max generations

**Features**:
- **Hybrid Precision**: FP8 during exploration (80%), FP64 during refinement (20%)
- **Adaptive Mutation**: Adjusts based on population diversity
- **Elite Preservation**: Top 10% survive unchanged
- **Convergence Detection**: Multi-criteria early stopping

**Complexity**: O(generations × population_size)

### Usage (Rust)

```rust
use kimsfinance_core::backtest::{
    GeneticOptimizer, BacktestEngine, ParameterGrid, ParameterRange,
};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create backtest engine
    let engine = BacktestEngine::new(10_000.0)?; // $10k initial capital

    // Define parameter grid (search space)
    let mut grid = ParameterGrid::new();
    grid.add_range("rsi_period", ParameterRange::Int {
        min: 10,
        max: 20,
        step: 1,
    });
    grid.add_range("buy_threshold", ParameterRange::Float {
        min: 20.0,
        max: 40.0,
        step: 1.0,
    });
    grid.add_range("sell_threshold", ParameterRange::Float {
        min: 60.0,
        max: 80.0,
        step: 1.0,
    });

    // Create optimizer
    let optimizer = GeneticOptimizer::new()
        .population_size(100)           // 100 individuals per generation
        .generations(50)                // 50 generations
        .mutation_rate(0.1)             // 10% mutation probability
        .crossover_rate(0.8)            // 80% crossover probability
        .fp8_exploration_ratio(0.8)     // 80% FP8, 20% FP64
        .elitism_rate(0.1)              // Top 10% survive
        .tournament_size(5);            // Tournament selection size

    // Load OHLCV data
    let timestamps: Vec<i64> = vec![/* ... */];
    let open = Array1::from_vec(vec![/* ... */]);
    let high = Array1::from_vec(vec![/* ... */]);
    let low = Array1::from_vec(vec![/* ... */]);
    let close = Array1::from_vec(vec![/* ... */]);
    let volume = Array1::from_vec(vec![/* ... */]);

    // Define strategy (must implement Strategy + Clone)
    let strategy = /* your strategy implementation */;

    // Run optimization
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

    // Print results
    println!("Best Parameters:");
    for (key, value) in &result.best_parameters {
        println!("  {}: {:.2}", key, value);
    }
    println!("Best Fitness: {:.2}", result.best_fitness);
    println!("Converged at generation: {:?}", result.convergence_stats.generation_converged);
    println!("Final diversity: {:.4}", result.convergence_stats.final_diversity);

    Ok(())
}
```

### Performance Characteristics

**Target**: 50 generations × 100 individuals = 5000 evaluations

**FP8 Acceleration**:
- FP8 exploration (80% of gens): 4-6x faster
- FP64 refinement (20% of gens): High accuracy
- **Overall speedup**: 2-3x vs pure FP64

**GPU Batch Evaluation** (50+ individuals):
- Single GPU kernel evaluates all individuals
- 20-40x faster than CPU parallel
- Automatic fallback to CPU if GPU unavailable

**Adaptive Mutation**:
- Low diversity (<10%): Increase mutation → more exploration
- High diversity (>30%): Decrease mutation → more exploitation
- Prevents premature convergence

### Convergence Detection

Genetic algorithm stops when **2+ criteria are met**:

1. **Fitness Plateau**: <0.1% improvement over last 15 generations
2. **Low Diversity**: Coefficient of variation < 1%
3. **Consecutive Same**: Best fitness unchanged for 14+ generations

```rust
if result.convergence_stats.generation_converged.is_some() {
    let gen = result.convergence_stats.generation_converged.unwrap();
    println!("✓ Converged early at generation {}", gen);
} else {
    println!("✗ Ran all {} generations", optimizer.generations);
}
```

---

## Comparison Table

| Feature | Grid Search | Euler Search | Genetic Algorithm |
|---------|-------------|--------------|-------------------|
| **Guarantees** | Global optimum | Local optimum | Approximate |
| **Speed** | Slow (exhaustive) | Fast (iterative) | Medium (evolutionary) |
| **Evaluations** | n^d (all combos) | 90% fewer | Generations × population |
| **Best for** | <1000 combos | 100-10K combos | >10K combos |
| **Parameters** | <4 discrete | <6 continuous | Any (8+ ok) |
| **GPU Speedup** | 40x (batch) | 40x (batch) | 100x+ (FP8 + batch) |
| **Convergence** | N/A (always complete) | 5-10 iterations | 20-50 generations |
| **Use When** | Need exact optimum | Need fast convergence | Large search space |
| **Avoid When** | >10K combinations | Many local optima | Need guaranteed optimum |

### Example Problem Sizes

**3-parameter RSI strategy**:
- Grid: 6 × 5 × 5 = 150 combinations → **Grid Search** (0.5s)
- Euler: ~500 evaluations → **Euler Search** (1.5s, but explores more)
- Genetic: 50 × 100 = 5000 evaluations → **Too slow** (15s)

**5-parameter MA crossover**:
- Grid: 10^5 = 100,000 combinations → **Too slow** (5 minutes)
- Euler: ~1,500 evaluations → **Euler Search** (4s)
- Genetic: 50 × 100 = 5000 evaluations → **Also good** (15s)

**8-parameter multi-indicator**:
- Grid: 10^8 = 100 million combinations → **Impossible**
- Euler: ~10,000 evaluations → **Risky** (many local optima)
- Genetic: 100 × 200 = 20,000 evaluations → **Genetic Algorithm** (1 minute)

---

## Python API

### Installation

```bash
pip install kimsfinance-core
```

### Grid Search (Python)

```python
import kimsfinance_core as kf
import numpy as np

# Load OHLCV data
timestamps = np.array([...])  # Unix timestamps
open_prices = np.array([...])
high_prices = np.array([...])
low_prices = np.array([...])
close_prices = np.array([...])
volumes = np.array([...])

# Define parameter grid
param_grid = {
    "rsi_period": {"type": "int", "min": 10, "max": 20, "step": 2},
    "buy_threshold": {"type": "float", "min": 20.0, "max": 40.0, "step": 5.0},
    "sell_threshold": {"type": "float", "min": 60.0, "max": 80.0, "step": 5.0},
}

# Run grid search
result = kf.grid_search_optimize(
    timestamps=timestamps,
    open=open_prices,
    high=high_prices,
    low=low_prices,
    close=close_prices,
    volume=volumes,
    param_grid=param_grid,
    strategy_type="rsi_crossover",
    batch_size=500,
    initial_capital=10000.0,
    trading_fee=0.001,
    slippage=0.0005,
)

print(f"Best Parameters: {result['best_parameters']}")
print(f"Best Sharpe: {result['best_fitness']:.2f}")
print(f"Combinations Evaluated: {result['total_combinations']}")
```

### Euler Search (Python)

```python
import kimsfinance_core as kf
import numpy as np

# Load OHLCV data (same as above)
timestamps = np.array([...])
open_prices = np.array([...])
high_prices = np.array([...])
low_prices = np.array([...])
close_prices = np.array([...])
volumes = np.array([...])

# Define parameter search space
# (name, min, max, initial_step, min_step)
parameters = [
    ("rsi_period", 5.0, 30.0, 5.0, 1.0),
    ("buy_threshold", 20.0, 40.0, 5.0, 1.0),
    ("sell_threshold", 60.0, 80.0, 5.0, 1.0),
]

# Run Euler search
result = kf.euler_search_optimize(
    timestamps=timestamps,
    open=open_prices,
    high=high_prices,
    low=low_prices,
    close=close_prices,
    volume=volumes,
    parameters=parameters,
    strategy_type="rsi_crossover",
    segment_amount=4,
    max_iterations=20,
    batch_size=1000,
    early_stopping_patience=3,
    initial_capital=10000.0,
    trading_fee=0.001,
    slippage=0.0005,
)

print(f"Best Parameters: {result['best_parameters']}")
print(f"Best Sharpe: {result['best_fitness']:.2f}")
print(f"Converged: {result['is_converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Total Evaluations: {result['total_evaluations']}")
```

### Genetic Algorithm (Python)

```python
import kimsfinance_core as kf
import numpy as np

# Load OHLCV data (same as above)
timestamps = np.array([...])
open_prices = np.array([...])
high_prices = np.array([...])
low_prices = np.array([...])
close_prices = np.array([...])
volumes = np.array([...])

# Define parameter ranges
param_grid = {
    "rsi_period": {"type": "int", "min": 10, "max": 20, "step": 1},
    "buy_threshold": {"type": "float", "min": 20.0, "max": 40.0, "step": 1.0},
    "sell_threshold": {"type": "float", "min": 60.0, "max": 80.0, "step": 1.0},
}

# Run genetic optimization
result = kf.genetic_optimize(
    timestamps=timestamps,
    open=open_prices,
    high=high_prices,
    low=low_prices,
    close=close_prices,
    volume=volumes,
    param_grid=param_grid,
    strategy_type="rsi_crossover",
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    fp8_exploration_ratio=0.8,
    elitism_rate=0.1,
    initial_capital=10000.0,
    trading_fee=0.001,
    slippage=0.0005,
)

print(f"Best Parameters: {result['best_parameters']}")
print(f"Best Fitness: {result['best_fitness']:.2f}")
print(f"Converged at generation: {result['converged_generation']}")
print(f"FP8 generations: {result['fp8_generations']}")
print(f"FP64 generations: {result['fp64_generations']}")
```

---

## Performance Tuning

### GPU Batch Size Selection

**Grid Search & Euler Search**:
```rust
let optimizer = GridSearchOptimizer::new()
    .batch_size(500); // Adjust based on VRAM
```

**Guidelines**:
- **100**: Safe for 4GB VRAM
- **500**: Optimal for 8-12GB VRAM (RTX 3500 Ada)
- **1000**: For 16GB+ VRAM or small datasets

**Trade-off**: Larger batches = better GPU utilization but more VRAM usage

### Genetic Algorithm Population Size

```rust
let optimizer = GeneticOptimizer::new()
    .population_size(100)  // Adjust based on problem complexity
    .generations(50);
```

**Guidelines**:
- **50-100**: Simple strategies (3-4 parameters)
- **100-200**: Medium strategies (5-7 parameters)
- **200-500**: Complex strategies (8+ parameters)

**Trade-off**: Larger populations explore more but take longer per generation

### FP8 Exploration Ratio

```rust
let optimizer = GeneticOptimizer::new()
    .fp8_exploration_ratio(0.8); // 80% FP8, 20% FP64
```

**Guidelines**:
- **0.8 (default)**: Balanced speed/accuracy
- **0.9**: Maximum speed (slightly less accurate)
- **0.6-0.7**: More accurate (slower)
- **0.0**: Pure FP64 (no speedup)

**Trade-off**: Higher FP8 ratio = faster but less precise exploration

### Execution Mode Selection

All three optimizers use `BatchBacktestSweep` which supports 3 execution modes:

```rust
// Automatic mode selection (default)
let config = BacktestConfig {
    use_gpu: true,
    force_cpu: false,
    // ... other config
};

// Traditional: 4 separate kernel launches
// Fused: 1 kernel launch (2x faster)
// Async: Triple-buffered (3x faster, requires compute 7.0+)
```

**Mode selection is automatic** based on:
- Dataset size
- GPU compute capability
- Number of strategies in batch

### Parameter Grid Design

**For Grid Search**:
```rust
// ❌ Too many combinations (10^5 = 100,000)
grid.add_range("ma_fast", ParameterRange::Int { min: 5, max: 50, step: 1 });
grid.add_range("ma_slow", ParameterRange::Int { min: 20, max: 200, step: 1 });

// ✅ Reasonable combinations (10 × 19 = 190)
grid.add_range("ma_fast", ParameterRange::Int { min: 5, max: 50, step: 5 });
grid.add_range("ma_slow", ParameterRange::Int { min: 20, max: 200, step: 10 });
```

**Tip**: Start with coarse grid, refine with finer grid around best region

**For Euler Search**:
```rust
// Set appropriate step sizes
optimizer.add_parameter(
    "threshold",
    0.0,        // min
    100.0,      // max
    10.0,       // initial_step (wide exploration)
    0.1,        // min_step (fine-grained convergence)
);
```

**Tip**: `initial_step` should be ~10% of range, `min_step` should be desired precision

---

## Troubleshooting

### Out of Memory Errors

**Symptom**: `GpuError::AllocationError` or CUDA out-of-memory

**Solutions**:
1. Reduce batch size:
   ```rust
   let optimizer = GridSearchOptimizer::new()
       .batch_size(100); // Reduce from 500
   ```

2. Use smaller dataset (fewer candles)

3. Check VRAM usage:
   ```bash
   nvidia-smi
   ```

4. Close other GPU applications

### Slow Convergence

**Euler Search**:

**Symptom**: Takes many iterations (>20) without converging

**Solutions**:
1. Adjust segment amount (trade-off: speed vs precision):
   ```rust
   let optimizer = EulerSearchOptimizer::new(device)
       .segment_amount(6); // Increase for finer grids (slower but more thorough)
   ```

2. Check for local optima:
   ```rust
   // Run multiple times with different starting points
   // Compare results
   ```

3. Increase batch size for better GPU utilization:
   ```rust
   .batch_size(2000) // If VRAM allows
   ```

**Genetic Algorithm**:

**Symptom**: Fitness plateaus early, low diversity

**Solutions**:
1. Increase mutation rate:
   ```rust
   let optimizer = GeneticOptimizer::new()
       .mutation_rate(0.2); // Increase from 0.1
   ```

2. Increase population size:
   ```rust
   .population_size(200) // More diversity
   ```

3. Use island model for better exploration:
   ```rust
   use kimsfinance_core::backtest::IslandGeneticOptimizer;

   let island_optimizer = IslandGeneticOptimizer::new(base_optimizer)
       .num_islands(4)
       .migration_interval(10);
   ```

### Sub-Optimal Results

**Symptom**: Best parameters don't perform well out-of-sample

**Causes**:
1. **Overfitting**: Optimized on too little data
2. **Look-ahead bias**: Strategy uses future information
3. **Poor fitness metric**: Sharpe ratio doesn't capture risk

**Solutions**:
1. **Use Walk-Forward Analysis**:
   ```rust
   use kimsfinance_core::backtest::walkforward_optimize;

   let wf_result = walkforward_optimize(
       &optimizer,
       &data,
       in_sample_ratio: 0.7,  // 70% training, 30% validation
       num_folds: 5,
   )?;
   ```

2. **Add regularization** to fitness function:
   ```rust
   // Penalize excessive trading
   let fitness = sharpe_ratio - 0.01 * num_trades;
   ```

3. **Multi-objective optimization**:
   ```rust
   use kimsfinance_core::backtest::MultiObjectiveOptimizer;

   let objectives = vec![
       Objective::MaximizeSharpe,
       Objective::MinimizeDrawdown,
       Objective::MinimizeTrades,
   ];
   ```

### GPU Not Being Used

**Symptom**: Optimization falls back to CPU, or `force_cpu: false` but using CPU

**Solutions**:
1. Check GPU availability:
   ```bash
   nvidia-smi
   cargo run --features gpu --example test_gpu_detection
   ```

2. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

3. Check population size threshold:
   ```rust
   // GPU batch evaluation requires 50+ individuals
   let optimizer = GeneticOptimizer::new()
       .population_size(50); // Minimum for GPU batch
   ```

4. Verify feature flag:
   ```bash
   cargo build --features gpu
   ```

---

## Benchmarks

All benchmarks run on:
- **GPU**: NVIDIA RTX 3500 Ada (12GB VRAM)
- **CPU**: Intel i9-13980HX (24 cores, 32 threads)
- **Dataset**: 10,000 candles (1 month of 5min bars)

### Grid Search Performance

| Combinations | Time (GPU) | Time (CPU) | Speedup | GPU Utilization |
|--------------|------------|------------|---------|-----------------|
| 100 | 0.3s | 12s | 40x | 88% |
| 500 | 1.5s | 60s | 40x | 92% |
| 1000 | 2.8s | 120s | 43x | 94% |
| 5000 | 14s | 600s | 43x | 95% |

**Batch size**: 500 strategies per GPU call

### Euler Search Performance

| Parameters | Iterations | Evaluations | Time | Speedup vs Grid |
|------------|------------|-------------|------|-----------------|
| 3 | 6 | 486 | 1.5s | 10x (vs 4,860 grid) |
| 4 | 8 | 1,280 | 3.8s | 18x (vs 23,000 grid) |
| 5 | 10 | 2,560 | 7.5s | 40x (vs 100,000 grid) |

**Grid comparison**: Assumes 10 steps per parameter

### Genetic Algorithm Performance

| Population | Generations | Total Evals | Time (FP8) | Time (FP64) | Speedup |
|------------|-------------|-------------|------------|-------------|---------|
| 50 | 20 | 1,000 | 3s | 7s | 2.3x |
| 100 | 50 | 5,000 | 15s | 38s | 2.5x |
| 200 | 100 | 20,000 | 60s | 155s | 2.6x |

**FP8 ratio**: 80% exploration, 20% refinement

### GPU Batch Evaluation Speedup

| Population | Time (Sequential CPU) | Time (Parallel CPU) | Time (GPU Batch) | GPU Speedup |
|------------|----------------------|---------------------|------------------|-------------|
| 20 | 1.8s | 0.15s | 0.08s | 1.9x vs parallel |
| 50 | 4.5s | 0.35s | 0.12s | 2.9x vs parallel |
| 100 | 9.0s | 0.70s | 0.20s | 3.5x vs parallel |
| 200 | 18s | 1.4s | 0.35s | 4.0x vs parallel |

**Note**: GPU batch evaluation automatically kicks in at population ≥50

---

## See Also

- [Euler Search Algorithm Deep-Dive](./EULER_SEARCH_ALGORITHM.md) - Mathematical formulation and convergence analysis
- [Quick Start Guide](./OPTIMIZER_QUICKSTART.md) - 5-minute getting started
- [Batch Backtest API](../src/backtest/batch.rs) - GPU batch backtesting infrastructure
- [Walk-Forward Analysis](../examples/walkforward_demo.rs) - Out-of-sample validation
- [Multi-Objective Optimization](../examples/multi_objective_demo.rs) - Pareto frontier optimization

---

**Last Updated**: 2025-11-04
**Version**: 0.2.0
**Maintained By**: kimsfinance team

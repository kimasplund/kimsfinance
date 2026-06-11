# Euler Search Algorithm Deep-Dive

**Comprehensive analysis of QuantConnect's iterative grid refinement algorithm with GPU acceleration**

## Table of Contents

- [Overview](#overview)
- [Algorithm Description](#algorithm-description)
- [Mathematical Formulation](#mathematical-formulation)
- [Convergence Analysis](#convergence-analysis)
- [Comparison with Other Methods](#comparison-with-other-methods)
- [Implementation Details](#implementation-details)
- [Performance Analysis](#performance-analysis)
- [Best Practices](#best-practices)
- [References](#references)

---

## Overview

Euler Search is an **iterative grid refinement algorithm** developed by QuantConnect for parameter optimization. It combines the thoroughness of grid search with the efficiency of gradient-free optimization.

### Key Characteristics

- **Iterative Refinement**: Narrows search space around best solutions
- **Grid-Based**: Uses discrete grid points (no gradient required)
- **Adaptive Step Size**: Reduces step size each iteration
- **Fast Convergence**: Typically 5-10 iterations
- **90% Fewer Evaluations**: Compared to exhaustive grid search

### When to Use

✅ **Best for**:
- Medium-sized parameter spaces (100-10,000 combinations as grid search)
- Fast convergence requirements (<1 minute)
- Continuous parameter ranges
- Strategies with smooth fitness landscapes

❌ **Avoid when**:
- Need guaranteed global optimum (grid search is better)
- Highly non-convex fitness landscapes (many local optima)
- Very large parameter spaces (genetic algorithm is better)

---

## Algorithm Description

### High-Level Process

```text
1. Initialize parameters with broad search space
2. Repeat until convergence:
   a. Generate grid of parameter combinations
   b. Evaluate all combinations (GPU batch)
   c. Find best combination
   d. Refine search space around best:
      - Reduce step size
      - Narrow boundaries
3. Return best parameters found
```

### Example Iteration Sequence

**Iteration 1** (Initial exploration):
```
Parameter: rsi_period
Range: [5.0, 30.0]
Step: 5.0
Grid: [5, 10, 15, 20, 25, 30]  (6 points)
Best: 15.0
```

**Iteration 2** (First refinement):
```
Range: [10.0, 20.0]  (narrowed around 15.0)
Step: 1.25
Grid: [10.0, 11.25, 12.5, 13.75, 15.0, 16.25, 17.5, 18.75, 20.0]  (9 points)
Best: 13.75
```

**Iteration 3** (Further refinement):
```
Range: [12.19, 15.31]  (narrowed around 13.75)
Step: 0.31
Grid: [12.19, 12.5, 12.81, 13.12, 13.43, 13.74, 14.05, 14.36, 14.67, 14.98, 15.29]
Best: 13.43
```

**Iteration 4** (Fine-grained):
```
Range: [13.12, 13.74]
Step: 0.08
Grid: [13.12, 13.20, 13.28, ..., 13.66, 13.74]
...continues until step ≤ min_step
```

---

## Mathematical Formulation

### Refinement Formula

For each parameter at iteration `i`, given the best value `v_best` from iteration `i-1`:

#### Step Size Reduction

```
step[i] = max(min_step, step[i-1] / segment_amount)
```

Where:
- `step[i]` = new step size
- `step[i-1]` = previous step size
- `segment_amount` = refinement factor (typically 4, QuantConnect default)
- `min_step` = minimum step size (convergence threshold)

#### Range Narrowing

```
fractal = step[i] × (segment_amount / 2)
min[i] = v_best - fractal
max[i] = v_best + fractal
```

Where:
- `fractal` = half-width of new search range
- `min[i]`, `max[i]` = new boundaries

#### Grid Generation

```
grid[i] = {v | v = min[i] + k × step[i], k ∈ ℕ, v ≤ max[i]}
```

### Multi-Parameter Case

For n parameters, the algorithm maintains independent refinement for each:

```
search_space[i] = grid₁[i] × grid₂[i] × ... × gridₙ[i]
```

**Size of search space**:
```
|search_space[i]| = ∏ⱼ₌₁ⁿ |gridⱼ[i]|
                  ≈ ∏ⱼ₌₁ⁿ (segment_amount + 1)
                  ≈ (segment_amount + 1)ⁿ
```

For `segment_amount = 4` and `n = 3` parameters:
```
|search_space[i]| ≈ 5³ = 125 evaluations per iteration
```

### Example Calculation

**Initial Setup**:
```
Parameter: threshold
min₀ = 0.0
max₀ = 100.0
step₀ = 10.0
segment_amount = 4
min_step = 0.1
```

**Iteration 1**:
```
grid₁ = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  (11 points)
v_best = 60.0
```

**Iteration 2**:
```
step₂ = max(0.1, 10.0 / 4) = 2.5
fractal = 2.5 × (4 / 2) = 5.0
min₂ = 60.0 - 5.0 = 55.0
max₂ = 60.0 + 5.0 = 65.0
grid₂ = [55.0, 57.5, 60.0, 62.5, 65.0]  (5 points)
v_best = 62.5
```

**Iteration 3**:
```
step₃ = max(0.1, 2.5 / 4) = 0.625
fractal = 0.625 × 2 = 1.25
min₃ = 62.5 - 1.25 = 61.25
max₃ = 62.5 + 1.25 = 63.75
grid₃ = [61.25, 61.875, 62.5, 63.125, 63.75]  (5 points)
v_best = 63.125
```

**Iteration 4**:
```
step₄ = max(0.1, 0.625 / 4) = 0.156
fractal = 0.156 × 2 = 0.312
min₄ = 63.125 - 0.312 = 62.813
max₄ = 63.125 + 0.312 = 63.437
grid₄ = [62.813, 62.969, 63.125, 63.281, 63.437]  (5 points)
v_best = 63.281
```

**Iteration 5**:
```
step₅ = max(0.1, 0.156 / 4) = 0.1  ← Converged! (step = min_step)
```

**Total evaluations**: 11 + 5 + 5 + 5 + 5 = **31 evaluations**

Compare to grid search with step=0.1: 1000 evaluations (**32x fewer!**)

---

## Convergence Analysis

### Convergence Criteria

The algorithm converges when **all parameters** satisfy:

```
step[i] ≤ min_step + ε
```

Where `ε = 1e-9` (floating-point tolerance).

### Early Stopping

Additionally, early stopping triggers when:

```
best_fitness[i] - best_fitness[i-patience] < threshold
```

Default: `patience = 3`, `threshold = 0.01%`

### Convergence Rate

The step size reduces geometrically:

```
step[i] = step₀ / segment_amount^i  (when step > min_step)
```

**Number of iterations to convergence**:

```
n_iter = ⌈log(step₀ / min_step) / log(segment_amount)⌉
```

**Example**: `step₀ = 10.0`, `min_step = 0.1`, `segment_amount = 4`
```
n_iter = ⌈log(10.0 / 0.1) / log(4)⌉ = ⌈log(100) / log(4)⌉ = ⌈3.32⌉ = 4 iterations
```

### Total Evaluations

For `n` parameters with `segment_amount = s`:

```
total_evals = ∑ᵢ₌₀ⁿ_ⁱᵗᵉʳ (s + 1)ⁿ
            ≈ n_iter × (s + 1)ⁿ
```

**Example**: 3 parameters, 4 iterations, s=4
```
total_evals ≈ 4 × 5³ = 500 evaluations
```

Compare to grid search:
```
grid_evals = (step₀ / min_step + 1)ⁿ = (100 + 1)³ = 1,030,301 evaluations
```

**Speedup**: 2,060x fewer evaluations! 🚀

### Limitations

#### 1. Local Optima

Euler Search can get trapped in local optima if:
- Initial grid misses global optimum region
- Fitness landscape is highly non-convex

**Mitigation**:
- Use coarse grid search first to identify promising regions
- Run multiple Euler searches with different starting points
- Increase `segment_amount` for more thorough exploration (slower)

#### 2. Boundary Effects

If optimal parameter is near boundary:
- Refinement may push search range outside valid bounds
- Algorithm respects original `[min, max]` bounds

**Example**:
```
Original: [0, 100]
Best at iteration 3: v_best = 2.0
Refinement: [1.0, 3.0]  ✓ Within bounds
```

#### 3. Discrete Parameters

For discrete parameters (e.g., `rsi_period ∈ {10, 11, 12, ..., 20}`):
- Grid may skip integer values as step size reduces
- Use `min_step = 1.0` for integer parameters

---

## Comparison with Other Methods

### Euler Search vs Grid Search

| Aspect | Euler Search | Grid Search |
|--------|--------------|-------------|
| **Evaluations** | O(n_iter × sⁿ) | O(mⁿ) |
| **Typical** | 500-2000 | 10,000-1,000,000 |
| **Speedup** | 100-1000x fewer | Baseline |
| **Guarantee** | Local optimum | Global optimum |
| **Use Case** | Medium search spaces | Small search spaces |

Where:
- `s` = segment_amount (typically 4)
- `m` = steps per parameter in grid (typically 10-100)
- `n` = number of parameters

### Euler Search vs Gradient Descent

| Aspect | Euler Search | Gradient Descent |
|--------|--------------|------------------|
| **Gradient** | Not required | Required |
| **Discrete** | Handles naturally | Requires approximation |
| **Convergence** | Fixed iterations | Variable (learning rate dependent) |
| **Noise** | Robust | Sensitive |
| **GPU Batch** | Natural (grid eval) | Difficult (sequential steps) |

**Euler Search advantage**: Trading strategies have **non-differentiable** fitness landscapes (due to discrete trades, position sizing, etc.)

### Euler Search vs Genetic Algorithm

| Aspect | Euler Search | Genetic Algorithm |
|--------|--------------|-------------------|
| **Convergence** | 5-10 iterations | 20-50 generations |
| **Evaluations** | 500-2000 | 5000-20000 |
| **Speed** | Fast (seconds) | Medium (minutes) |
| **Large Spaces** | Limited (n ≤ 6) | Excellent (n ≤ 20) |
| **Guarantees** | Local optimum | Approximate solution |

**Rule of thumb**:
- Euler Search for ≤6 parameters
- Genetic Algorithm for >6 parameters

### Euler Search vs Bayesian Optimization

| Aspect | Euler Search | Bayesian Optimization |
|--------|--------------|----------------------|
| **Model** | None | Gaussian Process |
| **Sequential** | No (batch-friendly) | Yes (inherently sequential) |
| **GPU Batch** | ✓ Native support | ✗ Requires special techniques |
| **Overhead** | Low | High (GP inference) |
| **Sample Efficiency** | Medium | High |

**Euler Search advantage**: **GPU batch evaluation** provides 40x speedup, offsetting higher sample count.

---

## Implementation Details

### Data Structures

```rust
pub struct Parameter {
    name: String,
    min: f64,          // Current minimum
    max: f64,          // Current maximum
    step: f64,         // Current step size
    min_step: f64,     // Convergence threshold
}

pub struct EulerSearchOptimizer {
    device: Arc<GpuDevice>,
    parameters: Vec<Parameter>,
    segment_amount: usize,              // Refinement factor (default: 4)
    max_iterations: usize,              // Stop after N iterations
    batch_size: usize,                  // GPU batch size (100-1000)
    early_stopping_patience: Option<usize>, // Stop if no improvement for N iterations
}
```

### Grid Generation

```rust
impl Parameter {
    fn generate_grid(&self) -> Vec<f64> {
        let mut values = Vec::new();
        let mut value = self.min;

        while value <= self.max {
            values.push(value);
            value += self.step;
        }

        // Always include max value
        if let Some(&last) = values.last() {
            if (last - self.max).abs() > 1e-9 {
                values.push(self.max);
            }
        }

        values
    }
}
```

### Refinement Logic

```rust
impl Parameter {
    fn refine(&mut self, best_value: f64, segment_amount: usize) {
        // Reduce step size (geometric decay)
        let new_step = (self.step / segment_amount as f64).max(self.min_step);

        // Calculate fractal (half-width of new range)
        let fractal = new_step * (segment_amount as f64 / 2.0);

        // Narrow boundaries around best value
        self.min = best_value - fractal;
        self.max = best_value + fractal;
        self.step = new_step;
    }

    fn is_converged(&self) -> bool {
        self.step <= self.min_step + 1e-9
    }
}
```

### Cartesian Product

```rust
fn generate_parameter_grid(parameters: &[Parameter]) -> Vec<Vec<f64>> {
    // Generate grid points for each parameter
    let grids: Vec<Vec<f64>> = parameters.iter()
        .map(|p| p.generate_grid())
        .collect();

    // Compute Cartesian product
    let mut result = vec![vec![]];
    for grid in grids {
        let mut new_result = Vec::new();
        for existing in &result {
            for &value in &grid {
                let mut new_combo = existing.clone();
                new_combo.push(value);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }

    result
}
```

### GPU Batch Evaluation

```rust
fn evaluate_batch_gpu(
    device: &Arc<GpuDevice>,
    param_grid: &[Vec<f64>],
    strategy_type: StrategyType,
    ohlcv: &OhlcvData,
    config: BacktestConfig,
) -> Result<Vec<BacktestResult>, GpuError> {
    // Use BatchBacktestSweep for parallel GPU evaluation
    BatchBacktestSweep::new(device.clone())
        .strategy_type(strategy_type)
        .data_ohlcv(&ohlcv.timestamps, &ohlcv.open, &ohlcv.high, &ohlcv.low, &ohlcv.close, &ohlcv.volume)
        .parameters_batch(param_grid)
        .config(config)
        .execute()
}
```

---

## Performance Analysis

### Time Complexity

**Per iteration**:
```
T_iter = T_grid_gen + T_batch_eval + T_find_best + T_refine
```

Breaking down:

1. **Grid Generation**: O(∏ᵢ |gridᵢ|)
   - Typically ~100-500 combinations
   - Negligible (<1ms)

2. **Batch Evaluation**: O(N_combos × N_candles)
   - Dominates runtime
   - **GPU**: ~200-500μs per batch (1000 strategies × 10K candles)
   - **CPU**: ~10-20s per batch (sequential)

3. **Find Best**: O(N_combos)
   - Single pass through results
   - Negligible (<1ms)

4. **Refine**: O(N_params)
   - Update boundaries and step sizes
   - Negligible (<1μs)

**Total time**:
```
T_total ≈ N_iter × T_batch_eval
        ≈ 10 × 300ms = 3 seconds (typical)
```

### Space Complexity

**GPU Memory**:
```
VRAM = N_strategies × N_candles × sizeof(OHLCV) + indicator_buffers + trade_buffers
```

For RTX 3500 Ada (12GB VRAM):
```
Max strategies = 12GB / (10K candles × 40 bytes/candle) ≈ 30,000 strategies
```

**Typical batch** (1000 strategies):
```
VRAM_used = 1000 × 10K × 40 bytes ≈ 400MB
```

**CPU Memory**: Negligible (parameter grids are small)

### Scalability

**Dataset Size**:
```
10K candles:  ~250ms per iteration
50K candles:  ~800ms per iteration
100K candles: ~1.5s per iteration
```

**Number of Parameters**:
```
2 params: ~25 combos/iter   → 250ms total
3 params: ~125 combos/iter  → 300ms total
4 params: ~625 combos/iter  → 500ms total
5 params: ~3125 combos/iter → 2s total
```

**Recommended limits**:
- ≤6 parameters (otherwise use Genetic Algorithm)
- ≤100K candles (otherwise use tick-level optimization)

---

## Best Practices

### 1. Initial Step Size Selection

**Rule of thumb**: `initial_step = (max - min) / 10`

```rust
let range = max - min;
let initial_step = range / 10.0;
let min_step = range / 1000.0; // Or desired precision
```

**Example**:
```rust
// RSI period optimization
optimizer.add_parameter(
    "rsi_period",
    5.0,        // min
    30.0,       // max
    2.5,        // initial_step = (30-5)/10
    0.1,        // min_step (0.1 granularity acceptable)
);
```

### 2. Segment Amount Tuning

**Default**: `segment_amount = 4` (QuantConnect)

**Trade-offs**:
- **Larger** (6-8): More thorough, slower convergence, more evaluations
- **Smaller** (2-3): Faster convergence, fewer evaluations, more risk of missing optimum

```rust
// Conservative (thorough search)
let optimizer = EulerSearchOptimizer::new(device)
    .segment_amount(6);

// Aggressive (fast search)
let optimizer = EulerSearchOptimizer::new(device)
    .segment_amount(2);
```

### 3. Multi-Start Optimization

To avoid local optima, run multiple searches with different starting ranges:

```rust
fn multi_start_euler(
    device: Arc<GpuDevice>,
    param_ranges: &[(f64, f64)], // (min, max) per parameter
    num_starts: usize,
) -> Result<OptimizerResult, GpuError> {
    let mut best_overall = None;

    for i in 0..num_starts {
        // Partition search space
        let partition_width = (param_ranges[0].1 - param_ranges[0].0) / num_starts as f64;
        let start_min = param_ranges[0].0 + i as f64 * partition_width;
        let start_max = start_min + partition_width;

        // Create optimizer for this partition
        let mut optimizer = EulerSearchOptimizer::new(device.clone());
        optimizer.add_parameter("param1", start_min, start_max, partition_width / 10.0, 0.1);
        // ... add other parameters

        let result = optimizer.optimize(/* ... */)?;

        // Track best across all starts
        if best_overall.is_none() || result.best_fitness > best_overall.as_ref().unwrap().best_fitness {
            best_overall = Some(result);
        }
    }

    Ok(best_overall.unwrap())
}
```

### 4. Hybrid Grid-Euler Strategy

For large parameter spaces, use coarse grid search followed by Euler refinement:

```rust
// Step 1: Coarse grid search (1000 combinations)
let mut grid = ParameterGrid::new();
grid.add_range("rsi_period", ParameterRange::Int { min: 5, max: 30, step: 5 });
grid.add_range("threshold", ParameterRange::Float { min: 20.0, max: 80.0, step: 10.0 });

let grid_result = GridSearchOptimizer::new()
    .batch_size(500)
    .optimize(/* ... */)?;

// Step 2: Euler refinement around best region
let best_rsi = grid_result.best_parameters["rsi_period"];
let best_threshold = grid_result.best_parameters["threshold"];

let mut euler = EulerSearchOptimizer::new(device.clone());
euler.add_parameter("rsi_period", best_rsi - 5.0, best_rsi + 5.0, 2.0, 0.5);
euler.add_parameter("threshold", best_threshold - 10.0, best_threshold + 10.0, 2.0, 0.5);

let final_result = euler.optimize(/* ... */)?;
```

### 5. Convergence Monitoring

Track convergence to diagnose issues:

```rust
let result = optimizer.optimize(/* ... */)?;

// Check convergence quality
if result.is_converged() {
    println!("✓ Converged to local optimum");

    // Analyze convergence history
    let improvement = result.convergence_history.last().unwrap()
        - result.convergence_history.first().unwrap();
    println!("Total improvement: {:.2}%", improvement * 100.0);

    // Check refinement trajectory
    for (i, step) in result.refinement_history.iter().enumerate() {
        println!("Iter {}: step_sizes = {:?}, best = {:.4}",
            i, step.step_sizes, step.best_fitness);
    }
} else {
    println!("✗ Did not converge (reached max iterations)");
    println!("  Consider: increasing max_iterations or adjusting min_step");
}
```

---

## References

### Papers & Articles

1. **QuantConnect Euler Search**
   - Source: QuantConnect Lean Engine
   - Algorithm: Iterative grid refinement
   - URL: https://www.quantconnect.com/docs/v2/writing-algorithms/optimization

2. **Grid Search Optimization**
   - Bergstra, J., & Bengio, Y. (2012)
   - "Random Search for Hyper-Parameter Optimization"
   - Journal of Machine Learning Research, 13(1), 281-305

3. **Parameter Optimization in Trading**
   - Pardo, R. (2008)
   - "The Evaluation and Optimization of Trading Strategies"
   - Wiley Trading Series

### Related Documentation

- [Optimization Guide](./OPTIMIZATION_GUIDE.md) - Complete guide to all three optimizers
- [Grid Search Implementation](../src/backtest/grid_search.rs) - Source code
- [Euler Search Implementation](../src/backtest/euler_search.rs) - Source code
- [Batch Backtest API](../src/backtest/batch.rs) - GPU batch evaluation
- [Quick Start Guide](./OPTIMIZER_QUICKSTART.md) - 5-minute getting started

### Example Code

- [Euler Search Demo](../examples/euler_search_demo.rs) - Comprehensive example
- [Grid Search Demo](../examples/grid_search_demo.rs) - Grid search comparison
- [Hybrid Strategy](../examples/parameter_sweep_demo.rs) - Grid + Euler combination

---

**Last Updated**: 2025-11-04
**Version**: 0.2.0
**Maintained By**: kimsfinance team
**QuantConnect Attribution**: Algorithm inspired by QuantConnect Lean Engine

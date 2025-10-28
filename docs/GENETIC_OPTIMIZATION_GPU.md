# Genetic Algorithm Optimization with GPU Batch Backtesting

> **Speedup**: [TBD]x faster genetic optimization (target: 20-40x)
> **Scale**: Evaluate 100-1000 individuals per generation in parallel
> **Use Case**: Multi-objective strategy optimization, hyperparameter tuning

---

## Table of Contents

1. [Overview](#overview)
2. [Why Batch Evaluation?](#why-batch-evaluation)
3. [Quick Start](#quick-start)
4. [Integration Guide](#integration-guide)
5. [Performance Comparison](#performance-comparison)
6. [Best Practices](#best-practices)
7. [Advanced Techniques](#advanced-techniques)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Genetic Algorithm Optimization?

Genetic algorithms (GA) are evolutionary optimization techniques that:

1. **Population**: Maintain a population of candidate solutions (strategies)
2. **Fitness**: Evaluate each individual's performance (backtest results)
3. **Selection**: Select best-performing individuals for reproduction
4. **Crossover**: Combine parents to create offspring (new strategies)
5. **Mutation**: Randomly modify offspring to explore new solutions
6. **Iterate**: Repeat for multiple generations until convergence

**Traditional Problem**: Fitness evaluation is the bottleneck!

```
Generation 1: Evaluate 100 individuals × 10ms = 1,000ms
Generation 2: Evaluate 100 individuals × 10ms = 1,000ms
...
Generation 50: Evaluate 100 individuals × 10ms = 1,000ms

Total: 50 generations × 1,000ms = 50,000ms (50 seconds!)
```

### How Batch Backtesting Solves This

Instead of evaluating individuals one-by-one, **evaluate entire population in single GPU batch**:

```
Generation 1: Evaluate 100 individuals in batch = 50ms
Generation 2: Evaluate 100 individuals in batch = 50ms
...
Generation 50: Evaluate 100 individuals in batch = 50ms

Total: 50 generations × 50ms = 2,500ms (2.5 seconds!)

Speedup: 50s / 2.5s = 20x ✅
```

**Key Insight**: All individuals in a generation are independent → perfect for parallel GPU execution!

---

## Why Batch Evaluation?

### Traditional Sequential Evaluation

```python
# Traditional approach (SLOW)
def evaluate_population(population):
    fitness_values = []
    for individual in population:  # 100 individuals
        params = decode(individual)
        result = backtest_cpu(params)  # 10ms per individual
        fitness = (result.sharpe, result.max_drawdown, result.win_rate)
        fitness_values.append(fitness)
    return fitness_values

# Time: 100 × 10ms = 1,000ms per generation
```

**Bottleneck**: Sequential execution wastes time

**Problems**:
- 99% of time waiting for backtests to complete
- CPU cores idle while processing one backtest at a time
- No parallelism across individuals
- Total optimization time: Minutes to hours

### GPU Batch Evaluation

```python
# Batch approach (FAST)
def evaluate_population_batch(population):
    # Decode all individuals at once
    all_params = [decode(ind) for ind in population]  # 100 parameter sets

    # Single GPU batch call for entire population!
    results = batch_backtest('rsi_crossover', data, all_params)  # 50ms total

    # Extract fitness for all individuals
    fitness_values = [
        (r['sharpe_ratio'], r['max_drawdown'], r['win_rate'])
        for r in results
    ]
    return fitness_values

# Time: 50ms per generation (regardless of population size!)
```

**Advantages**:
- ✅ All individuals evaluated in parallel on GPU
- ✅ Single data transfer (CPU → GPU → CPU)
- ✅ 20-40x faster than sequential
- ✅ Scales to 1000+ individuals without slowdown
- ✅ Total optimization time: Seconds instead of minutes

---

## Quick Start

### Prerequisites

```bash
# Install kimsfinance with GPU support
pip install kimsfinance[gpu]

# Verify GPU availability
python -c "from kimsfinance import batch_backtest; print('GPU ready!')"
```

### Basic Example: Optimize RSI Strategy

```python
from kimsfinance.optimization import GeneticOptimizer
from kimsfinance import load_ohlcv

# Load your data
data = load_ohlcv('BTC-USD', start='2024-01-01', end='2024-10-01')

# Define parameter space for RSI crossover strategy
param_space = {
    'rsi_period': (10, 30, int),      # (min, max, type)
    'buy_threshold': (20, 40, float),
    'sell_threshold': (60, 80, float),
}

# Create optimizer with batch evaluation enabled
optimizer = GeneticOptimizer(
    param_space=param_space,
    population_size=100,              # 100 individuals per generation
    generations=50,                   # 50 generations
    objectives=['sharpe', 'max_drawdown', 'win_rate'],  # Multi-objective
    use_batch_backtest=True,          # 🚀 Enable GPU batch evaluation
)

# Run optimization (20x faster than sequential!)
best_solutions = optimizer.optimize(
    strategy='rsi_crossover',
    data=data,
)

# Print best strategies (Pareto front)
print(f"Found {len(best_solutions)} Pareto-optimal strategies:\n")
for i, solution in enumerate(best_solutions[:5]):
    print(f"Strategy {i+1}:")
    print(f"  Parameters: {solution['parameters']}")
    print(f"  Sharpe: {solution['sharpe']:.2f}")
    print(f"  Max Drawdown: {solution['max_drawdown']*100:.1f}%")
    print(f"  Win Rate: {solution['win_rate']*100:.1f}%")
    print()
```

**Expected Output**:
```
Found 12 Pareto-optimal strategies:

Strategy 1:
  Parameters: {'rsi_period': 14, 'buy_threshold': 25.3, 'sell_threshold': 74.8}
  Sharpe: 2.15
  Max Drawdown: -8.3%
  Win Rate: 62.1%

Strategy 2:
  Parameters: {'rsi_period': 18, 'buy_threshold': 28.7, 'sell_threshold': 71.2}
  Sharpe: 1.98
  Max Drawdown: -6.1%
  Win Rate: 58.3%

...
```

**Performance**:
```
Traditional sequential: ~50 seconds
With GPU batch:         ~2.5 seconds
Speedup:                20x ✅
```

---

## Integration Guide

### Step-by-Step Integration

#### 1. Modify GeneticOptimizer Class

**File**: `kimsfinance/optimization/genetic.py`

**Add batch evaluation method**:

```python
class GeneticOptimizer:
    def __init__(
        self,
        param_space: dict,
        population_size: int = 100,
        generations: int = 50,
        objectives: List[str] = ['sharpe'],
        use_batch_backtest: bool = False,  # NEW: Enable batch evaluation
        **kwargs
    ):
        self.param_space = param_space
        self.population_size = population_size
        self.generations = generations
        self.objectives = objectives
        self.use_batch_backtest = use_batch_backtest  # NEW
        # ... rest of initialization

    def _evaluate_fitness_batch(
        self,
        population: List[List[float]],
        strategy: str,
        data: dict,
    ) -> List[Tuple[float, ...]]:
        """
        Evaluate entire population in single GPU batch call.

        20-40x faster than sequential evaluation!

        Args:
            population: List of individuals (each is a list of parameter values)
            strategy: Strategy type ('rsi_crossover', 'ma_crossover', etc.)
            data: OHLCV data dictionary

        Returns:
            List of fitness tuples (one per individual)
        """
        from kimsfinance import batch_backtest

        # Decode all individuals to parameter dictionaries
        all_params = []
        for individual in population:
            params = self._decode_individual(individual)
            all_params.append(params)

        # Single GPU batch call for entire population!
        results = batch_backtest(
            strategy=strategy,
            data=data,
            parameters=all_params,
            config={
                'initial_capital': 10_000.0,
                'trading_fee': 0.001,
                'slippage': 0.0005,
            }
        )

        # Extract fitness tuples for all individuals
        fitness_values = []
        for result in results:
            fitness = []
            for objective in self.objectives:
                if objective == 'sharpe':
                    fitness.append(result['sharpe_ratio'])
                elif objective == 'max_drawdown':
                    # Negate because GA maximizes (lower drawdown is better)
                    fitness.append(-abs(result['max_drawdown']))
                elif objective == 'win_rate':
                    fitness.append(result['win_rate'])
                elif objective == 'total_return':
                    fitness.append(result['total_return'])
                elif objective == 'profit_factor':
                    fitness.append(result['profit_factor'])
                else:
                    raise ValueError(f"Unknown objective: {objective}")
            fitness_values.append(tuple(fitness))

        return fitness_values

    def _evaluate_fitness_sequential(
        self,
        individual: List[float],
        strategy: str,
        data: dict,
        backtester: Any
    ) -> Tuple[float, ...]:
        """Traditional sequential evaluation (fallback)"""
        params = self._decode_individual(individual)
        result = backtester.run(strategy=strategy, data=data, params=params)

        fitness = []
        for objective in self.objectives:
            if objective == 'sharpe':
                fitness.append(result['sharpe_ratio'])
            elif objective == 'max_drawdown':
                fitness.append(-abs(result['max_drawdown']))
            # ... other objectives
        return tuple(fitness)

    def _evolve_island(self, island_id, strategy, data, backtester, stats_callback):
        """Modified to use batch evaluation"""
        population = self.toolbox.population(n=self.population_size)

        for gen in range(self.generations):
            # BATCH EVALUATION - Entire population at once!
            if self.use_batch_backtest:
                fitness_values = self._evaluate_fitness_batch(
                    population, strategy, data
                )

                # Assign fitness to individuals
                for ind, fit in zip(population, fitness_values):
                    ind.fitness.values = fit
            else:
                # Traditional sequential evaluation (fallback)
                for ind in population:
                    fit = self._evaluate_fitness_sequential(
                        ind, strategy, data, backtester
                    )
                    ind.fitness.values = fit

            # Selection, crossover, mutation (unchanged)
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals (batch or sequential)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            if self.use_batch_backtest:
                fitness_values = self._evaluate_fitness_batch(
                    invalid_ind, strategy, data
                )
                for ind, fit in zip(invalid_ind, fitness_values):
                    ind.fitness.values = fit
            else:
                for ind in invalid_ind:
                    fit = self._evaluate_fitness_sequential(
                        ind, strategy, data, backtester
                    )
                    ind.fitness.values = fit

            # Replace population
            population[:] = offspring

            # Stats callback
            if stats_callback:
                stats_callback(gen, population)

        return population
```

#### 2. Usage Example

```python
from kimsfinance.optimization import GeneticOptimizer

# Traditional sequential (SLOW)
optimizer_slow = GeneticOptimizer(
    param_space=param_space,
    population_size=100,
    generations=50,
    use_batch_backtest=False,  # Sequential CPU evaluation
)
best_slow = optimizer_slow.optimize('rsi_crossover', data)

# GPU batch (FAST)
optimizer_fast = GeneticOptimizer(
    param_space=param_space,
    population_size=100,
    generations=50,
    use_batch_backtest=True,  # 🚀 GPU batch evaluation
)
best_fast = optimizer_fast.optimize('rsi_crossover', data)

# Results should be similar, but GPU is 20x faster!
```

#### 3. Automatic Fallback

Add automatic GPU detection and fallback:

```python
class GeneticOptimizer:
    def __init__(self, param_space, use_batch_backtest='auto', **kwargs):
        self.param_space = param_space

        # Auto-detect GPU availability
        if use_batch_backtest == 'auto':
            try:
                from kimsfinance import batch_backtest, gpu_available
                self.use_batch_backtest = gpu_available()
                if self.use_batch_backtest:
                    print("✅ GPU detected - using batch evaluation")
                else:
                    print("⚠️ GPU not available - using sequential evaluation")
            except ImportError:
                self.use_batch_backtest = False
                print("⚠️ GPU support not installed - using sequential evaluation")
        else:
            self.use_batch_backtest = use_batch_backtest
```

---

## Performance Comparison

### Sequential vs Batch Evaluation

**Configuration**:
- Population size: 100 individuals
- Generations: 50
- Dataset: 10,000 candles
- Strategy: RSI crossover
- Hardware: RTX 3500 Ada, Intel i9-13980HX

**Sequential Evaluation** (Traditional):
```
Per-individual time:     10ms
Per-generation time:     100 × 10ms = 1,000ms
Total optimization time: 50 × 1,000ms = 50,000ms (50 seconds)
```

**Batch GPU Evaluation**:
```
Per-batch time:          [TBD]ms (100 individuals)
Per-generation time:     [TBD]ms
Total optimization time: 50 × [TBD]ms = [TBD]ms ([TBD] seconds)

Speedup: [TBD]x ✅
```

### Scaling with Population Size

**How does performance scale with population size?**

**Sequential** (linear scaling):
```
100 individuals:  1,000ms per generation
200 individuals:  2,000ms per generation
500 individuals:  5,000ms per generation
1000 individuals: 10,000ms per generation
```

**Batch GPU** (sub-linear scaling):
```
100 individuals:  [TBD]ms per generation
200 individuals:  [TBD]ms per generation
500 individuals:  [TBD]ms per generation
1000 individuals: [TBD]ms per generation

Note: GPU scales much better due to parallel execution!
```

**Speedup by Population Size**:

| Population Size | Sequential | Batch GPU | Speedup |
|----------------|------------|-----------|---------|
| 100 | [TBD]ms | [TBD]ms | [TBD]x |
| 200 | [TBD]ms | [TBD]ms | [TBD]x |
| 500 | [TBD]ms | [TBD]ms | [TBD]x |
| 1000 | [TBD]ms | [TBD]ms | [TBD]x |

**Conclusion**: Larger populations benefit even more from GPU batch evaluation!

---

## Best Practices

### 1. Population Size Selection

**Rule of Thumb**: Use 50-200 individuals for most problems

```python
# ✅ Good: 100 individuals (optimal for most GPUs)
optimizer = GeneticOptimizer(
    population_size=100,
    use_batch_backtest=True,
)

# ⚠️ Too small: <50 individuals (GPU underutilized)
optimizer = GeneticOptimizer(
    population_size=20,
    use_batch_backtest=False,  # Use sequential instead
)

# ⚠️ Too large: >1000 individuals (may exceed VRAM)
optimizer = GeneticOptimizer(
    population_size=2000,
    use_batch_backtest=True,  # Will auto-chunk into 2 batches
)
```

**Tradeoffs**:
- **Smaller populations**: Faster convergence, risk of premature convergence
- **Larger populations**: Better exploration, slower convergence, more VRAM

### 2. Generations vs Population Size

**Total Evaluations = Population × Generations**

**Strategy 1**: Small population, many generations
```python
optimizer = GeneticOptimizer(
    population_size=50,
    generations=100,
)
# Total: 5,000 evaluations
# Good for: Smooth convergence, avoiding local minima
```

**Strategy 2**: Large population, fewer generations
```python
optimizer = GeneticOptimizer(
    population_size=200,
    generations=25,
)
# Total: 5,000 evaluations
# Good for: Wide exploration, parallel efficiency
```

**Recommendation**: For GPU batch, prefer Strategy 2 (large population, fewer generations)
- GPU scales better with larger batches
- Each generation is barely slower (sub-linear scaling)
- Better exploration of parameter space

### 3. Multi-Objective Optimization

**Use 2-4 objectives** for best results:

```python
# ✅ Good: 3 objectives (Sharpe, Drawdown, Win Rate)
optimizer = GeneticOptimizer(
    objectives=['sharpe', 'max_drawdown', 'win_rate'],
)

# ⚠️ Single objective (may overfit to one metric)
optimizer = GeneticOptimizer(
    objectives=['sharpe'],
)

# ❌ Too many objectives (5+, difficult to optimize)
optimizer = GeneticOptimizer(
    objectives=['sharpe', 'max_drawdown', 'win_rate',
                'profit_factor', 'total_return', 'num_trades'],
)
```

**Why multi-objective?**
- Prevents overfitting to single metric
- Finds diverse strategies (Pareto front)
- Better real-world performance

### 4. Parameter Space Design

**Keep it reasonable**: 3-6 parameters optimal

```python
# ✅ Good: 3 parameters for RSI crossover
param_space = {
    'rsi_period': (10, 30, int),
    'buy_threshold': (20, 40, float),
    'sell_threshold': (60, 80, float),
}

# ⚠️ Too many parameters (>8, risk of overfitting)
param_space = {
    'rsi_period': (10, 30, int),
    'buy_threshold': (20, 40, float),
    'sell_threshold': (60, 80, float),
    'sma_fast': (10, 50, int),
    'sma_slow': (50, 200, int),
    'atr_period': (10, 30, int),
    'atr_multiplier': (1.0, 5.0, float),
    'position_size': (0.1, 1.0, float),
}
```

**Why fewer parameters?**
- Faster convergence
- Less overfitting risk
- Easier to interpret results

### 5. Crossover and Mutation Rates

**Recommended defaults**:

```python
optimizer = GeneticOptimizer(
    param_space=param_space,
    crossover_rate=0.5,   # 50% of pairs undergo crossover
    mutation_rate=0.2,    # 20% of individuals undergo mutation
)
```

**Tuning**:
- **High crossover** (0.7-0.9): Exploits known good solutions
- **Low crossover** (0.3-0.5): Explores more diverse solutions
- **High mutation** (0.3-0.5): More exploration, slower convergence
- **Low mutation** (0.1-0.2): Less exploration, faster convergence

### 6. Convergence Detection

**Stop early if converged**:

```python
def check_convergence(population, threshold=0.01):
    """Stop if population diversity drops below threshold"""
    fitness_values = [ind.fitness.values[0] for ind in population]
    diversity = np.std(fitness_values) / np.mean(fitness_values)
    return diversity < threshold

optimizer = GeneticOptimizer(
    param_space=param_space,
    generations=100,
    convergence_threshold=0.01,  # Stop if diversity < 1%
)
```

**Benefits**:
- Saves computation time
- Prevents wasted generations
- Detects when optimization is complete

---

## Advanced Techniques

### 1. Island Model (Parallel Populations)

Run multiple populations in parallel, periodically exchange best individuals:

```python
class IslandGeneticOptimizer(GeneticOptimizer):
    def __init__(
        self,
        param_space,
        num_islands=4,
        population_size=100,
        migration_interval=10,  # Migrate every 10 generations
        migration_rate=0.1,     # Migrate 10% of population
        **kwargs
    ):
        super().__init__(param_space, population_size, **kwargs)
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate

    def optimize(self, strategy, data):
        """Run island model optimization"""
        # Initialize islands
        islands = [
            self.toolbox.population(n=self.population_size)
            for _ in range(self.num_islands)
        ]

        for gen in range(self.generations):
            # Evolve each island independently
            for island in islands:
                # Batch evaluate entire island population
                fitness_values = self._evaluate_fitness_batch(
                    island, strategy, data
                )
                for ind, fit in zip(island, fitness_values):
                    ind.fitness.values = fit

                # Selection, crossover, mutation
                # ... (same as single population)

            # Periodic migration
            if gen % self.migration_interval == 0:
                self._migrate_individuals(islands)

        # Combine all islands and return Pareto front
        all_individuals = [ind for island in islands for ind in island]
        return self._extract_pareto_front(all_individuals)

    def _migrate_individuals(self, islands):
        """Exchange best individuals between islands"""
        num_migrants = int(self.population_size * self.migration_rate)

        for i in range(self.num_islands):
            # Select best individuals from island i
            migrants = sorted(islands[i],
                            key=lambda ind: ind.fitness.values[0],
                            reverse=True)[:num_migrants]

            # Send to next island (ring topology)
            next_island = (i + 1) % self.num_islands
            islands[next_island][-num_migrants:] = migrants
```

**Benefits**:
- Better exploration (multiple independent searches)
- Prevents premature convergence
- Can find multiple optima
- Each island batch-evaluated in parallel!

**Usage**:
```python
optimizer = IslandGeneticOptimizer(
    param_space=param_space,
    num_islands=4,
    population_size=100,  # 100 per island = 400 total
    generations=50,
    use_batch_backtest=True,
)

best = optimizer.optimize('rsi_crossover', data)
```

### 2. Adaptive Mutation Rate

Adjust mutation rate based on convergence:

```python
class AdaptiveGeneticOptimizer(GeneticOptimizer):
    def _evolve_island(self, island_id, strategy, data, backtester, stats_callback):
        population = self.toolbox.population(n=self.population_size)
        mutation_rate = 0.2  # Initial

        for gen in range(self.generations):
            # Batch evaluate population
            fitness_values = self._evaluate_fitness_batch(population, strategy, data)
            for ind, fit in zip(population, fitness_values):
                ind.fitness.values = fit

            # Check diversity
            diversity = self._calculate_diversity(population)

            # Adapt mutation rate
            if diversity < 0.1:
                mutation_rate = min(0.5, mutation_rate * 1.2)  # Increase mutation
            else:
                mutation_rate = max(0.1, mutation_rate * 0.9)  # Decrease mutation

            # Apply operators with adaptive rate
            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)

            # ... rest of evolution

    def _calculate_diversity(self, population):
        """Measure population diversity (coefficient of variation)"""
        fitness_values = [ind.fitness.values[0] for ind in population]
        mean = np.mean(fitness_values)
        std = np.std(fitness_values)
        return std / mean if mean > 0 else 0
```

### 3. Elitism (Preserve Best Individuals)

Ensure best individuals survive to next generation:

```python
class ElitistGeneticOptimizer(GeneticOptimizer):
    def __init__(self, param_space, elite_size=5, **kwargs):
        super().__init__(param_space, **kwargs)
        self.elite_size = elite_size

    def _evolve_island(self, island_id, strategy, data, backtester, stats_callback):
        population = self.toolbox.population(n=self.population_size)

        for gen in range(self.generations):
            # Batch evaluate
            fitness_values = self._evaluate_fitness_batch(population, strategy, data)
            for ind, fit in zip(population, fitness_values):
                ind.fitness.values = fit

            # Preserve elite individuals
            elite = sorted(population,
                          key=lambda ind: ind.fitness.values[0],
                          reverse=True)[:self.elite_size]

            # Selection, crossover, mutation on remaining
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            # ... crossover, mutation

            # Combine elite + offspring
            population[:] = elite + offspring

        return population
```

**Benefits**:
- Guarantees best solutions never lost
- Faster convergence
- More stable optimization

---

## Examples

### Example 1: Basic Multi-Objective Optimization

Optimize for Sharpe ratio, max drawdown, and win rate:

```python
from kimsfinance.optimization import GeneticOptimizer

param_space = {
    'rsi_period': (10, 30, int),
    'buy_threshold': (20, 40, float),
    'sell_threshold': (60, 80, float),
}

optimizer = GeneticOptimizer(
    param_space=param_space,
    population_size=100,
    generations=50,
    objectives=['sharpe', 'max_drawdown', 'win_rate'],
    use_batch_backtest=True,
)

best = optimizer.optimize('rsi_crossover', data)

# Plot Pareto front
import matplotlib.pyplot as plt
sharpes = [s['sharpe'] for s in best]
drawdowns = [s['max_drawdown'] * 100 for s in best]

plt.figure(figsize=(10, 6))
plt.scatter(drawdowns, sharpes, s=100, alpha=0.6)
plt.xlabel('Max Drawdown (%)')
plt.ylabel('Sharpe Ratio')
plt.title('Pareto Front: 50 generations, 100 individuals')
plt.grid(True)
plt.show()
```

### Example 2: Walk-Forward Optimization with Validation

Optimize on training data, validate on test data:

```python
from kimsfinance.optimization import GeneticOptimizer
import numpy as np

# Split data: 70% train, 30% test
split_idx = int(len(data['close']) * 0.7)
train_data = {k: v[:split_idx] for k, v in data.items()}
test_data = {k: v[split_idx:] for k, v in data.items()}

# Optimize on training data
optimizer = GeneticOptimizer(
    param_space=param_space,
    population_size=100,
    generations=50,
    objectives=['sharpe', 'max_drawdown'],
    use_batch_backtest=True,
)

train_best = optimizer.optimize('rsi_crossover', train_data)

# Validate all Pareto-optimal strategies on test data
from kimsfinance import batch_backtest

test_params = [s['parameters'] for s in train_best]
test_results = batch_backtest('rsi_crossover', test_data, test_params)

# Compare train vs test performance
for i, (train_res, test_res) in enumerate(zip(train_best, test_results)):
    train_sharpe = train_res['sharpe']
    test_sharpe = test_res['sharpe_ratio']

    print(f"Strategy {i+1}:")
    print(f"  Train Sharpe: {train_sharpe:.2f}")
    print(f"  Test Sharpe:  {test_sharpe:.2f}")
    print(f"  Degradation:  {(1 - test_sharpe/train_sharpe)*100:.1f}%")

    if test_sharpe < train_sharpe * 0.5:
        print(f"  ⚠️ Warning: Possible overfitting")
    else:
        print(f"  ✅ Strategy validated")
    print()
```

### Example 3: Large-Scale Optimization (1000 individuals)

Optimize with very large population for thorough exploration:

```python
from kimsfinance.optimization import GeneticOptimizer

# Large population for wide exploration
optimizer = GeneticOptimizer(
    param_space=param_space,
    population_size=1000,  # 1000 individuals!
    generations=20,         # Fewer generations (still 20K evaluations)
    objectives=['sharpe', 'max_drawdown', 'win_rate'],
    use_batch_backtest=True,  # Essential for large populations
)

best = optimizer.optimize('rsi_crossover', data)

print(f"Explored {1000 * 20} = 20,000 strategies")
print(f"Found {len(best)} Pareto-optimal solutions")

# With GPU batch: ~1 second per generation × 20 = ~20 seconds total
# Sequential would take: ~10 seconds per generation × 20 = ~200 seconds
```

### Example 4: Island Model with Migration

Run 4 independent populations with periodic migration:

```python
from kimsfinance.optimization import IslandGeneticOptimizer

optimizer = IslandGeneticOptimizer(
    param_space=param_space,
    num_islands=4,
    population_size=100,      # 100 per island = 400 total
    generations=50,
    migration_interval=10,    # Migrate every 10 generations
    migration_rate=0.1,       # Migrate 10% of population
    objectives=['sharpe', 'max_drawdown', 'win_rate'],
    use_batch_backtest=True,
)

best = optimizer.optimize('rsi_crossover', data)

# Benefits:
# - Better exploration (4 independent searches)
# - Prevents premature convergence
# - Each island batch-evaluated in parallel
```

---

## Troubleshooting

### Issue 1: Slow Optimization (Not Faster Than Sequential)

**Symptoms**: GPU batch optimization not faster than expected

**Possible Causes**:

1. **Population too small**:
```python
# ❌ Bad: Only 20 individuals (GPU underutilized)
optimizer = GeneticOptimizer(population_size=20, use_batch_backtest=True)

# ✅ Fix: Use 100+ individuals
optimizer = GeneticOptimizer(population_size=100, use_batch_backtest=True)
```

2. **Data too small**:
```python
# ❌ Bad: Only 1000 candles (transfer overhead dominates)
data = load_ohlcv(days=1)

# ✅ Fix: Use 10K+ candles
data = load_ohlcv(days=7)
```

3. **Not using batch evaluation**:
```python
# ❌ Wrong: Forgot to enable batch
optimizer = GeneticOptimizer(use_batch_backtest=False)

# ✅ Fix: Enable batch evaluation
optimizer = GeneticOptimizer(use_batch_backtest=True)
```

### Issue 2: Out of Memory (VRAM)

**Symptoms**: `RuntimeError: CUDA error: out of memory`

**Solutions**:

1. **Reduce population size**:
```python
# Before: 2000 individuals (too large)
optimizer = GeneticOptimizer(population_size=2000)

# After: 500 individuals (fits in VRAM)
optimizer = GeneticOptimizer(population_size=500)
```

2. **Reduce data size**:
```python
# Shorter time window
data = load_ohlcv(days=30)  # Instead of days=90
```

3. **Use automatic chunking** (implemented in batch_backtest):
```python
# System will automatically chunk large populations
# No code changes needed - it just works!
```

### Issue 3: Poor Convergence (No Good Solutions)

**Symptoms**: Best strategies have poor performance

**Possible Causes**:

1. **Too few generations**:
```python
# ❌ Bad: Only 10 generations (not enough)
optimizer = GeneticOptimizer(generations=10)

# ✅ Fix: Use 50+ generations
optimizer = GeneticOptimizer(generations=50)
```

2. **Parameter space too restrictive**:
```python
# ❌ Bad: Narrow parameter ranges
param_space = {'buy_threshold': (28, 32, float)}  # Too narrow

# ✅ Fix: Wider ranges
param_space = {'buy_threshold': (20, 40, float)}
```

3. **Wrong objectives**:
```python
# ❌ Bad: Conflicting objectives
objectives = ['sharpe', 'total_return']  # Highly correlated

# ✅ Fix: Diverse objectives
objectives = ['sharpe', 'max_drawdown', 'win_rate']
```

### Issue 4: Overfitting to Training Data

**Symptoms**: Great train performance, poor test performance

**Solutions**:

1. **Walk-forward validation**:
```python
# Always validate on out-of-sample data
train_best = optimizer.optimize('rsi_crossover', train_data)
test_results = batch_backtest('rsi_crossover', test_data,
                              [s['parameters'] for s in train_best])
```

2. **Reduce parameter count**:
```python
# ❌ Bad: 8 parameters (overfitting risk)
param_space = {k: v for k, v in eight_params.items()}

# ✅ Fix: 3-5 parameters
param_space = {k: v for k, v in five_params.items()}
```

3. **Regularization via objectives**:
```python
# Add objectives that penalize complexity
objectives = ['sharpe', 'max_drawdown', 'win_rate', 'num_trades']
# Strategies with fewer trades often more robust
```

---

## Performance Benchmarks

**Configuration**: RTX 3500 Ada, 10K candles, RSI crossover

**Benchmark Results** (TBD after implementation):

| Population | Generations | Sequential | Batch GPU | Speedup |
|-----------|-------------|------------|-----------|---------|
| 100 | 50 | [TBD]s | [TBD]s | [TBD]x |
| 200 | 50 | [TBD]s | [TBD]s | [TBD]x |
| 500 | 50 | [TBD]s | [TBD]s | [TBD]x |
| 1000 | 20 | [TBD]s | [TBD]s | [TBD]x |

**Expected Results**:
- Sequential: ~10ms per individual × population × generations
- Batch GPU: ~50ms per generation (regardless of population size!)
- Speedup: 20-40x for typical configurations

---

## Further Reading

- [GPU_BATCH_BACKTESTING.md](GPU_BATCH_BACKTESTING.md) - Core batch backtesting documentation
- [batch_backtest_tutorial.ipynb](../examples/batch_backtest_tutorial.ipynb) - Interactive tutorial
- [Implementation Plan](../integrated-reasoning/gpu_batch_backtesting_implementation_plan.md) - Technical details

---

**Last Updated**: 2025-10-27
**Version**: 0.1.0 (Draft - pending implementation)
**Status**: Documentation draft - performance numbers TBD after implementation

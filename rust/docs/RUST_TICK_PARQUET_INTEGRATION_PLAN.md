# Rust Tick-Level Parquet Integration Plan

**Date**: 2025-11-01
**Status**: 📋 **IMPLEMENTATION PLAN**
**Target**: Integrate tick-level Parquet data loading with existing genetic optimizer

---

## Executive Summary

This document provides a complete implementation plan for adding tick-level Parquet data support to the existing Rust genetic optimizer, targeting **5-10M ticks/sec** processing speed (8-15x faster than Python baseline of 648K ticks/sec).

**Current State**:
- ✅ Python prototype working (648K ticks/sec)
- ✅ Rust infrastructure exists (TickStrategy, GeneticOptimizer, Trade struct)
- ✅ Parquet dependency available
- ✅ 20.7B tick dataset ready (12 pairs, Parquet format)

**Missing**:
- ❌ Parquet file loading function
- ❌ Integration with genetic optimizer
- ❌ Benchmarking and validation

**Estimated Effort**: 40-80 hours

---

## Architecture Overview

### Current Python Implementation

```python
# Load Parquet (Polars)
df = pl.read_parquet("/path/to/trades_parquet/2024-01/*.parquet")

# Backtest
for row in df.iter_rows(named=True):
    strategy.on_tick(row['price'], row['qty'], row['side'], row['timestamp'])

# Genetic optimization
result = run_genetic_optimization_tick(df, generations=10, population=20)
```

**Performance**: 648,081 ticks/sec

---

### Proposed Rust Implementation

```rust
// Load Parquet (Arrow + Parquet crates)
let trades = load_tick_data_parquet(
    "/path/to/trades_parquet/2024-01",
    Some(1_000_000) // Optional limit
)?;

// Backtest
let mut strategy = SimpleMAStrategy::new(10, 30);
let result = backtest_ticks(&trades, &mut strategy)?;

// Genetic optimization
let mut optimizer = GeneticOptimizer::new()
    .population_size(100)
    .generations(50);

let result = optimizer.optimize_tick_strategy(
    &trades,
    &parameter_grid
)?;
```

**Target Performance**: 5-10M ticks/sec (8-15x Python)

---

## Implementation Steps

### Phase 1: Parquet Loading (8-12 hours)

#### Step 1.1: Add Parquet Reader Module

**File**: `rust/src/binance/parquet_loader.rs` (NEW)

**Dependencies** (already in Cargo.toml):
```toml
arrow = { version = "54.0", features = ["ipc"] }
parquet = { version = "54.0", features = ["arrow"], optional = true }
```

**Implementation**:

```rust
//! Parquet file loader for tick-level trade data
//!
//! Loads Binance tick data from month-partitioned Parquet files.
//! Zero-copy reads using Apache Arrow for maximum performance.

use crate::binance::{Trade, BinanceError};
use arrow::array::{Array, Float64Array, Int64Array, UInt64Array, BooleanArray, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Load tick data from a single Parquet file
///
/// # Performance
/// - Zero-copy reads via Arrow
/// - Batch processing (10K records at a time)
/// - Target: 10-20M records/sec
///
/// # Schema
/// Expects Parquet with columns:
/// - `id`: UInt64 (trade ID)
/// - `price`: Float64
/// - `qty`: Float64
/// - `quote_qty`: Float64
/// - `time`: Int64 (Unix timestamp ms)
/// - `is_buyer_maker`: Boolean
///
/// # Example
/// ```rust,ignore
/// let trades = load_parquet_file("BTCUSDT-trades-2024-01.parquet")?;
/// println!("Loaded {} trades", trades.len());
/// ```
pub fn load_parquet_file<P: AsRef<Path>>(
    parquet_path: P
) -> Result<Vec<Trade>, BinanceError> {
    let file = File::open(parquet_path.as_ref())
        .map_err(|e| BinanceError::IoError(e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| BinanceError::ParseError(format!("Parquet open error: {}", e)))?;

    let reader = builder.build()
        .map_err(|e| BinanceError::ParseError(format!("Parquet build error: {}", e)))?;

    let mut trades = Vec::new();

    // Process batches
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| BinanceError::ParseError(format!("Batch read error: {}", e)))?;

        // Extract columns (zero-copy via Arrow)
        let ids = batch.column_by_name("id")
            .ok_or_else(|| BinanceError::InvalidData("Missing 'id' column".to_string()))?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| BinanceError::InvalidData("Invalid 'id' type".to_string()))?;

        let prices = extract_float64_column(&batch, "price")?;
        let quantities = extract_float64_column(&batch, "qty")?;
        let quote_qtys = extract_float64_column(&batch, "quote_qty")?;
        let timestamps = extract_int64_column(&batch, "time")?;
        let is_buyer_makers = extract_boolean_column(&batch, "is_buyer_maker")?;

        // Convert to Trade structs
        for i in 0..batch.num_rows() {
            trades.push(Trade {
                trade_id: ids.value(i),
                price: prices.value(i),
                quantity: quantities.value(i),
                quote_quantity: quote_qtys.value(i),
                timestamp_ms: timestamps.value(i),
                is_buyer_maker: is_buyer_makers.value(i),
            });
        }
    }

    Ok(trades)
}

/// Load all Parquet files from a month directory
///
/// # Performance
/// - Parallel file loading (via Rayon)
/// - Memory-mapped I/O for large files
/// - Target: 5-10M records/sec aggregate
///
/// # File Pattern
/// Expects directory structure:
/// ```text
/// /trades_parquet/2024-01/
///   ├── BTCUSDT-trades-2024-01.parquet
///   └── (or multiple daily/weekly files)
/// ```
///
/// # Example
/// ```rust,ignore
/// let trades = load_parquet_month(
///     "/home/user/binance-data/futures/BTCUSDT/trades_parquet/2024-01",
///     None // No limit
/// )?;
/// ```
pub fn load_parquet_month<P: AsRef<Path>>(
    month_dir: P,
    max_trades: Option<usize>,
) -> Result<Vec<Trade>, BinanceError> {
    let month_path = month_dir.as_ref();

    // Find all .parquet files
    let parquet_files: Vec<PathBuf> = std::fs::read_dir(month_path)
        .map_err(|e| BinanceError::IoError(e))?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "parquet").unwrap_or(false))
        .collect();

    if parquet_files.is_empty() {
        return Err(BinanceError::InvalidData(
            format!("No Parquet files found in {:?}", month_path)
        ));
    }

    let mut all_trades = Vec::new();

    // Load each file (TODO: parallelize with Rayon)
    for file_path in parquet_files {
        let trades = load_parquet_file(&file_path)?;

        if let Some(limit) = max_trades {
            let remaining = limit.saturating_sub(all_trades.len());
            if remaining == 0 {
                break;
            }
            all_trades.extend(trades.into_iter().take(remaining));
        } else {
            all_trades.extend(trades);
        }
    }

    // Sort by timestamp (files may not be sorted)
    all_trades.sort_unstable_by_key(|t| t.timestamp_ms);

    Ok(all_trades)
}

// Helper functions for column extraction

fn extract_float64_column(
    batch: &RecordBatch,
    name: &str
) -> Result<&Float64Array, BinanceError> {
    batch.column_by_name(name)
        .ok_or_else(|| BinanceError::InvalidData(format!("Missing '{}' column", name)))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| BinanceError::InvalidData(format!("Invalid '{}' type", name)))
}

fn extract_int64_column(
    batch: &RecordBatch,
    name: &str
) -> Result<&Int64Array, BinanceError> {
    batch.column_by_name(name)
        .ok_or_else(|| BinanceError::InvalidData(format!("Missing '{}' column", name)))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| BinanceError::InvalidData(format!("Invalid '{}' type", name)))
}

fn extract_boolean_column(
    batch: &RecordBatch,
    name: &str
) -> Result<&BooleanArray, BinanceError> {
    batch.column_by_name(name)
        .ok_or_else(|| BinanceError::InvalidData(format!("Missing '{}' column", name)))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| BinanceError::InvalidData(format!("Invalid '{}' type", name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires actual Parquet files
    fn test_load_parquet_file() {
        let trades = load_parquet_file(
            "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01/BTCUSDT-trades-2024-01.parquet"
        ).unwrap();

        assert!(trades.len() > 0);
        assert!(trades[0].price > 0.0);
        assert!(trades[0].timestamp_ms > 0);
    }

    #[test]
    #[ignore] // Requires actual Parquet files
    fn test_load_parquet_month() {
        let trades = load_parquet_month(
            "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01",
            Some(100_000) // Limit for fast test
        ).unwrap();

        assert_eq!(trades.len(), 100_000);

        // Verify sorted
        for i in 1..trades.len() {
            assert!(trades[i].timestamp_ms >= trades[i-1].timestamp_ms);
        }
    }
}
```

**File**: `rust/src/binance/mod.rs` (MODIFY)

Add re-export:
```rust
#[cfg(feature = "data-downloaders")]
pub mod parquet_loader;

#[cfg(feature = "data-downloaders")]
pub use parquet_loader::{load_parquet_file, load_parquet_month};
```

---

### Phase 2: Tick Backtesting Engine (12-16 hours)

#### Step 2.1: Add Tick Backtesting Function

**File**: `rust/src/backtest/tick_engine.rs` (MODIFY)

Add high-performance tick backtesting function:

```rust
/// Backtest a tick strategy on raw trade data
///
/// Processes every single trade through the strategy's `on_tick()` method.
/// Significantly more realistic than OHLCV backtesting for high-frequency strategies.
///
/// # Performance
/// - Target: 5-10M ticks/sec
/// - Uses vectorization where possible
/// - Minimal allocations in hot path
///
/// # Example
/// ```rust,ignore
/// let trades = load_parquet_month("....", Some(1_000_000))?;
/// let mut strategy = IntraCandleMomentum::new(0.5);
/// let result = backtest_ticks(&trades, &mut strategy)?;
///
/// println!("Return: {:.2}%", result.total_return * 100.0);
/// println!("Trades: {}", result.num_trades);
/// ```
pub fn backtest_ticks<S: TickStrategy>(
    trades: &[Trade],
    strategy: &mut S,
) -> Result<BacktestResult, BinanceError> {
    if trades.is_empty() {
        return Err(BinanceError::InvalidData("No trades provided".to_string()));
    }

    let mut position: f64 = 0.0;
    let mut equity: f64 = 10000.0;
    let initial_equity = equity;
    let mut executed_trades: Vec<ExecutedTrade> = Vec::new();

    // Track incomplete candle for strategy
    let mut candle = IncompleteCandle::new(Timeframe::minutes(1));

    // Process each tick
    for trade in trades {
        // Update incomplete candle
        candle.update(trade);

        // Get strategy signal
        let signal = strategy.on_tick(trade, &candle);

        // Execute signal
        match signal {
            Signal::Buy if position == 0.0 => {
                position = equity / trade.price;
                executed_trades.push(ExecutedTrade {
                    timestamp: trade.timestamp_ms,
                    direction: TradeDirection::Buy,
                    price: trade.price,
                    quantity: position,
                    equity,
                });
            }
            Signal::Sell if position > 0.0 => {
                equity = position * trade.price;
                executed_trades.push(ExecutedTrade {
                    timestamp: trade.timestamp_ms,
                    direction: TradeDirection::Sell,
                    price: trade.price,
                    quantity: position,
                    equity,
                });
                position = 0.0;
            }
            _ => {}
        }
    }

    // Close final position
    if position > 0.0 {
        let last_price = trades.last().unwrap().price;
        equity = position * last_price;
        position = 0.0;
    }

    // Calculate metrics
    let total_return = (equity - initial_equity) / initial_equity;
    let num_trades = executed_trades.len();

    // Calculate win rate
    let mut wins = 0;
    for i in (1..executed_trades.len()).step_by(2) {
        if executed_trades[i].equity > executed_trades[i-1].equity {
            wins += 1;
        }
    }
    let win_rate = if num_trades >= 2 {
        wins as f64 / (num_trades / 2) as f64
    } else {
        0.0
    };

    Ok(BacktestResult {
        total_return,
        sharpe_ratio: 0.0, // TODO: Calculate
        max_drawdown: 0.0, // TODO: Calculate
        win_rate,
        num_trades,
        final_equity: equity,
        trades: executed_trades,
    })
}

struct ExecutedTrade {
    timestamp: i64,
    direction: TradeDirection,
    price: f64,
    quantity: f64,
    equity: f64,
}

#[derive(Debug, Clone, Copy)]
enum TradeDirection {
    Buy,
    Sell,
}
```

---

### Phase 3: Genetic Optimizer Integration (12-16 hours)

#### Step 3.1: Add Tick-Level Optimizer Method

**File**: `rust/src/backtest/optimizer.rs` (MODIFY)

Add new method to `GeneticOptimizer`:

```rust
impl GeneticOptimizer {
    /// Optimize a tick-level strategy on raw trade data
    ///
    /// # Performance
    /// - Parallel evaluation (Rayon)
    /// - Target: 5-10M ticks/sec per worker
    /// - Full month (100M ticks) in 20-200 seconds
    ///
    /// # Example
    /// ```rust,ignore
    /// let trades = load_parquet_month("...", None)?;
    /// let mut grid = ParameterGrid::new();
    /// grid.add_range("fast_ma", ParameterRange::Int { min: 5, max: 50, step: 5 });
    /// grid.add_range("slow_ma", ParameterRange::Int { min: 20, max: 200, step: 10 });
    ///
    /// let optimizer = GeneticOptimizer::new()
    ///     .population_size(50)
    ///     .generations(20);
    ///
    /// let result = optimizer.optimize_tick_strategy::<IntraCandleMomentum>(
    ///     &trades,
    ///     &grid
    /// )?;
    ///
    /// println!("Best params: {:?}", result.best_parameters);
    /// println!("Best return: {:.2}%", result.best_fitness * 100.0);
    /// ```
    pub fn optimize_tick_strategy<S: TickStrategy + Clone + Send>(
        &self,
        trades: &[Trade],
        param_grid: &ParameterGrid,
    ) -> Result<OptimizerResult, GpuError> {
        // Initialize random population
        let mut population: Vec<HashMap<String, f64>> = Vec::with_capacity(self.population_size);
        let mut rng = rand::thread_rng();

        for _ in 0..self.population_size {
            let mut individual = HashMap::new();
            for (name, range) in &param_grid.ranges {
                let value = match range {
                    ParameterRange::Int { min, max, step: _ } => {
                        rng.gen_range(*min..=*max) as f64
                    }
                    ParameterRange::Float { min, max, step: _ } => {
                        rng.gen_range(*min..=*max)
                    }
                };
                individual.insert(name.clone(), value);
            }
            population.push(individual);
        }

        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        // Run genetic algorithm
        for generation in 0..self.generations {
            // Evaluate fitness for all individuals (PARALLEL)
            let fitness_scores: Vec<(f64, HashMap<String, f64>)> = if population.len() >= PARALLEL_THRESHOLD {
                // Parallel evaluation using Rayon
                population.par_iter()
                    .map(|params| {
                        let mut strategy = S::new(); // TODO: Pass parameters
                        let result = backtest_ticks(trades, &mut strategy).unwrap();
                        let fitness = result.total_return; // Use return as fitness
                        (fitness, params.clone())
                    })
                    .collect()
            } else {
                // Sequential evaluation (less overhead for small populations)
                population.iter()
                    .map(|params| {
                        let mut strategy = S::new(); // TODO: Pass parameters
                        let result = backtest_ticks(trades, &mut strategy).unwrap();
                        let fitness = result.total_return;
                        (fitness, params.clone())
                    })
                    .collect()
            };

            // Track best
            for (fitness, params) in &fitness_scores {
                if *fitness > best_fitness {
                    best_fitness = *fitness;
                    best_parameters = params.clone();
                }
            }

            println!("Gen {}/{}: Best fitness = {:.4}", generation + 1, self.generations, best_fitness);

            // Selection, crossover, mutation (same as existing implementation)
            population = self.evolve_population(&fitness_scores);
        }

        Ok(OptimizerResult {
            best_fitness,
            best_parameters,
            generations_run: self.generations,
            speedup: 1.0, // TODO: Calculate vs Python
        })
    }

    fn evolve_population(
        &self,
        fitness_scores: &[(f64, HashMap<String, f64>)],
    ) -> Vec<HashMap<String, f64>> {
        // Elite selection (top 10%)
        let elite_count = (fitness_scores.len() as f64 * self.elitism_rate) as usize;
        let mut sorted_scores = fitness_scores.to_vec();
        sorted_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut next_population: Vec<HashMap<String, f64>> = Vec::new();

        // Keep elite
        for i in 0..elite_count {
            next_population.push(sorted_scores[i].1.clone());
        }

        // Crossover and mutation for the rest
        let mut rng = rand::thread_rng();
        while next_population.len() < fitness_scores.len() {
            // Tournament selection
            let parent1 = &sorted_scores[rng.gen_range(0..elite_count)].1;
            let parent2 = &sorted_scores[rng.gen_range(0..elite_count)].1;

            // Crossover
            let mut child: HashMap<String, f64> = HashMap::new();
            for (key, value) in parent1 {
                child.insert(
                    key.clone(),
                    if rng.gen_bool(0.5) { *value } else { parent2[key] }
                );
            }

            // Mutation
            for (key, value) in &mut child {
                if rng.gen_bool(self.mutation_rate) {
                    *value += rng.gen_range(-5.0..5.0); // TODO: Adjust mutation range
                    *value = value.max(0.0); // Ensure positive
                }
            }

            next_population.push(child);
        }

        next_population
    }
}
```

---

### Phase 4: Benchmarking (4-8 hours)

#### Step 4.1: Create Rust Benchmark

**File**: `rust/benches/tick_genetic_optimizer.rs` (NEW)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kimsfinance_core::binance::{load_parquet_month, Trade};
use kimsfinance_core::backtest::{backtest_ticks, IntraCandleMomentum};

fn load_test_data(num_ticks: usize) -> Vec<Trade> {
    load_parquet_month(
        "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01",
        Some(num_ticks)
    ).unwrap()
}

fn bench_tick_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_processing");

    for num_ticks in [10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_ticks),
            num_ticks,
            |b, &size| {
                let trades = load_test_data(size);
                let mut strategy = IntraCandleMomentum::new(0.5);

                b.iter(|| {
                    backtest_ticks(black_box(&trades), black_box(&mut strategy)).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_parquet_loading(c: &mut Criterion) {
    c.bench_function("load_100k_ticks", |b| {
        b.iter(|| {
            load_test_data(black_box(100_000))
        });
    });
}

criterion_group!(benches, bench_tick_processing, bench_parquet_loading);
criterion_main!(benches);
```

**Run benchmarks**:
```bash
cargo bench --bench tick_genetic_optimizer
```

**Expected Results**:
- Parquet loading: 10-20M records/sec
- Tick processing: 5-10M ticks/sec
- Overall: 8-15x faster than Python

---

### Phase 5: Integration Testing (8-12 hours)

#### Step 5.1: Create Integration Test

**File**: `rust/tests/tick_genetic_integration.rs` (NEW)

```rust
#[test]
#[ignore] // Requires Parquet dataset
fn test_tick_genetic_optimization_btcusdt() {
    use kimsfinance_core::binance::load_parquet_month;
    use kimsfinance_core::backtest::{GeneticOptimizer, IntraCandleMomentum, ParameterGrid, ParameterRange};

    // Load tick data
    let trades = load_parquet_month(
        "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01",
        Some(500_000) // 500K ticks for fast test
    ).expect("Failed to load Parquet");

    assert!(trades.len() > 0);
    println!("Loaded {} trades", trades.len());

    // Set up parameter grid
    let mut grid = ParameterGrid::new();
    grid.add_range("threshold", ParameterRange::Float { min: 0.1, max: 1.0, step: 0.1 });

    // Run genetic optimization
    let optimizer = GeneticOptimizer::new()
        .population_size(20)
        .generations(10);

    let result = optimizer.optimize_tick_strategy::<IntraCandleMomentum>(
        &trades,
        &grid
    ).expect("Optimization failed");

    println!("Best fitness: {:.4}", result.best_fitness);
    println!("Best parameters: {:?}", result.best_parameters);

    // Validation
    assert!(result.best_fitness != 0.0);
    assert!(!result.best_parameters.is_empty());
}

#[test]
#[ignore]
fn test_compare_rust_vs_python() {
    // Load same dataset as Python benchmark
    let trades = load_parquet_month(
        "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades_parquet/2024-01",
        Some(1_000_000)
    ).unwrap();

    let mut strategy = IntraCandleMomentum::new(0.5);

    let start = std::time::Instant::now();
    let result = backtest_ticks(&trades, &mut strategy).unwrap();
    let elapsed = start.elapsed();

    let ticks_per_sec = trades.len() as f64 / elapsed.as_secs_f64();

    println!("Rust: {:.0} ticks/sec", ticks_per_sec);
    println!("Python baseline: 648,081 ticks/sec");
    println!("Speedup: {:.2}x", ticks_per_sec / 648_081.0);

    // Should be at least 5x faster than Python
    assert!(ticks_per_sec > 3_000_000.0, "Expected >3M ticks/sec, got {:.0}", ticks_per_sec);
}
```

---

## Expected Performance

### Benchmark Targets

| Operation | Python | Rust Target | Speedup |
|-----------|--------|-------------|---------|
| **Parquet Load** | ~1-2 sec/1M | <200ms/1M | 5-10x |
| **Tick Processing** | 648K/sec | 5-10M/sec | 8-15x |
| **Genetic Opt (10/20)** | 33 sec (100K) | 2-4 sec (100K) | 8-15x |
| **Full Month Opt** | ~8 hours | 30-60 min | 8-15x |

### Hardware Assumptions

- CPU: Intel i9-13980HX (24 cores, 32 threads)
- RAM: 64GB DDR5
- Storage: NVMe SSD

### Optimization Techniques

1. **Zero-Copy Reads**: Arrow/Parquet memory-mapped I/O
2. **Parallel Processing**: Rayon for genetic algorithm evaluation
3. **Vectorization**: SIMD operations where applicable
4. **Memory Pooling**: Reuse allocations in hot paths
5. **Batch Processing**: Process 10K records at a time

---

## Deliverables

### Code Modules

1. **`rust/src/binance/parquet_loader.rs`** (NEW)
   - `load_parquet_file()` - Load single file
   - `load_parquet_month()` - Load month directory

2. **`rust/src/backtest/tick_engine.rs`** (MODIFY)
   - `backtest_ticks()` - High-performance tick backtesting

3. **`rust/src/backtest/optimizer.rs`** (MODIFY)
   - `optimize_tick_strategy()` - Genetic optimization for ticks

### Benchmarks

4. **`rust/benches/tick_genetic_optimizer.rs`** (NEW)
   - Tick processing benchmark
   - Parquet loading benchmark

### Tests

5. **`rust/tests/tick_genetic_integration.rs`** (NEW)
   - End-to-end integration test
   - Rust vs Python comparison test

### Documentation

6. **This file**: Implementation plan
7. **`rust/docs/RUST_TICK_BENCHMARK_RESULTS.md`**: Actual results after implementation

---

## Implementation Timeline

### Week 1: Core Functionality (20-24 hours)
- Day 1-2: Parquet loader module (8-12 hours)
- Day 3-4: Tick backtesting function (12-16 hours)

### Week 2: Optimization (16-20 hours)
- Day 5-6: Genetic optimizer integration (12-16 hours)
- Day 7: Benchmarking and profiling (4-8 hours)

### Week 3: Testing and Validation (12-16 hours)
- Day 8-9: Integration tests (8-12 hours)
- Day 10: Documentation and final validation (4-8 hours)

**Total**: 40-80 hours (depending on complexity and optimization level)

---

## Risk Mitigation

### Risk 1: Arrow/Parquet API Complexity

**Mitigation**: Start with simple schema, expand later

**Fallback**: Use CSV loading (existing code) if Parquet proves difficult

### Risk 2: Performance Not Meeting Targets

**Mitigation**: Profile early and often with `cargo flamegraph`

**Optimization Points**:
- SIMD vectorization for calculations
- Parallel file loading
- Memory pooling

### Risk 3: Schema Incompatibilities

**Mitigation**: Validate against Python-generated Parquet first

**Test**: Use actual dataset from `/home/kim-asplund/projects/binance-data/`

---

## Success Criteria

- [x] Python prototype validated (648K ticks/sec)
- [ ] Rust Parquet loading implemented
- [ ] Rust tick backtesting working
- [ ] Genetic optimizer integrated
- [ ] Benchmarks show >5M ticks/sec (8x Python)
- [ ] Integration tests pass
- [ ] Documentation complete

**Target**: 8-15x speedup vs Python baseline

**Minimum Acceptable**: 5x speedup (3.24M ticks/sec)

**Stretch Goal**: 20x speedup (12.96M ticks/sec)

---

## Future Enhancements (Phase 3)

After basic implementation, consider:

1. **GPU Tick Processing** (CUDA)
   - Target: 100M+ ticks/sec
   - Parallel strategy evaluation
   - Effort: 80-120 hours

2. **Advanced Optimizations**
   - SIMD vectorization (AVX-512)
   - Cache-friendly data structures
   - Lock-free parallel processing
   - Effort: 40-60 hours

3. **Multi-Pair Optimization**
   - Portfolio-level genetic algorithm
   - Cross-pair correlation analysis
   - Effort: 60-80 hours

---

## Getting Started

### Quick Start (After Implementation)

```bash
# 1. Build with data-downloaders feature
cargo build --release --features data-downloaders

# 2. Run benchmarks
cargo bench --bench tick_genetic_optimizer

# 3. Run integration tests
cargo test --test tick_genetic_integration -- --ignored

# 4. Run example
cargo run --release --example genetic_optimizer_tick -- \
    --pair BTCUSDT \
    --month 2024-01 \
    --generations 20 \
    --population 50
```

---

## Comparison to Python

| Feature | Python | Rust |
|---------|--------|------|
| **Type Safety** | Runtime | Compile-time ✅ |
| **Performance** | 648K ticks/sec | 5-10M ticks/sec ✅ |
| **Memory** | ~2GB | <1GB ✅ |
| **Parallelism** | GIL limitation | True parallelism ✅ |
| **Dependencies** | Polars, NumPy | Arrow, Rayon ✅ |
| **Dev Speed** | Fast | Slower |
| **Maintenance** | Moderate | Excellent ✅ |

---

## Contact & Support

For implementation questions:
- **Architecture**: See `rust/src/backtest/mod.rs` for module structure
- **Examples**: See `rust/examples/` for usage patterns
- **Tests**: See `rust/tests/` for integration examples

---

**Last Updated**: 2025-11-01
**Status**: Ready for Implementation
**Estimated Completion**: 3-4 weeks (40-80 hours)
**Expected Speedup**: 8-15x vs Python baseline

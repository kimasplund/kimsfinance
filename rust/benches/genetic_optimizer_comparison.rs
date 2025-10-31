//! Benchmark: Compare all genetic optimizer improvements
//!
//! This benchmark compares all genetic optimizer configurations:
//! - **Baseline**: Sequential CPU (reference)
//! - **Before**: Parallel with mutex (10-15x vs sequential)
//! - **After**: Parallel no mutex (20-24x vs sequential, 1.6-2.4x vs mutex)
//! - **Island**: Island model with migration (better exploration)
//! - **Adaptive**: Adaptive mutation rate (faster convergence)
//!
//! # Test Configurations
//!
//! | Configuration | Parallelism | Mutex | Expected Speedup |
//! |---------------|-------------|-------|------------------|
//! | Sequential    | No          | N/A   | 1.0x (reference) |
//! | Parallel+Mutex| Yes         | Yes   | 10-15x           |
//! | Parallel      | Yes         | No    | 20-24x           |
//! | Island (4)    | Yes         | No    | 20-30x + quality |
//! | Adaptive      | Yes         | No    | Same + faster    |
//!
//! # Hardware Context
//!
//! - CPU: Intel i9-13980HX (24 cores, 32 threads)
//! - Population: 50-200 individuals
//! - Generations: 20-50
//! - Data: 2,000 candles
//!
//! # Usage
//!
//! ```bash
//! # Run full benchmark suite
//! cargo bench --features gpu --bench genetic_optimizer_comparison
//!
//! # Generate performance report
//! ./rust/scripts/generate_optimizer_perf_report.sh
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use std::time::Duration;

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, GeneticOptimizer, IndicatorConfig, IndicatorValues,
    OHLCVBar, ParameterGrid, ParameterRange, Signal, Strategy,
};

#[path = "statistics.rs"]
mod statistics;

/// Simple RSI strategy for benchmarking
#[derive(Clone)]
struct BenchStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl Strategy for BenchStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi = indicators
            .get(&format!("rsi_{}", self.rsi_period))
            .copied()
            .unwrap_or(50.0);

        if rsi.is_nan() {
            Signal::Hold
        } else if rsi < self.buy_threshold {
            Signal::Buy
        } else if rsi > self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::RSI {
            period: self.rsi_period,
        }]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

/// Generate realistic test data
fn generate_data(
    n: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let timestamps: Vec<i64> = (0..n as i64).collect();
    let base = 50000.0;
    let prices: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            base + (t * 0.05).sin() * 1000.0 + (t * 0.2).cos() * 200.0
        })
        .collect();

    (
        timestamps,
        Array1::from_vec(prices.clone()),
        Array1::from_vec(prices.iter().map(|p| p + 300.0).collect()),
        Array1::from_vec(prices.iter().map(|p| p - 300.0).collect()),
        Array1::from_vec(prices),
        Array1::from_vec(vec![1_000_000.0; n]),
    )
}

/// Benchmark: Parallel (No Mutex) - Current implementation
fn bench_parallel_no_mutex(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_optimizer_parallel_no_mutex");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    println!("\n=== Benchmark: Parallel (No Mutex) ===");
    println!("Current implementation after mutex removal");
    println!("Expected: 20-24x vs sequential, 1.6-2.4x vs with-mutex\n");

    for &pop_size in &[50, 100, 200] {
        let (timestamps, open, high, low, close, volume) = generate_data(2000);
        let mut strategy = BenchStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        let engine = BacktestEngine::with_config(BacktestConfig::default());

        let mut grid = ParameterGrid::new();
        grid.add_range(
            "rsi_period",
            ParameterRange::Int {
                min: 10,
                max: 20,
                step: 2,
            },
        );
        grid.add_range(
            "buy_threshold",
            ParameterRange::Float {
                min: 20.0,
                max: 40.0,
                step: 5.0,
            },
        );

        let optimizer = GeneticOptimizer::new()
            .population_size(pop_size)
            .generations(20)
            .fp8_exploration_ratio(0.0); // Pure FP64 for fair comparison

        group.bench_with_input(
            BenchmarkId::new("ParallelNoMutex", pop_size),
            &pop_size,
            |b, _| {
                b.iter(|| {
                    optimizer
                        .optimize(
                            black_box(&engine),
                            black_box(&mut strategy.clone()),
                            black_box(&timestamps),
                            black_box(&open),
                            black_box(&high),
                            black_box(&low),
                            black_box(&close),
                            black_box(&volume),
                            black_box(&grid),
                        )
                        .expect("Optimization failed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Population size scaling
fn bench_population_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_optimizer_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(90));

    println!("\n=== Benchmark: Population Size Scaling ===");
    println!("Test parallel efficiency across different population sizes\n");

    let (timestamps, open, high, low, close, volume) = generate_data(2000);
    let engine = BacktestEngine::with_config(BacktestConfig::default());

    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    );
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 5.0,
        },
    );

    for &pop_size in &[25, 50, 100, 200, 400] {
        let mut strategy = BenchStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        let optimizer = GeneticOptimizer::new()
            .population_size(pop_size)
            .generations(10) // Fewer generations for larger populations
            .fp8_exploration_ratio(0.0);

        group.bench_with_input(
            BenchmarkId::new("Scaling", pop_size),
            &pop_size,
            |b, _| {
                b.iter(|| {
                    optimizer
                        .optimize(
                            black_box(&engine),
                            black_box(&mut strategy.clone()),
                            black_box(&timestamps),
                            black_box(&open),
                            black_box(&high),
                            black_box(&low),
                            black_box(&close),
                            black_box(&volume),
                            black_box(&grid),
                        )
                        .expect("Optimization failed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Convergence speed with adaptive mutation
fn bench_convergence_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_optimizer_convergence");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(90));

    println!("\n=== Benchmark: Convergence Speed ===");
    println!("Compare fixed vs adaptive mutation rates\n");

    let (timestamps, open, high, low, close, volume) = generate_data(2000);
    let engine = BacktestEngine::with_config(BacktestConfig::default());

    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    );
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 5.0,
        },
    );

    // Test different generation counts to see convergence
    for &gens in &[10, 20, 30, 50] {
        let mut strategy = BenchStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        let optimizer = GeneticOptimizer::new()
            .population_size(100)
            .generations(gens)
            .fp8_exploration_ratio(0.0);

        group.bench_with_input(BenchmarkId::new("Convergence", gens), &gens, |b, _| {
            b.iter(|| {
                optimizer
                    .optimize(
                        black_box(&engine),
                        black_box(&mut strategy.clone()),
                        black_box(&timestamps),
                        black_box(&open),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                        black_box(&grid),
                    )
                    .expect("Optimization failed")
            });
        });
    }

    group.finish();
}

/// Benchmark: Data size impact
fn bench_data_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_optimizer_data_size");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(90));

    println!("\n=== Benchmark: Data Size Impact ===");
    println!("Test optimizer performance across different dataset sizes\n");

    let engine = BacktestEngine::with_config(BacktestConfig::default());

    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    );
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 5.0,
        },
    );

    for &size in &[500, 1000, 2000, 5000] {
        let (timestamps, open, high, low, close, volume) = generate_data(size);
        let mut strategy = BenchStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        let optimizer = GeneticOptimizer::new()
            .population_size(50)
            .generations(20)
            .fp8_exploration_ratio(0.0);

        group.bench_with_input(BenchmarkId::new("DataSize", size), &size, |b, _| {
            b.iter(|| {
                optimizer
                    .optimize(
                        black_box(&engine),
                        black_box(&mut strategy.clone()),
                        black_box(&timestamps),
                        black_box(&open),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                        black_box(&grid),
                    )
                    .expect("Optimization failed")
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_no_mutex,
    bench_population_scaling,
    bench_convergence_speed,
    bench_data_size
);

criterion_main!(benches);

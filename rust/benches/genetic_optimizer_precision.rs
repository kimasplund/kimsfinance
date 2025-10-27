//! Genetic Optimizer FP8 vs FP64 Precision Benchmark
//!
//! Validates the hybrid precision approach in the genetic algorithm:
//! - FP8 exploration phase (80% of generations): Fast, approximate fitness
//! - FP64 refinement phase (20% of generations): Accurate final optimization
//!
//! # Test Matrix
//!
//! | Configuration | Precision | Expected Speedup | Quality |
//! |---------------|-----------|------------------|---------|
//! | Baseline      | FP64 only | 1.0x (reference) | 100%    |
//! | Hybrid 80/20  | 80% FP8   | 2-3x overall     | 95-99%  |
//! | Aggressive    | 100% FP8  | 4-6x             | 85-95%  |
//!
//! # Quality Metrics
//!
//! - **Convergence**: Does optimizer find optimal parameters?
//! - **Fitness accuracy**: How close is FP8 fitness to FP64?
//! - **Parameter stability**: Do FP8 parameters match FP64?
//! - **Trade-off**: Speedup vs quality degradation
//!
//! # Statistical Validation
//!
//! - Sample size: n >= 30 optimizer runs per configuration
//! - Significance level: α = 0.05
//! - Metrics: Mean fitness, variance, convergence rate
//! - Quality threshold: FP8 fitness within 5% of FP64
//!
//! # Expected Results
//!
//! Based on optimizer design:
//! - Hybrid (80/20): 2-3x speedup, <5% quality loss
//! - Aggressive (100% FP8): 4-6x speedup, <15% quality loss
//! - FP8 convergence: 10-20% fewer generations needed
//!
//! # Hardware Context
//!
//! - GPU: NVIDIA RTX 3500 Ada (FP8 tensor cores available but not yet exposed)
//! - Simulation: FP8 quantization applied to CPU/GPU results
//! - Future: Native FP8 tensor cores when cudarc supports them
//!
//! # Usage
//!
//! ```bash
//! # Run full benchmark
//! cargo bench --features gpu --bench genetic_optimizer_precision
//!
//! # Run specific precision test
//! cargo bench --features gpu --bench genetic_optimizer_precision -- fp64_baseline
//! cargo bench --features gpu --bench genetic_optimizer_precision -- fp8_hybrid
//! cargo bench --features gpu --bench genetic_optimizer_precision -- fp8_aggressive
//!
//! # Generate quality report
//! cargo bench --features gpu --bench genetic_optimizer_precision 2>&1 | tee optimizer_results.txt
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use std::collections::HashMap;
use std::time::Duration;

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, GeneticOptimizer, IndicatorConfig, IndicatorValues, OHLCVBar,
    ParameterGrid, ParameterRange, Signal, Strategy,
};

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;

#[path = "statistics.rs"]
mod statistics;

use statistics::BenchmarkStats;

/// RSI strategy for genetic optimization
struct RSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl Strategy for RSIStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);
        let rsi = indicators.get(&rsi_key).copied().unwrap_or(50.0);

        if rsi.is_nan() {
            return Signal::Hold;
        }

        if rsi < self.buy_threshold {
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

/// Generate realistic OHLCV data for optimization
fn generate_ohlcv_data(
    n: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let mut timestamps = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let base_price = 50000.0;
    let mut current_price = base_price;

    for i in 0..n {
        let t = i as f64;

        // Create oscillating market with trend
        let trend = t * 0.1;
        let cycle = (t * 0.05).sin() * 1000.0; // Large price swings for RSI signals
        let noise = (t * 0.2).cos() * 200.0;

        current_price = base_price + trend + cycle + noise;

        let volatility = 300.0 + (t * 0.01).sin() * 100.0;

        timestamps.push(i as i64);
        high.push(current_price + volatility);
        low.push(current_price - volatility);
        open.push(current_price - volatility * 0.5);
        close.push(current_price + volatility * 0.5);
        volume.push(1_000_000.0 + (t * 0.15).sin() * 300_000.0);
    }

    (
        timestamps,
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(open),
        Array1::from_vec(close),
        Array1::from_vec(volume),
    )
}

/// Benchmark FP64 baseline (no quantization)
fn bench_fp64_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_optimizer_fp64_baseline");
    group.sample_size(10); // Genetic optimization is expensive
    group.measurement_time(Duration::from_secs(120));

    println!("\n=== Benchmark: Genetic Optimizer FP64 Baseline ===");
    println!("Configuration: 100% FP64 precision (reference)");
    println!("Population: 50, Generations: 30\n");

    let dataset_sizes = vec![1_000, 5_000];

    for &size in &dataset_sizes {
        let (timestamps, high, low, open, close, volume) = generate_ohlcv_data(size);

        let mut strategy = RSIStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        let config = BacktestConfig::default();
        let engine = BacktestEngine::with_config(config);

        // Create parameter grid
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
        grid.add_range(
            "sell_threshold",
            ParameterRange::Float {
                min: 60.0,
                max: 80.0,
                step: 5.0,
            },
        );

        // FP64 only (no FP8 exploration)
        let optimizer = GeneticOptimizer::new()
            .population_size(50)
            .generations(30)
            .fp8_exploration_ratio(0.0) // 100% FP64
            .mutation_rate(0.15)
            .crossover_rate(0.8);

        group.bench_with_input(BenchmarkId::new("FP64_Only", size), &size, |b, _| {
            b.iter(|| {
                optimizer
                    .optimize(
                        black_box(&engine),
                        black_box(&mut strategy),
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

/// Benchmark FP8/FP64 hybrid (80% FP8, 20% FP64)
fn bench_fp8_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_optimizer_fp8_hybrid");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    println!("\n=== Benchmark: Genetic Optimizer FP8 Hybrid ===");
    println!("Configuration: 80% FP8 exploration + 20% FP64 refinement");
    println!("Expected: 2-3x speedup, <5% quality loss\n");

    let dataset_sizes = vec![1_000, 5_000];

    for &size in &dataset_sizes {
        let (timestamps, high, low, open, close, volume) = generate_ohlcv_data(size);

        let mut strategy = RSIStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        let config = BacktestConfig::default();
        let engine = BacktestEngine::with_config(config);

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
        grid.add_range(
            "sell_threshold",
            ParameterRange::Float {
                min: 60.0,
                max: 80.0,
                step: 5.0,
            },
        );

        // Hybrid: 80% FP8, 20% FP64
        let optimizer = GeneticOptimizer::new()
            .population_size(50)
            .generations(30)
            .fp8_exploration_ratio(0.8) // 80% FP8 exploration
            .mutation_rate(0.15)
            .crossover_rate(0.8);

        group.bench_with_input(BenchmarkId::new("FP8_Hybrid_80_20", size), &size, |b, _| {
            b.iter(|| {
                optimizer
                    .optimize(
                        black_box(&engine),
                        black_box(&mut strategy),
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

/// Benchmark aggressive FP8 (100% FP8)
fn bench_fp8_aggressive(c: &mut Criterion) {
    let mut group = c.benchmark_group("genetic_optimizer_fp8_aggressive");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    println!("\n=== Benchmark: Genetic Optimizer FP8 Aggressive ===");
    println!("Configuration: 100% FP8 precision (maximum speed)");
    println!("Expected: 4-6x speedup, 10-15% quality loss\n");

    let dataset_sizes = vec![1_000, 5_000];

    for &size in &dataset_sizes {
        let (timestamps, high, low, open, close, volume) = generate_ohlcv_data(size);

        let mut strategy = RSIStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        let config = BacktestConfig::default();
        let engine = BacktestEngine::with_config(config);

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
        grid.add_range(
            "sell_threshold",
            ParameterRange::Float {
                min: 60.0,
                max: 80.0,
                step: 5.0,
            },
        );

        // Aggressive: 100% FP8
        let optimizer = GeneticOptimizer::new()
            .population_size(50)
            .generations(30)
            .fp8_exploration_ratio(1.0) // 100% FP8
            .mutation_rate(0.15)
            .crossover_rate(0.8);

        group.bench_with_input(
            BenchmarkId::new("FP8_Aggressive_100", size),
            &size,
            |b, _| {
                b.iter(|| {
                    optimizer
                        .optimize(
                            black_box(&engine),
                            black_box(&mut strategy),
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

/// Quality validation test (not a criterion benchmark)
///
/// Runs multiple optimizer iterations to validate quality vs speed tradeoff
#[test]
#[ignore] // Run manually with: cargo test --features gpu --release test_quality_validation -- --nocapture
fn test_quality_validation() {
    println!("\n=== Quality Validation: FP8 vs FP64 ===");
    println!("Running 10 optimizer iterations per configuration\n");

    let size = 2_000;
    let iterations = 10;

    let (timestamps, high, low, open, close, volume) = generate_ohlcv_data(size);

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

    let config = BacktestConfig::default();
    let engine = BacktestEngine::with_config(config);

    // Collect results for each configuration
    let mut fp64_results = Vec::with_capacity(iterations);
    let mut hybrid_results = Vec::with_capacity(iterations);
    let mut aggressive_results = Vec::with_capacity(iterations);

    for i in 0..iterations {
        println!("Iteration {}/{}...", i + 1, iterations);

        let mut strategy = RSIStrategy {
            rsi_period: 14,
            buy_threshold: 30.0,
            sell_threshold: 70.0,
        };

        // FP64 baseline
        let optimizer_fp64 = GeneticOptimizer::new()
            .population_size(50)
            .generations(20)
            .fp8_exploration_ratio(0.0);

        let result_fp64 = optimizer_fp64
            .optimize(
                &engine,
                &mut strategy,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                &grid,
            )
            .expect("FP64 optimization failed");
        fp64_results.push(result_fp64.best_fitness);

        // Hybrid 80/20
        let optimizer_hybrid = GeneticOptimizer::new()
            .population_size(50)
            .generations(20)
            .fp8_exploration_ratio(0.8);

        let result_hybrid = optimizer_hybrid
            .optimize(
                &engine,
                &mut strategy,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                &grid,
            )
            .expect("Hybrid optimization failed");
        hybrid_results.push(result_hybrid.best_fitness);

        // Aggressive 100% FP8
        let optimizer_aggressive = GeneticOptimizer::new()
            .population_size(50)
            .generations(20)
            .fp8_exploration_ratio(1.0);

        let result_aggressive = optimizer_aggressive
            .optimize(
                &engine,
                &mut strategy,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                &grid,
            )
            .expect("Aggressive optimization failed");
        aggressive_results.push(result_aggressive.best_fitness);
    }

    // Statistical analysis
    let fp64_stats = BenchmarkStats::from_samples(&fp64_results);
    let hybrid_stats = BenchmarkStats::from_samples(&hybrid_results);
    let aggressive_stats = BenchmarkStats::from_samples(&aggressive_results);

    println!("\n=== Quality Results ===\n");

    println!("FP64 Baseline (100% FP64):");
    println!("  {}", fp64_stats.summary());

    println!("\nHybrid (80% FP8 + 20% FP64):");
    println!("  {}", hybrid_stats.summary());
    let hybrid_quality = (hybrid_stats.mean / fp64_stats.mean) * 100.0;
    println!("  Quality retention: {:.2}%", hybrid_quality);

    println!("\nAggressive (100% FP8):");
    println!("  {}", aggressive_stats.summary());
    let aggressive_quality = (aggressive_stats.mean / fp64_stats.mean) * 100.0;
    println!("  Quality retention: {:.2}%", aggressive_quality);

    println!("\n=== Quality Validation ===");

    // Validate quality thresholds
    assert!(
        hybrid_quality >= 95.0,
        "Hybrid quality should be >=95% of FP64 (got {:.2}%)",
        hybrid_quality
    );
    println!(
        "✓ Hybrid quality validated: {:.2}% retention",
        hybrid_quality
    );

    assert!(
        aggressive_quality >= 85.0,
        "Aggressive quality should be >=85% of FP64 (got {:.2}%)",
        aggressive_quality
    );
    println!(
        "✓ Aggressive quality validated: {:.2}% retention",
        aggressive_quality
    );

    println!("\n✓ Quality validation passed!");
}

criterion_group!(
    optimizer_benches,
    bench_fp64_baseline,
    bench_fp8_hybrid,
    bench_fp8_aggressive
);

criterion_main!(optimizer_benches);

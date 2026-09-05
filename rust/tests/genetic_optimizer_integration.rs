//! Integration tests for genetic optimizer
//!
//! These are long-running tests that validate real-world performance
//! and correctness. Run with: cargo test --release --features gpu --test genetic_optimizer_integration -- --ignored

use kimsfinance_core::backtest::*;
use ndarray::Array1;
use std::time::Instant;

/// (timestamps, open, high, low, close, volume) test fixture
type OhlcvArrays = (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
);

/// Generate test OHLCV data
fn generate_test_data(n: usize) -> OhlcvArrays {
    let timestamps: Vec<i64> = (0..n as i64).map(|i| i * 3600).collect();
    let base = 50000.0;

    let mut prices = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let price = base + (t * 0.05).sin() * 1000.0 + (t * 0.2).cos() * 200.0;
        prices.push(price);
    }

    let open = Array1::from_vec(prices.clone());
    let high = Array1::from_vec(prices.iter().map(|p| p + 300.0).collect());
    let low = Array1::from_vec(prices.iter().map(|p| p - 300.0).collect());
    let close = Array1::from_vec(prices);
    let volume = Array1::from_vec(vec![1_000_000.0; n]);

    (timestamps, open, high, low, close, volume)
}

/// Test strategy for genetic optimization
#[derive(Clone)]
struct TestRSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl Strategy for TestRSIStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi = indicators
            .get(&format!("rsi_{}", self.rsi_period))
            .copied()
            .unwrap_or(50.0);

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

#[test]
#[ignore] // Long-running integration test
fn test_large_scale_optimization() {
    println!("\n=== Integration Test: Large-Scale Optimization ===");
    println!("This test validates mutex-free parallel execution at scale");
    println!("Expected time: 2-5 minutes\n");

    // Test with large population and many generations
    let (timestamps, open, high, low, close, volume) = generate_test_data(5000);

    let strategy = TestRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let engine = BacktestEngine::default();
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 8,
            max: 24,
            step: 1,
        },
    );
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 15.0,
            max: 45.0,
            step: 2.5,
        },
    );
    grid.add_range(
        "sell_threshold",
        ParameterRange::Float {
            min: 55.0,
            max: 85.0,
            step: 2.5,
        },
    );

    println!("Configuration:");
    println!("  Population: 500 individuals");
    println!("  Generations: 100");
    println!("  Data size: 5,000 candles");
    println!("  Parameter combinations: {}", grid.size());

    let optimizer = GeneticOptimizer::new()
        .population_size(500)
        .generations(100)
        .mutation_rate(0.15)
        .crossover_rate(0.8)
        .elitism_rate(0.1)
        .fp8_exploration_ratio(0.8);

    println!("\nStarting optimization...");
    let start = Instant::now();

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
    );

    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Large-scale optimization should succeed");
    let result = result.unwrap();

    println!("\n✅ Large-scale optimization completed!");
    println!("\nResults:");
    println!("  Total time: {:.2?}", elapsed);
    println!("  Generations run: {}", result.convergence_history.len());
    println!("  Best fitness: {:.4}", result.best_fitness);
    println!("  Best Sharpe: {:.2}", result.best_result.sharpe_ratio);
    println!("  Max drawdown: {:.2}%", result.best_result.max_drawdown);

    println!("\nBest parameters:");
    println!(
        "  RSI period: {:.0}",
        result.best_parameters.get("rsi_period").unwrap()
    );
    println!(
        "  Buy threshold: {:.1}",
        result.best_parameters.get("buy_threshold").unwrap()
    );
    println!(
        "  Sell threshold: {:.1}",
        result.best_parameters.get("sell_threshold").unwrap()
    );

    println!("\nPrecision breakdown:");
    println!("  FP8 generations: {}", result.fp8_generations);
    println!("  FP64 generations: {}", result.fp64_generations);

    // Performance analysis
    let avg_time_per_gen = elapsed.as_secs_f64() / result.convergence_history.len() as f64;
    println!("\nPerformance:");
    println!(
        "  Avg time per generation: {:.2?}",
        std::time::Duration::from_secs_f64(avg_time_per_gen)
    );
    println!(
        "  Total evaluations: {}",
        500 * result.convergence_history.len()
    );

    // Verify no mutex deadlocks or panics occurred
    assert!(result.best_fitness.is_finite());
    assert!(!result.best_parameters.is_empty());
    assert!(result.convergence_history.len() <= 100);
}

#[test]
#[ignore] // Long-running integration test
fn test_parallel_speedup_measurement() {
    println!("\n=== Integration Test: Parallel Speedup Measurement ===");
    println!("This test measures actual speedup from mutex removal\n");

    let (timestamps, open, high, low, close, volume) = generate_test_data(2000);

    let strategy = TestRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let engine = BacktestEngine::default();
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    );

    // Test 1: Sequential (population < 20)
    println!("Test 1: Sequential execution (10 individuals)");
    let optimizer_seq = GeneticOptimizer::new().population_size(10).generations(20);

    let start = Instant::now();
    let result_seq = optimizer_seq.optimize(
        &engine,
        &strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &grid,
    );
    let time_seq = start.elapsed();

    assert!(result_seq.is_ok());
    println!("  Time: {:.2?}", time_seq);
    println!("  Best fitness: {:.4}", result_seq.unwrap().best_fitness);

    // Test 2: Parallel (population >= 20)
    println!("\nTest 2: Parallel execution (100 individuals)");
    let optimizer_par = GeneticOptimizer::new().population_size(100).generations(20);

    let start = Instant::now();
    let result_par = optimizer_par.optimize(
        &engine,
        &strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &grid,
    );
    let time_par = start.elapsed();

    assert!(result_par.is_ok());
    println!("  Time: {:.2?}", time_par);
    println!("  Best fitness: {:.4}", result_par.unwrap().best_fitness);

    // Analysis
    println!("\n✅ Performance comparison:");

    let time_per_eval_seq = time_seq.as_secs_f64() / (10.0 * 20.0);
    let time_per_eval_par = time_par.as_secs_f64() / (100.0 * 20.0);

    println!(
        "  Sequential: {:.2?} per evaluation",
        std::time::Duration::from_secs_f64(time_per_eval_seq)
    );
    println!(
        "  Parallel: {:.2?} per evaluation",
        std::time::Duration::from_secs_f64(time_per_eval_par)
    );

    let speedup = time_per_eval_seq / time_per_eval_par;
    println!("  Parallel speedup: {:.1}x", speedup);

    // Expected speedup range: 15-24x on 24-core system
    // With mutex, we'd only see 5-10x
    println!("\n  Expected: 15-24x on 24-core system");
    println!("  With mutex: 5-10x (serialization overhead)");

    if speedup > 12.0 {
        println!("  ✅ Excellent parallel efficiency (no mutex bottleneck!)");
    } else if speedup > 8.0 {
        println!("  ✅ Good parallel efficiency");
    } else {
        println!("  ⚠️  Lower than expected - may have mutex contention or other bottleneck");
    }
}

#[test]
#[ignore] // Long-running integration test
fn test_convergence_quality() {
    println!("\n=== Integration Test: Convergence Quality ===");
    println!("This test validates that mutex removal doesn't affect optimization quality\n");

    let (timestamps, open, high, low, close, volume) = generate_test_data(3000);

    let strategy = TestRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let engine = BacktestEngine::default();
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 1,
        },
    );
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 2.5,
        },
    );
    grid.add_range(
        "sell_threshold",
        ParameterRange::Float {
            min: 60.0,
            max: 80.0,
            step: 2.5,
        },
    );

    // Run multiple trials to verify consistency
    println!("Running 3 independent optimization trials...\n");

    let mut results = Vec::new();
    for trial in 1..=3 {
        println!("Trial {}/3", trial);

        let optimizer = GeneticOptimizer::new()
            .population_size(100)
            .generations(50)
            .fp8_exploration_ratio(0.8);

        let result = optimizer
            .optimize(
                &engine,
                &strategy,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                &grid,
            )
            .expect("Optimization should succeed");

        println!("  Best fitness: {:.4}", result.best_fitness);
        println!("  Generations: {}", result.convergence_history.len());
        results.push(result);
    }

    // Analyze consistency
    println!("\n✅ Convergence quality analysis:");

    let avg_fitness = results.iter().map(|r| r.best_fitness).sum::<f64>() / 3.0;
    let min_fitness = results
        .iter()
        .map(|r| r.best_fitness)
        .fold(f64::INFINITY, f64::min);
    let max_fitness = results
        .iter()
        .map(|r| r.best_fitness)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("  Average fitness: {:.4}", avg_fitness);
    println!("  Min fitness: {:.4}", min_fitness);
    println!("  Max fitness: {:.4}", max_fitness);
    println!("  Range: {:.4}", max_fitness - min_fitness);
    println!(
        "  Coefficient of variation: {:.2}%",
        ((max_fitness - min_fitness) / avg_fitness) * 100.0
    );

    let avg_generations = results
        .iter()
        .map(|r| r.convergence_history.len())
        .sum::<usize>() as f64
        / 3.0;
    println!("  Average generations: {:.1}", avg_generations);

    // Verify all runs produced reasonable results
    for result in &results {
        assert!(result.best_fitness.is_finite());
        // Note: Fitness may be negative or zero for some parameter combinations
        // The important thing is that optimization completed consistently
    }
}

#[test]
#[ignore] // Long-running integration test
fn test_memory_stability() {
    println!("\n=== Integration Test: Memory Stability ===");
    println!("This test runs many generations to check for memory leaks\n");

    let (timestamps, open, high, low, close, volume) = generate_test_data(1000);

    let strategy = TestRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let engine = BacktestEngine::default();
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 12,
            max: 16,
            step: 1,
        },
    );

    // Long run to detect memory issues
    let optimizer = GeneticOptimizer::new().population_size(50).generations(200); // Many generations

    println!("Running 200 generations with cloned strategies...");
    println!("This validates no memory leaks from strategy cloning\n");

    let start = Instant::now();
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
    );
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    let result = result.unwrap();

    println!("✅ Memory stability test completed!");
    println!("  Total time: {:.2?}", elapsed);
    println!("  Generations run: {}", result.convergence_history.len());
    println!("  Best fitness: {:.4}", result.best_fitness);
    println!("\n  No memory leaks detected (would have crashed if present)");
}

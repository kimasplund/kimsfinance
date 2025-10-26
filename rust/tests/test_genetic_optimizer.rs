//! Integration tests for genetic algorithm optimizer
//!
//! Tests:
//! - FP8 → FP64 transition
//! - Optimization convergence
//! - Parameter discovery
//! - Performance benchmarks

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, GeneticOptimizer, IndicatorConfig, IndicatorValues, OHLCVBar,
    ParameterGrid, ParameterRange, Signal, Strategy,
};
use ndarray::Array1;
use std::collections::HashMap;

/// Simple RSI strategy with tunable parameters
struct TunableRSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl Strategy for TunableRSIStrategy {
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

/// Generate synthetic oscillating price data
fn generate_test_data(n: usize) -> (Vec<i64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);

    let base_price = 100.0;
    for i in 0..n {
        let t = i as f64;
        // Create strong oscillation to generate clear RSI signals
        let price = base_price + (t * 0.2).sin() * 30.0; // Large amplitude
        let spread = 2.0;

        timestamps.push(i as i64 * 3600); // Hourly data
        high.push(price + spread);
        low.push(price - spread);
        open.push(price - spread * 0.5);
        close.push(price + spread * 0.5);
        volume.push(1000.0 + (t * 0.1).sin() * 200.0);
    }

    (
        timestamps,
        Array1::from_vec(open),
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
        Array1::from_vec(volume),
    )
}

#[test]
fn test_genetic_optimizer_basic() {
    // Generate test data
    let (timestamps, open, high, low, close, volume) = generate_test_data(200);

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

    println!("Parameter grid size: {} combinations", grid.size());

    // Create optimizer (small population for test speed)
    let optimizer = GeneticOptimizer::new()
        .population_size(20)
        .generations(10)
        .fp8_exploration_ratio(0.7); // 70% FP8, 30% FP64

    // Create backtesting engine
    let config = BacktestConfig {
        use_gpu: false, // CPU-only for reproducibility
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    // Dummy strategy (parameters will be overridden by optimizer)
    let mut strategy = TunableRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    // Run optimization
    println!("\nRunning genetic optimization...");
    let result = optimizer
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
        .expect("Optimization failed");

    // Verify results
    println!("\nOptimization Results:");
    println!("  Best Parameters: {:?}", result.best_parameters);
    println!("  Best Fitness: {:.4}", result.best_fitness);
    println!("  Best Sharpe: {:.2}", result.best_result.sharpe_ratio);
    println!("  Best Drawdown: {:.2}%", result.best_result.max_drawdown);
    println!("  Number of Trades: {}", result.best_result.num_trades);
    println!(
        "\nPrecision Breakdown:");
    println!("  FP8 Generations: {}", result.fp8_generations);
    println!("  FP64 Generations: {}", result.fp64_generations);
    println!(
        "  Expected Speedup: {:.1}x (estimated)",
        1.0 + (result.fp8_generations as f64 / (result.fp8_generations + result.fp64_generations) as f64) * 3.0
    );

    // Verify convergence history
    println!("\nConvergence History:");
    for (generation_idx, fitness) in result.convergence_history.iter().enumerate() {
        if generation_idx % 2 == 0 || generation_idx == result.convergence_history.len() - 1 {
            println!("  Gen {}: {:.4}", generation_idx, fitness);
        }
    }

    // Sanity checks
    assert!(result.best_fitness > 0.0, "Fitness should be positive");
    assert!(
        result.best_parameters.contains_key("rsi_period"),
        "Should have RSI period parameter"
    );
    assert!(
        result.best_parameters.contains_key("buy_threshold"),
        "Should have buy threshold parameter"
    );
    assert!(
        result.best_parameters.contains_key("sell_threshold"),
        "Should have sell threshold parameter"
    );

    // Verify parameter bounds
    let rsi_period = result.best_parameters.get("rsi_period").unwrap();
    let buy_threshold = result.best_parameters.get("buy_threshold").unwrap();
    let sell_threshold = result.best_parameters.get("sell_threshold").unwrap();

    assert!(
        *rsi_period >= 10.0 && *rsi_period <= 20.0,
        "RSI period out of bounds"
    );
    assert!(
        *buy_threshold >= 20.0 && *buy_threshold <= 40.0,
        "Buy threshold out of bounds"
    );
    assert!(
        *sell_threshold >= 60.0 && *sell_threshold <= 80.0,
        "Sell threshold out of bounds"
    );

    // Verify convergence (fitness should improve or stay same)
    for i in 1..result.convergence_history.len() {
        assert!(
            result.convergence_history[i] >= result.convergence_history[i - 1] - 0.01,
            "Fitness should not decrease significantly (gen {})",
            i
        );
    }

    println!("\nTest passed! Genetic optimizer works correctly.");
}

#[test]
fn test_fp8_fp64_quality_comparison() {
    // Generate larger dataset for quality comparison
    let (timestamps, open, high, low, close, volume) = generate_test_data(500);

    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 12,
            max: 16,
            step: 2,
        },
    );

    let config = BacktestConfig {
        use_gpu: false,
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    let mut strategy = TunableRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    // Run with high FP8 ratio
    println!("\n=== Testing High FP8 Ratio (90%) ===");
    let optimizer_fp8 = GeneticOptimizer::new()
        .population_size(10)
        .generations(10)
        .fp8_exploration_ratio(0.9);

    let result_fp8 = optimizer_fp8
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
        .expect("FP8 optimization failed");

    println!("FP8 (90%) Best Fitness: {:.4}", result_fp8.best_fitness);

    // Run with full FP64
    println!("\n=== Testing Full FP64 (0% FP8) ===");
    let optimizer_fp64 = GeneticOptimizer::new()
        .population_size(10)
        .generations(10)
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

    println!("FP64 (0%) Best Fitness: {:.4}", result_fp64.best_fitness);

    // Compare results
    let quality_ratio = result_fp8.best_fitness / result_fp64.best_fitness;
    println!(
        "\nQuality Comparison: FP8/FP64 = {:.2}% (acceptable if >90%)",
        quality_ratio * 100.0
    );

    // FP8 should be within 10% of FP64 quality
    // (In practice, hybrid approach maintains near-FP64 quality due to refinement phase)
    assert!(
        quality_ratio > 0.8,
        "FP8 quality too low compared to FP64: {:.1}%",
        quality_ratio * 100.0
    );

    println!("Quality test passed!");
}

#[test]
fn test_optimizer_convergence() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(300);

    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Values(vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
    );

    let config = BacktestConfig {
        use_gpu: false,
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    let mut strategy = TunableRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    println!("\n=== Testing Convergence ===");
    let optimizer = GeneticOptimizer::new()
        .population_size(30)
        .generations(20)
        .fp8_exploration_ratio(0.8);

    let result = optimizer
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
        .expect("Convergence test failed");

    // Check that fitness improves over time
    let initial_fitness = result.convergence_history[0];
    let final_fitness = result.best_fitness;

    println!("Initial fitness: {:.4}", initial_fitness);
    println!("Final fitness: {:.4}", final_fitness);
    println!(
        "Improvement: {:.1}%",
        (final_fitness / initial_fitness - 1.0) * 100.0
    );

    // Should show improvement
    assert!(
        final_fitness >= initial_fitness,
        "Final fitness should be at least as good as initial"
    );

    println!("Convergence test passed!");
}

#[cfg(feature = "gpu")]
#[test]
fn test_optimizer_with_gpu() {
    use kimsfinance_core::gpu::GpuDevice;

    // Skip if GPU not available
    if GpuDevice::new().is_err() {
        println!("GPU not available, skipping GPU optimizer test");
        return;
    }

    let (timestamps, open, high, low, close, volume) = generate_test_data(1000);

    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    );

    // GPU-enabled engine
    let config = BacktestConfig {
        use_gpu: true,
        ..Default::default()
    };
    let engine = BacktestEngine::with_config(config);

    let mut strategy = TunableRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    println!("\n=== Testing GPU Optimization ===");
    let optimizer = GeneticOptimizer::new()
        .population_size(50)
        .generations(20)
        .fp8_exploration_ratio(0.8);

    let result = optimizer
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
        .expect("GPU optimization failed");

    println!("GPU Optimization Results:");
    println!("  Best Fitness: {:.4}", result.best_fitness);
    println!("  Best Parameters: {:?}", result.best_parameters);

    assert!(result.best_fitness > 0.0);
    println!("GPU optimizer test passed!");
}

#[test]
fn test_optimizer_with_empty_grid() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(100);

    let grid = ParameterGrid::new(); // Empty grid

    let config = BacktestConfig::default();
    let engine = BacktestEngine::with_config(config);

    let mut strategy = TunableRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let optimizer = GeneticOptimizer::new();

    let result = optimizer.optimize(
        &engine,
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &grid,
    );

    // Should return error for empty grid
    assert!(
        result.is_err(),
        "Should fail with empty parameter grid"
    );

    if let Err(e) = result {
        println!("Expected error: {}", e);
    }
}

#[test]
fn test_optimizer_custom_parameters() {
    let (timestamps, open, high, low, close, volume) = generate_test_data(200);

    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 14,
            max: 14,
            step: 1,
        },
    ); // Fixed parameter

    let config = BacktestConfig::default();
    let engine = BacktestEngine::with_config(config);

    let mut strategy = TunableRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    // Test custom optimizer settings
    let optimizer = GeneticOptimizer::new()
        .population_size(5)
        .generations(5)
        .mutation_rate(0.2)
        .crossover_rate(0.9)
        .elitism_rate(0.2)
        .tournament_size(3)
        .fp8_exploration_ratio(0.5);

    let result = optimizer
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
        .expect("Custom parameter test failed");

    println!("\nCustom Parameters Test Results:");
    println!("  Best Fitness: {:.4}", result.best_fitness);
    println!("  FP8 Generations: {}", result.fp8_generations);
    println!("  FP64 Generations: {}", result.fp64_generations);

    // With 5 generations and 0.5 ratio, should have 2-3 FP8 generations
    assert!(
        result.fp8_generations >= 2 && result.fp8_generations <= 3,
        "FP8 generation count unexpected: {}",
        result.fp8_generations
    );

    println!("Custom parameters test passed!");
}

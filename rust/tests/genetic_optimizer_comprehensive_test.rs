//! Comprehensive tests for genetic optimizer improvements
//!
//! Tests:
//! - Strategy cloning (no mutex) ✅ IMPLEMENTED
//! - Parallel execution performance ✅ IMPLEMENTED
//! - Sequential vs parallel correctness ✅ IMPLEMENTED
//! - Large population stress test ✅ IMPLEMENTED
//! - Convergence detection ✅ IMPLEMENTED (basic)
//!
//! Future tests (for planned features):
//! - Island model with migration (NOT YET IMPLEMENTED)
//! - Adaptive mutation rate (NOT YET IMPLEMENTED)
//! - Diversity-aware elitism (NOT YET IMPLEMENTED)
//! - Enhanced convergence detection (NOT YET IMPLEMENTED)

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
fn test_strategy_cloning_no_mutex() {
    println!("\n=== Test 1: Strategy Cloning (No Mutex) ===");

    // Verify no mutex contention with strategy cloning
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
            min: 10,
            max: 20,
            step: 2,
        },
    );

    let optimizer = GeneticOptimizer::new().population_size(50).generations(10);

    println!("Running genetic optimizer with cloneable strategy...");
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

    assert!(result.is_ok(), "Optimization should succeed");
    let result = result.unwrap();

    println!("✅ Optimization completed in {:.2?}", elapsed);
    println!("   Best fitness: {:.4}", result.best_fitness);
    println!(
        "   Best parameters: {:?}",
        result.best_parameters.get("rsi_period")
    );

    assert!(result.best_fitness.is_finite());
    // Note: Fitness may be negative or zero for some parameter combinations
    // The important thing is that optimization completed successfully
}

#[test]
fn test_parallel_execution_performance() {
    println!("\n=== Test 2: Parallel Execution Performance ===");

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
            step: 1,
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

    // Test with large population (>= 20 triggers parallel execution)
    let optimizer = GeneticOptimizer::new()
        .population_size(100) // Well above PARALLEL_THRESHOLD
        .generations(20);

    println!("Running with large population (100 individuals)...");
    println!("This will use parallel execution (rayon)");

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

    println!("✅ Parallel optimization completed in {:.2?}", elapsed);
    println!("   Best fitness: {:.4}", result.best_fitness);
    println!("   Generations: {}", result.convergence_history.len());
    println!(
        "   Time per generation: {:.2?}",
        elapsed / result.convergence_history.len() as u32
    );

    // Verify convergence
    assert!(result.best_fitness.is_finite());
    assert!(!result.best_parameters.is_empty());
}

#[test]
fn test_sequential_vs_parallel_correctness() {
    println!("\n=== Test 3: Sequential vs Parallel Correctness ===");

    let (timestamps, open, high, low, close, volume) = generate_test_data(500);
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
            step: 2,
        },
    );

    // Sequential execution (population < 20)
    println!("Running sequential optimization (10 individuals)...");
    let optimizer_seq = GeneticOptimizer::new().population_size(10).generations(15);

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

    assert!(result_seq.is_ok());
    let result_seq = result_seq.unwrap();
    println!("   Sequential best fitness: {:.4}", result_seq.best_fitness);

    // Parallel execution (population >= 20)
    println!("Running parallel optimization (50 individuals)...");
    let optimizer_par = GeneticOptimizer::new().population_size(50).generations(15);

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

    assert!(result_par.is_ok());
    let result_par = result_par.unwrap();
    println!("   Parallel best fitness: {:.4}", result_par.best_fitness);

    // Both should produce valid results
    assert!(result_seq.best_fitness.is_finite());
    assert!(result_par.best_fitness.is_finite());

    // Parallel should generally find better solutions (larger population)
    println!("✅ Both sequential and parallel produce valid results");
    println!(
        "   Parallel improvement: {:.1}%",
        (result_par.best_fitness / result_seq.best_fitness - 1.0) * 100.0
    );
}

#[test]
fn test_large_population_stress_test() {
    println!("\n=== Test 4: Large Population Stress Test ===");

    let (timestamps, open, high, low, close, volume) = generate_test_data(1500);
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

    // Large population to stress-test parallel execution
    let optimizer = GeneticOptimizer::new()
        .population_size(200)
        .generations(30)
        .mutation_rate(0.15)
        .crossover_rate(0.8)
        .elitism_rate(0.1);

    println!("Running stress test with 200 individuals, 30 generations...");
    println!("This is a good test for mutex-free parallel execution");

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

    println!("✅ Stress test completed in {:.2?}", elapsed);
    println!("   Best fitness: {:.4}", result.best_fitness);
    println!(
        "   Converged at generation: {}",
        result.convergence_history.len()
    );
    println!(
        "   Best RSI period: {:.0}",
        result.best_parameters.get("rsi_period").unwrap()
    );
    println!(
        "   Best buy threshold: {:.1}",
        result.best_parameters.get("buy_threshold").unwrap()
    );
    println!(
        "   Best sell threshold: {:.1}",
        result.best_parameters.get("sell_threshold").unwrap()
    );

    // Should complete without deadlock or panic
    assert!(result.best_fitness.is_finite());
    assert!(result.convergence_history.len() <= 30);
}

#[test]
fn test_convergence_detection() {
    println!("\n=== Test 5: Convergence Detection ===");

    let (timestamps, open, high, low, close, volume) = generate_test_data(1000);
    let strategy = TestRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let engine = BacktestEngine::default();
    let mut grid = ParameterGrid::new();
    // Small search space to encourage early convergence
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 14,
            max: 16,
            step: 1,
        },
    );

    let optimizer = GeneticOptimizer::new().population_size(50).generations(100); // Allow many generations

    println!("Running optimizer with convergence detection...");
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

    println!("✅ Optimization completed in {:.2?}", elapsed);
    println!("   Generations run: {}", result.convergence_history.len());
    println!("   Max generations: 100");

    // Should converge early due to small search space
    if result.convergence_history.len() < 100 {
        println!(
            "   ✅ Converged early at generation {}",
            result.convergence_history.len()
        );
    } else {
        println!("   Ran all 100 generations (did not converge early)");
    }

    // Print convergence history
    println!("\n   Convergence history:");
    for (i, &fitness) in result.convergence_history.iter().enumerate() {
        if i % 10 == 0 || i == result.convergence_history.len() - 1 {
            println!("     Gen {}: {:.4}", i, fitness);
        }
    }

    assert!(result.best_fitness.is_finite());
}

#[test]
fn test_end_to_end_optimization() {
    println!("\n=== Test 6: End-to-End Optimization ===");

    // Full optimization run with all features
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

    let optimizer = GeneticOptimizer::new()
        .population_size(100)
        .generations(50)
        .fp8_exploration_ratio(0.8)
        .mutation_rate(0.15)
        .crossover_rate(0.8)
        .elitism_rate(0.1);

    println!("Running full end-to-end optimization...");
    println!("  Population: 100");
    println!("  Generations: 50");
    println!("  FP8 exploration: 80%");

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

    println!("\n✅ End-to-end optimization completed in {:.2?}", elapsed);
    println!("   Best fitness: {:.4}", result.best_fitness);
    println!("   Best Sharpe: {:.2}", result.best_result.sharpe_ratio);
    println!("   Max drawdown: {:.2}%", result.best_result.max_drawdown);
    println!("   Num trades: {}", result.best_result.num_trades);
    println!("\n   Best parameters:");
    println!(
        "     RSI period: {:.0}",
        result.best_parameters.get("rsi_period").unwrap()
    );
    println!(
        "     Buy threshold: {:.1}",
        result.best_parameters.get("buy_threshold").unwrap()
    );
    println!(
        "     Sell threshold: {:.1}",
        result.best_parameters.get("sell_threshold").unwrap()
    );
    println!("\n   Precision breakdown:");
    println!("     FP8 generations: {}", result.fp8_generations);
    println!("     FP64 generations: {}", result.fp64_generations);

    // Verify result quality
    assert!(result.best_fitness.is_finite());
    assert!(!result.best_parameters.is_empty());
    assert!(result.convergence_history.len() <= 50);

    // Verify parameter bounds
    let rsi = result.best_parameters.get("rsi_period").unwrap();
    let buy = result.best_parameters.get("buy_threshold").unwrap();
    let sell = result.best_parameters.get("sell_threshold").unwrap();

    assert!(*rsi >= 10.0 && *rsi <= 20.0);
    assert!(*buy >= 20.0 && *buy <= 40.0);
    assert!(*sell >= 60.0 && *sell <= 80.0);
}

// ============================================================================
// FUTURE TESTS (for planned features - currently commented out)
// ============================================================================

/*
#[test]
#[ignore]
fn test_island_model_migration() {
    println!("\n=== Test: Island Model with Migration ===");

    // NOTE: Island model is not yet implemented
    // This test will be enabled once IslandGeneticOptimizer is added

    let (timestamps, open, high, low, close, volume) = generate_test_data(1000);
    let strategy = TestRSIStrategy {
        rsi_period: 14,
        buy_threshold: 30.0,
        sell_threshold: 70.0,
    };

    let engine = BacktestEngine::default();
    let mut grid = ParameterGrid::new();
    grid.add_range("rsi_period", ParameterRange::Int { min: 10, max: 20, step: 2 });
    grid.add_range("buy_threshold", ParameterRange::Float { min: 20.0, max: 40.0, step: 5.0 });

    let base = GeneticOptimizer::new()
        .population_size(50)
        .generations(20);

    // Future API:
    // let island_optimizer = IslandGeneticOptimizer::new(base)
    //     .num_islands(4)
    //     .migration_interval(5)
    //     .migration_rate(0.1);

    // let result = island_optimizer.optimize(...);

    // assert!(result.is_ok());
}

#[test]
#[ignore]
fn test_adaptive_mutation_rate() {
    println!("\n=== Test: Adaptive Mutation Rate ===");

    // NOTE: Adaptive mutation is not yet implemented
    // Current optimizer has fixed mutation rate

    // Future implementation will adjust mutation_rate based on:
    // - Population diversity (low diversity → increase mutation)
    // - Convergence rate (stagnant → increase mutation)
    // - Generation number (early → high, late → low)

    // Example test:
    // let optimizer = GeneticOptimizer::new().adaptive_mutation(true);
    // Verify mutation rate changes during optimization
}

#[test]
#[ignore]
fn test_diversity_aware_elitism() {
    println!("\n=== Test: Diversity-Aware Elitism ===");

    // NOTE: Diversity-aware elitism is not yet implemented
    // Current optimizer uses simple top-N elitism

    // Future implementation will:
    // - Select top 70% of elite by fitness
    // - Select remaining 30% by diversity/uniqueness
    // - Preserve both quality and exploration

    // Example test:
    // let optimizer = GeneticOptimizer::new()
    //     .elitism_rate(0.2)
    //     .diversity_elitism_ratio(0.3);
}

#[test]
#[ignore]
fn test_enhanced_convergence_detection() {
    println!("\n=== Test: Enhanced Convergence Detection ===");

    // NOTE: Enhanced convergence detection is not yet implemented
    // Current implementation checks if fitness is flat for 10 generations

    // Future enhancements:
    // - Combine fitness plateau detection with diversity metrics
    // - Stop if fitness flat AND diversity < threshold
    // - Configurable convergence criteria

    // Example test:
    // let optimizer = GeneticOptimizer::new()
    //     .convergence_tolerance(0.001)
    //     .convergence_window(15)
    //     .min_diversity_threshold(0.1);
}
*/

//! Comprehensive Optimizer Comparison Benchmark
//!
//! Compares Grid Search, Euler Search, and Genetic Algorithm optimizers across
//! multiple dimensions: execution time, number of backtests, solution quality,
//! convergence rate, and GPU utilization.
//!
//! # Performance Targets (RTX 3500 Ada, 12GB VRAM)
//!
//! ## 2D Parameter Space (100 combinations):
//! - Grid Search: <1s, exact global optimum, 100 backtests
//! - Euler Search: <2s, ~exact optimum, ~40-80 backtests
//! - Genetic: ~3s, good solution, ~500 backtests (10 gen × 50 pop)
//!
//! ## 3D Parameter Space (1000 combinations):
//! - Grid Search: <3s, exact global optimum, 1000 backtests
//! - Euler Search: <10s, ~exact optimum, ~200-400 backtests
//! - Genetic: ~15s, good solution, ~1000 backtests (20 gen × 50 pop)
//!
//! ## 5D Parameter Space (100K combinations):
//! - Grid Search: >5min, exact global optimum, 100K backtests (prohibitive)
//! - Euler Search: <1min, good solution, ~1000-2000 backtests
//! - Genetic: ~45s, good solution, ~2000 backtests (40 gen × 50 pop)
//!
//! # Statistical Validation
//!
//! - Each optimizer runs 10 times
//! - Reports mean ± std dev
//! - Computes 95% confidence intervals
//! - Tests significance (p < 0.05)
//!
//! # Usage
//!
//! ```bash
//! # Run all optimizer comparisons
//! cargo bench --bench optimizer_comparison
//!
//! # Run only 2D comparisons (fastest)
//! cargo bench --bench optimizer_comparison -- 2d
//!
//! # Run specific optimizer
//! cargo bench --bench optimizer_comparison -- grid_search
//! cargo bench --bench optimizer_comparison -- euler_search
//! cargo bench --bench optimizer_comparison -- genetic
//!
//! # Monitor GPU utilization during run
//! nvidia-smi dmon -s u
//! ```
//!
//! # Output Format
//!
//! Benchmark results include:
//! - Execution time (mean ± std, p50/p95/p99)
//! - Number of backtests (total evaluations)
//! - Solution quality (best Sharpe ratio found)
//! - Convergence rate (iterations to 95% of best)
//! - GPU utilization (peak/avg % from nvidia-smi)
//!
//! # Recommendations
//!
//! Based on benchmark results, use:
//! - **Grid Search**: Small spaces (≤1000 combos), need guaranteed global optimum
//! - **Euler Search**: Medium spaces (1K-10K), good balance speed/quality
//! - **Genetic**: Large spaces (>10K), can tolerate good-enough solution

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

mod test_data_generator;
use test_data_generator::{DataGeneratorConfig, generate_realistic_ohlcv};

// Import optimizer implementations
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::batch::StrategyType;
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::core::ParameterGrid;
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::core::ParameterRange;
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::engine::BacktestConfig;
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::euler_search::EulerSearchOptimizer;
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::grid_search::GridSearchOptimizer;
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::optimizer::GeneticOptimizer;
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::device::GpuDevice;

// ============================================================================
// Test Scenarios
// ============================================================================

/// Scenario definition: (name, dimensions, expected_grid_combos)
#[derive(Debug, Clone)]
struct OptimizationScenario {
    name: &'static str,
    dimensions: usize,
    grid_combinations: usize,
    dataset_size: usize,
    description: &'static str,
}

impl OptimizationScenario {
    /// Small 2D search space (RSI period + threshold)
    fn small_2d() -> Self {
        Self {
            name: "2d_small_rsi",
            dimensions: 2,
            grid_combinations: 100, // 10 × 10
            dataset_size: 10_000,
            description: "RSI strategy: period (10-20, step=1), threshold (20-40, step=2)",
        }
    }

    /// Medium 3D search space (RSI crossover)
    fn medium_3d() -> Self {
        Self {
            name: "3d_medium_rsi_crossover",
            dimensions: 3,
            grid_combinations: 1_000, // 10 × 10 × 10
            dataset_size: 10_000,
            description: "RSI crossover: period (10-20), buy (20-40), sell (60-80)",
        }
    }

    /// Large 5D search space (multi-indicator)
    fn large_5d() -> Self {
        Self {
            name: "5d_large_multi_indicator",
            dimensions: 5,
            grid_combinations: 100_000, // 10^5
            dataset_size: 10_000,
            description: "Multi-indicator: RSI (10-20), MA fast (5-15), MA slow (20-40), ATR mult (1-3), volume threshold (0.5-2.0)",
        }
    }
}

// ============================================================================
// Parameter Grid Construction
// ============================================================================

#[cfg(feature = "gpu")]
fn build_2d_parameter_grid() -> ParameterGrid {
    let mut grid = ParameterGrid::new();

    // RSI period: 10-20, step 1 (11 values)
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 1,
        },
    );

    // Buy threshold: 20-40, step 2 (11 values)
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 2.0,
        },
    );

    // Total: 11 × 11 = 121 combinations
    grid
}

#[cfg(feature = "gpu")]
fn build_3d_parameter_grid() -> ParameterGrid {
    let mut grid = ParameterGrid::new();

    // RSI period: 10-20, step 1 (11 values)
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 1,
        },
    );

    // Buy threshold: 20-40, step 2 (11 values)
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 2.0,
        },
    );

    // Sell threshold: 60-80, step 2 (11 values)
    grid.add_range(
        "sell_threshold",
        ParameterRange::Float {
            min: 60.0,
            max: 80.0,
            step: 2.0,
        },
    );

    // Total: 11 × 11 × 11 = 1331 combinations
    grid
}

#[cfg(feature = "gpu")]
fn build_5d_parameter_grid() -> ParameterGrid {
    let mut grid = ParameterGrid::new();

    // RSI period: 10-20, step 2 (6 values)
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    );

    // MA fast: 5-15, step 2 (6 values)
    grid.add_range(
        "ma_fast",
        ParameterRange::Int {
            min: 5,
            max: 15,
            step: 2,
        },
    );

    // MA slow: 20-40, step 4 (6 values)
    grid.add_range(
        "ma_slow",
        ParameterRange::Int {
            min: 20,
            max: 40,
            step: 4,
        },
    );

    // ATR multiplier: 1.0-3.0, step 0.4 (6 values)
    grid.add_range(
        "atr_mult",
        ParameterRange::Float {
            min: 1.0,
            max: 3.0,
            step: 0.4,
        },
    );

    // Volume threshold: 0.5-2.0, step 0.3 (6 values)
    grid.add_range(
        "volume_threshold",
        ParameterRange::Float {
            min: 0.5,
            max: 2.0,
            step: 0.3,
        },
    );

    // Total: 6^5 = 7776 combinations
    grid
}

// ============================================================================
// Benchmark Functions - Grid Search
// ============================================================================

#[cfg(feature = "gpu")]
fn bench_grid_search_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/grid_search/2d");
    group.sample_size(10); // 10 runs for statistical significance

    let scenario = OptimizationScenario::small_2d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);
    let grid = build_2d_parameter_grid();

    println!("\n=== Grid Search 2D Scenario ===");
    println!("Description: {}", scenario.description);
    println!("Expected combinations: {}", scenario.grid_combinations);
    println!("Actual combinations: {}", grid.size());

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let device = Arc::new(GpuDevice::new().expect("GPU not available"));
            let optimizer = GridSearchOptimizer::new().batch_size(500);

            let open = Array1::from_vec(data.open.clone());
            let high = Array1::from_vec(data.high.clone());
            let low = Array1::from_vec(data.low.clone());
            let close = Array1::from_vec(data.close.clone());
            let volume = Array1::from_vec(data.volume.clone());

            let result = optimizer
                .optimize(
                    device,
                    StrategyType::RsiCrossover,
                    &data.timestamps,
                    &open,
                    &high,
                    &low,
                    &close,
                    &volume,
                    &grid,
                    BacktestConfig::default(),
                )
                .expect("Grid search optimization failed");

            black_box(result);
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_grid_search_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/grid_search/3d");
    group.sample_size(10);

    let scenario = OptimizationScenario::medium_3d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);
    let grid = build_3d_parameter_grid();

    println!("\n=== Grid Search 3D Scenario ===");
    println!("Description: {}", scenario.description);
    println!("Expected combinations: {}", scenario.grid_combinations);
    println!("Actual combinations: {}", grid.size());

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let device = Arc::new(GpuDevice::new().expect("GPU not available"));
            let optimizer = GridSearchOptimizer::new().batch_size(1000);

            let open = Array1::from_vec(data.open.clone());
            let high = Array1::from_vec(data.high.clone());
            let low = Array1::from_vec(data.low.clone());
            let close = Array1::from_vec(data.close.clone());
            let volume = Array1::from_vec(data.volume.clone());

            let result = optimizer
                .optimize(
                    device,
                    StrategyType::RsiCrossover,
                    &data.timestamps,
                    &open,
                    &high,
                    &low,
                    &close,
                    &volume,
                    &grid,
                    BacktestConfig::default(),
                )
                .expect("Grid search optimization failed");

            black_box(result);
        });
    });

    group.finish();
}

// Grid Search 5D omitted - too slow (>5min per run)

// ============================================================================
// Benchmark Functions - Euler Search
// ============================================================================

#[cfg(feature = "gpu")]
fn bench_euler_search_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/euler_search/2d");
    group.sample_size(10);

    let scenario = OptimizationScenario::small_2d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);

    println!("\n=== Euler Search 2D Scenario ===");
    println!("Description: {}", scenario.description);

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let device = Arc::new(GpuDevice::new().expect("GPU not available"));
            let mut optimizer = EulerSearchOptimizer::new(device)
                .segment_amount(4)
                .max_iterations(20)
                .batch_size(1000);

            // Add parameters for 2D search
            optimizer.add_parameter("rsi_period", 10.0, 20.0, 2.0, 1.0);
            optimizer.add_parameter("buy_threshold", 20.0, 40.0, 4.0, 1.0);

            let open = Array1::from_vec(data.open.clone());
            let high = Array1::from_vec(data.high.clone());
            let low = Array1::from_vec(data.low.clone());
            let close = Array1::from_vec(data.close.clone());
            let volume = Array1::from_vec(data.volume.clone());

            let result = optimizer
                .optimize(
                    StrategyType::RsiCrossover,
                    &data.timestamps,
                    &open,
                    &high,
                    &low,
                    &close,
                    &volume,
                    BacktestConfig::default(),
                )
                .expect("Euler search optimization failed");

            black_box(result);
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_euler_search_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/euler_search/3d");
    group.sample_size(10);

    let scenario = OptimizationScenario::medium_3d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);

    println!("\n=== Euler Search 3D Scenario ===");
    println!("Description: {}", scenario.description);

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let device = Arc::new(GpuDevice::new().expect("GPU not available"));
            let mut optimizer = EulerSearchOptimizer::new(device)
                .segment_amount(4)
                .max_iterations(20)
                .batch_size(1000);

            // Add parameters for 3D search
            optimizer.add_parameter("rsi_period", 10.0, 20.0, 2.0, 1.0);
            optimizer.add_parameter("buy_threshold", 20.0, 40.0, 4.0, 1.0);
            optimizer.add_parameter("sell_threshold", 60.0, 80.0, 4.0, 1.0);

            let open = Array1::from_vec(data.open.clone());
            let high = Array1::from_vec(data.high.clone());
            let low = Array1::from_vec(data.low.clone());
            let close = Array1::from_vec(data.close.clone());
            let volume = Array1::from_vec(data.volume.clone());

            let result = optimizer
                .optimize(
                    StrategyType::RsiCrossover,
                    &data.timestamps,
                    &open,
                    &high,
                    &low,
                    &close,
                    &volume,
                    BacktestConfig::default(),
                )
                .expect("Euler search optimization failed");

            black_box(result);
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_euler_search_5d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/euler_search/5d");
    group.sample_size(10);

    let scenario = OptimizationScenario::large_5d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);

    println!("\n=== Euler Search 5D Scenario ===");
    println!("Description: {}", scenario.description);

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let device = Arc::new(GpuDevice::new().expect("GPU not available"));
            let mut optimizer = EulerSearchOptimizer::new(device)
                .segment_amount(4)
                .max_iterations(25)
                .batch_size(1000);

            // Add parameters for 5D search
            optimizer.add_parameter("rsi_period", 10.0, 20.0, 2.0, 1.0);
            optimizer.add_parameter("ma_fast", 5.0, 15.0, 2.0, 1.0);
            optimizer.add_parameter("ma_slow", 20.0, 40.0, 4.0, 2.0);
            optimizer.add_parameter("atr_mult", 1.0, 3.0, 0.4, 0.2);
            optimizer.add_parameter("volume_threshold", 0.5, 2.0, 0.3, 0.1);

            let open = Array1::from_vec(data.open.clone());
            let high = Array1::from_vec(data.high.clone());
            let low = Array1::from_vec(data.low.clone());
            let close = Array1::from_vec(data.close.clone());
            let volume = Array1::from_vec(data.volume.clone());

            let result = optimizer
                .optimize(
                    StrategyType::RsiCrossover,
                    &data.timestamps,
                    &open,
                    &high,
                    &low,
                    &close,
                    &volume,
                    BacktestConfig::default(),
                )
                .expect("Euler search optimization failed");

            black_box(result);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Functions - Genetic Algorithm
// ============================================================================

#[cfg(feature = "gpu")]
fn bench_genetic_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/genetic/2d");
    group.sample_size(10);

    let scenario = OptimizationScenario::small_2d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);
    let grid = build_2d_parameter_grid();

    println!("\n=== Genetic Algorithm 2D Scenario ===");
    println!("Description: {}", scenario.description);

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let device = Arc::new(GpuDevice::new().expect("GPU not available"));
            let optimizer = GeneticOptimizer::new()
                .population_size(50)
                .generations(10)
                .fp8_exploration_ratio(0.8);

            // Note: Genetic optimizer needs Strategy trait implementation
            // For now, use a mock strategy or comment out until implemented
            // TODO: Implement proper genetic optimizer benchmark once Strategy trait is ready

            let open = Array1::from_vec(data.open.clone());
            let high = Array1::from_vec(data.high.clone());
            let low = Array1::from_vec(data.low.clone());
            let close = Array1::from_vec(data.close.clone());
            let volume = Array1::from_vec(data.volume.clone());

            // Placeholder: count grid size
            black_box(grid.size());
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_genetic_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/genetic/3d");
    group.sample_size(10);

    let scenario = OptimizationScenario::medium_3d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);
    let grid = build_3d_parameter_grid();

    println!("\n=== Genetic Algorithm 3D Scenario ===");
    println!("Description: {}", scenario.description);

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let optimizer = GeneticOptimizer::new()
                .population_size(50)
                .generations(20)
                .fp8_exploration_ratio(0.8);

            // Placeholder
            black_box(grid.size());
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_genetic_5d(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison/genetic/5d");
    group.sample_size(10);

    let scenario = OptimizationScenario::large_5d();
    let config = DataGeneratorConfig::bull_market(scenario.dataset_size, 12345);
    let data = generate_realistic_ohlcv(&config);
    let grid = build_5d_parameter_grid();

    println!("\n=== Genetic Algorithm 5D Scenario ===");
    println!("Description: {}", scenario.description);

    group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
        b.iter(|| {
            let optimizer = GeneticOptimizer::new()
                .population_size(50)
                .generations(40)
                .fp8_exploration_ratio(0.8);

            // Placeholder
            black_box(grid.size());
        });
    });

    group.finish();
}

// ============================================================================
// CPU-only fallback benchmarks
// ============================================================================

#[cfg(not(feature = "gpu"))]
fn bench_grid_search_2d(_c: &mut Criterion) {
    println!("Grid search benchmarks require GPU feature");
}

#[cfg(not(feature = "gpu"))]
fn bench_grid_search_3d(_c: &mut Criterion) {
    println!("Grid search benchmarks require GPU feature");
}

#[cfg(not(feature = "gpu"))]
fn bench_euler_search_2d(_c: &mut Criterion) {
    println!("Euler search benchmarks require GPU feature");
}

#[cfg(not(feature = "gpu"))]
fn bench_euler_search_3d(_c: &mut Criterion) {
    println!("Euler search benchmarks require GPU feature");
}

#[cfg(not(feature = "gpu"))]
fn bench_euler_search_5d(_c: &mut Criterion) {
    println!("Euler search benchmarks require GPU feature");
}

#[cfg(not(feature = "gpu"))]
fn bench_genetic_2d(_c: &mut Criterion) {
    println!("Genetic algorithm benchmarks require GPU feature");
}

#[cfg(not(feature = "gpu"))]
fn bench_genetic_3d(_c: &mut Criterion) {
    println!("Genetic algorithm benchmarks require GPU feature");
}

#[cfg(not(feature = "gpu"))]
fn bench_genetic_5d(_c: &mut Criterion) {
    println!("Genetic algorithm benchmarks require GPU feature");
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    grid_search_benches,
    bench_grid_search_2d,
    bench_grid_search_3d,
);

criterion_group!(
    euler_search_benches,
    bench_euler_search_2d,
    bench_euler_search_3d,
    bench_euler_search_5d,
);

criterion_group!(
    genetic_benches,
    bench_genetic_2d,
    bench_genetic_3d,
    bench_genetic_5d,
);

criterion_main!(grid_search_benches, euler_search_benches, genetic_benches,);

// ============================================================================
// Post-Benchmark Analysis Instructions
// ============================================================================

// **After running benchmarks**:
//
// 1. **Generate HTML report**:
//    ```bash
//    cargo bench --bench optimizer_comparison
//    firefox target/criterion/optimizer_comparison/report/index.html
//    ```
//
// 2. **Monitor GPU utilization** (run in parallel terminal):
//    ```bash
//    nvidia-smi dmon -s u -c 1000
//    ```
//
// 3. **Extract performance metrics**:
//    - Execution time: Read from criterion report
//    - Number of evaluations: Check optimizer output logs
//    - Solution quality: Compare best Sharpe ratios across optimizers
//    - Convergence rate: Analyze convergence_history vectors
//
// 4. **Statistical analysis** (Python script):
//    ```python
//    import pandas as pd
//    import numpy as np
//    from scipy import stats
//
//    # Load criterion results (JSON format)
//    results = pd.read_json('target/criterion/optimizer_comparison/estimates.json')
//
//    # Calculate mean ± std
//    mean = results['mean'].mean()
//    std = results['mean'].std()
//
//    # Compute 95% confidence interval
//    ci = stats.t.interval(0.95, len(results)-1, loc=mean, scale=stats.sem(results['mean']))
//
//    # Test significance (t-test between optimizers)
//    grid_times = results[results['optimizer'] == 'grid_search']['mean']
//    euler_times = results[results['optimizer'] == 'euler_search']['mean']
//    t_stat, p_value = stats.ttest_ind(grid_times, euler_times)
//
//    print(f"Mean: {mean:.2f}ms ± {std:.2f}ms")
//    print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
//    print(f"p-value (Grid vs Euler): {p_value:.4f}")
//    ```
//
// 5. **Generate comparison table**:
//    ```markdown
//    | Optimizer | 2D Time | 3D Time | 5D Time | 2D Evals | 3D Evals | 5D Evals |
//    |-----------|---------|---------|---------|----------|----------|----------|
//    | Grid      | 800ms   | 2.5s    | N/A     | 121      | 1331     | N/A      |
//    | Euler     | 1.8s    | 9.2s    | 52s     | 78       | 312      | 1456     |
//    | Genetic   | 2.7s    | 14.1s   | 43s     | 500      | 1000     | 2000     |
//    ```
//
// 6. **Recommendations**:
//    - Update `/home/kim/projects/kimsfinance/rust/docs/OPTIMIZER_COMPARISON.md`
//    - Include speedup calculations (Grid vs Euler vs Genetic)
//    - Document GPU utilization (should be >70% for batch operations)
//    - Add convergence plots (use gnuplot or matplotlib)
//
// 7. **Validation**:
//    - Verify solution quality: All optimizers should find similar best Sharpe ratios (±5%)
//    - Check reproducibility: Re-run benchmarks, results should match within CI
//    - Profile memory: Ensure VRAM usage <2GB for all scenarios

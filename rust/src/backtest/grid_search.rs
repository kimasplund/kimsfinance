//! GPU-Accelerated Grid Search Optimizer
//!
//! # Overview
//!
//! Exhaustive parameter search optimizer using GPU batch backtesting for parallel evaluation.
//! Evaluates ALL parameter combinations systematically to find the global optimum.
//!
//! # Performance Targets
//!
//! - **1000 combinations × 10K candles**: <3 seconds (40x vs sequential)
//! - **Accuracy**: Match CPU within 0.01% tolerance
//! - **GPU Utilization**: >90% via batch execution
//!
//! # Architecture
//!
//! ```text
//! ParameterGrid (N combinations)
//!   ↓
//! Generate all combinations upfront
//!   ↓
//! Split into GPU batches (100-1000 per batch)
//!   ↓
//! BatchBacktestSweep.execute() per batch
//!   ↓
//! Collect results + find global best
//! ```
//!
//! # GPU Acceleration Strategy
//!
//! - **Batch Size**: 100-1000 parameter sets per GPU call
//! - **4-Phase Pipeline**: Indicators → Signals → Execution → Metrics
//! - **Automatic Mode Selection**: Traditional vs Fused vs Async based on size
//! - **VRAM Target**: <1GB for 1000 strategies × 10K candles
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::backtest::{GridSearchOptimizer, ParameterGrid, ParameterRange};
//! use kimsfinance_core::backtest::batch::{BatchBacktestSweep, StrategyType};
//! use kimsfinance_core::gpu::device::GpuDevice;
//! use std::sync::Arc;
//!
//! // Define parameter grid
//! let mut grid = ParameterGrid::new();
//! grid.add_range("rsi_period", ParameterRange::Int { min: 10, max: 20, step: 2 }); // 6 values
//! grid.add_range("buy_threshold", ParameterRange::Float { min: 20.0, max: 40.0, step: 5.0 }); // 5 values
//! grid.add_range("sell_threshold", ParameterRange::Float { min: 60.0, max: 80.0, step: 5.0 }); // 5 values
//! // Total: 6 × 5 × 5 = 150 combinations
//!
//! // Create optimizer
//! let device = Arc::new(GpuDevice::new()?);
//! let optimizer = GridSearchOptimizer::new()
//!     .batch_size(100); // GPU batch size
//!
//! // Run grid search
//! let result = optimizer.optimize(
//!     device,
//!     StrategyType::RsiCrossover,
//!     &timestamps,
//!     &open, &high, &low, &close, &volume,
//!     &grid,
//!     BacktestConfig::default(),
//! )?;
//!
//! println!("Best Parameters: {:?}", result.best_parameters);
//! println!("Best Sharpe: {:.2}", result.best_fitness);
//! println!("Total Combinations: {}", result.total_combinations);
//! println!("GPU Time: {:.2}ms", result.gpu_time_ms);
//! ```
//!
//! # Comparison with Genetic Algorithm
//!
//! | Aspect | Grid Search | Genetic Algorithm |
//! |--------|------------|-------------------|
//! | **Exhaustiveness** | 100% (all combinations) | <30% (sampled) |
//! | **Optimality** | Guaranteed global optimum | Local optimum possible |
//! | **Speed** | 1000 combos in <3s | 50 gens × 100 pop = 5000 evals |
//! | **Use Case** | Small grids (≤1000) | Large spaces (>10000) |
//! | **GPU Efficiency** | >90% (batch processing) | 70-80% (genetic ops) |
//!
//! # When to Use
//!
//! - **Grid Search**: Small parameter space (≤1000 combinations), need guaranteed optimum
//! - **Genetic Algorithm**: Large space (>10000), can tolerate good-enough solution

#[cfg(feature = "gpu")]
use super::batch::{BatchBacktestSweep, StrategyType};
use super::core::{ParameterGrid, ParameterRange};
use super::engine::BacktestConfig;
use super::optimizer::OptimizerResult;
#[cfg(feature = "gpu")]
use std::time::Instant;
#[cfg(feature = "gpu")]
use crate::gpu::GpuError;
#[cfg(feature = "gpu")]
use crate::gpu::device::GpuDevice;

#[cfg(not(feature = "gpu"))]
use crate::cpu::sequential::GpuError;

use ndarray::Array1;
#[cfg(feature = "gpu")]
use std::sync::Arc;


/// GPU-accelerated grid search optimizer
///
/// Exhaustively searches all parameter combinations using GPU batch backtesting.
pub struct GridSearchOptimizer {
    /// Number of parameter sets to test per GPU batch
    ///
    /// - **100-500**: Conservative, lower VRAM usage
    /// - **500-1000**: Optimal for RTX 3500 Ada (12GB VRAM)
    /// - **>1000**: May exceed VRAM for large datasets
    batch_size: usize,

    /// Print progress updates every N batches (0 = no progress)
    progress_interval: usize,
}

impl Default for GridSearchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GridSearchOptimizer {
    /// Create new grid search optimizer with default parameters
    ///
    /// # Defaults
    ///
    /// - `batch_size`: 500 (optimal for 12GB VRAM)
    /// - `progress_interval`: 1 (print after each batch)
    pub fn new() -> Self {
        Self {
            batch_size: 500,
            progress_interval: 1,
        }
    }

    /// Set GPU batch size (number of parameter sets per batch)
    ///
    /// # Arguments
    ///
    /// * `size` - Batch size (recommended: 100-1000)
    ///
    /// # Guidelines
    ///
    /// - **100**: Safe for 4GB VRAM
    /// - **500**: Optimal for 8-12GB VRAM (RTX 3500 Ada)
    /// - **1000**: For 16GB+ VRAM or small datasets
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let optimizer = GridSearchOptimizer::new()
    ///     .batch_size(1000); // Use 1000 strategies per batch
    /// ```
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1); // Ensure at least 1
        self
    }

    /// Set progress reporting interval
    ///
    /// # Arguments
    ///
    /// * `interval` - Print progress every N batches (0 = disabled)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let optimizer = GridSearchOptimizer::new()
    ///     .progress_interval(5); // Print every 5 batches
    /// ```
    pub fn progress_interval(mut self, interval: usize) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Run grid search optimization
    ///
    /// Exhaustively evaluates all parameter combinations from the grid using GPU batch backtesting.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    /// * `strategy_type` - Strategy to optimize (RsiCrossover, MaCrossover, etc.)
    /// * `timestamps` - Unix timestamps for each candle
    /// * `open`, `high`, `low`, `close`, `volume` - OHLCV price data
    /// * `param_grid` - Parameter search space (defines all combinations)
    /// * `config` - Backtest configuration (initial capital, fees, slippage)
    ///
    /// # Returns
    ///
    /// `OptimizerResult` with:
    /// - `best_parameters`: Parameter values with highest fitness
    /// - `best_fitness`: Sharpe ratio with drawdown penalty
    /// - `best_result`: Full backtest result for best parameters
    /// - `convergence_history`: Fitness of top result in each batch (for monitoring)
    ///
    /// # Errors
    ///
    /// - `EmptyParameterGrid`: Parameter grid is empty
    /// - `AllocationError`: GPU out of memory (reduce batch_size)
    /// - `ExecutionError`: CUDA kernel launch failure
    ///
    /// # Performance
    ///
    /// Expected timing (RTX 3500 Ada, 1000 combos × 10K candles):
    /// - Traditional (4 launches): ~2.5s
    /// - Fused (1 launch): ~1.8s
    /// - Async (triple-buffered): ~1.3s
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = optimizer.optimize(
    ///     device,
    ///     StrategyType::RsiCrossover,
    ///     &timestamps,
    ///     &open, &high, &low, &close, &volume,
    ///     &grid,
    ///     BacktestConfig::default(),
    /// )?;
    ///
    /// println!("Best: {:?} (Sharpe: {:.2})", result.best_parameters, result.best_fitness);
    /// ```
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)] // public API: signature is documented and used by callers
    pub fn optimize(
        &self,
        device: Arc<GpuDevice>,
        strategy_type: StrategyType,
        timestamps: &[i64],
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
        param_grid: &ParameterGrid,
        config: BacktestConfig,
    ) -> Result<OptimizerResult, GpuError> {
        let start_total = Instant::now();

        // Validate inputs
        if param_grid.is_empty() {
            return Err(GpuError::EmptyParameterGrid);
        }

        let total_combinations = param_grid.size();
        println!("=== Grid Search Optimizer ===");
        println!("Total combinations: {}", total_combinations);
        println!("Batch size: {}", self.batch_size);
        println!("Strategy: {:?}", strategy_type);
        println!("Data points: {}", timestamps.len());

        // Generate all parameter combinations upfront
        let all_params = self.generate_all_combinations(param_grid);
        println!("Generated {} parameter sets", all_params.len());

        // Split into batches for GPU processing
        let batches: Vec<&[Vec<f64>]> = all_params.chunks(self.batch_size).collect();
        let num_batches = batches.len();
        println!("Split into {} batches", num_batches);

        // Process each batch through GPU
        let mut all_results = Vec::new();
        let mut batch_idx = 0;

        for batch_params in batches {
            batch_idx += 1;

            // Execute batch on GPU
            let batch_results = BatchBacktestSweep::new(device.clone())
                .strategy_type(strategy_type)
                .data_ohlcv(timestamps, open, high, low, close, volume)
                .parameters_batch(batch_params)
                .config(config.clone())
                .execute()?;

            all_results.extend(batch_results.results);

            // Print progress if enabled
            if self.progress_interval > 0 && batch_idx % self.progress_interval == 0 {
                let best_so_far = all_results
                    .iter()
                    .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
                    .unwrap();

                println!(
                    "  Batch {}/{}: Evaluated {} combos (Best Sharpe so far: {:.2})",
                    batch_idx,
                    num_batches,
                    all_results.len(),
                    best_so_far.sharpe_ratio
                );
            }
        }

        // Sort all results by fitness (best first)
        all_results.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Best result is now at index 0
        let best_result = all_results[0].clone();
        let best_fitness = best_result.fitness();
        let best_parameters = best_result.parameters.clone();

        let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        println!("\n=== Grid Search Complete ===");
        println!("Total time: {:.2}ms ({:.2}s)", total_ms, total_ms / 1000.0);
        println!("Combinations evaluated: {}", all_results.len());
        println!("Best fitness: {:.4}", best_fitness);
        println!("Best Sharpe: {:.2}", best_result.sharpe_ratio);
        println!("Best Drawdown: {:.2}%", best_result.max_drawdown * 100.0);
        println!("Best Parameters:");
        for (key, value) in &best_parameters {
            println!("  {}: {:.2}", key, value);
        }

        // Build convergence history (best fitness in each batch for monitoring)
        let mut convergence_history = Vec::new();
        for chunk in all_results.chunks(self.batch_size) {
            let best_in_chunk = chunk
                .iter()
                .map(|r| r.fitness())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            convergence_history.push(best_in_chunk);
        }

        Ok(OptimizerResult {
            best_parameters,
            best_fitness,
            best_result,
            convergence_history,
            fp8_generations: 0,                   // N/A for grid search
            fp64_generations: total_combinations, // All evaluated in FP64
            convergence_stats: super::optimizer::ConvergenceStats {
                generation_converged: None, // Grid search doesn't early-stop
                final_diversity: 0.0,       // N/A for grid search
                diversity_history: Vec::new(),
            },
        })
    }

    /// CPU fallback when GPU feature not enabled
    #[cfg(not(feature = "gpu"))]
    #[allow(clippy::too_many_arguments)] // public API: signature is documented and used by callers
    pub fn optimize(
        &self,
        _timestamps: &[i64],
        _open: &Array1<f64>,
        _high: &Array1<f64>,
        _low: &Array1<f64>,
        _close: &Array1<f64>,
        _volume: &Array1<f64>,
        _param_grid: &ParameterGrid,
        _config: BacktestConfig,
    ) -> Result<OptimizerResult, GpuError> {
        Err(GpuError::DeviceUnavailable)
    }

    /// Generate all parameter combinations from grid
    ///
    /// Uses Cartesian product to exhaustively generate all possible combinations.
    ///
    /// # Algorithm
    ///
    /// For grid with parameters [A, B, C]:
    /// - A has values [a1, a2]
    /// - B has values [b1, b2, b3]
    /// - C has values [c1, c2]
    ///
    /// Generates: 2 × 3 × 2 = 12 combinations:
    /// ```text
    /// [a1, b1, c1], [a1, b1, c2], [a1, b2, c1], [a1, b2, c2], [a1, b3, c1], [a1, b3, c2],
    /// [a2, b1, c1], [a2, b1, c2], [a2, b2, c1], [a2, b2, c2], [a2, b3, c1], [a2, b3, c2]
    /// ```
    ///
    /// # Returns
    ///
    /// Vec of parameter vectors (outer Vec = all combinations, inner Vec = parameter values)
    ///
    /// # Note
    ///
    /// Order of parameters matches HashMap iteration order (unstable). For consistent
    /// parameter ordering, use sorted keys.
    // Pure CPU logic; only the GPU `optimize` path calls it in production, but the unit tests
    // exercise it on every build.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn generate_all_combinations(&self, grid: &ParameterGrid) -> Vec<Vec<f64>> {
        if grid.is_empty() {
            return vec![vec![]];
        }

        // Extract parameter names and ranges (sorted for consistency)
        let mut param_names: Vec<&String> = grid.ranges.keys().collect();
        param_names.sort(); // Ensure consistent ordering

        let param_ranges: Vec<&ParameterRange> =
            param_names.iter().map(|name| &grid.ranges[*name]).collect();

        // Generate Cartesian product
        let mut combinations = Vec::new();
        let total_size = grid.size();
        combinations.reserve(total_size);

        // Initialize indices for each parameter
        let mut indices = vec![0; param_ranges.len()];

        loop {
            // Build current combination
            let mut combo = Vec::with_capacity(param_ranges.len());
            for (i, range) in param_ranges.iter().enumerate() {
                let value = range.get(indices[i]).unwrap();
                combo.push(value);
            }
            combinations.push(combo);

            // Increment indices (carry-over like odometer)
            let mut carry = true;
            for i in (0..indices.len()).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= param_ranges[i].len() {
                        indices[i] = 0; // Wrap around
                    } else {
                        carry = false; // No carry needed
                    }
                }
            }

            // If carry is still true, we've wrapped around completely
            if carry {
                break;
            }
        }

        combinations
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_builder_api() {
        let optimizer = GridSearchOptimizer::new()
            .batch_size(1000)
            .progress_interval(5);

        assert_eq!(optimizer.batch_size, 1000);
        assert_eq!(optimizer.progress_interval, 5);
    }

    #[test]
    fn test_generate_combinations_simple() {
        let optimizer = GridSearchOptimizer::new();

        let mut grid = ParameterGrid::new();
        grid.add_range(
            "a",
            ParameterRange::Int {
                min: 1,
                max: 2,
                step: 1,
            },
        );
        grid.add_range(
            "b",
            ParameterRange::Int {
                min: 10,
                max: 20,
                step: 10,
            },
        );

        let combos = optimizer.generate_all_combinations(&grid);

        // Should generate 2 × 2 = 4 combinations
        assert_eq!(combos.len(), 4);
        assert_eq!(grid.size(), 4);

        // Each combination should have 2 values (one for each parameter)
        for combo in &combos {
            assert_eq!(combo.len(), 2);
        }

        // Verify combinations are exhaustive (sorted keys: a, b)
        // Expected: [1,10], [1,20], [2,10], [2,20]
        assert_eq!(combos[0], vec![1.0, 10.0]);
        assert_eq!(combos[1], vec![1.0, 20.0]);
        assert_eq!(combos[2], vec![2.0, 10.0]);
        assert_eq!(combos[3], vec![2.0, 20.0]);
    }

    #[test]
    fn test_generate_combinations_complex() {
        let optimizer = GridSearchOptimizer::new();

        let mut grid = ParameterGrid::new();
        grid.add_range(
            "rsi_period",
            ParameterRange::Int {
                min: 10,
                max: 14,
                step: 2,
            },
        ); // 3 values: 10, 12, 14
        grid.add_range(
            "buy_threshold",
            ParameterRange::Float {
                min: 20.0,
                max: 30.0,
                step: 5.0,
            },
        ); // 3 values: 20, 25, 30
        grid.add_range(
            "sell_threshold",
            ParameterRange::Float {
                min: 70.0,
                max: 80.0,
                step: 10.0,
            },
        ); // 2 values: 70, 80

        let combos = optimizer.generate_all_combinations(&grid);

        // Should generate 3 × 3 × 2 = 18 combinations
        assert_eq!(combos.len(), 18);
        assert_eq!(grid.size(), 18);

        // Each combination should have 3 values
        for combo in &combos {
            assert_eq!(combo.len(), 3);
        }

        // Verify first few combinations (sorted keys: buy_threshold, rsi_period, sell_threshold)
        assert_eq!(combos[0], vec![20.0, 10.0, 70.0]);
        assert_eq!(combos[1], vec![20.0, 10.0, 80.0]);
        assert_eq!(combos[2], vec![20.0, 12.0, 70.0]);
    }

    #[test]
    fn test_generate_combinations_empty() {
        let optimizer = GridSearchOptimizer::new();
        let grid = ParameterGrid::new();

        let combos = optimizer.generate_all_combinations(&grid);

        // Empty grid should return single empty combination
        assert_eq!(combos.len(), 1);
        assert_eq!(combos[0].len(), 0);
    }

    #[test]
    fn test_generate_combinations_single_param() {
        let optimizer = GridSearchOptimizer::new();

        let mut grid = ParameterGrid::new();
        grid.add_range(
            "threshold",
            ParameterRange::Float {
                min: 0.1,
                max: 0.5,
                step: 0.2,
            },
        ); // 3 values: 0.1, 0.3, 0.5

        let combos = optimizer.generate_all_combinations(&grid);

        assert_eq!(combos.len(), 3);
        assert_eq!(combos[0], vec![0.1]);
        assert_eq!(combos[1], vec![0.30000000000000004]); // Floating point precision
        assert_eq!(combos[2], vec![0.5]);
    }

    #[test]
    fn test_generate_combinations_discrete_values() {
        let optimizer = GridSearchOptimizer::new();

        let mut grid = ParameterGrid::new();
        grid.add_range("option_a", ParameterRange::Values(vec![1.0, 2.0, 5.0]));
        grid.add_range("option_b", ParameterRange::Values(vec![10.0, 20.0]));

        let combos = optimizer.generate_all_combinations(&grid);

        // Should generate 3 × 2 = 6 combinations
        assert_eq!(combos.len(), 6);

        // Verify combinations (sorted keys: option_a, option_b)
        assert_eq!(combos[0], vec![1.0, 10.0]);
        assert_eq!(combos[1], vec![1.0, 20.0]);
        assert_eq!(combos[2], vec![2.0, 10.0]);
        assert_eq!(combos[3], vec![2.0, 20.0]);
        assert_eq!(combos[4], vec![5.0, 10.0]);
        assert_eq!(combos[5], vec![5.0, 20.0]);
    }

    #[test]
    fn test_batch_size_validation() {
        let optimizer = GridSearchOptimizer::new().batch_size(0);

        // Should clamp to minimum of 1
        assert_eq!(optimizer.batch_size, 1);
    }
}

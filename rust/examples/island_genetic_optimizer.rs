//! Island Model Genetic Optimizer Example
//!
//! This example demonstrates the Island Model genetic algorithm optimizer which runs
//! multiple independent populations (islands) that periodically exchange best individuals.
//!
//! # Benefits over Standard Genetic Optimizer
//!
//! - **Better Exploration**: Multiple independent search spaces prevent premature convergence
//! - **Diversity Preservation**: Islands maintain unique genetic diversity
//! - **Migration**: Periodic exchange of best solutions spreads good traits across islands
//! - **Parallel Evolution**: Each island can potentially evolve different optimal solutions
//!
//! # Architecture
//!
//! ```text
//! Island 1: [100 individuals] ──┐
//! Island 2: [100 individuals] ──┼─→ Migration every 10 generations
//! Island 3: [100 individuals] ──┤   (Ring topology: 1→2→3→4→1)
//! Island 4: [100 individuals] ──┘
//! ```
//!
//! # Usage
//!
//! ```bash
//! # CPU-only mode
//! cargo run --example island_genetic_optimizer --release
//!
//! # With GPU acceleration (20-40x faster)
//! cargo run --example island_genetic_optimizer --release --features gpu
//! ```

use kimsfinance_core::backtest::{
    BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid, ParameterRange,
    Signal, Strategy, IslandGeneticOptimizer, GeneticOptimizer,
};
use kimsfinance_core::binance::{process_binance_month, Timeframe};
use ndarray::Array1;
use std::error::Error;
use std::time::Instant;

/// Simple RSI strategy for optimization
///
/// Trading logic:
/// - Buy when RSI < buy_threshold (oversold)
/// - Sell when RSI > sell_threshold (overbought)
#[derive(Debug, Clone)]
struct RSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl RSIStrategy {
    fn new() -> Self {
        Self {
            rsi_period: 14,         // Will be optimized
            buy_threshold: 30.0,    // Will be optimized
            sell_threshold: 70.0,   // Will be optimized
        }
    }
}

impl Strategy for RSIStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);
        let rsi = indicators.get(&rsi_key).unwrap_or(&50.0);

        if rsi.is_nan() {
            return Signal::Hold;
        }

        if *rsi < self.buy_threshold {
            Signal::Buy
        } else if *rsi > self.sell_threshold {
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

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Island Model Genetic Optimizer Example ===\n");

    // Load Binance BTCUSDT data
    let data_path = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/BTCUSDT-trades-2024-05-31.zip";

    println!("Loading Binance BTCUSDT data...");
    let load_start = Instant::now();

    let timeframe = Timeframe::minutes(15);
    let candles = process_binance_month(data_path, timeframe)?;

    println!("Loaded {} candles in {:.2}s\n", candles.len(), load_start.elapsed().as_secs_f64());

    // Convert to vectors and ndarray
    let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp).collect();
    let open = Array1::from_vec(candles.iter().map(|c| c.open).collect());
    let high = Array1::from_vec(candles.iter().map(|c| c.high).collect());
    let low = Array1::from_vec(candles.iter().map(|c| c.low).collect());
    let close = Array1::from_vec(candles.iter().map(|c| c.close).collect());
    let volume = Array1::from_vec(candles.iter().map(|c| c.volume).collect());

    // Create backtesting engine
    let engine = BacktestEngine::default();

    // Define parameter search space
    let mut param_grid = ParameterGrid::new();

    // RSI period: 10-20 in steps of 1
    param_grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 1,
        },
    );

    // Buy threshold: 20-40 in steps of 5
    param_grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 5.0,
        },
    );

    // Sell threshold: 60-80 in steps of 5
    param_grid.add_range(
        "sell_threshold",
        ParameterRange::Float {
            min: 60.0,
            max: 80.0,
            step: 5.0,
        },
    );

    println!("Parameter search space:");
    println!("  RSI period: 10-20 (step=1)");
    println!("  Buy threshold: 20-40 (step=5)");
    println!("  Sell threshold: 60-80 (step=5)\n");

    // Create base genetic optimizer
    let base_optimizer = GeneticOptimizer::new()
        .population_size(100)          // 100 individuals per island
        .generations(50)                // 50 generations
        .mutation_rate(0.15)           // 15% mutation rate
        .crossover_rate(0.8)           // 80% crossover rate
        .fp8_exploration_ratio(0.8)    // 80% FP8, 20% FP64
        .elitism_rate(0.1);            // Top 10% survive

    // Create island model optimizer
    let island_optimizer = IslandGeneticOptimizer::new(base_optimizer)
        .num_islands(4)                // 4 independent populations
        .migration_interval(10)        // Migrate every 10 generations
        .migration_rate(0.1);          // Migrate top 10% of each island

    println!("Island Model Configuration:");
    println!("  Islands: 4");
    println!("  Population per island: 100");
    println!("  Total individuals: 400");
    println!("  Generations: 50");
    println!("  Migration interval: 10 generations");
    println!("  Migration rate: 10% (top 10 individuals per island)");
    println!("  FP8/FP64 split: 80%/20%\n");

    // Run optimization
    println!("Starting island model optimization...\n");
    let opt_start = Instant::now();

    let strategy = RSIStrategy::new();
    let result = island_optimizer.optimize(
        &engine,
        &strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &param_grid,
    )?;

    let opt_duration = opt_start.elapsed();

    // Print results
    println!("\n=== Optimization Results ===");
    println!("Total time: {:.2}s", opt_duration.as_secs_f64());
    println!("FP8 generations: {}", result.fp8_generations);
    println!("FP64 generations: {}", result.fp64_generations);
    println!("\nBest Parameters:");
    for (name, value) in &result.best_parameters {
        println!("  {}: {:.2}", name, value);
    }
    println!("\nBest Fitness (Sharpe Ratio): {:.4}", result.best_fitness);
    println!("\nBacktest Results:");
    println!("  Total Return: {:.2}%", result.best_result.total_return * 100.0);
    println!("  Max Drawdown: {:.2}%", result.best_result.max_drawdown * 100.0);
    println!("  Win Rate: {:.2}%", result.best_result.win_rate * 100.0);
    println!("  Profit Factor: {:.2}", result.best_result.profit_factor);
    println!("  Total Trades: {}", result.best_result.num_trades);

    // Print convergence history (every 10 generations)
    println!("\nConvergence History:");
    for (generation, fitness) in result.convergence_history.iter().enumerate().step_by(10) {
        println!("  Gen {:3}: Fitness={:.4}", generation + 1, fitness);
    }

    println!("\n=== Comparison to Standard Genetic Optimizer ===");
    println!("Island Model Benefits:");
    println!("  - Better exploration via multiple independent populations");
    println!("  - Prevents premature convergence to local optima");
    println!("  - Migration spreads good solutions across islands");
    println!("  - More robust to parameter settings");
    println!("\nTradeoff:");
    println!("  - Same total evaluations as standard GA");
    println!("  - But distributed across 4 islands for better coverage");

    Ok(())
}

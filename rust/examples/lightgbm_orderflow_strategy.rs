//! LightGBM Orderflow Strategy Example
//!
//! Complete example showing how to integrate LightGBM orderflow models with
//! kimsfinance tick backtesting engine.
//!
//! # Usage
//!
//! ```bash
//! # Single backtest with default parameters
//! cargo run --release --features data-downloaders --example lightgbm_orderflow_strategy -- \
//!     --model models/orderflow_model.lgb \
//!     --data /data/trades_parquet/2024-01 \
//!     --threshold 0.6
//!
//! # Genetic optimization
//! cargo run --release --features data-downloaders --example lightgbm_orderflow_strategy -- \
//!     --model models/orderflow_model.lgb \
//!     --data /data/trades_parquet/2024-01 \
//!     --optimize \
//!     --population 100 \
//!     --generations 50
//! ```

use kimsfinance_core::backtest::{
    BacktestConfig, GeneticOptimizer, ParameterGrid, ParameterRange, Signal, TickEngine,
    TickStrategy,
};
use kimsfinance_core::binance::{Candle, IncompleteCandle, Timeframe, Trade};

#[cfg(feature = "data-downloaders")]
use kimsfinance_core::binance::load_parquet_month;

use clap::Parser;
use ndarray::Array1;
use std::collections::HashMap;

/// Command-line arguments
#[derive(Parser, Debug)]
#[clap(name = "lightgbm_orderflow_strategy")]
#[clap(about = "LightGBM orderflow strategy backtesting and optimization")]
struct Args {
    /// Path to trained LightGBM model file
    #[clap(long)]
    model: String,

    /// Path to Parquet data directory (e.g., /data/trades_parquet/2024-01)
    #[clap(long)]
    data: String,

    /// Signal threshold (0.0-1.0)
    #[clap(long, default_value = "0.6")]
    threshold: f64,

    /// Run genetic optimization
    #[clap(long)]
    optimize: bool,

    /// Population size for optimization
    #[clap(long, default_value = "100")]
    population: usize,

    /// Number of generations for optimization
    #[clap(long, default_value = "50")]
    generations: usize,

    /// Limit number of trades (for testing)
    #[clap(long)]
    max_trades: Option<usize>,
}

// ============================================================================
// Orderflow Feature Extraction
// ============================================================================

/// Orderflow features extracted from tick data
#[derive(Debug, Clone)]
pub struct OrderflowFeatures {
    trade_window: Vec<Trade>,
    window_size: usize,
    last_candle: Option<Candle>,
}

impl OrderflowFeatures {
    pub fn new(window_size: usize) -> Self {
        Self {
            trade_window: Vec::with_capacity(window_size),
            window_size,
            last_candle: None,
        }
    }

    pub fn update_tick(&mut self, trade: &Trade) {
        self.trade_window.push(trade.clone());
        if self.trade_window.len() > self.window_size {
            self.trade_window.remove(0);
        }
    }

    pub fn update_candle(&mut self, candle: &Candle) {
        self.last_candle = Some(candle.clone());
    }

    pub fn extract(&self, current_trade: &Trade, current_candle: &IncompleteCandle) -> Array1<f64> {
        let mut features = vec![];

        // Feature 1: Order imbalance
        features.push(self.calculate_order_imbalance());

        // Feature 2: Volume delta
        features.push(self.calculate_volume_delta());

        // Feature 3: Price momentum
        let price_momentum = if current_candle.volume > 0.0 {
            (current_trade.price / (current_candle.quote_volume / current_candle.volume)) - 1.0
        } else {
            0.0
        };
        features.push(price_momentum);

        // Feature 4: Trade intensity
        features.push(self.calculate_trade_intensity(current_trade.timestamp_ms, 5000));

        // Feature 5: Candle progress
        features.push(current_candle.num_trades as f64 / self.window_size as f64);

        // Feature 6: Previous candle return
        let prev_return = if let Some(ref last) = self.last_candle {
            (last.close - last.open) / last.open
        } else {
            0.0
        };
        features.push(prev_return);

        Array1::from_vec(features)
    }

    fn calculate_order_imbalance(&self) -> f64 {
        if self.trade_window.is_empty() {
            return 0.5;
        }

        let buy_volume: f64 = self
            .trade_window
            .iter()
            .filter(|t| !t.is_buyer_maker)
            .map(|t| t.quantity)
            .sum();

        let total_volume: f64 = self.trade_window.iter().map(|t| t.quantity).sum();

        if total_volume > 0.0 {
            buy_volume / total_volume
        } else {
            0.5
        }
    }

    fn calculate_volume_delta(&self) -> f64 {
        let buy_volume: f64 = self
            .trade_window
            .iter()
            .filter(|t| !t.is_buyer_maker)
            .map(|t| t.quantity)
            .sum();

        let sell_volume: f64 = self
            .trade_window
            .iter()
            .filter(|t| t.is_buyer_maker)
            .map(|t| t.quantity)
            .sum();

        buy_volume - sell_volume
    }

    fn calculate_trade_intensity(&self, current_time_ms: i64, lookback_ms: i64) -> f64 {
        let cutoff = current_time_ms - lookback_ms;
        self.trade_window
            .iter()
            .filter(|t| t.timestamp_ms >= cutoff)
            .count() as f64
    }
}

// ============================================================================
// Mock LightGBM Model (replace with actual lightgbm crate in production)
// ============================================================================

/// Mock LightGBM model for demonstration
/// In production, replace with: use lightgbm::Booster;
pub struct MockLightGBMModel {
    threshold: f64,
}

impl MockLightGBMModel {
    pub fn from_file(_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { threshold: 0.6 })
    }

    pub fn predict(&self, features: &[f64]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        // Mock prediction: Simple rule based on order imbalance
        let order_imbalance = features[0];
        let volume_delta = features[1];

        let prediction = if order_imbalance > 0.55 && volume_delta > 0.0 {
            0.75 // Strong buy signal
        } else if order_imbalance < 0.45 && volume_delta < 0.0 {
            0.25 // Strong sell signal
        } else {
            0.50 // Neutral
        };

        Ok(vec![vec![prediction]])
    }
}

// ============================================================================
// LightGBM Orderflow Strategy
// ============================================================================

pub struct LightGBMOrderflowStrategy {
    model: MockLightGBMModel,
    features: OrderflowFeatures,
    threshold: f64,
    position_size: f64,
}

impl LightGBMOrderflowStrategy {
    pub fn load(model_path: &str, threshold: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let model = MockLightGBMModel::from_file(model_path)?;

        Ok(Self {
            model,
            features: OrderflowFeatures::new(100),
            threshold,
            position_size: 1.0,
        })
    }

    pub fn new(
        model_path: &str,
        threshold: f64,
        window_size: usize,
        position_size: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = MockLightGBMModel::from_file(model_path)?;

        Ok(Self {
            model,
            features: OrderflowFeatures::new(window_size),
            threshold,
            position_size,
        })
    }
}

impl TickStrategy for LightGBMOrderflowStrategy {
    fn on_tick(&mut self, trade: &Trade, candle: &IncompleteCandle) -> Signal {
        self.features.update_tick(trade);
        let feature_array = self.features.extract(trade, candle);
        let features_vec: Vec<f64> = feature_array.to_vec();

        let prediction = match self.model.predict(&features_vec) {
            Ok(preds) => preds[0][0],
            Err(e) => {
                eprintln!("Prediction error: {}", e);
                return Signal::Flat;
            }
        };

        if prediction > self.threshold {
            Signal::Long {
                size: self.position_size,
            }
        } else if prediction < (1.0 - self.threshold) {
            Signal::Short {
                size: self.position_size,
            }
        } else {
            Signal::Flat
        }
    }

    fn on_candle_complete(&mut self, candle: &Candle) {
        self.features.update_candle(candle);
    }
}

// ============================================================================
// Single Backtest
// ============================================================================

#[cfg(feature = "data-downloaders")]
fn run_single_backtest(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Single Backtest Mode ===\n");

    // Configure backtesting
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        max_position_size: 1.0,
        risk_per_trade: 0.02,
        ..Default::default()
    };

    // Create engine
    let engine = TickEngine::new(config);

    // Load strategy
    let mut strategy = LightGBMOrderflowStrategy::load(&args.model, args.threshold)?;

    // Load tick data
    println!("Loading tick data from: {}", args.data);
    let trades = load_parquet_month(&args.data, args.max_trades)?;
    println!("Loaded {} trades\n", trades.len());

    // Run backtest
    println!("Running backtest...");
    let start = std::time::Instant::now();
    let result = engine.run(&mut strategy, &trades, Timeframe::minutes(1))?;
    let elapsed = start.elapsed();

    // Print results
    println!("\n=== Backtest Results ===");
    println!("Processing time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Throughput: {:.2}M ticks/sec\n",
        trades.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );

    println!("Performance Metrics:");
    println!("  Final Equity: ${:.2}", result.final_equity);
    println!("  Total Return: {:.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("  Win Rate: {:.2}%", result.win_rate * 100.0);
    println!("  Num Trades: {}", result.num_trades);
    println!("  Profit Factor: {:.2}", result.profit_factor);

    Ok(())
}

// ============================================================================
// Genetic Optimization
// ============================================================================

#[cfg(feature = "data-downloaders")]
fn run_optimization(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Genetic Optimization Mode ===\n");

    // Load tick data
    println!("Loading tick data from: {}", args.data);
    let trades = load_parquet_month(&args.data, args.max_trades)?;
    println!("Loaded {} trades\n", trades.len());

    // Define parameter search space
    let mut param_grid = ParameterGrid::new();

    param_grid.add_range(
        "threshold",
        ParameterRange::Float {
            min: 0.5,
            max: 0.75,
            step: 0.05,
        },
    );

    param_grid.add_range(
        "window_size",
        ParameterRange::Int {
            min: 50,
            max: 200,
            step: 50,
        },
    );

    param_grid.add_range(
        "position_size",
        ParameterRange::Float {
            min: 0.5,
            max: 1.5,
            step: 0.25,
        },
    );

    // Create optimizer
    let optimizer = GeneticOptimizer::new()
        .population_size(args.population)
        .generations(args.generations)
        .mutation_rate(0.15)
        .crossover_rate(0.8)
        .elitism_rate(0.1);

    // Strategy factory
    let model_path = args.model.clone();
    let strategy_factory = move |params: &HashMap<String, f64>| {
        let threshold = params["threshold"];
        let window_size = params["window_size"] as usize;
        let position_size = params["position_size"];

        Box::new(
            LightGBMOrderflowStrategy::new(&model_path, threshold, window_size, position_size)
                .unwrap(),
        ) as Box<dyn TickStrategy>
    };

    // Run optimization
    println!("Starting genetic optimization...");
    println!("  Population: {}", args.population);
    println!("  Generations: {}", args.generations);
    println!("  Search space: {} combinations\n", param_grid.size());

    let start = std::time::Instant::now();
    let result = optimizer.optimize_tick_strategy(
        &trades,
        Timeframe::minutes(1),
        &param_grid,
        strategy_factory,
    )?;
    let elapsed = start.elapsed();

    // Print results
    println!("\n=== Optimization Results ===");
    println!("Time: {:.2} minutes", elapsed.as_secs_f64() / 60.0);
    println!("Evaluations: {} backtests", result.evaluations);
    println!(
        "Throughput: {:.2} backtests/sec",
        result.evaluations as f64 / elapsed.as_secs_f64()
    );

    println!("\nBest Parameters:");
    for (param, value) in &result.best_parameters {
        println!("  {}: {:.3}", param, value);
    }

    println!("\nBest Performance:");
    println!("  Fitness (Sharpe): {:.2}", result.best_fitness);
    println!("  Return: {:.2}%", result.best_return * 100.0);

    println!("\nConvergence:");
    println!("  Stagnant generations: {}", result.stagnant_generations);
    println!(
        "  Early stop: {}",
        if result.early_stopped { "Yes" } else { "No" }
    );

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "data-downloaders"))]
    {
        eprintln!("Error: This example requires the 'data-downloaders' feature");
        eprintln!(
            "Run with: cargo run --features data-downloaders --example lightgbm_orderflow_strategy"
        );
        std::process::exit(1);
    }

    #[cfg(feature = "data-downloaders")]
    {
        let args = Args::parse();

        if args.optimize {
            run_optimization(&args)?;
        } else {
            run_single_backtest(&args)?;
        }
    }

    Ok(())
}

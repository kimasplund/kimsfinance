//! Advanced Momentum Strategy with Orderflow Analysis
//!
//! High-performance tick-level strategy combining:
//! - Orderflow imbalance detection
//! - Volume delta analysis
//! - Price momentum signals
//! - Trade intensity tracking
//! - Dynamic position sizing
//!
//! # Features
//! - Adaptive thresholds based on market conditions
//! - Multi-timeframe analysis (tick + candle)
//! - Risk-adjusted position sizing
//! - Comprehensive metrics collection
//!
//! # Usage
//!
//! ```bash
//! # Generate test data
//! cargo run --release --features data-downloaders --example advanced_momentum_strategy -- generate --trades 10000000
//!
//! # Single backtest
//! cargo run --release --features data-downloaders --example advanced_momentum_strategy -- \
//!     backtest --data data/test_trades.parquet
//!
//! # Full optimization with metrics
//! cargo run --release --features data-downloaders --example advanced_momentum_strategy -- \
//!     optimize --data data/test_trades.parquet \
//!     --population 100 --generations 50 --metrics
//! ```

use kimsfinance_core::backtest::{
    BacktestConfig, GeneticOptimizer, ParameterGrid, ParameterRange, Signal, TickEngine,
    TickStrategy,
};
use kimsfinance_core::binance::{Candle, IncompleteCandle, Timeframe, Trade};

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "data-downloaders")]
use kimsfinance_core::binance::{load_parquet_file, load_parquet_month};

// ============================================================================
// Orderflow Features
// ============================================================================

#[derive(Debug, Clone)]
pub struct OrderflowFeatures {
    trade_window: Vec<Trade>,
    window_size: usize,
    last_candle: Option<Candle>,
    volume_ema: f64,
    price_ema: f64,
}

impl OrderflowFeatures {
    pub fn new(window_size: usize) -> Self {
        Self {
            trade_window: Vec::with_capacity(window_size),
            window_size,
            last_candle: None,
            volume_ema: 0.0,
            price_ema: 0.0,
        }
    }

    pub fn update_tick(&mut self, trade: &Trade) {
        // Update window
        self.trade_window.push(trade.clone());
        if self.trade_window.len() > self.window_size {
            self.trade_window.remove(0);
        }

        // Update EMAs
        let alpha = 2.0 / (self.window_size as f64 + 1.0);
        self.volume_ema = alpha * trade.quantity + (1.0 - alpha) * self.volume_ema;
        self.price_ema = alpha * trade.price + (1.0 - alpha) * self.price_ema;
    }

    pub fn update_candle(&mut self, candle: &Candle) {
        self.last_candle = Some(candle.clone());
    }

    /// Calculate order imbalance (0.0-1.0, 0.5 = balanced)
    pub fn order_imbalance(&self) -> f64 {
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

    /// Calculate volume delta (buy - sell)
    pub fn volume_delta(&self) -> f64 {
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

    /// Calculate trade intensity (trades per second in recent period)
    pub fn trade_intensity(&self, current_time_ms: i64, lookback_ms: i64) -> f64 {
        let cutoff = current_time_ms - lookback_ms;
        self.trade_window
            .iter()
            .filter(|t| t.timestamp_ms >= cutoff)
            .count() as f64
            / (lookback_ms as f64 / 1000.0)
    }

    /// Calculate price momentum relative to EMA
    pub fn price_momentum(&self, current_price: f64) -> f64 {
        if self.price_ema > 0.0 {
            (current_price - self.price_ema) / self.price_ema
        } else {
            0.0
        }
    }

    /// Calculate volume momentum relative to EMA
    pub fn volume_momentum(&self, current_volume: f64) -> f64 {
        if self.volume_ema > 0.0 {
            (current_volume - self.volume_ema) / self.volume_ema
        } else {
            0.0
        }
    }

    /// Get candle return if available
    pub fn candle_return(&self) -> f64 {
        if let Some(ref candle) = self.last_candle {
            (candle.close - candle.open) / candle.open
        } else {
            0.0
        }
    }
}

// ============================================================================
// Advanced Momentum Strategy
// ============================================================================

pub struct AdvancedMomentumStrategy {
    features: OrderflowFeatures,

    // Strategy parameters
    imbalance_threshold: f64,
    volume_delta_threshold: f64,
    momentum_threshold: f64,
    intensity_threshold: f64,

    // Position sizing
    base_position_size: f64,
    use_dynamic_sizing: bool,

    // Risk management
    max_position_size: f64,
    stop_loss_pct: f64,
}

impl AdvancedMomentumStrategy {
    pub fn new(
        window_size: usize,
        imbalance_threshold: f64,
        volume_delta_threshold: f64,
        momentum_threshold: f64,
        intensity_threshold: f64,
        base_position_size: f64,
    ) -> Self {
        Self {
            features: OrderflowFeatures::new(window_size),
            imbalance_threshold,
            volume_delta_threshold,
            momentum_threshold,
            intensity_threshold,
            base_position_size,
            use_dynamic_sizing: true,
            max_position_size: 2.0,
            stop_loss_pct: 0.02,
        }
    }

    fn calculate_position_size(&self, signal_strength: f64) -> f64 {
        if !self.use_dynamic_sizing {
            return self.base_position_size;
        }

        // Scale position size by signal strength (0.0-2.0)
        let size = self.base_position_size * signal_strength;
        size.min(self.max_position_size).max(0.1)
    }

    fn calculate_signal_strength(&self, trade: &Trade) -> (bool, bool, f64) {
        let imbalance = self.features.order_imbalance();
        let volume_delta = self.features.volume_delta();
        let momentum = self.features.price_momentum(trade.price);
        let intensity = self.features.trade_intensity(trade.timestamp_ms, 5000);

        // Bull signal: High buy pressure + positive momentum
        let bull_imbalance = imbalance > (0.5 + self.imbalance_threshold);
        let bull_volume = volume_delta > self.volume_delta_threshold;
        let bull_momentum = momentum > self.momentum_threshold;
        let high_intensity = intensity > self.intensity_threshold;

        let is_bull = bull_imbalance && bull_volume && (bull_momentum || high_intensity);

        // Bear signal: High sell pressure + negative momentum
        let bear_imbalance = imbalance < (0.5 - self.imbalance_threshold);
        let bear_volume = volume_delta < -self.volume_delta_threshold;
        let bear_momentum = momentum < -self.momentum_threshold;

        let is_bear = bear_imbalance && bear_volume && (bear_momentum || high_intensity);

        // Calculate signal strength (number of conditions met / total conditions)
        let bull_strength = (bull_imbalance as u8 + bull_volume as u8 + bull_momentum as u8
            + high_intensity as u8) as f64
            / 4.0;
        let bear_strength = (bear_imbalance as u8 + bear_volume as u8 + bear_momentum as u8
            + high_intensity as u8) as f64
            / 4.0;

        let strength = if is_bull {
            1.0 + bull_strength // 1.0-2.0
        } else if is_bear {
            1.0 + bear_strength
        } else {
            0.0
        };

        (is_bull, is_bear, strength)
    }
}

impl TickStrategy for AdvancedMomentumStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        self.features.update_tick(trade);

        // Need minimum window size
        if self.features.trade_window.len() < self.features.window_size / 2 {
            return Signal::Hold;
        }

        let (is_bull, is_bear, _strength) = self.calculate_signal_strength(trade);

        if is_bull {
            Signal::Buy
        } else if is_bear {
            Signal::Short
        } else {
            Signal::Hold
        }
    }

    fn on_candle_complete(&mut self, candle: &Candle) -> Signal {
        self.features.update_candle(candle);
        Signal::Hold  // Just update state, don't trade on candle completion
    }
}

// ============================================================================
// Test Data Generation
// ============================================================================

#[cfg(feature = "data-downloaders")]
fn generate_synthetic_trades(num_trades: usize, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use arrow::array::{Float64Array, Int64Array, UInt64Array, BooleanArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use rand::Rng;

    println!("Generating {} synthetic trades...", num_trades);
    let start = Instant::now();

    let mut rng = rand::thread_rng();

    // Generate realistic BTC price walk
    let mut price: f64 = 50000.0;
    let mut prices = Vec::with_capacity(num_trades);
    let mut quantities = Vec::with_capacity(num_trades);
    let mut timestamps = Vec::with_capacity(num_trades);
    let mut ids = Vec::with_capacity(num_trades);
    let mut is_buyer_maker = Vec::with_capacity(num_trades);

    let base_time = 1704067200000i64; // 2024-01-01 00:00:00 UTC

    for i in 0..num_trades {
        // Price walk with trend and mean reversion
        let trend = if i % 10000 < 5000 { 0.0001 } else { -0.0001 };
        let volatility = 0.0002;
        let change = trend + rng.gen_range(-volatility..volatility);
        price *= 1.0 + change;
        price = price.max(45000.0).min(55000.0); // Keep in range

        // Realistic quantities (0.001 to 1.0 BTC)
        let qty = rng.gen_range(0.001..1.0);

        // Time: ~2ms per trade (realistic for BTC)
        let time = base_time + (i as i64 * 2);

        // 50% buy, 50% sell with slight imbalance in trending periods
        let is_buy = if i % 10000 < 5000 {
            rng.gen_bool(0.55) // Slight buy pressure during uptrend
        } else {
            rng.gen_bool(0.45) // Slight sell pressure during downtrend
        };

        prices.push(price);
        quantities.push(qty);
        timestamps.push(time);
        ids.push(i as u64);
        is_buyer_maker.push(is_buy);
    }

    // Create Arrow schema matching Binance format
    let schema = Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("qty", DataType::Float64, false),
        Field::new("quote_qty", DataType::Float64, false),
        Field::new("time", DataType::Int64, false),
        Field::new("is_buyer_maker", DataType::Boolean, false),
    ]);

    // Calculate quote quantities
    let quote_qtys: Vec<f64> = prices.iter().zip(&quantities).map(|(p, q)| p * q).collect();

    // Create record batch
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema.clone()),
        vec![
            std::sync::Arc::new(UInt64Array::from(ids)),
            std::sync::Arc::new(Float64Array::from(prices)),
            std::sync::Arc::new(Float64Array::from(quantities)),
            std::sync::Arc::new(Float64Array::from(quote_qtys)),
            std::sync::Arc::new(Int64Array::from(timestamps)),
            std::sync::Arc::new(BooleanArray::from(is_buyer_maker)),
        ],
    )?;

    // Write to Parquet
    let file = File::create(output_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, std::sync::Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    let elapsed = start.elapsed();
    println!("✓ Generated {} trades in {:.2}s", num_trades, elapsed.as_secs_f64());
    println!("✓ Output: {}", output_path);
    println!("✓ File size: {:.2} MB", std::fs::metadata(output_path)?.len() as f64 / 1_000_000.0);

    Ok(())
}

// ============================================================================
// Single Backtest
// ============================================================================

#[cfg(feature = "data-downloaders")]
fn run_backtest(data_path: &str, max_trades: Option<usize>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Single Backtest ===\n");

    // Load data
    println!("Loading data from: {}", data_path);
    let load_start = Instant::now();
    let trades = if data_path.ends_with(".parquet") {
        load_parquet_file(data_path)?
    } else {
        load_parquet_month(data_path, max_trades)?
    };
    let load_time = load_start.elapsed();

    let num_trades = trades.len();
    println!("✓ Loaded {} trades in {:.2}s", num_trades, load_time.as_secs_f64());
    println!("  Throughput: {:.2}M records/sec\n", num_trades as f64 / load_time.as_secs_f64() / 1_000_000.0);

    // Create strategy with default parameters
    let mut strategy = AdvancedMomentumStrategy::new(
        100,    // window_size
        0.1,    // imbalance_threshold
        10.0,   // volume_delta_threshold
        0.001,  // momentum_threshold
        5.0,    // intensity_threshold
        1.0,    // base_position_size
    );

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        execution_latency_ms: 10,  // 10ms realistic execution latency
        use_gpu: false,  // Tick backtesting uses CPU
        force_cpu: true,
    };

    let engine = TickEngine::new(config);

    // Run backtest
    println!("Running backtest...");
    let backtest_start = Instant::now();
    let result = engine.run(&mut strategy, &trades, Timeframe::minutes(1))?;
    let backtest_time = backtest_start.elapsed();

    // Print results
    println!("\n=== Performance Metrics ===");
    println!("Processing time: {:.2}s", backtest_time.as_secs_f64());
    println!("Throughput: {:.2}M ticks/sec", num_trades as f64 / backtest_time.as_secs_f64() / 1_000_000.0);
    println!("Ticks per ms: {:.2}", num_trades as f64 / backtest_time.as_millis() as f64);

    println!("\n=== Trading Results ===");
    println!("Final Equity: ${:.2}", result.final_equity);
    println!("Total Return: {:.2}%", result.total_return * 100.0);
    println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("Win Rate: {:.2}%", result.win_rate * 100.0);
    println!("Num Trades: {}", result.num_trades);
    println!("Profit Factor: {:.2}", result.profit_factor);

    Ok(())
}

// ============================================================================
// Genetic Optimization
// ============================================================================

#[cfg(feature = "data-downloaders")]
fn run_optimization(
    data_path: &str,
    population: usize,
    generations: usize,
    max_trades: Option<usize>,
    collect_metrics: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Genetic Optimization ===\n");

    // Load data
    println!("Loading data from: {}", data_path);
    let load_start = Instant::now();
    let trades = if data_path.ends_with(".parquet") {
        load_parquet_file(data_path)?
    } else {
        load_parquet_month(data_path, max_trades)?
    };
    let load_time = load_start.elapsed();

    println!("✓ Loaded {} trades in {:.2}s\n", trades.len(), load_time.as_secs_f64());

    // Define parameter space
    let mut param_grid = ParameterGrid::new();

    param_grid.add_range(
        "window_size",
        ParameterRange::Int { min: 50, max: 200, step: 50 },
    );
    param_grid.add_range(
        "imbalance_threshold",
        ParameterRange::Float { min: 0.05, max: 0.2, step: 0.05 },
    );
    param_grid.add_range(
        "volume_delta_threshold",
        ParameterRange::Float { min: 5.0, max: 20.0, step: 5.0 },
    );
    param_grid.add_range(
        "momentum_threshold",
        ParameterRange::Float { min: 0.0005, max: 0.002, step: 0.0005 },
    );
    param_grid.add_range(
        "intensity_threshold",
        ParameterRange::Float { min: 2.0, max: 10.0, step: 2.0 },
    );
    param_grid.add_range(
        "base_position_size",
        ParameterRange::Float { min: 0.5, max: 1.5, step: 0.25 },
    );

    println!("Parameter Space:");
    println!("  Search space: {} combinations", param_grid.size());
    println!("  Population: {}", population);
    println!("  Generations: {}", generations);
    println!("  Total evaluations: {}\n", population * generations);

    // Create optimizer
    let optimizer = GeneticOptimizer::new()
        .population_size(population)
        .generations(generations)
        .mutation_rate(0.15)
        .crossover_rate(0.8)
        .elitism_rate(0.1);

    // Strategy factory
    let strategy_factory = |params: &HashMap<String, f64>| {
        let window_size = params["window_size"] as usize;
        let imbalance_threshold = params["imbalance_threshold"];
        let volume_delta_threshold = params["volume_delta_threshold"];
        let momentum_threshold = params["momentum_threshold"];
        let intensity_threshold = params["intensity_threshold"];
        let base_position_size = params["base_position_size"];

        Box::new(AdvancedMomentumStrategy::new(
            window_size,
            imbalance_threshold,
            volume_delta_threshold,
            momentum_threshold,
            intensity_threshold,
            base_position_size,
        )) as Box<dyn TickStrategy>
    };

    // Run optimization
    println!("Starting genetic optimization with {} CPU cores...\n", rayon::current_num_threads());
    let opt_start = Instant::now();
    let result = optimizer.optimize_tick_strategy(
        &trades,
        Timeframe::minutes(1),
        &param_grid,
        strategy_factory,
    )?;
    let opt_time = opt_start.elapsed();

    // Calculate total evaluations
    let total_evaluations = population * generations;

    // Print results
    println!("\n=== Optimization Results ===");
    println!("Total time: {:.2} minutes", opt_time.as_secs_f64() / 60.0);
    println!("Evaluations: {}", total_evaluations);
    println!("Throughput: {:.2} backtests/sec", total_evaluations as f64 / opt_time.as_secs_f64());
    println!("Time per backtest: {:.2}ms", opt_time.as_millis() as f64 / total_evaluations as f64);

    println!("\n=== Best Parameters ===");
    for (param, value) in &result.best_parameters {
        println!("  {}: {:.4}", param, value);
    }

    println!("\n=== Best Performance ===");
    println!("  Fitness (Sharpe): {:.2}", result.best_fitness);
    println!("  Total Return: {:.2}%", result.best_result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", result.best_result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.best_result.max_drawdown * 100.0);

    println!("\n=== Convergence ===");
    if let Some(generation) = result.convergence_stats.generation_converged {
        println!("  Converged at generation: {}", generation);
    } else {
        println!("  Did not converge early");
    }
    println!("  Final diversity: {:.4}", result.convergence_stats.final_diversity);

    // Save metrics if requested
    if collect_metrics {
        let metrics_path = "optimization_metrics.txt";
        let mut file = File::create(metrics_path)?;

        writeln!(file, "=== Optimization Metrics ===")?;
        writeln!(file, "Date: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"))?;
        writeln!(file, "\n=== Configuration ===")?;
        writeln!(file, "Population: {}", population)?;
        writeln!(file, "Generations: {}", generations)?;
        writeln!(file, "Total evaluations: {}", total_evaluations)?;
        writeln!(file, "Dataset size: {} trades", trades.len())?;
        writeln!(file, "\n=== Performance ===")?;
        writeln!(file, "Total time: {:.2} minutes", opt_time.as_secs_f64() / 60.0)?;
        writeln!(file, "Throughput: {:.2} backtests/sec", total_evaluations as f64 / opt_time.as_secs_f64())?;
        writeln!(file, "Time per backtest: {:.2}ms", opt_time.as_millis() as f64 / total_evaluations as f64)?;
        writeln!(file, "CPU cores used: {} (detected by Rayon)", rayon::current_num_threads())?;
        writeln!(file, "\n=== Best Parameters ===")?;
        for (param, value) in &result.best_parameters {
            writeln!(file, "{}: {:.4}", param, value)?;
        }
        writeln!(file, "\n=== Best Results ===")?;
        writeln!(file, "Fitness (Sharpe): {:.2}", result.best_fitness)?;
        writeln!(file, "Total Return: {:.2}%", result.best_result.total_return * 100.0)?;
        writeln!(file, "Sharpe Ratio: {:.2}", result.best_result.sharpe_ratio)?;
        writeln!(file, "Max Drawdown: {:.2}%", result.best_result.max_drawdown * 100.0)?;
        writeln!(file, "Win Rate: {:.2}%", result.best_result.win_rate * 100.0)?;

        println!("\n✓ Metrics saved to: {}", metrics_path);
    }

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn print_usage() {
    eprintln!("Advanced Momentum Strategy - Tick-level Backtesting");
    eprintln!("\nUsage:");
    eprintln!("  cargo run --release --features data-downloaders --example advanced_momentum_strategy <MODE> [OPTIONS]");
    eprintln!("\nModes:");
    eprintln!("  generate [TRADES] [OUTPUT]    - Generate synthetic test data");
    eprintln!("  backtest [DATA]               - Run single backtest");
    eprintln!("  optimize [DATA] [POP] [GENS]  - Run genetic optimization");
    eprintln!("\nExamples:");
    eprintln!("  Generate 10M trades:");
    eprintln!("    cargo run --release --features data-downloaders --example advanced_momentum_strategy generate 10000000 data/test_trades.parquet");
    eprintln!("\n  Run backtest:");
    eprintln!("    cargo run --release --features data-downloaders --example advanced_momentum_strategy backtest data/test_trades.parquet");
    eprintln!("\n  Run optimization:");
    eprintln!("    cargo run --release --features data-downloaders --example advanced_momentum_strategy optimize data/test_trades.parquet 100 50");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "data-downloaders"))]
    {
        eprintln!("Error: This example requires the 'data-downloaders' feature");
        eprintln!("Run with: cargo run --release --features data-downloaders --example advanced_momentum_strategy");
        std::process::exit(1);
    }

    #[cfg(feature = "data-downloaders")]
    {
        let args: Vec<String> = env::args().collect();

        if args.len() < 2 {
            print_usage();
            std::process::exit(1);
        }

        let mode = args[1].as_str();

        match mode {
            "generate" => {
                let trades = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
                let output = args.get(3).map(|s| s.as_str()).unwrap_or("data/test_trades.parquet");
                generate_synthetic_trades(trades, output)?;
            }
            "backtest" => {
                let data = args.get(2).map(|s| s.as_str()).unwrap_or("data/test_trades.parquet");
                run_backtest(data, None)?;
            }
            "optimize" => {
                let data = args.get(2).map(|s| s.as_str()).unwrap_or("data/test_trades.parquet");
                let population = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
                let generations = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50);
                run_optimization(data, population, generations, None, true)?;
            }
            _ => {
                eprintln!("Error: Unknown mode '{}'", mode);
                print_usage();
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

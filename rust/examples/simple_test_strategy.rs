/// Simple test strategy for GPU infrastructure validation
/// Generates predictable signals to ensure trading logic works

use kimsfinance_core::{
    backtest::{
        engine::TickEngine,
        optimizer::GeneticOptimizer,
        tick_strategy::{Signal, TickStrategy},
    },
    binance::{load_trade_data, Trade},
    strategy::incomplete_candle::IncompleteCandle,
    timeframe::Timeframe,
};
use std::collections::HashMap;

/// Simple momentum strategy that ALWAYS trades (for testing)
pub struct SimpleTestStrategy {
    /// Lookback period for momentum
    window: usize,
    /// Threshold for entry (very low to ensure trading)
    threshold: f64,
    /// Price history buffer
    prices: Vec<f64>,
}

impl SimpleTestStrategy {
    pub fn new(window: usize, threshold: f64) -> Self {
        Self {
            window,
            threshold,
            prices: Vec::with_capacity(window),
        }
    }
}

impl TickStrategy for SimpleTestStrategy {
    fn on_tick(&mut self, trade: &Trade, _candle: &IncompleteCandle) -> Signal {
        self.prices.push(trade.price);

        // Keep only recent prices
        if self.prices.len() > self.window {
            self.prices.remove(0);
        }

        if self.prices.len() < 2 {
            return Signal::Hold;
        }

        // Simple momentum: buy if price increased, sell if decreased
        let momentum = (self.prices.last().unwrap() - self.prices.first().unwrap())
            / self.prices.first().unwrap();

        if momentum > self.threshold {
            Signal::Buy
        } else if momentum < -self.threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn name(&self) -> &str {
        "SimpleTest"
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: {} <mode> <parquet_path> <num_trades_limit>", args[0]);
        eprintln!("Modes:");
        eprintln!("  test     - Run single backtest");
        eprintln!("  optimize - Run genetic optimization");
        std::process::exit(1);
    }

    let mode = &args[1];
    let parquet_path = &args[2];
    let limit: Option<usize> = args[3].parse().ok();

    println!("\n=== Simple Test Strategy ===\n");

    // Load data
    println!("Loading data from: {}", parquet_path);
    let trades = load_trade_data(parquet_path, limit)
        .expect("Failed to load trade data");
    println!("✓ Loaded {} trades", trades.len());

    match mode.as_str() {
        "test" => run_single_test(&trades),
        "optimize" => run_optimization(&trades),
        _ => {
            eprintln!("Unknown mode: {}", mode);
            std::process::exit(1);
        }
    }
}

fn run_single_test(trades: &[Trade]) {
    println!("\n=== Single Backtest Test ===\n");

    // Very simple strategy (window=10, threshold=0.0001 = 0.01%)
    let mut strategy = SimpleTestStrategy::new(10, 0.0001);

    let timeframe = Timeframe::from_duration(std::time::Duration::from_secs(60));
    let mut engine = TickEngine::new(
        10_000.0,  // initial capital
        0.001,     // fee
        0.0005,    // slippage
        Some(10),  // 10ms latency
        timeframe,
    );

    println!("Processing {} trades...", trades.len());
    let start = std::time::Instant::now();

    engine.run(trades, &mut strategy);

    let elapsed = start.elapsed();
    let result = engine.result();

    println!("\n=== Results ===\n");
    println!("Time:           {:.2}s", elapsed.as_secs_f64());
    println!("Throughput:     {:.2}M trades/sec",
             trades.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    println!("Total Return:   {:.2}%", result.total_return * 100.0);
    println!("Sharpe Ratio:   {:.4}", result.sharpe_ratio);
    println!("Max Drawdown:   {:.2}%", result.max_drawdown * 100.0);
    println!("Win Rate:       {:.2}%", result.win_rate * 100.0);
    println!("Total Trades:   {}", result.total_trades);
    println!("Final Equity:   ${:.2}", result.final_equity);
}

fn run_optimization(trades: &[Trade]) {
    println!("\n=== Genetic Optimization ===\n");

    // Define parameter space
    let mut param_space = HashMap::new();
    param_space.insert("window".to_string(), vec![5.0, 10.0, 20.0, 50.0]);
    param_space.insert("threshold".to_string(), vec![0.0001, 0.0005, 0.001, 0.005]);

    let total_combinations: usize = param_space.values()
        .map(|v| v.len())
        .product();

    println!("Parameter Space:");
    println!("  window: {:?}", param_space.get("window").unwrap());
    println!("  threshold: {:?}", param_space.get("threshold").unwrap());
    println!("  Total combinations: {}", total_combinations);
    println!();

    // Create optimizer
    let optimizer = GeneticOptimizer::builder()
        .population_size(20)
        .generations(10)
        .mutation_rate(0.15)
        .crossover_rate(0.8)
        .elitism_rate(0.1)
        .parameter_space(param_space)
        .build();

    println!("Starting optimization...");
    println!("  Population: 20");
    println!("  Generations: 10");
    println!("  Parallelism: Rayon (auto)");
    println!();

    let start = std::time::Instant::now();

    // Run optimization
    let result = optimizer.optimize_tick(
        trades,
        |params| {
            let window = params.get("window").unwrap_or(&10.0).round() as usize;
            let threshold = *params.get("threshold").unwrap_or(&0.001);
            Box::new(SimpleTestStrategy::new(window, threshold))
        },
        Timeframe::from_duration(std::time::Duration::from_secs(60)),
    );

    let elapsed = start.elapsed();

    println!("\n=== Optimization Complete ===\n");
    println!("Time: {:.2}s ({:.2} minutes)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);

    match result {
        Ok((best_params, best_result)) => {
            println!("\n=== Best Strategy ===\n");
            println!("Parameters:");
            for (key, value) in &best_params {
                println!("  {}: {:.4}", key, value);
            }
            println!("\nPerformance:");
            println!("  Sharpe Ratio:   {:.4}", best_result.sharpe_ratio);
            println!("  Total Return:   {:.2}%", best_result.total_return * 100.0);
            println!("  Max Drawdown:   {:.2}%", best_result.max_drawdown * 100.0);
            println!("  Win Rate:       {:.2}%", best_result.win_rate * 100.0);
            println!("  Total Trades:   {}", best_result.total_trades);
        }
        Err(e) => {
            eprintln!("Optimization failed: {:?}", e);
            std::process::exit(1);
        }
    }
}

//! Comprehensive backtest example using real Binance BTCUSDT futures data
//!
//! This example demonstrates:
//! - Loading real tick-level trade data from Binance
//! - Aggregating trades into OHLCV candles
//! - Running multiple RSI strategies in parallel
//! - Comparing CPU vs GPU performance
//! - Generating detailed backtest results
//!
//! # Data Source
//! Binance BTCUSDT futures trade data from `/home/kim/projects/binance-data/futures/BTCUSDT/trades`
//!
//! # Usage
//! ```bash
//! # CPU-only mode
//! cargo run --example backtest_binance_futures --release
//!
//! # With GPU acceleration
//! cargo run --example backtest_binance_futures --release --features gpu
//! ```

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, Signal, Strategy,
};
use kimsfinance_core::binance::{Timeframe, process_binance_month};
use ndarray::Array1;
use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

/// Simple RSI strategy implementation
///
/// Trading logic:
/// - Buy when RSI < buy_threshold (oversold)
/// - Sell when RSI > sell_threshold (overbought)
/// - Hold otherwise
#[derive(Debug, Clone)]
struct RSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
    name: String,
}

impl RSIStrategy {
    fn new(period: usize, buy: f64, sell: f64) -> Self {
        Self {
            rsi_period: period,
            buy_threshold: buy,
            sell_threshold: sell,
            name: format!("RSI({}, {}, {})", period, buy, sell),
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

/// ATR-based volatility strategy
///
/// Trading logic:
/// - Buy when ATR is rising (increasing volatility)
/// - Sell when ATR is falling (decreasing volatility)
#[derive(Debug, Clone)]
struct ATRStrategy {
    atr_period: usize,
    name: String,
    last_atr: Option<f64>,
}

impl ATRStrategy {
    fn new(period: usize) -> Self {
        Self {
            atr_period: period,
            name: format!("ATR({})", period),
            last_atr: None,
        }
    }
}

impl Strategy for ATRStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let atr_key = format!("atr_{}", self.atr_period);
        let atr = indicators.get(&atr_key).unwrap_or(&0.0);

        if atr.is_nan() {
            return Signal::Hold;
        }

        let signal = if let Some(last) = self.last_atr {
            if *atr > last * 1.1 {
                // ATR increased by 10%
                Signal::Buy
            } else if *atr < last * 0.9 {
                // ATR decreased by 10%
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        };

        self.last_atr = Some(*atr);
        signal
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::ATR {
            period: self.atr_period,
        }]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Binance BTCUSDT Futures Backtesting ===\n");

    // Configuration
    let data_path = "/home/kim/projects/binance-data/futures/BTCUSDT/trades/BTCUSDT-trades-2024-05-31.zip";
    let timeframe = Timeframe::minutes(5);

    println!("Loading data from: {}", data_path);
    println!("Timeframe: {:?}", timeframe);

    // Load and aggregate trade data
    let start = Instant::now();
    let candles = process_binance_month(data_path, timeframe)?;
    let load_duration = start.elapsed();

    println!("\nData loaded in {:.2}s", load_duration.as_secs_f64());
    println!("Candles: {}", candles.len());
    println!(
        "Date range: {} to {}",
        candles.first().unwrap().timestamp,
        candles.last().unwrap().timestamp
    );
    println!(
        "Total volume: {:.2} BTC",
        candles.iter().map(|c| c.volume).sum::<f64>()
    );

    // Convert candles to OHLCV arrays
    let n = candles.len();
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    for candle in &candles {
        timestamps.push(candle.timestamp);
        open.push(candle.open);
        high.push(candle.high);
        low.push(candle.low);
        close.push(candle.close);
        volume.push(candle.volume);
    }

    let open_arr = Array1::from_vec(open);
    let high_arr = Array1::from_vec(high);
    let low_arr = Array1::from_vec(low);
    let close_arr = Array1::from_vec(close);
    let volume_arr = Array1::from_vec(volume);

    // Define strategies to test with names
    let mut strategies: Vec<(String, Box<dyn Strategy>)> = vec![];

    // RSI strategies with different parameters
    let rsi_14_30_70 = RSIStrategy::new(14, 30.0, 70.0);
    let name = rsi_14_30_70.name.clone();
    strategies.push((name, Box::new(rsi_14_30_70)));

    let rsi_14_25_75 = RSIStrategy::new(14, 25.0, 75.0);
    let name = rsi_14_25_75.name.clone();
    strategies.push((name, Box::new(rsi_14_25_75)));

    let rsi_21_30_70 = RSIStrategy::new(21, 30.0, 70.0);
    let name = rsi_21_30_70.name.clone();
    strategies.push((name, Box::new(rsi_21_30_70)));

    let rsi_7_30_70 = RSIStrategy::new(7, 30.0, 70.0);
    let name = rsi_7_30_70.name.clone();
    strategies.push((name, Box::new(rsi_7_30_70)));

    // ATR volatility strategies
    let atr_14 = ATRStrategy::new(14);
    let name = atr_14.name.clone();
    strategies.push((name, Box::new(atr_14)));

    let atr_7 = ATRStrategy::new(7);
    let name = atr_7.name.clone();
    strategies.push((name, Box::new(atr_7)));

    println!("\n=== Running Backtests ===\n");

    // Create backtest engine
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001, // 0.1% Binance futures taker fee
        slippage: 0.0005,   // 0.05% slippage
        use_gpu: cfg!(feature = "gpu"),
        force_cpu: false,
    };

    let engine = BacktestEngine::with_config(config.clone());

    #[cfg(feature = "gpu")]
    println!("GPU acceleration: ENABLED");
    #[cfg(not(feature = "gpu"))]
    println!("GPU acceleration: DISABLED (CPU fallback)");

    println!("Config: {:?}\n", config);

    let mut results = Vec::new();

    for (i, (strategy_name, mut strategy)) in strategies.into_iter().enumerate() {
        print!("Strategy {}: {} ... ", i + 1, strategy_name);
        std::io::Write::flush(&mut std::io::stdout())?;

        let start = Instant::now();
        let result = engine.run(
            strategy.as_mut(),
            &timestamps,
            &open_arr,
            &high_arr,
            &low_arr,
            &close_arr,
            &volume_arr,
        )?;
        let duration = start.elapsed();

        println!("DONE ({:.2}ms)", duration.as_secs_f64() * 1000.0);

        results.push((strategy_name, result, duration));
    }

    // Display results
    println!("\n=== Backtest Results ===\n");
    println!(
        "{:<30} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Strategy", "Return %", "Sharpe", "Max DD %", "Win Rate %", "Trades", "Time (ms)"
    );
    println!("{:-<120}", "");

    for (strategy_name, result, duration) in &results {
        println!(
            "{:<30} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>10} {:>10.2}",
            strategy_name,
            result.total_return,
            result.sharpe_ratio,
            result.max_drawdown,
            result.win_rate,
            result.num_trades,
            duration.as_secs_f64() * 1000.0
        );
    }

    // Find best strategy
    if let Some((best_strategy_name, best_result, _)) = results
        .iter()
        .max_by(|a, b| a.1.sharpe_ratio.partial_cmp(&b.1.sharpe_ratio).unwrap())
    {
        println!("\n=== Best Strategy ===\n");
        println!("Strategy: {}", best_strategy_name);
        println!("Total Return: {:.2}%", best_result.total_return);
        println!("Sharpe Ratio: {:.2}", best_result.sharpe_ratio);
        println!("Max Drawdown: {:.2}%", best_result.max_drawdown);
        println!("Win Rate: {:.2}%", best_result.win_rate);
        println!("Profit Factor: {:.2}", best_result.profit_factor);
        println!("Total Trades: {}", best_result.num_trades);
        println!("Final Equity: ${:.2}", best_result.final_equity);

        // Show first 10 trades
        if !best_result.trades.is_empty() {
            println!("\nFirst 10 trades:");
            println!(
                "{:<20} {:>12} {:>12} {:>12} {:>10}",
                "Exit Time", "Entry Price", "Exit Price", "P&L", "P&L %"
            );
            println!("{:-<70}", "");

            for trade in best_result.trades.iter().take(10) {
                println!(
                    "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>10.2}",
                    trade.exit_time,
                    trade.entry_price,
                    trade.exit_price,
                    trade.pnl,
                    trade.pnl_percent
                );
            }

            if best_result.trades.len() > 10 {
                println!("... and {} more trades", best_result.trades.len() - 10);
            }
        }
    }

    // Performance summary
    let total_backtest_time: std::time::Duration = results.iter().map(|(_, _, d)| *d).sum();
    println!("\n=== Performance Summary ===\n");
    println!("Total strategies tested: {}", results.len());
    println!(
        "Total backtest time: {:.2}ms",
        total_backtest_time.as_secs_f64() * 1000.0
    );
    println!(
        "Average time per strategy: {:.2}ms",
        total_backtest_time.as_secs_f64() * 1000.0 / results.len() as f64
    );
    println!(
        "Candles per second: {:.0}",
        (n * results.len()) as f64 / total_backtest_time.as_secs_f64()
    );

    Ok(())
}

//! Comprehensive backtest example using real Binance BTCUSDT futures data
//!
//! This example demonstrates:
//! - Loading real tick-level trade data from Binance
//! - Aggregating trades into OHLCV candles (1min, 5min, 15min)
//! - Testing multiple strategies (RSI, ATR, MACD)
//! - Comparing performance across different timeframes
//! - Generating comprehensive markdown report
//!
//! # Data Source
//! Binance BTCUSDT futures trade data from `/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades`
//!
//! # Usage
//! ```bash
//! # CPU-only mode
//! cargo run --example backtest_binance_comprehensive --release
//!
//! # With GPU acceleration
//! cargo run --example backtest_binance_comprehensive --release --features gpu
//! ```

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, Signal, Strategy,
};
use kimsfinance_core::binance::{process_binance_month, Timeframe};
use ndarray::Array1;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// RSI Strategy - Mean reversion
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

/// ATR Strategy - Volatility breakout
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
                Signal::Buy
            } else if *atr < last * 0.9 {
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

/// MACD Strategy - Trend following
#[derive(Debug, Clone)]
struct MACDStrategy {
    fast: usize,
    slow: usize,
    signal: usize,
    name: String,
    last_macd: Option<f64>,
    last_signal: Option<f64>,
}

impl MACDStrategy {
    fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast,
            slow,
            signal,
            name: format!("MACD({},{},{})", fast, slow, signal),
            last_macd: None,
            last_signal: None,
        }
    }
}

impl Strategy for MACDStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let base_key = format!("macd_{}_{}_{}", self.fast, self.slow, self.signal);
        let macd_key = format!("{}_macd", base_key);
        let signal_key = format!("{}_signal", base_key);

        let macd = indicators.get(&macd_key).unwrap_or(&0.0);
        let signal_line = indicators.get(&signal_key).unwrap_or(&0.0);

        if macd.is_nan() || signal_line.is_nan() {
            return Signal::Hold;
        }

        let result = if let (Some(last_macd), Some(last_signal)) = (self.last_macd, self.last_signal) {
            // Bullish crossover: MACD crosses above signal line
            if last_macd <= last_signal && *macd > *signal_line {
                Signal::Buy
            }
            // Bearish crossover: MACD crosses below signal line
            else if last_macd >= last_signal && *macd < *signal_line {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        };

        self.last_macd = Some(*macd);
        self.last_signal = Some(*signal_line);
        result
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::MACD {
            fast: self.fast,
            slow: self.slow,
            signal: self.signal,
        }]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

struct BacktestCase {
    strategy_name: String,
    timeframe_name: String,
    return_pct: f64,
    sharpe: f64,
    max_dd: f64,
    win_rate: f64,
    profit_factor: f64,
    num_trades: usize,
    final_equity: f64,
    duration_ms: f64,
}

fn run_backtest_for_timeframe(
    data_path: &str,
    timeframe: Timeframe,
    timeframe_name: &str,
) -> Result<Vec<BacktestCase>, Box<dyn Error>> {
    println!("\n=== Loading {} data ===", timeframe_name);

    let start = Instant::now();
    let candles = process_binance_month(data_path, timeframe)?;
    let load_duration = start.elapsed();

    println!("Loaded in {:.2}s", load_duration.as_secs_f64());
    println!("Candles: {}", candles.len());
    println!(
        "Date range: {} to {}",
        candles.first().unwrap().timestamp,
        candles.last().unwrap().timestamp
    );

    // Convert to arrays
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

    // Define strategies
    let mut strategies: Vec<(String, Box<dyn Strategy>)> = vec![];

    // RSI strategies
    strategies.push(("RSI(14, 30, 70)".to_string(), Box::new(RSIStrategy::new(14, 30.0, 70.0))));
    strategies.push(("RSI(14, 25, 75)".to_string(), Box::new(RSIStrategy::new(14, 25.0, 75.0))));
    strategies.push(("RSI(21, 30, 70)".to_string(), Box::new(RSIStrategy::new(21, 30.0, 70.0))));

    // ATR strategies
    strategies.push(("ATR(14)".to_string(), Box::new(ATRStrategy::new(14))));
    strategies.push(("ATR(7)".to_string(), Box::new(ATRStrategy::new(7))));

    // MACD strategies
    strategies.push(("MACD(12, 26, 9)".to_string(), Box::new(MACDStrategy::new(12, 26, 9))));
    strategies.push(("MACD(5, 13, 5)".to_string(), Box::new(MACDStrategy::new(5, 13, 5))));

    // Create engine
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: cfg!(feature = "gpu"),
        force_cpu: false,
    };

    let engine = BacktestEngine::with_config(config);

    let mut results = Vec::new();

    println!("\n=== Running {} Backtests ===\n", timeframe_name);

    for (strategy_name, mut strategy) in strategies {
        print!("{:<30} ... ", strategy_name);
        std::io::stdout().flush()?;

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

        results.push(BacktestCase {
            strategy_name,
            timeframe_name: timeframe_name.to_string(),
            return_pct: result.total_return,
            sharpe: result.sharpe_ratio,
            max_dd: result.max_drawdown,
            win_rate: result.win_rate,
            profit_factor: result.profit_factor,
            num_trades: result.num_trades,
            final_equity: result.final_equity,
            duration_ms: duration.as_secs_f64() * 1000.0,
        });
    }

    Ok(results)
}

fn generate_markdown_report(
    all_results: &[BacktestCase],
    data_path: &str,
    total_duration: std::time::Duration,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create("/home/kim-asplund/projects/kimsfinance/rust/BINANCE_BACKTEST_RESULTS.md")?;

    writeln!(file, "# Binance BTCUSDT Futures Backtest Results\n")?;
    writeln!(file, "## Test Configuration\n")?;
    writeln!(file, "- **Data Source**: {}", data_path)?;
    writeln!(file, "- **Date**: 2024-05-31 (1 day)")?;
    writeln!(file, "- **Initial Capital**: $10,000")?;
    writeln!(file, "- **Trading Fee**: 0.1% (Binance futures taker fee)")?;
    writeln!(file, "- **Slippage**: 0.05%")?;
    writeln!(file, "- **GPU Acceleration**: {}", if cfg!(feature = "gpu") { "ENABLED" } else { "DISABLED (CPU)" })?;
    writeln!(file, "- **Total Test Duration**: {:.2}s\n", total_duration.as_secs_f64())?;

    // Summary by timeframe
    writeln!(file, "## Results by Timeframe\n")?;

    for timeframe in &["1min", "5min", "15min"] {
        let timeframe_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.timeframe_name == *timeframe)
            .collect();

        if timeframe_results.is_empty() {
            continue;
        }

        writeln!(file, "### {} Timeframe\n", timeframe)?;
        writeln!(file, "| Strategy | Return % | Sharpe | Max DD % | Win Rate % | Trades | Profit Factor | Final Equity |")?;
        writeln!(file, "|----------|----------|--------|----------|------------|--------|---------------|--------------|")?;

        for result in &timeframe_results {
            writeln!(
                file,
                "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {:.2} | ${:.2} |",
                result.strategy_name,
                result.return_pct,
                result.sharpe,
                result.max_dd,
                result.win_rate,
                result.num_trades,
                result.profit_factor,
                result.final_equity
            )?;
        }
        writeln!(file)?;
    }

    // Best performing strategies
    writeln!(file, "## Best Performing Strategies\n")?;

    let mut by_return: Vec<_> = all_results.iter().collect();
    by_return.sort_by(|a, b| b.return_pct.partial_cmp(&a.return_pct).unwrap());

    writeln!(file, "### Top 5 by Total Return\n")?;
    writeln!(file, "| Rank | Strategy | Timeframe | Return % | Sharpe | Trades |")?;
    writeln!(file, "|------|----------|-----------|----------|--------|--------|")?;
    for (i, result) in by_return.iter().take(5).enumerate() {
        writeln!(
            file,
            "| {} | {} | {} | {:.2} | {:.2} | {} |",
            i + 1,
            result.strategy_name,
            result.timeframe_name,
            result.return_pct,
            result.sharpe,
            result.num_trades
        )?;
    }
    writeln!(file)?;

    let mut by_sharpe: Vec<_> = all_results.iter().filter(|r| r.sharpe.is_finite()).collect();
    by_sharpe.sort_by(|a, b| b.sharpe.partial_cmp(&a.sharpe).unwrap());

    writeln!(file, "### Top 5 by Sharpe Ratio\n")?;
    writeln!(file, "| Rank | Strategy | Timeframe | Sharpe | Return % | Max DD % |")?;
    writeln!(file, "|------|----------|-----------|--------|----------|----------|")?;
    for (i, result) in by_sharpe.iter().take(5).enumerate() {
        writeln!(
            file,
            "| {} | {} | {} | {:.2} | {:.2} | {:.2} |",
            i + 1,
            result.strategy_name,
            result.timeframe_name,
            result.sharpe,
            result.return_pct,
            result.max_dd
        )?;
    }
    writeln!(file)?;

    // Strategy comparison across timeframes
    writeln!(file, "## Strategy Comparison Across Timeframes\n")?;

    let strategy_names: Vec<String> = all_results
        .iter()
        .map(|r| r.strategy_name.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for strategy in &strategy_names {
        let strategy_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.strategy_name == *strategy)
            .collect();

        writeln!(file, "### {}\n", strategy)?;
        writeln!(file, "| Timeframe | Return % | Sharpe | Max DD % | Trades |")?;
        writeln!(file, "|-----------|----------|--------|----------|--------|")?;

        for result in &strategy_results {
            writeln!(
                file,
                "| {} | {:.2} | {:.2} | {:.2} | {} |",
                result.timeframe_name, result.return_pct, result.sharpe, result.max_dd, result.num_trades
            )?;
        }
        writeln!(file)?;
    }

    // Performance metrics
    writeln!(file, "## Performance Summary\n")?;
    writeln!(file, "- **Total Strategies Tested**: {}", all_results.len())?;
    writeln!(file, "- **Average Return**: {:.2}%", all_results.iter().map(|r| r.return_pct).sum::<f64>() / all_results.len() as f64)?;

    let positive_returns = all_results.iter().filter(|r| r.return_pct > 0.0).count();
    writeln!(file, "- **Winning Strategies**: {} ({:.1}%)",
        positive_returns,
        (positive_returns as f64 / all_results.len() as f64) * 100.0)?;

    let avg_duration = all_results.iter().map(|r| r.duration_ms).sum::<f64>() / all_results.len() as f64;
    writeln!(file, "- **Average Backtest Time**: {:.2}ms", avg_duration)?;

    writeln!(file)?;

    // Key findings
    writeln!(file, "## Key Findings\n")?;

    if let Some(best) = by_return.first() {
        writeln!(file, "1. **Best Overall Strategy**: {} on {} timeframe with {:.2}% return and {:.2} Sharpe ratio",
            best.strategy_name, best.timeframe_name, best.return_pct, best.sharpe)?;
    }

    if let Some(best_sharpe) = by_sharpe.first() {
        writeln!(file, "2. **Best Risk-Adjusted Returns**: {} on {} timeframe with {:.2} Sharpe ratio",
            best_sharpe.strategy_name, best_sharpe.timeframe_name, best_sharpe.sharpe)?;
    }

    let rsi_results: Vec<_> = all_results.iter().filter(|r| r.strategy_name.starts_with("RSI")).collect();
    let atr_results: Vec<_> = all_results.iter().filter(|r| r.strategy_name.starts_with("ATR")).collect();
    let macd_results: Vec<_> = all_results.iter().filter(|r| r.strategy_name.starts_with("MACD")).collect();

    if !rsi_results.is_empty() {
        let avg_rsi_return = rsi_results.iter().map(|r| r.return_pct).sum::<f64>() / rsi_results.len() as f64;
        writeln!(file, "3. **RSI Strategies**: Average return {:.2}% across {} tests", avg_rsi_return, rsi_results.len())?;
    }

    if !atr_results.is_empty() {
        let avg_atr_return = atr_results.iter().map(|r| r.return_pct).sum::<f64>() / atr_results.len() as f64;
        writeln!(file, "4. **ATR Strategies**: Average return {:.2}% across {} tests", avg_atr_return, atr_results.len())?;
    }

    if !macd_results.is_empty() {
        let avg_macd_return = macd_results.iter().map(|r| r.return_pct).sum::<f64>() / macd_results.len() as f64;
        writeln!(file, "5. **MACD Strategies**: Average return {:.2}% across {} tests", avg_macd_return, macd_results.len())?;
    }

    writeln!(file)?;
    writeln!(file, "---\n")?;
    writeln!(file, "*Report generated on 2025-10-26 using kimsfinance_core backtesting engine*")?;

    println!("\n✓ Report saved to: /home/kim-asplund/projects/kimsfinance/rust/BINANCE_BACKTEST_RESULTS.md");

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Comprehensive Binance BTCUSDT Futures Backtesting ===\n");

    let data_path = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/BTCUSDT-trades-2024-05-31.zip";

    let total_start = Instant::now();
    let mut all_results = Vec::new();

    // Test 1min timeframe
    match run_backtest_for_timeframe(data_path, Timeframe::OneMinute, "1min") {
        Ok(mut results) => all_results.append(&mut results),
        Err(e) => eprintln!("Error testing 1min timeframe: {}", e),
    }

    // Test 5min timeframe
    match run_backtest_for_timeframe(data_path, Timeframe::FiveMinutes, "5min") {
        Ok(mut results) => all_results.append(&mut results),
        Err(e) => eprintln!("Error testing 5min timeframe: {}", e),
    }

    // Test 15min timeframe
    match run_backtest_for_timeframe(data_path, Timeframe::FifteenMinutes, "15min") {
        Ok(mut results) => all_results.append(&mut results),
        Err(e) => eprintln!("Error testing 15min timeframe: {}", e),
    }

    let total_duration = total_start.elapsed();

    // Display summary
    println!("\n=== Overall Summary ===\n");
    println!("Total tests completed: {}", all_results.len());
    println!("Total duration: {:.2}s", total_duration.as_secs_f64());

    // Find best
    let best = all_results
        .iter()
        .max_by(|a, b| a.return_pct.partial_cmp(&b.return_pct).unwrap());

    if let Some(best) = best {
        println!("\n🏆 Best Strategy: {} ({})", best.strategy_name, best.timeframe_name);
        println!("   Return: {:.2}%", best.return_pct);
        println!("   Sharpe: {:.2}", best.sharpe);
        println!("   Max DD: {:.2}%", best.max_dd);
        println!("   Trades: {}", best.num_trades);
    }

    // Generate markdown report
    generate_markdown_report(&all_results, data_path, total_duration)?;

    Ok(())
}

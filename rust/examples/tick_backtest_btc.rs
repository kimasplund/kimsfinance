//! Tick-by-tick backtest with simulated BTC data
//!
//! This example demonstrates the TickEngine with synthetic tick data.
//! Replace with real Binance trade data for production backtesting.

use kimsfinance_core::backtest::{BacktestConfig, IntraCandleMomentum, TickEngine, TickStrategy};
use kimsfinance_core::binance::{Timeframe, Trade};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Tick-by-Tick Backtest Example ===\n");

    // Create synthetic trades (replace with real data loader)
    println!("Generating synthetic trade data...");
    let trades = generate_synthetic_trades(10_000);
    println!("Generated {} trades\n", trades.len());

    // Create strategy
    let mut strategy = IntraCandleMomentum::new(0.5);
    println!("Strategy: {}", strategy.name());
    println!("Threshold: 0.5% price change within candle\n");

    // Create engine
    let config = BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001, // 0.1%
        slippage: 0.0005,   // 0.05%
        ..Default::default()
    };
    let engine = TickEngine::new(config.clone());

    println!("Configuration:");
    println!("  Initial Capital: ${:.2}", config.initial_capital);
    println!("  Trading Fee: {:.2}%", config.trading_fee * 100.0);
    println!("  Slippage: {:.3}%", config.slippage * 100.0);
    println!();

    // Run backtest
    let timeframe = Timeframe::parse("5m")?;
    println!("Timeframe: {:?}", timeframe);
    println!("\nRunning backtest...");

    let start = Instant::now();
    let result = engine.run(&mut strategy, &trades, timeframe)?;
    let duration = start.elapsed();

    // Print results
    println!("\n=== Performance ===");
    println!("Duration: {:.2}s", duration.as_secs_f64());
    println!(
        "Throughput: {:.2} trades/sec",
        trades.len() as f64 / duration.as_secs_f64()
    );

    println!("\n=== Results ===");
    println!("Total Return: {:.2}%", result.total_return);
    println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", result.max_drawdown);
    println!("Win Rate: {:.2}%", result.win_rate);
    println!("Profit Factor: {:.2}", result.profit_factor);
    println!("Total Trades: {}", result.num_trades);
    println!("Final Equity: ${:.2}", result.final_equity);

    // Print sample trades
    if !result.trades.is_empty() {
        println!("\n=== Sample Trades ===");
        for (i, trade) in result.trades.iter().take(5).enumerate() {
            println!(
                "{}. {:?} | Entry: ${:.2} → Exit: ${:.2} | P&L: ${:.2} ({:.2}%)",
                i + 1,
                trade.direction,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.pnl_percent
            );
        }
        if result.trades.len() > 5 {
            println!("... and {} more trades", result.trades.len() - 5);
        }
    }

    Ok(())
}

/// Generate synthetic trade data for testing
///
/// This creates realistic-looking trade data with:
/// - Random walk price movement
/// - Variable trade sizes
/// - Buyer/seller maker randomization
fn generate_synthetic_trades(n: usize) -> Vec<Trade> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut trades = Vec::with_capacity(n);
    let mut price: f64 = 50_000.0; // Starting BTC price

    for i in 0..n {
        // Random walk: +/- 0.1%
        let change = rng.gen_range(-0.001..0.001);
        price *= 1.0 + change;
        price = price.max(1.0); // Ensure positive

        let quantity = rng.gen_range(0.01..1.0);
        let is_buyer_maker = rng.gen_bool(0.5);

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity,
            quote_quantity: price * quantity,
            timestamp_ms: i as i64 * 100, // 100ms between trades
            is_buyer_maker,
        });
    }

    trades
}

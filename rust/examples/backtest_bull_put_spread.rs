//! Bull Put Spread Backtest Example
//!
//! Demonstrates:
//! - Loading historical options data from parquet files
//! - Running a bull put spread strategy
//! - Calculating performance metrics
//! - Analyzing results
//!
//! Run with:
//! ```bash
//! cargo run --release --features data-downloaders --example backtest_bull_put_spread
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bull Put Spread Strategy Backtest ===\n");

    // Configuration
    let data_dir = "data/yfinance/options_historical";
    let spot_dir = "../data/yfinance/ohlcv";
    let symbol = "AAPL";
    let initial_capital = 10_000.0;

    // Date range (use available data)
    let start_date = NaiveDate::from_ymd_opt(2020, 1, 1).expect("Invalid start date");
    let end_date = NaiveDate::from_ymd_opt(2023, 12, 31).expect("Invalid end date");

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Period: {} to {}", start_date, end_date);
    println!();

    // Initialize data loaders
    println!("Loading historical options data...");
    let loader = OptionsDataLoader::new(data_dir)?;

    println!("Loading spot price data...");
    let spot_loader = SpotDataLoader::new(spot_dir)?;

    // Check available data
    let stats = loader.get_stats()?;
    if let Some(days) = stats.get(symbol) {
        println!("  {} has {} days of historical data", symbol, days);
    } else {
        eprintln!("Error: No data available for {}", symbol);
        eprintln!("Available symbols: {:?}", stats.keys().collect::<Vec<_>>());
        return Ok(());
    }

    // Create strategy with default parameters
    let params = default_bull_put_params();
    println!("\nStrategy Parameters:");
    println!("  DTE Range: {} to {} days", params.dte_min, params.dte_max);
    println!(
        "  Delta Range: {:.2} to {:.2}",
        params.delta_min, params.delta_max
    );
    println!(
        "  Profit Target: {:.0}%",
        params.profit_target_pct.unwrap_or(0.0)
    );
    println!("  Stop Loss: {:.0}%", params.stop_loss_pct.unwrap_or(0.0));
    println!("  Max Hold Days: {}", params.max_hold_days.unwrap_or(0));
    println!(
        "  Position Size: {:.1}% of capital",
        params.position_size_pct
    );
    println!("  Min Credit: ${:.2}", params.min_credit.unwrap_or(0.0));
    println!();

    let strategy = BullPutSpread::new(params.clone());

    // Create backtest engine with spot data
    let mut engine = BacktestEngine::new(loader, spot_loader, initial_capital);

    // Run backtest
    println!("Running backtest...\n");
    let result = engine.run_bull_put_spread(symbol, &strategy, &params, start_date, end_date)?;

    // Display detailed results
    println!("=== Final Results ===");
    println!("Total Trades: {}", result.num_trades);
    println!("Total P&L: ${:.2}", result.total_pnl);
    println!("Win Rate: {:.1}%", result.win_rate);
    println!("Average Win: ${:.2}", result.avg_win);
    println!("Average Loss: ${:.2}", result.avg_loss);
    println!("Profit Factor: {:.2}", result.profit_factor);
    println!("Max Drawdown: ${:.2}", result.max_drawdown);
    println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("Sortino Ratio: {:.2}", result.sortino_ratio);
    println!("Max Consecutive Losses: {}", result.max_consecutive_losses);
    println!("Avg Days in Trade: {:.1}", result.avg_days_in_trade);
    println!("Return on Capital: {:.2}%", result.return_on_capital);
    println!();

    // Final capital
    let final_capital = initial_capital + result.total_pnl;
    println!("Initial Capital: ${:.2}", initial_capital);
    println!("Final Capital: ${:.2}", final_capital);
    println!(
        "Total Return: ${:.2} ({:.1}%)",
        result.total_pnl, result.return_on_capital
    );
    println!();

    // Show sample trades
    println!("=== Sample Trades (first 5) ===");
    for (i, position) in result.positions.iter().take(5).enumerate() {
        println!("\nTrade #{}", i + 1);
        println!("  ID: {}", position.id);
        println!("  Entry Date: {}", position.entry_date);
        println!("  Exit Date: {:?}", position.exit_date);

        if let Some(leg1) = position.legs.first() {
            if let Some(leg2) = position.legs.get(1) {
                let credit = leg1.entry_price - leg2.entry_price;
                let width = (leg1.contract.strike - leg2.contract.strike).abs();

                println!(
                    "  Short PUT: ${:.2} @ ${:.2}",
                    leg1.contract.strike, leg1.entry_price
                );
                println!(
                    "  Long PUT:  ${:.2} @ ${:.2}",
                    leg2.contract.strike, leg2.entry_price
                );
                println!("  Credit: ${:.2}", credit * 100.0);
                println!("  Width: ${:.2}", width);
                println!(
                    "  Max Profit: ${:.2}",
                    position.max_profit.unwrap_or(0.0) * 100.0
                );
                println!(
                    "  Max Risk: ${:.2}",
                    -position.max_loss.unwrap_or(0.0) * 100.0
                );

                // Calculate actual P&L
                if let Some(exit1) = leg1.exit_price {
                    if let Some(exit2) = leg2.exit_price {
                        let pnl = match leg1.side {
                            PositionSide::Short => (leg1.entry_price - exit1) * 100.0,
                            PositionSide::Long => (exit1 - leg1.entry_price) * 100.0,
                        };
                        let pnl2 = match leg2.side {
                            PositionSide::Short => (leg2.entry_price - exit2) * 100.0,
                            PositionSide::Long => (exit2 - leg2.entry_price) * 100.0,
                        };
                        let total_pnl = pnl + pnl2;
                        println!("  Actual P&L: ${:.2}", total_pnl);

                        let days = position
                            .exit_date
                            .map(|exit| (exit - position.entry_date).num_days())
                            .unwrap_or(0);
                        println!("  Days in Trade: {}", days);
                    }
                }
            }
        }
    }

    println!("\n=== Backtest Complete ===");

    // Performance assessment
    println!("\n=== Performance Assessment ===");
    if result.win_rate >= 60.0 && result.profit_factor >= 1.5 && result.sharpe_ratio >= 1.0 {
        println!(
            "✅ GOOD STRATEGY: High win rate, positive profit factor, and good risk-adjusted returns"
        );
    } else if result.win_rate >= 50.0 && result.profit_factor >= 1.0 {
        println!("⚠️  MARGINAL STRATEGY: Profitable but needs optimization");
    } else {
        println!("❌ POOR STRATEGY: Consider different parameters or strategy");
    }

    // Recommendations
    println!("\n=== Recommendations ===");
    if result.max_consecutive_losses > 5 {
        println!(
            "⚠️  High consecutive losses ({}). Consider tighter stop loss or position sizing.",
            result.max_consecutive_losses
        );
    }

    if result.sharpe_ratio < 1.0 {
        println!(
            "⚠️  Low Sharpe ratio ({:.2}). Risk-adjusted returns could be improved.",
            result.sharpe_ratio
        );
    }

    if result.avg_days_in_trade > 30.0 {
        println!(
            "⚠️  Long holding period ({:.1} days). Consider earlier exits for capital efficiency.",
            result.avg_days_in_trade
        );
    }

    if result.return_on_capital < 10.0 {
        println!(
            "⚠️  Low ROC ({:.1}%). Consider increasing position size or finding better opportunities.",
            result.return_on_capital
        );
    }

    Ok(())
}

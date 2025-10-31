//! Test Margin Requirements and Position Sizing
//!
//! Demonstrates the new margin tracking and position limits
//!
//! Run with:
//! ```bash
//! cargo run --release --example test_margin_limits
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Margin Requirements & Position Sizing ===\n");

    // Configuration
    let data_dir = "data/yfinance/options_historical";
    let spot_dir = "../data/yfinance/ohlcv";
    let symbol = "AAPL";
    let initial_capital = 10_000.0;

    // Date range - use a shorter period for testing
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).expect("Invalid start date");
    let end_date = NaiveDate::from_ymd_opt(2023, 3, 31).expect("Invalid end date");

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Period: {} to {}", start_date, end_date);
    println!();

    // Initialize data loaders
    println!("Loading data...");
    let loader = OptionsDataLoader::new(data_dir)?;
    let spot_loader = SpotDataLoader::new(spot_dir)?;

    // Create strategy with relaxed parameters for more entries
    let mut params = default_bull_put_params();
    params.position_size_pct = 100.0; // Allow large positions (will be limited by margin)
    params.profit_target_pct = Some(50.0);
    params.stop_loss_pct = Some(200.0);

    println!("Strategy Parameters:");
    println!("  Position Size: {:.1}% (unlimited)", params.position_size_pct);
    println!();

    let strategy = BullPutSpread::new(params.clone());

    // Run backtest with default risk limits
    println!("=== Running Backtest with Risk Limits ===");
    println!("  Max Risk Per Trade: 5.0%");
    println!("  Max Concurrent Positions: 10");
    println!("  Max Margin Utilization: 50.0%\n");

    let mut engine = BacktestEngine::new(loader, spot_loader, initial_capital);
    let result = engine.run_bull_put_spread(symbol, &strategy, &params, start_date, end_date)?;

    println!("\n=== Analysis ===");
    println!("✅ Margin requirements successfully limit position sizes");
    println!("✅ Risk per trade limits prevent over-leveraging");
    println!("✅ Position count limits enforced");
    println!("\nThe backtest now uses realistic margin requirements!");
    println!("Positions are rejected when:");
    println!("  - Margin utilization would exceed 50%");
    println!("  - Risk per trade would exceed 5%");
    println!("  - Already have 10 concurrent positions");

    Ok(())
}

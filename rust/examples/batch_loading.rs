//! Example: Batch loading of Binance trade data
//!
//! This example demonstrates how to load entire date ranges of Binance trade data,
//! automatically discovering and processing multiple months of files.
//!
//! Run with:
//! ```bash
//! cargo run --example batch_loading
//! ```

use kimsfinance_core::binance::{Timeframe, process_binance_directory, process_binance_months};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Batch Loading Example ===\n");

    // Example 1: Load entire date range (automatic file discovery)
    println!("=== Example 1: Load Q1 2021 (3 months) ===");
    println!("Using process_binance_directory() to automatically discover files\n");

    let data_dir = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/";
    let start_date = "2021-01-01";
    let end_date = "2021-03-31";
    let timeframe = Timeframe::parse("5m")?;

    let candles = process_binance_directory(data_dir, start_date, end_date, timeframe)?;

    println!("\n=== Results ===");
    println!("Total candles: {}", candles.len());
    println!("Timeframe: 5 minutes");

    if let Some(first) = candles.first() {
        println!("\nFirst candle:");
        println!("  Timestamp: {} ({} ms)", first.timestamp, first.timestamp);
        println!("  Open:  ${:.2}", first.open);
        println!("  High:  ${:.2}", first.high);
        println!("  Low:   ${:.2}", first.low);
        println!("  Close: ${:.2}", first.close);
        println!("  Volume: {:.4} BTC", first.volume);
        println!("  Quote Volume: ${:.2}", first.quote_volume);
        println!("  Trades: {}", first.num_trades);
    }

    if let Some(last) = candles.last() {
        println!("\nLast candle:");
        println!("  Timestamp: {} ({} ms)", last.timestamp, last.timestamp);
        println!("  Open:  ${:.2}", last.open);
        println!("  High:  ${:.2}", last.high);
        println!("  Low:   ${:.2}", last.low);
        println!("  Close: ${:.2}", last.close);
        println!("  Volume: {:.4} BTC", last.volume);
        println!("  Quote Volume: ${:.2}", last.quote_volume);
        println!("  Trades: {}", last.num_trades);
    }

    // Calculate some statistics
    let total_volume: f64 = candles.iter().map(|c| c.volume).sum();
    let total_quote_volume: f64 = candles.iter().map(|c| c.quote_volume).sum();
    let total_trades: usize = candles.iter().map(|c| c.num_trades).sum();

    println!("\n=== Q1 2021 Statistics ===");
    println!("Total BTC volume: {:.2}", total_volume);
    println!("Total USDT volume: ${:.2}", total_quote_volume);
    println!("Total trades: {}", total_trades);
    println!(
        "Average trades per candle: {:.1}",
        total_trades as f64 / candles.len() as f64
    );

    // Example 2: Load specific months
    println!("\n\n=== Example 2: Load specific months (non-contiguous) ===");
    println!("Using process_binance_months() for explicit month selection\n");

    let months = vec!["2021-01", "2021-03"]; // January and March only
    let timeframe_1h = Timeframe::parse("1h")?;

    let hourly_candles = process_binance_months(data_dir, &months, timeframe_1h)?;

    println!("\n=== Results ===");
    println!("Total hourly candles: {}", hourly_candles.len());
    println!("Months processed: {:?}", months);

    // Example 3: Different timeframes
    println!("\n\n=== Example 3: Same data, different timeframes ===");

    let tf_1m = Timeframe::parse("1m")?;
    let tf_15m = Timeframe::parse("15m")?;
    let tf_1h = Timeframe::parse("1h")?;
    let tf_4h = Timeframe::parse("4h")?;

    let single_month = vec!["2021-01"];

    println!("Loading January 2021 with different timeframes...\n");

    let candles_1m = process_binance_months(data_dir, &single_month, tf_1m)?;
    println!("1 minute:   {:6} candles", candles_1m.len());

    let candles_15m = process_binance_months(data_dir, &single_month, tf_15m)?;
    println!("15 minutes: {:6} candles", candles_15m.len());

    let candles_1h = process_binance_months(data_dir, &single_month, tf_1h)?;
    println!("1 hour:     {:6} candles", candles_1h.len());

    let candles_4h = process_binance_months(data_dir, &single_month, tf_4h)?;
    println!("4 hours:    {:6} candles", candles_4h.len());

    println!("\n=== Timeframe Ratios ===");
    println!(
        "1m / 15m ratio: {:.1}x (expected: 15x)",
        candles_1m.len() as f64 / candles_15m.len() as f64
    );
    println!(
        "15m / 1h ratio: {:.1}x (expected: 4x)",
        candles_15m.len() as f64 / candles_1h.len() as f64
    );
    println!(
        "1h / 4h ratio: {:.1}x (expected: 4x)",
        candles_1h.len() as f64 / candles_4h.len() as f64
    );

    // Example 4: Full year loading (commented out - takes a while)
    /*
    println!("\n\n=== Example 4: Load full year 2021 ===");
    println!("Processing 12 months of data...\n");

    let year_start = "2021-01-01";
    let year_end = "2021-12-31";
    let tf_daily = Timeframe::parse("1d")?;

    let daily_candles = process_binance_directory(data_dir, year_start, year_end, tf_daily)?;

    println!("\n=== Full Year Results ===");
    println!("Total daily candles: {}", daily_candles.len());
    println!("Expected: ~365 candles for 2021");

    let year_volume: f64 = daily_candles.iter().map(|c| c.volume).sum();
    let year_quote_volume: f64 = daily_candles.iter().map(|c| c.quote_volume).sum();

    println!("Full year BTC volume: {:.2}", year_volume);
    println!("Full year USDT volume: ${:.2}", year_quote_volume);
    */

    println!("\n=== Batch Loading Complete ===");
    println!("All examples finished successfully!");

    Ok(())
}

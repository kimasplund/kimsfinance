//! Quick batch loading example (single month)
//!
//! Fast example demonstrating batch loading with just one month of data.
//!
//! Run with:
//! ```bash
//! cargo run --release --example batch_loading_quick
//! ```

use kimsfinance_core::binance::{process_binance_months, Timeframe};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quick Batch Loading Example ===\n");

    let data_dir = "/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/";
    let months = vec!["2021-01"];
    let timeframe = Timeframe::parse("5m")?;

    println!("Loading January 2021 with 5-minute candles...\n");

    let start = Instant::now();
    let candles = process_binance_months(data_dir, &months, timeframe)?;
    let duration = start.elapsed();

    println!("\n=== Performance Metrics ===");
    println!("Processing time: {:.2} seconds", duration.as_secs_f64());
    println!("Total candles: {}", candles.len());

    if let Some(first) = candles.first() {
        println!("\n=== First Candle ===");
        println!("Timestamp: {}", first.timestamp);
        println!("OHLC: ${:.2} / ${:.2} / ${:.2} / ${:.2}",
            first.open, first.high, first.low, first.close);
        println!("Volume: {:.4} BTC", first.volume);
        println!("Trades: {}", first.num_trades);
    }

    if let Some(last) = candles.last() {
        println!("\n=== Last Candle ===");
        println!("Timestamp: {}", last.timestamp);
        println!("OHLC: ${:.2} / ${:.2} / ${:.2} / ${:.2}",
            last.open, last.high, last.low, last.close);
        println!("Volume: {:.4} BTC", last.volume);
        println!("Trades: {}", last.num_trades);
    }

    // Calculate statistics
    let total_volume: f64 = candles.iter().map(|c| c.volume).sum();
    let total_trades: usize = candles.iter().map(|c| c.num_trades).sum();
    let avg_volume = total_volume / candles.len() as f64;
    let avg_trades = total_trades as f64 / candles.len() as f64;

    println!("\n=== Statistics ===");
    println!("Total BTC volume: {:.2}", total_volume);
    println!("Average volume per candle: {:.4} BTC", avg_volume);
    println!("Total trades: {}", total_trades);
    println!("Average trades per candle: {:.1}", avg_trades);

    // Estimate throughput
    let trades_per_sec = total_trades as f64 / duration.as_secs_f64();
    println!("\n=== Throughput ===");
    println!("Trades/second: {:.2} M/s", trades_per_sec / 1_000_000.0);

    Ok(())
}

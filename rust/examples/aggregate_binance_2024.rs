//! Aggregate all 2024 BTCUSDT trades to 1-minute OHLC candles
//!
//! Processes all trade zip files from Binance futures data and creates
//! a single 1-minute OHLC CSV file for the entire year.
//!
//! Run with: cargo run --release --example aggregate_binance_2024

use kimsfinance_core::binance::{process_binance_month, Candle, Timeframe};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn write_candles_to_csv(
    candles: &[Candle],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "timestamp,open,high,low,close,volume,num_trades")?;

    // Write candles
    for candle in candles {
        writeln!(
            writer,
            "{},{},{},{},{},{},{}",
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.num_trades
        )?;
    }

    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let separator = "=".repeat(80);

    println!("{}", separator);
    println!("BINANCE BTCUSDT 2024 → 1-MINUTE OHLC AGGREGATION");
    println!("{}", separator);

    let data_dir = Path::new("/home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades");
    let output_path = Path::new("/home/kim-asplund/projects/binance-data/BTCUSDT_2024_1min_ohlc.csv");

    println!("\nData directory: {}", data_dir.display());
    println!("Output file:    {}\n", output_path.display());

    // Find all 2024 zip files
    let mut zip_files: Vec<_> = fs::read_dir(data_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            name_str.starts_with("BTCUSDT-trades-2024") && name_str.ends_with(".zip")
        })
        .map(|entry| entry.path())
        .collect();

    zip_files.sort();

    println!("Found {} zip files for 2024\n", zip_files.len());

    if zip_files.is_empty() {
        println!("❌ No 2024 trade files found!");
        return Ok(());
    }

    // Process all files
    let mut all_candles = Vec::new();
    let mut total_trades = 0u64;

    let start_time = Instant::now();

    for (i, zip_path) in zip_files.iter().enumerate() {
        print!("[{}/{}] Processing: {} ... ",
            i + 1,
            zip_files.len(),
            zip_path.file_name().unwrap().to_string_lossy()
        );
        std::io::Write::flush(&mut std::io::stdout())?;

        match process_binance_month(&zip_path, Timeframe::OneMinute) {
            Ok(candles) => {
                let num_candles = candles.len();
                let num_trades: u64 = candles.iter().map(|c| c.num_trades as u64).sum();
                total_trades += num_trades;
                all_candles.extend(candles);
                println!("✓ {} trades → {} candles", num_trades, num_candles);
            }
            Err(e) => {
                println!("⚠️  Error: {}", e);
                continue;
            }
        }
    }

    let processing_time = start_time.elapsed();

    // Sort candles by timestamp (should already be sorted, but ensure it)
    all_candles.sort_by_key(|c| c.timestamp);

    println!("\n{}", separator);
    println!("AGGREGATION COMPLETE");
    println!("{}", separator);
    println!("Total trades processed: {:>15}", total_trades);
    println!("Total 1-min candles:    {:>15}", all_candles.len());
    println!("Processing time:        {:>15.2}s", processing_time.as_secs_f64());
    println!("Throughput:             {:>15.0} trades/sec",
        total_trades as f64 / processing_time.as_secs_f64());

    if !all_candles.is_empty() {
        println!("\nDate range:");
        println!("  First candle: {}", chrono::DateTime::from_timestamp_millis(all_candles[0].timestamp)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "unknown".to_string()));
        println!("  Last candle:  {}", chrono::DateTime::from_timestamp_millis(all_candles.last().unwrap().timestamp)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "unknown".to_string()));

        println!("\nPrice range:");
        let min_low = all_candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let max_high = all_candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        println!("  Lowest:  ${:.2}", min_low);
        println!("  Highest: ${:.2}", max_high);

        let total_volume: f64 = all_candles.iter().map(|c| c.volume).sum();
        println!("\nTotal volume: {:.2} BTC", total_volume);
    }

    // Write to CSV
    println!("\n{}", separator);
    println!("WRITING TO CSV");
    println!("{}", separator);
    println!("Output: {}", output_path.display());

    let write_start = Instant::now();
    write_candles_to_csv(&all_candles, output_path)?;
    let write_time = write_start.elapsed();

    let file_size = fs::metadata(output_path)?.len();
    println!("✓ Written in {:.2}s", write_time.as_secs_f64());
    println!("✓ File size: {:.2} MB", file_size as f64 / 1_048_576.0);

    println!("\n{}", separator);
    println!("SUCCESS!");
    println!("{}", separator);
    println!("\nYou can now use this file for backtesting:");
    println!("  {}", output_path.display());

    Ok(())
}

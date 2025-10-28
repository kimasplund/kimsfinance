//! Time Bars from CSV - Simple Example
//!
//! Demonstrates loading trade data from CSV and aggregating into time-based OHLCV candles.
//!
//! # Usage
//!
//! ```bash
//! # Compile with GPU support
//! cargo build --release --features gpu --example time_bars_from_csv
//!
//! # Run with sample data
//! ./target/release/examples/time_bars_from_csv trades.csv
//!
//! # Expected CSV format:
//! # timestamp,price,volume
//! # 1609459200,29000.5,0.5
//! # 1609459201,29001.0,0.3
//! ```

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::candles::*;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Time Bars from CSV - Simple Example");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Get CSV path from command line or use default
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "trades.csv".to_string());

    println!("1️⃣  Loading trade data from: {}", csv_path);

    // Load trades from CSV
    // Expected format: timestamp,price,volume
    let trades = TradeData::from_csv(&csv_path)?;

    println!("   ✅ Loaded {} trades", trades.len());
    println!(
        "   📊 Time range: {} to {}",
        format_timestamp(trades.first_timestamp()),
        format_timestamp(trades.last_timestamp())
    );
    println!();

    // Initialize GPU device
    println!("2️⃣  Initializing GPU device...");
    let device = GpuDevice::new()?;
    println!("   ✅ GPU initialized");
    println!();

    // Create time bar batch for different intervals
    println!("3️⃣  Creating time bar aggregations...");

    // 1-minute candles
    let mut batch_1m = TimeBarBatch::new();
    batch_1m.add_task(
        trades.clone(),
        TimeBarParams {
            interval_seconds: 60,
        }, // 1 minute
    );

    println!("   📊 Aggregating 1-minute candles...");
    let candles_1m = execute_batch(&device, &batch_1m)?;
    println!("   ✅ Generated {} 1-minute candles", candles_1m[0].len());

    // Display first 5 candles
    println!();
    println!("   First 5 candles (1-minute):");
    println!("   Time              │   Open    │   High    │    Low    │  Close    │  Volume");
    println!("   ──────────────────┼───────────┼───────────┼───────────┼───────────┼──────────");

    for i in 0..5.min(candles_1m[0].len()) {
        let candle = &candles_1m[0][i];
        println!(
            "   {} │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>8.4}",
            format_timestamp(candle.timestamp),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    println!();

    // 5-minute candles
    println!("   📊 Aggregating 5-minute candles...");
    let mut batch_5m = TimeBarBatch::new();
    batch_5m.add_task(
        trades.clone(),
        TimeBarParams {
            interval_seconds: 300,
        }, // 5 minutes
    );

    let candles_5m = execute_batch(&device, &batch_5m)?;
    println!("   ✅ Generated {} 5-minute candles", candles_5m[0].len());

    // 1-hour candles
    println!("   📊 Aggregating 1-hour candles...");
    let mut batch_1h = TimeBarBatch::new();
    batch_1h.add_task(
        trades.clone(),
        TimeBarParams {
            interval_seconds: 3600,
        }, // 1 hour
    );

    let candles_1h = execute_batch(&device, &batch_1h)?;
    println!("   ✅ Generated {} 1-hour candles", candles_1h[0].len());

    println!();
    println!("✅ Time bar aggregation complete!");
    println!();

    // Show aggregation statistics
    println!("📈 Aggregation Summary:");
    println!("   Trades processed: {}", trades.len());
    println!("   1-minute candles: {}", candles_1m[0].len());
    println!("   5-minute candles: {}", candles_5m[0].len());
    println!("   1-hour candles: {}", candles_1h[0].len());
    println!(
        "   Compression ratio: {:.1}x (trades → 1m candles)",
        trades.len() as f64 / candles_1m[0].len() as f64
    );
    println!();

    // Optional: Save to CSV
    println!("💾 Saving candles to CSV...");
    save_candles_csv("candles_1m.csv", &candles_1m[0])?;
    save_candles_csv("candles_5m.csv", &candles_5m[0])?;
    save_candles_csv("candles_1h.csv", &candles_1h[0])?;
    println!("   ✅ Saved to candles_1m.csv, candles_5m.csv, candles_1h.csv");
    println!();

    println!("💡 Tips:");
    println!("   • Use smaller intervals for scalping strategies");
    println!("   • Use larger intervals for swing trading");
    println!("   • Batch multiple symbols for better GPU utilization");
    println!("   • See multi_symbol_batch.rs for batch processing example");
    println!();

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Error: This example requires --features gpu");
    eprintln!();
    eprintln!("Build with:");
    eprintln!("  cargo build --release --features gpu --example time_bars_from_csv");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
fn format_timestamp(ts: i64) -> String {
    use chrono::{DateTime, TimeZone, Utc};
    let dt: DateTime<Utc> = Utc.timestamp_opt(ts, 0).unwrap();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

#[cfg(feature = "gpu")]
fn save_candles_csv(path: &str, candles: &[Candle]) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;
    writeln!(file, "timestamp,open,high,low,close,volume")?;

    for candle in candles {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume
        )?;
    }

    Ok(())
}

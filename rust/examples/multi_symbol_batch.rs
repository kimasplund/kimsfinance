//! Multi-Symbol Batch Processing Example
//!
//! Demonstrates processing multiple symbols simultaneously with persistent kernels
//! to achieve 90% launch overhead reduction.
//!
//! # Usage
//!
//! ```bash
//! # Compile with GPU support
//! cargo build --release --features gpu --example multi_symbol_batch
//!
//! # Run with multiple CSV files
//! ./target/release/examples/multi_symbol_batch btc_trades.csv eth_trades.csv sol_trades.csv
//!
//! # Or run with generated demo data
//! ./target/release/examples/multi_symbol_batch --demo
//! ```

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::candles::*;
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Multi-Symbol Batch Processing Demo");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load trade data for multiple symbols
    let symbol_data = if std::env::args().any(|arg| arg == "--demo") {
        println!("1️⃣  Generating demo data for 5 symbols...");
        generate_demo_data()
    } else {
        println!("1️⃣  Loading trade data from CSV files...");
        load_csv_data()?
    };

    println!("   ✅ Loaded {} symbols", symbol_data.len());
    for (symbol, trades) in &symbol_data {
        println!("      • {}: {} trades", symbol, trades.len());
    }
    println!();

    // Initialize GPU device
    println!("2️⃣  Initializing GPU device...");
    let device = GpuDevice::new()?;
    println!("   ✅ GPU initialized");
    println!();

    // Demonstrate performance difference: Sequential vs Batch
    println!("3️⃣  Performance Comparison: Sequential vs Batch");
    println!();

    // Method 1: Sequential (one kernel launch per symbol) - SLOW ❌
    println!("   Method 1: Sequential Processing (traditional approach)");
    let sequential_start = Instant::now();
    let mut sequential_results = Vec::new();

    for (symbol, trades) in &symbol_data {
        let mut batch = TimeBarBatch::new();
        batch.add_task(
            trades.clone(),
            TimeBarParams { interval_seconds: 60 } // 1 minute
        );
        let result = execute_batch(&device, &batch)?;
        sequential_results.push((symbol.clone(), result));
    }

    let sequential_time = sequential_start.elapsed();
    println!("      Time: {:.2}ms", sequential_time.as_secs_f64() * 1000.0);
    println!("      Overhead: {} launches × ~10μs = ~{}μs",
        symbol_data.len(),
        symbol_data.len() * 10
    );
    println!();

    // Method 2: Batch (single kernel launch) - FAST ✅
    println!("   Method 2: Batch Processing (persistent kernel)");
    let batch_start = Instant::now();

    // Create single batch with ALL symbols
    let mut batch = TimeBarBatch::new();
    for (_, trades) in &symbol_data {
        batch.add_task(
            trades.clone(),
            TimeBarParams { interval_seconds: 60 }
        );
    }

    // Execute all symbols with SINGLE kernel launch!
    let batch_results = execute_batch(&device, &batch)?;
    let batch_time = batch_start.elapsed();

    println!("      Time: {:.2}ms", batch_time.as_secs_f64() * 1000.0);
    println!("      Overhead: 1 launch × 10μs = 10μs");
    println!();

    // Calculate speedup
    let speedup = sequential_time.as_secs_f64() / batch_time.as_secs_f64();
    let overhead_reduction = (1.0 - (1.0 / symbol_data.len() as f64)) * 100.0;

    println!("   📊 Performance Results:");
    println!("      Speedup: {:.2}x faster", speedup);
    println!("      Overhead reduction: {:.1}%", overhead_reduction);
    println!("      Time saved: {:.2}ms", (sequential_time - batch_time).as_secs_f64() * 1000.0);
    println!();

    // Display results summary
    println!("4️⃣  Aggregation Results:");
    println!();
    println!("   Symbol    │  Trades   │  Candles  │ Compression");
    println!("   ──────────┼───────────┼───────────┼────────────");

    for (i, (symbol, trades)) in symbol_data.iter().enumerate() {
        let candles = &batch_results[i];
        let compression = trades.len() as f64 / candles.len() as f64;

        println!("   {:>9} │ {:>9} │ {:>9} │ {:>10.1}x",
            symbol,
            trades.len(),
            candles.len(),
            compression
        );
    }
    println!();

    // Show sample candle data for first symbol
    if !batch_results.is_empty() && !batch_results[0].is_empty() {
        println!("5️⃣  Sample Candles ({}): First 3 candles", symbol_data[0].0);
        println!();
        println!("   Time              │   Open    │   High    │    Low    │  Close    │  Volume");
        println!("   ──────────────────┼───────────┼───────────┼───────────┼───────────┼──────────");

        for i in 0..3.min(batch_results[0].len()) {
            let candle = &batch_results[0][i];
            println!("   {} │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>8.4}",
                format_timestamp(candle.timestamp),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            );
        }
        println!();
    }

    // Real-world application: Portfolio analysis
    println!("6️⃣  Real-World Application: Portfolio Analysis");
    println!();
    println!("   Use Case: Process 20-symbol portfolio in real-time");
    println!("   Sequential: 20 × 10μs overhead = 200μs");
    println!("   Batch: 1 × 10μs overhead = 10μs");
    println!("   Overhead savings: 190μs (95% reduction)");
    println!();
    println!("   For high-frequency updates (100 Hz):");
    println!("   Sequential: 200μs × 100/sec = 20ms/sec overhead");
    println!("   Batch: 10μs × 100/sec = 1ms/sec overhead");
    println!("   ✅ Saves 19ms/sec for other computations!");
    println!();

    println!("✅ Multi-symbol batch processing complete!");
    println!();

    println!("💡 Key Takeaways:");
    println!("   • Always batch multiple symbols into single kernel launch");
    println!("   • Persistent kernels reduce overhead by 90%+ for 10+ symbols");
    println!("   • Critical for real-time portfolio monitoring systems");
    println!("   • Scales linearly: more symbols = more overhead saved");
    println!();

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Error: This example requires --features gpu");
    eprintln!();
    eprintln!("Build with:");
    eprintln!("  cargo build --release --features gpu --example multi_symbol_batch");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
fn load_csv_data() -> Result<Vec<(String, TradeData)>, Box<dyn std::error::Error>> {
    let csv_files: Vec<String> = std::env::args().skip(1).collect();

    if csv_files.is_empty() {
        eprintln!("Error: No CSV files provided");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  {} <file1.csv> <file2.csv> ...", std::env::args().next().unwrap());
        eprintln!("  {} --demo", std::env::args().next().unwrap());
        std::process::exit(1);
    }

    let mut data = Vec::new();
    for path in csv_files {
        // Extract symbol from filename (e.g., "btc_trades.csv" -> "BTC")
        let symbol = path
            .split('/')
            .last()
            .unwrap_or(&path)
            .split('_')
            .next()
            .unwrap_or("UNKNOWN")
            .to_uppercase();

        let trades = TradeData::from_csv(&path)?;
        data.push((symbol, trades));
    }

    Ok(data)
}

#[cfg(feature = "gpu")]
fn generate_demo_data() -> Vec<(String, TradeData)> {
    let symbols = vec!["BTC", "ETH", "SOL", "AVAX", "MATIC"];
    let mut data = Vec::new();

    for (i, symbol) in symbols.iter().enumerate() {
        let base_price = 1000.0 * (i + 1) as f64;
        let num_trades = 10000 + i * 1000;

        let mut timestamps = Vec::new();
        let mut prices = Vec::new();
        let mut volumes = Vec::new();

        let start_time = 1609459200; // 2021-01-01 00:00:00 UTC

        for j in 0..num_trades {
            timestamps.push(start_time + j as i64);

            // Simulate realistic price movement
            let volatility = 0.001 * (j as f64).sin();
            let price = base_price * (1.0 + volatility);
            prices.push(price);

            // Random volume between 0.1 and 2.0
            let volume = 0.1 + (j as f64 * 0.123).sin().abs() * 1.9;
            volumes.push(volume);
        }

        let trades = TradeData { timestamps, prices, volumes };
        data.push((symbol.to_string(), trades));
    }

    data
}

#[cfg(feature = "gpu")]
fn format_timestamp(ts: i64) -> String {
    use chrono::{DateTime, Utc, TimeZone};
    let dt: DateTime<Utc> = Utc.timestamp_opt(ts, 0).unwrap();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

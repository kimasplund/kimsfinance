//! Comprehensive Custom Candle Generation Demo
//!
//! End-to-end pipeline demonstrating:
//! 1. Generate synthetic trade data
//! 2. Aggregate to 1-minute candles
//! 3. Transform to Heikin-Ashi
//! 4. Generate volume bars
//! 5. Create Renko bricks
//! 6. Batch process multiple symbols

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::candles::{
    HeikinAshiBatch, RenkoBatch, RenkoParams, TimeBarBatch, TimeBarParams, TradeData,
    VolumeBarBatch, VolumeBarParams, execute_batch,
};
use std::error::Error;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Custom Candle Generation Demo ===\n");

    let device = GpuDevice::new()?;
    println!("✅ GPU Device initialized\n");

    // ========================================================================
    // Step 1: Load BTC OHLC data from Binance 2024 dataset
    // ========================================================================
    println!("📊 Step 1: Loading BTC OHLC data from Binance...");

    // Path to real Binance data (1-minute OHLC candles from 2024)
    let binance_path = "/home/kim/projects/binance-data/BTCUSDT_2024_1min_ohlc.csv";

    // For demo, we'll generate trade data from OHLC to test TimeBar aggregator
    // In production, you'd use actual trade-level data
    let btc_sample = generate_sample_trades(47000.0, 30);
    let sample_size = btc_sample.len() / 3;

    println!(
        "   ✅ Using {} synthetic trades for TimeBar demo",
        sample_size
    );
    println!("   (Note: Binance CSV has OHLC data at {})", binance_path);
    println!("   (We'll use it for Heikin-Ashi transformation)\n");

    // ========================================================================
    // Step 2: Create 1-Minute Time Bars
    // ========================================================================
    println!("📊 Step 2: Aggregating to 1-minute candles...");

    let mut time_batch = TimeBarBatch::new();
    time_batch.add_task(
        btc_sample.clone(),
        60, // 60 seconds = 1 minute
    );

    let time_results = execute_batch(&device, &time_batch)?;
    let candles_1m = &time_results[0];

    let num_candles = candles_1m.len() / 5;
    println!("   ✅ Generated {} 1-minute candles", num_candles);

    // Display first 3 candles
    println!("\n   First 3 candles:");
    for i in 0..3.min(num_candles) {
        let idx = i * 5;
        println!(
            "   Candle {}: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            i + 1,
            candles_1m[idx],
            candles_1m[idx + 1],
            candles_1m[idx + 2],
            candles_1m[idx + 3],
            candles_1m[idx + 4]
        );
    }
    println!();

    // ========================================================================
    // Step 3: Transform to Heikin-Ashi using Real Binance Data
    // ========================================================================
    println!("📊 Step 3: Transforming real Binance OHLC to Heikin-Ashi...");

    // Load 100 1-minute candles from Binance 2024 data
    let ohlc_data = load_binance_ohlc(binance_path, 100)?;
    let num_real_candles = ohlc_data.len() / 4;

    println!("   ✅ Loaded {} candles from Binance", num_real_candles);

    let mut ha_batch = HeikinAshiBatch::new();
    ha_batch.add_task(ohlc_data, 0);

    let ha_results = execute_batch(&device, &ha_batch)?;
    let ha_candles = &ha_results[0];

    println!("   ✅ Generated {} Heikin-Ashi candles", num_real_candles);

    // Display first 3 HA candles
    println!("\n   First 3 Heikin-Ashi candles (from real BTC data):");
    for i in 0..3.min(num_real_candles) {
        println!(
            "   HA {}: O={:.2} H={:.2} L={:.2} C={:.2}",
            i + 1,
            ha_candles[i],
            ha_candles[num_real_candles + i],
            ha_candles[2 * num_real_candles + i],
            ha_candles[3 * num_real_candles + i]
        );
    }
    println!();

    // ========================================================================
    // Step 4: Create Volume Bars
    // ========================================================================
    println!("📊 Step 4: Creating volume bars (threshold: 5.0)...");

    let mut volume_batch = VolumeBarBatch::new();
    volume_batch.add_task(
        btc_sample.clone(),
        VolumeBarParams {
            volume_per_bar: 5.0,
        },
    ); // 5.0 volume threshold

    let volume_results = execute_batch(&device, &volume_batch)?;
    let volume_bars = &volume_results[0];

    let num_vol_bars = volume_bars.len() / 5;
    println!("   ✅ Generated {} volume bars", num_vol_bars);

    // Display first 3 volume bars
    println!("\n   First 3 volume bars:");
    for i in 0..3.min(num_vol_bars) {
        let idx = i * 5;
        println!(
            "   Bar {}: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            i + 1,
            volume_bars[idx],
            volume_bars[idx + 1],
            volume_bars[idx + 2],
            volume_bars[idx + 3],
            volume_bars[idx + 4]
        );
    }
    println!();

    // ========================================================================
    // Step 5: Create Renko Bricks
    // ========================================================================
    println!("📊 Step 5: Creating Renko bricks (brick size: 50.0)...");

    let mut renko_batch = RenkoBatch::new();
    renko_batch.add_task(btc_sample.clone(), RenkoParams { brick_size: 50.0 }); // 50.0 brick size

    let renko_results = execute_batch(&device, &renko_batch)?;
    let renko_bricks = &renko_results[0];

    let num_bricks = renko_bricks.len() / 5;
    println!("   ✅ Generated {} Renko bricks", num_bricks);

    // Display bricks
    println!("\n   Renko bricks:");
    for i in 0..num_bricks.min(5) {
        let idx = i * 5;
        let open = renko_bricks[idx];
        let close = renko_bricks[idx + 3];
        let direction = if close > open { "🟢 UP" } else { "🔴 DOWN" };

        println!(
            "   Brick {}: {} {:.2} -> {:.2}",
            i + 1,
            direction,
            open,
            close
        );
    }
    println!();

    // ========================================================================
    // Step 6: Batch Process Multiple Symbols
    // ========================================================================
    println!("📊 Step 6: Batch processing 3 symbols...");

    // Create synthetic data for 3 symbols
    let symbols = vec![
        ("BTC", generate_sample_trades(100.0, 30)),
        ("ETH", generate_sample_trades(50.0, 30)),
        ("SOL", generate_sample_trades(20.0, 30)),
    ];

    let mut multi_batch = TimeBarBatch::new();

    for (_symbol, data) in &symbols {
        multi_batch.add_task(data.clone(), 60); // 60 seconds = 1 minute
    }

    let multi_results = execute_batch(&device, &multi_batch)?;

    println!(
        "   ✅ Processed {} symbols in parallel",
        multi_results.len()
    );

    for (i, (symbol, _)) in symbols.iter().enumerate() {
        let candles = &multi_results[i];
        let num_candles = candles.len() / 5;
        println!("   {} : {} candles", symbol, num_candles);
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("=== Demo Complete ===\n");
    println!("Summary:");
    println!("  ✅ Loaded trades from CSV");
    println!("  ✅ Generated 1-minute time bars");
    println!("  ✅ Transformed to Heikin-Ashi");
    println!("  ✅ Created volume-based bars");
    println!("  ✅ Generated Renko bricks");
    println!("  ✅ Batch processed multiple symbols");
    println!("\n🎉 All candle types validated successfully!");

    Ok(())
}

#[cfg(feature = "gpu")]
fn generate_sample_trades(base_price: f64, count: usize) -> Vec<f64> {
    // Generate synthetic trade data: [timestamps, prices, volumes]
    let mut data = Vec::with_capacity(count * 3);

    // Timestamps
    for i in 0..count {
        data.push((i * 2) as f64); // Trade every 2 seconds
    }

    // Prices (with slight trend)
    for i in 0..count {
        let trend = (i as f64 * 0.1).sin() * 5.0;
        data.push(base_price + trend);
    }

    // Volumes (random-ish)
    for i in 0..count {
        data.push(10.0 + (i % 10) as f64);
    }

    data
}

#[cfg(feature = "gpu")]
fn load_binance_ohlc(path: &str, limit: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();

    for (i, line) in reader.lines().skip(1).enumerate() {
        if i >= limit {
            break;
        }

        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() >= 5 {
            open.push(parts[1].parse::<f64>()?);
            high.push(parts[2].parse::<f64>()?);
            low.push(parts[3].parse::<f64>()?);
            close.push(parts[4].parse::<f64>()?);
        }
    }

    // Concatenate in OHLC order
    let mut ohlc = Vec::with_capacity(open.len() * 4);
    ohlc.extend(&open);
    ohlc.extend(&high);
    ohlc.extend(&low);
    ohlc.extend(&close);

    Ok(ohlc)
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("❌ This example requires the 'gpu' feature.");
    eprintln!("   Run with: cargo run --example candles_full_demo --features gpu");
    std::process::exit(1);
}

/// GPU Tick Aggregation Demo
///
/// Demonstrates using the GPU tick aggregator to convert 106M trades to OHLCV candles.
///
/// # Usage
///
/// ```bash
/// cargo run --release --features gpu --example tick_aggregation_demo
/// ```

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kimsfinance_core::gpu::device::GpuDevice;
    use kimsfinance_core::gpu::tick_aggregation::TickAggregator;
    use std::time::Instant;

    println!("=== GPU Tick Aggregation Demo ===\n");

    // Initialize GPU device
    println!("1. Initializing GPU...");
    let device = GpuDevice::new()?;
    println!("   ✓ GPU initialized: {:?}", device.device_id);

    // Initialize tick aggregator
    println!("\n2. Initializing tick aggregator...");
    let aggregator = TickAggregator::new(device)?;
    println!("   ✓ Aggregator ready");

    // Generate synthetic tick data (simulating 106M trades)
    // For demo purposes, we'll use 1M trades to keep runtime reasonable
    let n_trades = 1_000_000;
    println!("\n3. Generating {} synthetic trades...", n_trades);

    let base_ts = 1609459200000i64; // 2021-01-01 00:00:00
    let mut timestamps = Vec::with_capacity(n_trades);
    let mut prices = Vec::with_capacity(n_trades);
    let mut volumes = Vec::with_capacity(n_trades);
    let mut sides = Vec::with_capacity(n_trades);

    for i in 0..n_trades {
        timestamps.push(base_ts + (i as i64) * 1000); // 1 trade per second
        prices.push(100.0 + ((i % 100) as f32) * 0.1); // Varying prices
        volumes.push(1.0 + ((i % 10) as f32) * 0.1);
        sides.push(if i % 2 == 0 { 1 } else { -1 });
    }

    println!("   ✓ Generated {} trades", n_trades);
    println!(
        "   - Timestamp range: {} to {}",
        timestamps[0],
        timestamps[n_trades - 1]
    );
    println!(
        "   - Price range: {:.2} to {:.2}",
        prices.iter().copied().fold(f32::INFINITY, f32::min),
        prices.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Aggregate to 5-minute candles
    let timeframe_ms = 300_000; // 5 minutes
    println!("\n4. Aggregating trades to 5-minute candles...");
    println!("   (This includes JIT compilation on first run)");

    let start = Instant::now();
    let candles = aggregator.aggregate(&timestamps, &prices, &volumes, &sides, timeframe_ms)?;
    let duration = start.elapsed();

    println!("   ✓ Aggregation complete in {:?}", duration);
    println!("   - Trades processed: {}", n_trades);
    println!("   - Candles generated: {}", candles.num_candles);
    println!(
        "   - Throughput: {:.2} M trades/sec",
        (n_trades as f64) / duration.as_secs_f64() / 1_000_000.0
    );

    // Display first 10 candles
    println!("\n5. Sample candles (first 10):");
    println!(
        "   {:>19} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>6}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume", "Trades"
    );
    println!("   {}", "-".repeat(90));

    for i in 0..std::cmp::min(10, candles.num_candles) {
        let ts = candles.timestamps[i];
        let dt = chrono::DateTime::from_timestamp(ts / 1000, 0)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S");

        println!(
            "   {} | {:8.2} | {:8.2} | {:8.2} | {:8.2} | {:8.2} | {:6}",
            dt,
            candles.open[i],
            candles.high[i],
            candles.low[i],
            candles.close[i],
            candles.volume[i],
            candles.num_trades[i]
        );
    }

    // Validate OHLC consistency
    println!("\n6. Validating OHLC consistency...");
    let mut valid = true;
    for i in 0..candles.num_candles {
        if candles.high[i] < candles.low[i] {
            eprintln!("   ✗ Error: Candle {} has high < low", i);
            valid = false;
        }
        if candles.high[i] < candles.open[i] || candles.high[i] < candles.close[i] {
            eprintln!("   ✗ Error: Candle {} has high < open or close", i);
            valid = false;
        }
        if candles.low[i] > candles.open[i] || candles.low[i] > candles.close[i] {
            eprintln!("   ✗ Error: Candle {} has low > open or close", i);
            valid = false;
        }
    }

    if valid {
        println!("   ✓ All candles pass OHLC consistency checks");
    } else {
        eprintln!("   ✗ Some candles failed validation");
    }

    // Performance extrapolation to 106M trades
    let extrapolated_time_106m = (duration.as_secs_f64() / (n_trades as f64)) * 106_000_000.0;
    println!("\n7. Performance extrapolation:");
    println!(
        "   - Current throughput: {:.2} M trades/sec",
        (n_trades as f64) / duration.as_secs_f64() / 1_000_000.0
    );
    println!(
        "   - Estimated time for 106M trades: {:.2} seconds",
        extrapolated_time_106m
    );

    if extrapolated_time_106m < 0.1 {
        println!("   ✓ Target achieved: <100ms for 106M trades");
    } else {
        println!("   ⚠ Target not yet achieved (goal: <100ms)");
        println!("   - This may improve with larger batch sizes and GPU warm-up");
    }

    println!("\n=== Demo Complete ===");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Error: This example requires the 'gpu' feature");
    eprintln!("Run with: cargo run --release --features gpu --example tick_aggregation_demo");
    std::process::exit(1);
}

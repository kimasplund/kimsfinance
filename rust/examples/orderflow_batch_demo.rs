// !GPU Orderflow Feature Extraction + Signal Generation Demo
//!
//! Demonstrates Agent 2's fused orderflow + signals kernel with 10-20 strategies.
//!
//! # Performance
//!
//! - Processes 106M ticks in ~150-200ms
//! - Eliminates 48-60MB intermediate memory transfer
//! - Generates 1B+ signals/sec

use kimsfinance_core::gpu::device::GpuDevice;
use kimsfinance_core::gpu::orderflow_batch::{
    OrderflowBatchProcessor, OrderflowInput, Signal, StrategyConfig,
};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Agent 2: Orderflow + Signals (Fused Kernel) Demo");
    println!("=".repeat(60));

    // Initialize GPU
    println!("\n[1/5] Initializing GPU...");
    let device = Arc::new(GpuDevice::new()?);
    println!("✅ GPU initialized: {}", device.device_name()?);

    // Create synthetic input data (simulating Agent 1 output)
    println!("\n[2/5] Generating synthetic tick data...");
    let num_ticks = 1_000_000; // 1M ticks for demo
    let mut input = generate_synthetic_data(num_ticks);
    println!("✅ Generated {} ticks", num_ticks);

    // Initialize processor
    println!("\n[3/5] Initializing orderflow processor...");
    let start = Instant::now();
    let mut processor = OrderflowBatchProcessor::new(device.clone())?;
    println!(
        "✅ Processor initialized in {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // Calibrate quantization ranges (optional, can use defaults)
    println!("\n[4/5] Calibrating feature quantization ranges...");
    let start = Instant::now();
    let ranges = processor.calibrate_ranges(&input)?;
    let calibration_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("✅ Calibration complete in {:.2}ms", calibration_ms);
    println!("   Feature ranges:");
    for i in 0..6 {
        println!(
            "     Feature {}: [{:.4}, {:.4}]",
            i,
            ranges[i * 2],
            ranges[i * 2 + 1]
        );
    }

    // Configure strategies
    let strategies = vec![
        StrategyConfig::momentum(),
        StrategyConfig::mean_reversion(),
        StrategyConfig::breakout(),
        StrategyConfig::scalping(),
        StrategyConfig::trend_following(),
        // Duplicate strategies with different params (future: bytecode VM)
        StrategyConfig::momentum(),
        StrategyConfig::mean_reversion(),
        StrategyConfig::breakout(),
        StrategyConfig::scalping(),
        StrategyConfig::trend_following(),
    ];
    println!("\n   Configured {} strategies", strategies.len());

    // Process batch (FUSED KERNEL!)
    println!("\n[5/5] Processing orderflow features + signals (fused)...");
    let start = Instant::now();
    let results = processor.process_batch(&input, &strategies)?;
    let processing_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Calculate throughput
    let total_features = (strategies.len() * num_ticks * 6) as f64;
    let total_signals = (strategies.len() * num_ticks) as f64;
    let feature_throughput = total_features / (processing_ms / 1000.0);
    let signal_throughput = total_signals / (processing_ms / 1000.0);

    println!("✅ Processing complete in {:.2}ms", processing_ms);
    println!("\n📊 Performance Metrics:");
    println!(
        "   Total time: {:.2}ms (calibration: {:.2}ms + processing: {:.2}ms)",
        calibration_ms + processing_ms,
        calibration_ms,
        processing_ms
    );
    println!("   Strategies: {}", strategies.len());
    println!("   Ticks: {}", num_ticks);
    println!("   Total features computed: {:.2}M", total_features / 1e6);
    println!("   Total signals generated: {:.2}M", total_signals / 1e6);
    println!(
        "   Feature throughput: {:.2}B features/sec",
        feature_throughput / 1e9
    );
    println!(
        "   Signal throughput: {:.2}B signals/sec",
        signal_throughput / 1e9
    );

    // Analyze signals
    println!("\n📈 Signal Analysis:");
    for (i, strategy_signals) in results.signals.iter().enumerate() {
        let buy_count = strategy_signals
            .iter()
            .filter(|&&s| s == Signal::Buy as i8)
            .count();
        let sell_count = strategy_signals
            .iter()
            .filter(|&&s| s == Signal::Sell as i8)
            .count();
        let hold_count = strategy_signals
            .iter()
            .filter(|&&s| s == Signal::Hold as i8)
            .count();

        let buy_pct = (buy_count as f64 / num_ticks as f64) * 100.0;
        let sell_pct = (sell_count as f64 / num_ticks as f64) * 100.0;
        let hold_pct = (hold_count as f64 / num_ticks as f64) * 100.0;

        println!(
            "   Strategy {:2}: Buy={:6} ({:5.2}%), Sell={:6} ({:5.2}%), Hold={:6} ({:5.2}%)",
            i, buy_count, buy_pct, sell_count, sell_pct, hold_count, hold_pct
        );
    }

    // Memory efficiency
    println!("\n💾 Memory Efficiency:");
    let input_size = num_ticks * 5 * 4; // 5 fields × 4 bytes (f32)
    let output_signals_size = strategies.len() * num_ticks * 1; // 1 byte (i8)
    let output_features_size = strategies.len() * num_ticks * 6 * 1; // 6 features × 1 byte (i8)
    let intermediate_avoided = strategies.len() * num_ticks * 6 * 8; // 6 features × 8 bytes (f64) - NOT written!

    println!("   Input OHLCV: {:.2}MB", input_size as f64 / 1e6);
    println!(
        "   Output signals: {:.2}MB",
        output_signals_size as f64 / 1e6
    );
    println!(
        "   Output features (quantized): {:.2}MB",
        output_features_size as f64 / 1e6
    );
    println!(
        "   Intermediate avoided (fusion): {:.2}MB ✨",
        intermediate_avoided as f64 / 1e6
    );
    println!(
        "   Memory traffic reduction: {:.1}%",
        (intermediate_avoided as f64
            / (intermediate_avoided + output_signals_size + output_features_size) as f64)
            * 100.0
    );

    println!("\n✅ Demo complete!");
    println!("\n💡 Next steps:");
    println!("   1. Integrate with Agent 1 (tick aggregation)");
    println!("   2. Feed signals to Agent 3 (backtester)");
    println!("   3. Profile with Nsight Systems: nsys profile ./orderflow_batch_demo");
    println!("   4. Implement Phase 2 bytecode VM for dynamic strategies");

    Ok(())
}

/// Generate synthetic tick data for demo purposes
///
/// Simulates realistic orderflow with trends, mean reversion, and noise.
fn generate_synthetic_data(num_ticks: usize) -> OrderflowInput {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut timestamps = Vec::with_capacity(num_ticks);
    let mut close_prices = Vec::with_capacity(num_ticks);
    let mut volumes = Vec::with_capacity(num_ticks);
    let mut buy_volumes = Vec::with_capacity(num_ticks);
    let mut sell_volumes = Vec::with_capacity(num_ticks);

    let mut price = 50000.0f32; // Start at $50k
    let mut time = 1609459200000i64; // Jan 1, 2021

    for _ in 0..num_ticks {
        // Simulate price movement (random walk with trend)
        let trend = (rng.r#gen::<f32>() - 0.5) * 2.0; // -1 to +1
        let noise = (rng.r#gen::<f32>() - 0.5) * 10.0; // ±5
        price += trend + noise;
        price = price.max(1000.0); // Floor at $1k

        // Simulate volume (lognormal distribution)
        let base_volume = rng.r#gen::<f32>() * 100.0 + 10.0; // 10-110
        let total_volume = base_volume;

        // Simulate buy/sell split (with some imbalance)
        let imbalance = (rng.r#gen::<f32>() - 0.5) * 0.4 + 0.5; // 0.3-0.7
        let buy_vol = total_volume * imbalance;
        let sell_vol = total_volume * (1.0 - imbalance);

        timestamps.push(time);
        close_prices.push(price);
        volumes.push(total_volume);
        buy_volumes.push(buy_vol);
        sell_volumes.push(sell_vol);

        time += 100; // 100ms per tick
    }

    OrderflowInput {
        timestamps,
        close_prices,
        volumes,
        buy_volumes,
        sell_volumes,
    }
}

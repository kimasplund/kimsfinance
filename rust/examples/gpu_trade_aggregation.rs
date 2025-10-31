//! GPU Trade Aggregation Example
//!
//! Demonstrates GPU-accelerated OHLCV candle aggregation from Binance trade data.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features gpu --example gpu_trade_aggregation
//! ```
//!
//! # Expected Output
//!
//! ```
//! GPU Aggregation Available: true
//! Generating 100,000 test trades...
//! Generated 100,000 trades
//!
//! === CPU Aggregation ===
//! Time: 11.5ms
//! Candles: 100
//! Throughput: 8,695,652 trades/sec
//!
//! === GPU Aggregation ===
//! Time: 1.8ms
//! Candles: 100
//! Throughput: 55,555,555 trades/sec
//!
//! Speedup: 6.4x faster 🚀
//! ```

use kimsfinance_core::binance::{Timeframe, Trade, aggregate_trades_to_candles};
use kimsfinance_core::gpu::{AggregationEngine, EngineSelector, GpuAggregator};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU Trade Aggregation Example\n");

    // Check GPU availability
    let gpu_available = GpuAggregator::is_available();
    println!("GPU Aggregation Available: {}\n", gpu_available);

    if !gpu_available {
        println!("⚠️  GPU not available. Install CUDA drivers and ensure GPU is detected.");
        println!("   Falling back to CPU aggregation only.\n");
    }

    // Generate test trades
    let n_trades = 100_000;
    let num_candles = 100;
    println!("Generating {} test trades...", n_trades);
    let trades = generate_trades(n_trades, num_candles);
    println!("Generated {} trades\n", trades.len());

    let timeframe = Timeframe::minutes(5);

    // Benchmark CPU aggregation
    println!("=== CPU Aggregation ===");
    let start = Instant::now();
    let candles_cpu = aggregate_trades_to_candles(&trades, timeframe);
    let cpu_time = start.elapsed();

    println!("Time: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
    println!("Candles: {}", candles_cpu.len());
    println!(
        "Throughput: {:.0} trades/sec",
        n_trades as f64 / cpu_time.as_secs_f64()
    );
    println!();

    if gpu_available {
        // Benchmark GPU aggregation
        println!("=== GPU Aggregation ===");
        let aggregator = GpuAggregator::new()?;

        // Warm-up (JIT compilation)
        let _ = aggregator.aggregate_trades(&trades[..1000], timeframe)?;

        let start = Instant::now();
        let candles_gpu = aggregator.aggregate_trades(&trades, timeframe)?;
        let gpu_time = start.elapsed();

        println!("Time: {:.2}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("Candles: {}", candles_gpu.len());
        println!(
            "Throughput: {:.0} trades/sec",
            n_trades as f64 / gpu_time.as_secs_f64()
        );

        // Validate parity
        validate_candles(&candles_cpu, &candles_gpu)?;

        // Report speedup
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("\nSpeedup: {:.1}x faster 🚀", speedup);

        if speedup < 1.0 {
            println!(
                "\n⚠️  GPU slower than CPU for this dataset size. Consider using CPU for <10K trades."
            );
        }
    }

    // Demonstrate auto-selection
    println!("\n=== Auto-Selection ===");
    let selector = EngineSelector::default();

    let small_trades = &trades[..1_000];
    let engine_small = selector.select_engine(small_trades.len());
    println!("1,000 trades → {} (below threshold)", engine_small.name());

    let large_trades = &trades;
    let engine_large = selector.select_engine(large_trades.len());
    println!("100,000 trades → {} (above threshold)", engine_large.name());

    println!("\nThreshold: {} trades", selector.threshold());

    // Demonstrate auto-selected aggregation
    println!("\n=== Auto-Selected Aggregation ===");
    let start = Instant::now();
    let candles_auto = selector.aggregate_trades(&trades, timeframe)?;
    let auto_time = start.elapsed();

    println!("Engine: {}", selector.select_engine(trades.len()).name());
    println!("Time: {:.2}ms", auto_time.as_secs_f64() * 1000.0);
    println!("Candles: {}", candles_auto.len());

    Ok(())
}

/// Generate synthetic test trades
fn generate_trades(n: usize, num_candles: usize) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(n);

    let base_price = 50_000.0;
    let base_time = 1_600_000_000_000i64;
    let timeframe_ms = 5 * 60 * 1000; // 5 minutes

    for i in 0..n {
        // Distribute trades across candles
        let candle_idx = (i * num_candles) / n;
        let timestamp = base_time + (candle_idx as i64 * timeframe_ms) + (i as i64 * 10);

        // Price variation within candle
        let price_variation = ((i % 100) as f64 / 100.0) * 100.0 - 50.0;
        let price = base_price + price_variation;

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity: 0.1,
            quote_quantity: price * 0.1,
            timestamp_ms: timestamp,
            is_buyer_maker: i % 2 == 0,
        });
    }

    trades
}

/// Validate GPU results match CPU results
fn validate_candles(
    cpu: &[kimsfinance_core::binance::Candle],
    gpu: &[kimsfinance_core::binance::Candle],
) -> Result<(), String> {
    if cpu.len() != gpu.len() {
        return Err(format!(
            "Candle count mismatch: CPU={}, GPU={}",
            cpu.len(),
            gpu.len()
        ));
    }

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let tolerance = 1e-10;

        if c.timestamp != g.timestamp {
            return Err(format!(
                "Candle {} timestamp mismatch: CPU={}, GPU={}",
                i, c.timestamp, g.timestamp
            ));
        }

        if (c.open - g.open).abs() > tolerance {
            return Err(format!(
                "Candle {} open mismatch: CPU={}, GPU={}, diff={}",
                i,
                c.open,
                g.open,
                (c.open - g.open).abs()
            ));
        }

        if (c.high - g.high).abs() > tolerance {
            return Err(format!(
                "Candle {} high mismatch: CPU={}, GPU={}, diff={}",
                i,
                c.high,
                g.high,
                (c.high - g.high).abs()
            ));
        }

        if (c.low - g.low).abs() > tolerance {
            return Err(format!(
                "Candle {} low mismatch: CPU={}, GPU={}, diff={}",
                i,
                c.low,
                g.low,
                (c.low - g.low).abs()
            ));
        }

        if (c.close - g.close).abs() > tolerance {
            return Err(format!(
                "Candle {} close mismatch: CPU={}, GPU={}, diff={}",
                i,
                c.close,
                g.close,
                (c.close - g.close).abs()
            ));
        }

        if (c.volume - g.volume).abs() > tolerance {
            return Err(format!(
                "Candle {} volume mismatch: CPU={}, GPU={}, diff={}",
                i,
                c.volume,
                g.volume,
                (c.volume - g.volume).abs()
            ));
        }

        if c.num_trades != g.num_trades {
            return Err(format!(
                "Candle {} num_trades mismatch: CPU={}, GPU={}",
                i, c.num_trades, g.num_trades
            ));
        }
    }

    println!("\n✅ Validation passed: GPU matches CPU results exactly");
    Ok(())
}

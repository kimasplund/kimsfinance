//! Supertrend GPU Demo
//!
//! Demonstrates GPU-accelerated Supertrend indicator calculation.
//!
//! Run with:
//! ```bash
//! cargo run --example supertrend_gpu_demo --features gpu --release
//! ```

use kimsfinance_core::gpu::{GpuDevice, supertrend_gpu};
use ndarray::Array1;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Supertrend GPU Demo ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = Arc::new(GpuDevice::new()?);
    println!("GPU initialized: {:?}\n", device.context().ordinal());

    // Example 1: Basic usage with small dataset
    println!("Example 1: Basic Supertrend Calculation");
    println!("----------------------------------------");

    let high = vec![
        110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0, 133.0,
        136.0, 140.0, 138.0, 142.0, 145.0, 143.0, 146.0,
    ];
    let low = vec![
        105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0, 128.0,
        131.0, 135.0, 133.0, 137.0, 140.0, 138.0, 141.0,
    ];
    let close = vec![
        108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0, 131.0,
        134.0, 138.0, 136.0, 140.0, 143.0, 141.0, 144.0,
    ];

    let period = 10;
    let multiplier = 3.0;

    let (supertrend, signal) = supertrend_gpu(
        device.clone(),
        &high,
        &low,
        &close,
        period,
        multiplier,
        None,
    )?;

    println!("Period: {}", period);
    println!("Multiplier: {}", multiplier);
    println!("\nResults (last 10 values):");
    println!(
        "{:<6} {:<10} {:<15} {:<8}",
        "Index", "Close", "Supertrend", "Signal"
    );
    println!("{}", "-".repeat(45));

    for i in (10..20).rev() {
        let signal_str = match signal[i] {
            1 => "Uptrend",
            -1 => "Downtrend",
            _ => "Warmup",
        };
        println!(
            "{:<6} {:<10.2} {:<15.2} {:<8}",
            i, close[i], supertrend[i], signal_str
        );
    }

    // Example 2: Performance test with large dataset
    println!("\n\nExample 2: Performance Test");
    println!("----------------------------");

    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes {
        // Generate synthetic OHLC data
        let high: Vec<f64> = (0..size)
            .map(|i| {
                let x = i as f64 * 0.01;
                110.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();
        let low: Vec<f64> = (0..size)
            .map(|i| {
                let x = i as f64 * 0.01;
                100.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();
        let close: Vec<f64> = (0..size)
            .map(|i| {
                let x = i as f64 * 0.01;
                105.0 + 10.0 * (x * 0.1).sin()
            })
            .collect();

        let start = std::time::Instant::now();
        let (supertrend, signal) =
            supertrend_gpu(device.clone(), &high, &low, &close, 10, 3.0, None)?;
        let elapsed = start.elapsed();

        // Count trend changes
        let mut trend_changes = 0;
        for i in 1..signal.len() {
            if signal[i] != signal[i - 1] && signal[i] != 0 && signal[i - 1] != 0 {
                trend_changes += 1;
            }
        }

        let uptrend_count = signal.iter().filter(|&&s| s == 1).count();
        let downtrend_count = signal.iter().filter(|&&s| s == -1).count();

        println!("\nDataset size: {} candles", size);
        println!(
            "  Time: {:.2}ms ({:.2}μs per candle)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1_000_000.0 / size as f64
        );
        println!("  Trend changes: {}", trend_changes);
        println!(
            "  Uptrend: {} candles ({:.1}%)",
            uptrend_count,
            uptrend_count as f64 / size as f64 * 100.0
        );
        println!(
            "  Downtrend: {} candles ({:.1}%)",
            downtrend_count,
            downtrend_count as f64 / size as f64 * 100.0
        );
    }

    // Example 3: Different parameters comparison
    println!("\n\nExample 3: Parameter Comparison");
    println!("--------------------------------");

    let size = 10_000;
    let high: Vec<f64> = (0..size)
        .map(|i| {
            let x = i as f64 * 0.01;
            110.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let low: Vec<f64> = (0..size)
        .map(|i| {
            let x = i as f64 * 0.01;
            100.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let close: Vec<f64> = (0..size)
        .map(|i| {
            let x = i as f64 * 0.01;
            105.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();

    let configurations = vec![(10, 2.0), (10, 3.0), (10, 4.0), (14, 3.0), (20, 3.0)];

    println!("\nComparing different parameter configurations:");
    println!(
        "{:<8} {:<12} {:<15} {:<12}",
        "Period", "Multiplier", "Uptrend %", "Reversals"
    );
    println!("{}", "-".repeat(50));

    for (period, multiplier) in configurations {
        let (_, signal) = supertrend_gpu(
            device.clone(),
            &high,
            &low,
            &close,
            period,
            multiplier,
            None,
        )?;

        let uptrend_count = signal.iter().filter(|&&s| s == 1).count();
        let uptrend_pct = uptrend_count as f64 / size as f64 * 100.0;

        let mut reversals = 0;
        for i in 1..signal.len() {
            if signal[i] != signal[i - 1] && signal[i] != 0 && signal[i - 1] != 0 {
                reversals += 1;
            }
        }

        println!(
            "{:<8} {:<12.1} {:<15.1} {:<12}",
            period, multiplier, uptrend_pct, reversals
        );
    }

    // Example 4: Stream concurrency demonstration
    println!("\n\nExample 4: Stream Concurrency (Advanced)");
    println!("-----------------------------------------");
    println!("Note: Stream support allows concurrent execution with other GPU operations.");
    println!("Using default stream for this demo.\n");

    let size = 50_000;
    let high: Vec<f64> = (0..size)
        .map(|i| {
            let x = i as f64 * 0.01;
            110.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let low: Vec<f64> = (0..size)
        .map(|i| {
            let x = i as f64 * 0.01;
            100.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let close: Vec<f64> = (0..size)
        .map(|i| {
            let x = i as f64 * 0.01;
            105.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();

    let start = std::time::Instant::now();
    let (supertrend, signal) = supertrend_gpu(
        device.clone(),
        &high,
        &low,
        &close,
        10,
        3.0,
        None, // Uses default stream
    )?;
    let elapsed = start.elapsed();

    println!(
        "Processed {} candles in {:.2}ms",
        size,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "Average latency: {:.2}μs per candle",
        elapsed.as_secs_f64() * 1_000_000.0 / size as f64
    );

    // Show final trend
    let final_signal = signal[size - 1];
    let final_supertrend = supertrend[size - 1];
    let final_close = close[size - 1];

    println!("\nFinal state:");
    println!("  Close: {:.2}", final_close);
    println!("  Supertrend: {:.2}", final_supertrend);
    println!(
        "  Trend: {}",
        if final_signal == 1 {
            "Uptrend"
        } else {
            "Downtrend"
        }
    );

    println!("\n=== Demo Complete ===");

    Ok(())
}

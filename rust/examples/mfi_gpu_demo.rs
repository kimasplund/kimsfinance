//! MFI (Money Flow Index) GPU Acceleration Demo
//!
//! Demonstrates GPU-accelerated MFI calculation and compares with CPU implementation.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example mfi_gpu_demo --features gpu --release
//! ```
//!
//! # Expected Output
//!
//! - GPU initialization status
//! - MFI calculation results (first 20 and last 20 values)
//! - Performance comparison: CPU vs GPU
//! - Speedup factor (should be 10-20x for large datasets)

use kimsfinance_core::gpu::{GpuDevice, mfi_gpu};
use kimsfinance_core::indicators::volume::MFI;
use ndarray::Array1;
use std::time::Instant;

fn generate_sample_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    println!("Generating {} candles of sample OHLCV data...", n);

    // Generate realistic oscillating data with trend
    let high: Array1<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            let trend = i as f64 * 0.005; // Upward trend
            105.0 + trend + 5.0 * (x * 0.1).sin() + 0.5 * (x * 0.3).cos()
        })
        .collect();

    let low: Array1<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            let trend = i as f64 * 0.005;
            95.0 + trend + 5.0 * (x * 0.1).sin() + 0.5 * (x * 0.3).cos()
        })
        .collect();

    let close: Array1<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            let trend = i as f64 * 0.005;
            100.0 + trend + 5.0 * (x * 0.1).sin() + 0.5 * (x * 0.3).cos()
        })
        .collect();

    let volume: Array1<f64> = (0..n)
        .map(|i| {
            let base = 1000.0 + (i % 500) as f64;
            let noise = (i % 100) as f64 * 10.0;
            base + noise
        })
        .collect();

    (high, low, close, volume)
}

fn main() {
    println!("=== MFI (Money Flow Index) GPU Acceleration Demo ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = match GpuDevice::new() {
        Ok(dev) => {
            println!("✓ GPU device initialized successfully");
            dev
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize GPU: {:?}", e);
            eprintln!("  Make sure you have CUDA drivers installed and a compatible GPU");
            return;
        }
    };

    // Generate sample data
    let n = 100_000;
    let period = 14;
    let (high, low, close, volume) = generate_sample_data(n);

    println!("\nDataset size: {} candles", n);
    println!("MFI period: {}", period);
    println!("Price range: ${:.2} - ${:.2}", low[0], high[n - 1]);

    // Run CPU implementation
    println!("\n--- CPU Implementation ---");
    let mfi_cpu_impl = MFI::new(period).expect("Failed to create MFI");

    let start = Instant::now();
    let mfi_cpu = mfi_cpu_impl
        .calculate_hlcv(high.view(), low.view(), close.view(), volume.view())
        .expect("CPU MFI calculation failed");
    let cpu_time = start.elapsed();

    println!("CPU time: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
    println!(
        "CPU throughput: {:.0} candles/sec",
        n as f64 / cpu_time.as_secs_f64()
    );

    // Run GPU implementation
    println!("\n--- GPU Implementation ---");
    let start = Instant::now();
    let mfi_gpu_result = mfi_gpu(&device, &high, &low, &close, &volume, period, None)
        .expect("GPU MFI calculation failed");
    let gpu_time = start.elapsed();

    println!("GPU time: {:.2}ms", gpu_time.as_secs_f64() * 1000.0);
    println!(
        "GPU throughput: {:.0} candles/sec",
        n as f64 / gpu_time.as_secs_f64()
    );

    // Calculate speedup
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("\n--- Performance Comparison ---");
    println!("Speedup: {:.2}x", speedup);

    if speedup >= 10.0 {
        println!("✓ Excellent speedup (>10x)");
    } else if speedup >= 5.0 {
        println!("✓ Good speedup (5-10x)");
    } else if speedup >= 2.0 {
        println!("⚠ Moderate speedup (2-5x)");
    } else {
        println!("⚠ Limited speedup (<2x) - GPU overhead may dominate for this dataset size");
    }

    // Verify correctness
    println!("\n--- Correctness Verification ---");
    let mut max_diff: f64 = 0.0;
    let mut avg_diff: f64 = 0.0;
    let mut valid_count = 0;

    for i in period..n {
        if !mfi_cpu[i].is_nan() && !mfi_gpu_result[i].is_nan() {
            let diff = (mfi_cpu[i] - mfi_gpu_result[i]).abs();
            max_diff = f64::max(max_diff, diff);
            avg_diff += diff;
            valid_count += 1;
        }
    }

    avg_diff /= valid_count as f64;

    println!("Maximum difference: {:.6}", max_diff);
    println!("Average difference: {:.6}", avg_diff);

    if max_diff < 1e-6 {
        println!("✓ Results match perfectly (diff < 1e-6)");
    } else if max_diff < 1e-3 {
        println!("✓ Results match well (diff < 1e-3)");
    } else if max_diff < 0.1 {
        println!("⚠ Results differ slightly (diff < 0.1)");
    } else {
        println!("✗ Significant difference detected (diff >= 0.1)");
    }

    // Display sample results
    println!("\n--- Sample MFI Values ---");
    println!("First 20 valid values (after warmup):");
    for i in period..(period + 20).min(n) {
        println!(
            "  [{}] CPU: {:.2}, GPU: {:.2}, Diff: {:.6}",
            i,
            mfi_cpu[i],
            mfi_gpu_result[i],
            (mfi_cpu[i] - mfi_gpu_result[i]).abs()
        );
    }

    println!("\nLast 10 values:");
    for i in (n - 10)..n {
        println!(
            "  [{}] CPU: {:.2}, GPU: {:.2}, Diff: {:.6}",
            i,
            mfi_cpu[i],
            mfi_gpu_result[i],
            (mfi_cpu[i] - mfi_gpu_result[i]).abs()
        );
    }

    // MFI interpretation
    println!("\n--- MFI Interpretation ---");
    let last_mfi = mfi_gpu_result[n - 1];
    println!("Latest MFI: {:.2}", last_mfi);

    if last_mfi > 80.0 {
        println!("Signal: Overbought (MFI > 80) - Potential reversal to downside");
    } else if last_mfi < 20.0 {
        println!("Signal: Oversold (MFI < 20) - Potential reversal to upside");
    } else if last_mfi > 50.0 {
        println!("Signal: Bullish bias (MFI > 50) - Buying pressure dominates");
    } else {
        println!("Signal: Bearish bias (MFI < 50) - Selling pressure dominates");
    }

    println!("\n=== Demo Complete ===");
}

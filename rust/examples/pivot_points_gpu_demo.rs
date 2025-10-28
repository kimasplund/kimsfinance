//! Pivot Points GPU Demo
//!
//! Demonstrates GPU-accelerated Pivot Points calculation with performance comparison.
//!
//! # Features
//!
//! - GPU vs CPU performance comparison
//! - Multiple dataset sizes (1K, 10K, 100K candles)
//! - Validation of GPU results
//! - Visual output of pivot levels
//!
//! # Run Demo
//!
//! ```bash
//! cargo run --features gpu --example pivot_points_gpu_demo --release
//! ```

use kimsfinance_core::gpu::{GpuDevice, pivot_points_gpu};
use kimsfinance_core::indicators::trend::PivotPoints;
use ndarray::Array1;
use std::sync::Arc;
use std::time::Instant;

/// Generate realistic test OHLC data
fn generate_ohlc_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);

    let mut price = 100.0;

    for i in 0..n {
        // Simulate realistic price movement
        let trend = (i as f64 * 0.01).sin() * 5.0;
        let noise = ((i * 7919) % 100) as f64 * 0.1 - 5.0;

        price += trend + noise;
        price = price.max(50.0).min(150.0); // Keep in reasonable range

        high.push(price + 2.0);
        low.push(price - 2.0);
        close.push(price);
    }

    (high, low, close)
}

/// Calculate pivot points on CPU for comparison
fn pivot_points_cpu(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let n = high.len();
    let mut pp = Array1::from_elem(n, f64::NAN);
    let mut r1 = Array1::from_elem(n, f64::NAN);
    let mut r2 = Array1::from_elem(n, f64::NAN);
    let mut r3 = Array1::from_elem(n, f64::NAN);
    let mut s1 = Array1::from_elem(n, f64::NAN);
    let mut s2 = Array1::from_elem(n, f64::NAN);
    let mut s3 = Array1::from_elem(n, f64::NAN);

    let pivot_calc = PivotPoints::new();

    for i in 1..n {
        let levels = pivot_calc.calculate_single(high[i - 1], low[i - 1], close[i - 1]);
        pp[i] = levels[0];
        r1[i] = levels[1];
        r2[i] = levels[2];
        r3[i] = levels[3];
        s1[i] = levels[4];
        s2[i] = levels[5];
        s3[i] = levels[6];
    }

    (pp, r1, r2, r3, s1, s2, s3)
}

/// Validate GPU results against CPU
fn validate_results(
    gpu_pp: &Array1<f64>,
    gpu_r1: &Array1<f64>,
    gpu_s1: &Array1<f64>,
    cpu_pp: &Array1<f64>,
    cpu_r1: &Array1<f64>,
    cpu_s1: &Array1<f64>,
) -> bool {
    let n = gpu_pp.len();

    for i in 1..n {
        let pp_diff = (gpu_pp[i] - cpu_pp[i]).abs();
        let r1_diff = (gpu_r1[i] - cpu_r1[i]).abs();
        let s1_diff = (gpu_s1[i] - cpu_s1[i]).abs();

        if pp_diff > 1e-8 || r1_diff > 1e-8 || s1_diff > 1e-8 {
            eprintln!(
                "Validation failed at index {}: PP diff={}, R1 diff={}, S1 diff={}",
                i, pp_diff, r1_diff, s1_diff
            );
            return false;
        }
    }

    true
}

/// Display pivot levels for a specific candle
fn display_pivot_levels(
    index: usize,
    high: f64,
    low: f64,
    close: f64,
    pp: f64,
    r1: f64,
    r2: f64,
    r3: f64,
    s1: f64,
    s2: f64,
    s3: f64,
) {
    println!("\nCandle #{}", index);
    println!("  Previous: H={:.2}, L={:.2}, C={:.2}", high, low, close);
    println!("\n  Resistance Levels:");
    println!("    R3: {:.2}", r3);
    println!("    R2: {:.2}", r2);
    println!("    R1: {:.2}", r1);
    println!("\n  Pivot Point: {:.2}", pp);
    println!("\n  Support Levels:");
    println!("    S1: {:.2}", s1);
    println!("    S2: {:.2}", s2);
    println!("    S3: {:.2}", s3);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pivot Points GPU Demo ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = Arc::new(GpuDevice::new()?);
    println!("GPU device initialized successfully\n");

    // Test sizes
    let test_sizes = vec![("1K", 1_000), ("10K", 10_000), ("100K", 100_000)];

    println!("=== Performance Comparison ===\n");

    for (label, size) in test_sizes {
        println!("Dataset: {} candles", label);

        let (high, low, close) = generate_ohlc_data(size);

        // CPU benchmark
        let start = Instant::now();
        let (cpu_pp, cpu_r1, cpu_r2, cpu_r3, cpu_s1, cpu_s2, cpu_s3) =
            pivot_points_cpu(&high, &low, &close);
        let cpu_time = start.elapsed();

        // GPU benchmark (including compilation)
        let start = Instant::now();
        let gpu_result = pivot_points_gpu(device.clone(), &high, &low, &close, None)?;
        let gpu_time = start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("  CPU time: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
        println!("  GPU time: {:.2}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  Speedup: {:.2}x", speedup);

        // Validate GPU results
        let valid = validate_results(
            &gpu_result.pp,
            &gpu_result.r1,
            &gpu_result.s1,
            &cpu_pp,
            &cpu_r1,
            &cpu_s1,
        );

        if valid {
            println!("  Validation: PASS ✓");
        } else {
            println!("  Validation: FAIL ✗");
            return Err("GPU validation failed".into());
        }

        println!();
    }

    // Detailed example with small dataset
    println!("\n=== Detailed Example ===\n");

    let (high, low, close) = generate_ohlc_data(10);

    let result = pivot_points_gpu(device.clone(), &high, &low, &close, None)?;

    println!("Calculating pivot points for 10 candles:");

    // Display pivot levels for candles 5-7
    for i in 5..8 {
        display_pivot_levels(
            i,
            high[i - 1],
            low[i - 1],
            close[i - 1],
            result.pp[i],
            result.r1[i],
            result.r2[i],
            result.r3[i],
            result.s1[i],
            result.s2[i],
            result.s3[i],
        );
    }

    println!("\n=== Verification Tests ===\n");

    // Verify level ordering
    println!("Verifying level ordering (S3 < S2 < S1 < PP < R1 < R2 < R3)...");
    let mut all_valid = true;

    for i in 1..result.pp.len() {
        if result.s3[i] >= result.s2[i]
            || result.s2[i] >= result.s1[i]
            || result.s1[i] >= result.pp[i]
            || result.pp[i] >= result.r1[i]
            || result.r1[i] >= result.r2[i]
            || result.r2[i] >= result.r3[i]
        {
            eprintln!("  Invalid ordering at candle {}", i);
            all_valid = false;
        }
    }

    if all_valid {
        println!("  All levels correctly ordered ✓");
    } else {
        println!("  Some levels incorrectly ordered ✗");
    }

    // Verify symmetry
    println!("\nVerifying R1/S1 symmetry around PP...");
    let mut max_asymmetry = 0.0;

    for i in 1..result.pp.len() {
        let r1_dist = result.r1[i] - result.pp[i];
        let s1_dist = result.pp[i] - result.s1[i];
        let asymmetry = (r1_dist - s1_dist).abs();

        if asymmetry > max_asymmetry {
            max_asymmetry = asymmetry;
        }
    }

    println!("  Max asymmetry: {:.10}", max_asymmetry);
    if max_asymmetry < 1e-8 {
        println!("  Symmetry verification: PASS ✓");
    } else {
        println!("  Symmetry verification: FAIL ✗");
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}

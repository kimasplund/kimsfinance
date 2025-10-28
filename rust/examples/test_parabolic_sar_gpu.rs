//! Test: Parabolic SAR GPU Implementation
//!
//! Validates GPU implementation against CPU reference.
//!
//! # Run
//!
//! ```bash
//! cargo run --example test_parabolic_sar_gpu --features gpu
//! ```

use kimsfinance_core::gpu::{GpuDevice, parabolic_sar_gpu};
use kimsfinance_core::indicators::trend::ParabolicSAR;
use ndarray::Array1;

fn main() {
    println!("=== Parabolic SAR GPU Test ===\n");

    // Initialize GPU
    let device = match GpuDevice::new() {
        Ok(dev) => {
            println!("✓ GPU initialized successfully\n");
            dev
        }
        Err(e) => {
            eprintln!("✗ GPU initialization failed: {:?}", e);
            std::process::exit(1);
        }
    };

    // Test 1: Basic uptrend
    println!("Test 1: Basic uptrend");
    let high = Array1::from_vec(vec![
        110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0,
    ]);
    let low = Array1::from_vec(vec![
        105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0,
    ]);

    // CPU reference
    let psar_cpu = ParabolicSAR::new(0.02, 0.02, 0.2).unwrap();
    let sar_cpu = psar_cpu.calculate_hl(high.view(), low.view()).unwrap();

    // GPU implementation
    let (sar_gpu, signal_gpu) = parabolic_sar_gpu(&device, &high, &low, 0.02, 0.02, 0.2, None)
        .expect("GPU calculation failed");

    // Compare results
    let mut max_diff = 0.0f64;
    for i in 0..sar_cpu.len() {
        let diff = (sar_cpu[i] - sar_gpu[i]).abs();
        if !diff.is_nan() {
            max_diff = max_diff.max(diff);
        }
    }

    println!("  CPU SAR[0]: {:.4}", sar_cpu[0]);
    println!("  GPU SAR[0]: {:.4}", sar_gpu[0]);
    println!("  GPU Signal[0]: {}", signal_gpu[0]);
    println!("  Max difference: {:.10}", max_diff);

    if max_diff < 1e-8 {
        println!("✓ Test 1 PASSED: GPU matches CPU within tolerance\n");
    } else {
        println!("✗ Test 1 FAILED: Difference too large ({})\n", max_diff);
    }

    // Test 2: Trend reversal
    println!("Test 2: Trend reversal (uptrend -> downtrend)");
    let high2 = Array1::from_vec(vec![
        110.0, 115.0, 120.0, 125.0, 130.0, // Uptrend
        128.0, 123.0, 118.0, 113.0, 108.0, // Downtrend
    ]);
    let low2 = Array1::from_vec(vec![
        105.0, 110.0, 115.0, 120.0, 125.0, // Uptrend
        123.0, 118.0, 113.0, 108.0, 103.0, // Downtrend
    ]);

    let sar_cpu2 = psar_cpu.calculate_hl(high2.view(), low2.view()).unwrap();
    let (sar_gpu2, signal_gpu2) = parabolic_sar_gpu(&device, &high2, &low2, 0.02, 0.02, 0.2, None)
        .expect("GPU calculation failed");

    let mut max_diff2 = 0.0f64;
    for i in 0..sar_cpu2.len() {
        let diff = (sar_cpu2[i] - sar_gpu2[i]).abs();
        if !diff.is_nan() {
            max_diff2 = max_diff2.max(diff);
        }
    }

    // Check for reversal detection
    let has_uptrend = signal_gpu2.iter().any(|&s| s == 1);
    let has_downtrend = signal_gpu2.iter().any(|&s| s == -1);

    println!("  Has uptrend signal: {}", has_uptrend);
    println!("  Has downtrend signal: {}", has_downtrend);
    println!("  Max difference: {:.10}", max_diff2);

    if max_diff2 < 1e-8 && has_uptrend && has_downtrend {
        println!("✓ Test 2 PASSED: GPU correctly detects reversal\n");
    } else {
        println!("✗ Test 2 FAILED");
        if max_diff2 >= 1e-8 {
            println!("  Reason: Difference too large ({})", max_diff2);
        }
        if !has_uptrend || !has_downtrend {
            println!(
                "  Reason: Missing trend signals (up={}, down={})",
                has_uptrend, has_downtrend
            );
        }
        println!();
    }

    // Test 3: Large dataset
    println!("Test 3: Large dataset (10,000 candles)");
    let n = 10_000;
    let high3: Vec<f64> = (0..n)
        .map(|i| 100.0 + (i as f64 * 0.01).sin() * 10.0 + 5.0)
        .collect();
    let low3: Vec<f64> = (0..n)
        .map(|i| 100.0 + (i as f64 * 0.01).sin() * 10.0 - 5.0)
        .collect();
    let high3 = Array1::from_vec(high3);
    let low3 = Array1::from_vec(low3);

    let start = std::time::Instant::now();
    let sar_cpu3 = psar_cpu.calculate_hl(high3.view(), low3.view()).unwrap();
    let cpu_time = start.elapsed();

    let start = std::time::Instant::now();
    let (sar_gpu3, _signal_gpu3) = parabolic_sar_gpu(&device, &high3, &low3, 0.02, 0.02, 0.2, None)
        .expect("GPU calculation failed");
    let gpu_time = start.elapsed();

    let mut max_diff3 = 0.0f64;
    for i in 0..sar_cpu3.len() {
        let diff = (sar_cpu3[i] - sar_gpu3[i]).abs();
        if !diff.is_nan() {
            max_diff3 = max_diff3.max(diff);
        }
    }

    println!("  CPU time: {:.3} ms", cpu_time.as_secs_f64() * 1000.0);
    println!("  GPU time: {:.3} ms", gpu_time.as_secs_f64() * 1000.0);
    println!(
        "  Speedup: {:.2}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );
    println!("  Max difference: {:.10}", max_diff3);

    if max_diff3 < 1e-8 {
        println!("✓ Test 3 PASSED: GPU handles large datasets correctly\n");
    } else {
        println!("✗ Test 3 FAILED: Difference too large ({})\n", max_diff3);
    }

    // Test 4: Edge cases
    println!("Test 4: Edge cases");

    // Constant prices
    let high4 = Array1::from_vec(vec![110.0; 20]);
    let low4 = Array1::from_vec(vec![100.0; 20]);

    let (sar_gpu4, signal_gpu4) = parabolic_sar_gpu(&device, &high4, &low4, 0.02, 0.02, 0.2, None)
        .expect("GPU calculation failed");

    let all_valid = sar_gpu4
        .iter()
        .all(|&x| !x.is_nan() && x >= 100.0 && x <= 110.0);

    println!("  Constant prices: all SAR values valid = {}", all_valid);

    if all_valid {
        println!("✓ Test 4 PASSED: Edge cases handled correctly\n");
    } else {
        println!("✗ Test 4 FAILED: Invalid SAR values for constant prices\n");
    }

    println!("=== All Tests Complete ===");
}

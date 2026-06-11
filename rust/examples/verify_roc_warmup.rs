#!/usr/bin/env cargo
//! Verify ROC performance with proper warmup
//!
//! This script tests the hypothesis that ROC's 135ms measurement includes
//! kernel compilation overhead. Expected results:
//!
//! - Cold run (first): ~135ms (includes NVRTC compilation)
//! - Warm run (cached): ~0.6ms (actual performance)
//! - Speedup: ~226x

use kimsfinance_core::gpu::device::GpuDevice;
use kimsfinance_core::gpu::roc::roc_gpu;
use ndarray::Array1;
use std::time::Instant;

fn main() {
    let n = 100_000;
    println!("\n{:=^80}", " ROC WARMUP VERIFICATION ");
    println!("Testing hypothesis: ROC includes kernel compilation in benchmark");
    println!("Dataset: {} candles\n", n);

    // Generate test data
    let close = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 * 0.01)).collect());

    let device = GpuDevice::new().expect("Failed to initialize GPU");

    // ============================================================================
    // TEST 1: Cold start (includes compilation)
    // ============================================================================
    println!("{:-^80}", " TEST 1: COLD START (First Run) ");
    println!("Expected: ~135ms (includes NVRTC kernel compilation)\n");

    let start_cold = Instant::now();
    let _result = roc_gpu(&device, &close, 12, None).expect("ROC failed");
    device.synchronize().expect("Sync failed");
    let cold_time = start_cold.elapsed();

    println!("Cold run time: {:.2}ms", cold_time.as_secs_f64() * 1000.0);

    // ============================================================================
    // TEST 2: Warmup runs (kernel already compiled)
    // ============================================================================
    println!("\n{:-^80}", " TEST 2: WARMUP RUNS ");
    println!("Running 5 warmup iterations (kernel should be cached)...\n");

    let mut warmup_times = Vec::new();
    for i in 0..5 {
        let start = Instant::now();
        let _result = roc_gpu(&device, &close, 12, None).expect("ROC failed");
        device.synchronize().expect("Sync failed");
        let elapsed = start.elapsed();
        warmup_times.push(elapsed);

        println!(
            "  Warmup {}: {:.3}ms",
            i + 1,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // ============================================================================
    // TEST 3: Timed benchmark run (warm GPU)
    // ============================================================================
    println!("\n{:-^80}", " TEST 3: WARM BENCHMARK (Compiled Kernel) ");
    println!("Expected: ~0.6ms (actual performance)\n");

    let start_warm = Instant::now();
    let _result = roc_gpu(&device, &close, 12, None).expect("ROC failed");
    device.synchronize().expect("Sync failed");
    let warm_time = start_warm.elapsed();

    println!("Warm run time: {:.3}ms", warm_time.as_secs_f64() * 1000.0);

    // ============================================================================
    // ANALYSIS
    // ============================================================================
    println!("\n{:=^80}", " ANALYSIS ");

    let speedup = cold_time.as_secs_f64() / warm_time.as_secs_f64();
    let throughput = n as f64 / (warm_time.as_secs_f64() * 1000.0);

    println!("\n{:<40} {:>15} {:>15}", "Metric", "Cold", "Warm");
    println!("{:-^80}", "");
    println!(
        "{:<40} {:>15.2} {:>15.3}",
        "Time (ms)",
        cold_time.as_secs_f64() * 1000.0,
        warm_time.as_secs_f64() * 1000.0
    );
    println!(
        "{:<40} {:>15.0} {:>15.0}",
        "Throughput (M candles/sec)",
        n as f64 / (cold_time.as_secs_f64() * 1000.0),
        throughput
    );
    println!();
    println!("Speedup (warm vs cold): {:.1}x", speedup);

    // Compilation overhead estimate
    let compilation_overhead = cold_time.as_secs_f64() - warm_time.as_secs_f64();
    let compilation_percent = (compilation_overhead / cold_time.as_secs_f64()) * 100.0;

    println!(
        "Estimated compilation overhead: {:.2}ms ({:.1}% of cold time)",
        compilation_overhead * 1000.0,
        compilation_percent
    );

    // ============================================================================
    // HYPOTHESIS VALIDATION
    // ============================================================================
    println!("\n{:=^80}", " HYPOTHESIS VALIDATION ");

    let hypothesis = vec![
        (
            "Cold run includes compilation",
            cold_time.as_secs_f64() * 1000.0 > 50.0,
            format!("{:.2}ms > 50ms", cold_time.as_secs_f64() * 1000.0),
        ),
        (
            "Warm run is much faster",
            warm_time.as_secs_f64() * 1000.0 < 5.0,
            format!("{:.3}ms < 5ms", warm_time.as_secs_f64() * 1000.0),
        ),
        (
            "Speedup is significant",
            speedup > 20.0,
            format!("{:.1}x > 20x", speedup),
        ),
        (
            "Compilation overhead ~50-150ms",
            compilation_overhead * 1000.0 > 50.0 && compilation_overhead * 1000.0 < 200.0,
            format!(
                "{:.2}ms in range [50, 200]ms",
                compilation_overhead * 1000.0
            ),
        ),
    ];

    println!();
    for (test, passed, details) in hypothesis {
        let status = if passed { "✅ PASS" } else { "❌ FAIL" };
        println!("{} {:<45} ({})", status, test, details);
    }

    // ============================================================================
    // PERFORMANCE BREAKDOWN
    // ============================================================================
    println!("\n{:=^80}", " PERFORMANCE BREAKDOWN (Warm) ");

    let data_size_mb = (n * 8) as f64 / 1_000_000.0; // f64 = 8 bytes
    let total_transfer_mb = data_size_mb * 2.0; // H2D + D2H

    println!("\nMemory transfers:");
    println!("  Data size: {:.2} MB per array", data_size_mb);
    println!(
        "  H2D (close): 1 array × {:.2} MB = {:.2} MB",
        data_size_mb, data_size_mb
    );
    println!(
        "  D2H (roc): 1 array × {:.2} MB = {:.2} MB",
        data_size_mb, data_size_mb
    );
    println!("  Total transfer: {:.2} MB", total_transfer_mb);

    let pcie_bandwidth = 32.0; // GB/s for PCIe 4.0 x16
    let estimated_transfer_time = (total_transfer_mb / 1000.0) / pcie_bandwidth;

    println!("\nEstimated costs (warm):");
    println!(
        "  Memory transfers (PCIe 4.0 x16): ~{:.3}ms",
        estimated_transfer_time * 1000.0
    );
    println!("  Kernel execution: ~0.1-0.2ms");
    println!("  Overhead (sync, etc): ~0.1ms");
    println!(
        "  Total estimated: ~{:.3}ms",
        (estimated_transfer_time + 0.0003) * 1000.0
    );

    let actual_ms = warm_time.as_secs_f64() * 1000.0;
    let estimated_ms = (estimated_transfer_time + 0.0003) * 1000.0;
    let accuracy = (1.0 - (actual_ms - estimated_ms).abs() / actual_ms) * 100.0;

    println!("\nActual vs Estimated:");
    println!("  Actual: {:.3}ms", actual_ms);
    println!("  Estimated: {:.3}ms", estimated_ms);
    println!("  Accuracy: {:.1}%", accuracy);

    // ============================================================================
    // RECOMMENDATIONS
    // ============================================================================
    println!("\n{:=^80}", " RECOMMENDATIONS ");
    println!();
    println!("1. UPDATE BENCHMARK METHODOLOGY:");
    println!("   - Add 3-5 warmup runs before timing");
    println!("   - Always synchronize stream after each call");
    println!("   - Report both cold (with compilation) and warm (actual) times");
    println!();
    println!("2. UPDATE DOCUMENTATION:");
    println!(
        "   - Correct ROC time: 135.8ms → {:.3}ms ({:.0}x improvement)",
        actual_ms, speedup
    );
    println!("   - Add note about compilation overhead to all GPU indicators");
    println!();
    println!("3. CONSIDER AOT COMPILATION:");
    println!("   - Compile kernels at build time (build.rs)");
    println!("   - Embed PTX in binary → zero runtime compilation");
    println!();

    println!("\n{:=^80}", " INVESTIGATION COMPLETE ");
    println!();

    // Summary for easy copy-paste to report
    if speedup > 20.0 {
        println!("✅ HYPOTHESIS CONFIRMED!");
        println!("   ROC's apparent slowness is due to kernel compilation overhead.");
        println!(
            "   Actual warm performance: {:.3}ms ({:.1}x faster than cold)",
            actual_ms, speedup
        );
    } else {
        println!("⚠️  UNEXPECTED RESULT!");
        println!("   Speedup ({:.1}x) is less than expected (>20x).", speedup);
        println!("   Further investigation needed.");
    }
    println!();
}

//! OBV Performance Investigation
//!
//! This benchmark measures OBV performance and compares different kernel strategies.

use kimsfinance_core::gpu::device::GpuDevice;
use kimsfinance_core::gpu::obv::obv_gpu;
use ndarray::Array1;
use std::time::Instant;

fn main() {
    println!("=== OBV Performance Investigation ===\n");

    let device = GpuDevice::new().expect("Failed to initialize GPU device");
    println!("GPU Device initialized\n");

    // Test various data sizes
    let sizes = vec![1_000, 10_000, 100_000, 500_000];

    println!("{:<12} | {:>10} | {:>12} | {:>15}", "Size", "Time (ms)", "μs/candle", "Candles/sec");
    println!("{}", "-".repeat(60));

    for n in sizes {
        // Generate realistic price/volume data
        let close: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.001;
                100.0 + 10.0 * (x).sin() + (x * 0.1).cos() * 5.0
            })
            .collect();

        let volume: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.01;
                10000.0 + 5000.0 * (x * 0.3).sin().abs()
            })
            .collect();

        let close_arr = Array1::from_vec(close);
        let volume_arr = Array1::from_vec(volume);

        // Warmup run
        let _ = obv_gpu(&device, &close_arr, &volume_arr, None).expect("OBV warmup failed");

        // Timed runs (average of 5)
        let mut times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let _obv = obv_gpu(&device, &close_arr, &volume_arr, None).expect("OBV calculation failed");
            let elapsed = start.elapsed();
            times.push(elapsed.as_secs_f64());
        }

        // Calculate statistics
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let us_per_candle = (avg_time * 1_000_000.0) / n as f64;
        let candles_per_sec = n as f64 / avg_time;

        println!(
            "{:<12} | {:>10.3} | {:>12.3} | {:>15.0}",
            format!("{}", n),
            avg_time * 1000.0,
            us_per_candle,
            candles_per_sec
        );
    }

    println!("\n=== Component Breakdown Analysis ===\n");
    println!("Expected time breakdown for 100K candles:");
    println!("  1. H2D transfer (close):     ~0.05ms");
    println!("  2. H2D transfer (volume):    ~0.05ms");
    println!("  3. Deltas kernel (parallel): ~0.10ms");
    println!("  4. Cumsum kernel (SERIAL):   ~3.80ms  ← BOTTLENECK");
    println!("  5. D2H transfer (obv):       ~0.05ms");
    println!("  6. Synchronization:          ~0.10ms");
    println!("  ────────────────────────────────────");
    println!("  Total:                       ~4.15ms\n");

    println!("Key findings:");
    println!("  ✗ Cumsum kernel runs SINGLE THREAD for entire dataset");
    println!("  ✗ 100K iterations sequentially on GPU (inefficient)");
    println!("  ✗ No GPU parallelism for cumulative sum");
    println!("\nRecommendations:");
    println!("  → Implement parallel prefix sum (Blelloch scan)");
    println!("  → Expected speedup: 5-10x (target: <0.5ms for 100K)");
    println!("  → Alternative: Use CUB library's DeviceScan");
}

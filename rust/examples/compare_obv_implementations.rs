//! OBV Implementation Comparison
//!
//! Compares naive single-threaded cumsum vs parallel prefix sum

use kimsfinance_core::gpu::device::GpuDevice;
use kimsfinance_core::gpu::obv::obv_gpu;
use kimsfinance_core::gpu::obv_optimized::obv_gpu_optimized;
use ndarray::Array1;
use std::time::Instant;

fn main() {
    println!("=== OBV Implementation Comparison ===\n");

    let device = GpuDevice::new().expect("Failed to initialize GPU device");

    // Test sizes
    let sizes = vec![10_000, 50_000, 100_000, 200_000];

    println!("{:<12} | {:>12} | {:>12} | {:>10} | {:>15}", 
             "Size", "Naive (ms)", "Optimized (ms)", "Speedup", "Status");
    println!("{}", "-".repeat(75));

    for n in sizes {
        // Generate test data
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

        // Warmup
        let _ = obv_gpu(&device, &close_arr, &volume_arr, None);
        let _ = obv_gpu_optimized(&device, &close_arr, &volume_arr, None);

        // Benchmark naive implementation
        let mut naive_times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let _obv = obv_gpu(&device, &close_arr, &volume_arr, None)
                .expect("Naive OBV failed");
            naive_times.push(start.elapsed().as_secs_f64());
        }
        let naive_avg = naive_times.iter().sum::<f64>() / naive_times.len() as f64;

        // Benchmark optimized implementation
        let mut opt_times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let _obv = obv_gpu_optimized(&device, &close_arr, &volume_arr, None)
                .expect("Optimized OBV failed");
            opt_times.push(start.elapsed().as_secs_f64());
        }
        let opt_avg = opt_times.iter().sum::<f64>() / opt_times.len() as f64;

        let speedup = naive_avg / opt_avg;
        let status = if speedup >= 5.0 {
            "✓ Excellent"
        } else if speedup >= 3.0 {
            "✓ Good"
        } else if speedup >= 1.5 {
            "○ Moderate"
        } else {
            "✗ Poor"
        };

        println!(
            "{:<12} | {:>12.3} | {:>12.3} | {:>10.2}x | {:>15}",
            format!("{}", n),
            naive_avg * 1000.0,
            opt_avg * 1000.0,
            speedup,
            status
        );

        // Verify correctness for first size
        if n == 10_000 {
            let naive_result = obv_gpu(&device, &close_arr, &volume_arr, None)
                .expect("Naive verification failed");
            let opt_result = obv_gpu_optimized(&device, &close_arr, &volume_arr, None)
                .expect("Optimized verification failed");

            let mut max_error = 0.0;
            for i in 0..naive_result.len() {
                let error = (naive_result[i] - opt_result[i]).abs();
                if error > max_error {
                    max_error = error;
                }
            }

            println!("\n  Verification (n={}): max error = {:.2e}", n, max_error);
            if max_error < 1e-6 {
                println!("  ✓ Results match (< 1e-6 tolerance)\n");
            } else {
                println!("  ✗ Results differ! Investigation needed.\n");
            }
        }
    }

    println!("\n=== Analysis ===");
    println!("Target: 5-10x speedup over naive implementation");
    println!("Naive bottleneck: Single-threaded cumulative sum");
    println!("Optimized: Parallel prefix sum (Hillis-Steele scan)");
}

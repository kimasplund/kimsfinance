//! Benchmark: Parabolic SAR CPU vs GPU
//!
//! Compares performance of CPU-only vs hybrid CPU-GPU Parabolic SAR implementation.
//!
//! # Run
//!
//! ```bash
//! cargo run --example benchmark_parabolic_sar --features gpu --release
//! ```

use kimsfinance_core::indicators::trend::ParabolicSAR;
use ndarray::Array1;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{parabolic_sar_gpu, GpuDevice};

fn generate_trending_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Generate price data with trends and reversals
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);

    let mut base_price = 100.0;
    let mut direction = 1.0; // 1.0 = up, -1.0 = down

    for i in 0..n {
        // Change direction every 500 candles to create reversals
        if i % 500 == 0 && i > 0 {
            direction *= -1.0;
        }

        // Add some noise and trend
        let noise = (i as f64 * 0.01).sin() * 2.0;
        let trend = direction * 0.05;

        base_price += trend + noise * 0.1;

        high.push(base_price + 2.5);
        low.push(base_price - 2.5);
    }

    (high, low)
}

fn benchmark_cpu(high: &Array1<f64>, low: &Array1<f64>, iterations: usize) -> f64 {
    let psar = ParabolicSAR::new(0.02, 0.02, 0.2).expect("Failed to create Parabolic SAR");

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _result = psar
            .calculate_hl(high.view(), low.view())
            .expect("CPU calculation failed");
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

#[cfg(feature = "gpu")]
fn benchmark_gpu(
    device: &GpuDevice,
    high: &Array1<f64>,
    low: &Array1<f64>,
    iterations: usize,
) -> f64 {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _result = parabolic_sar_gpu(device, high, low, 0.02, 0.02, 0.2, None)
            .expect("GPU calculation failed");
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn main() {
    println!("=== Parabolic SAR Benchmark: CPU vs GPU ===\n");

    let dataset_sizes = vec![1_000, 5_000, 10_000, 50_000, 100_000];

    #[cfg(feature = "gpu")]
    let device = match GpuDevice::new() {
        Ok(dev) => {
            println!("GPU initialized successfully\n");
            Some(dev)
        }
        Err(e) => {
            println!("GPU initialization failed: {:?}", e);
            println!("Running CPU-only benchmarks\n");
            None
        }
    };

    #[cfg(not(feature = "gpu"))]
    let device: Option<()> = None;

    println!("{:<12} {:<15} {:<15} {:<15}", "Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<60}", "");

    for &n in &dataset_sizes {
        let (high_vec, low_vec) = generate_trending_data(n);
        let high = Array1::from_vec(high_vec);
        let low = Array1::from_vec(low_vec);

        // Determine iterations based on dataset size (fewer for larger datasets)
        let iterations = if n <= 10_000 {
            20
        } else if n <= 50_000 {
            10
        } else {
            5
        };

        // Warmup
        let psar = ParabolicSAR::new(0.02, 0.02, 0.2).unwrap();
        let _ = psar.calculate_hl(high.view(), low.view());

        // CPU benchmark
        let cpu_time = benchmark_cpu(&high, &low, iterations);

        #[cfg(feature = "gpu")]
        let gpu_time = if let Some(ref dev) = device {
            // Warmup GPU
            let _ = parabolic_sar_gpu(dev, &high, &low, 0.02, 0.02, 0.2, None);

            // GPU benchmark
            Some(benchmark_gpu(dev, &high, &low, iterations))
        } else {
            None
        };

        #[cfg(not(feature = "gpu"))]
        let gpu_time: Option<f64> = None;

        // Print results
        match gpu_time {
            Some(gpu_t) => {
                let speedup = cpu_time / gpu_t;
                println!(
                    "{:<12} {:<15.3} {:<15.3} {:<15.2}x",
                    n,
                    cpu_time * 1000.0,
                    gpu_t * 1000.0,
                    speedup
                );
            }
            None => {
                println!("{:<12} {:<15.3} {:<15} {:<15}", n, cpu_time * 1000.0, "N/A", "N/A");
            }
        }
    }

    println!("\n=== Analysis ===");
    println!("Parabolic SAR is inherently sequential due to trend state tracking.");
    println!("Expected speedup: 2-5x for large datasets (>10K candles).");
    println!("GPU benefit comes from batch processing within trend segments.");
    println!("Frequent reversals reduce GPU efficiency (more CPU state updates).");
}

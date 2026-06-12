//! Benchmark: Advanced Moving Averages (DEMA, TEMA, HMA, KAMA) CPU vs GPU
//!
//! Compares single-series and batch performance of CPU vs GPU implementations.
//!
//! # Run
//!
//! ```bash
//! cargo run --example benchmark_ma_advanced --features gpu --release
//! ```

use kimsfinance_core::indicators::core::Indicator;
use kimsfinance_core::indicators::moving_averages::{DEMA, HMA, TEMA};
use kimsfinance_core::indicators::moving_averages_advanced::KAMA;
use ndarray::{Array1, Array2};
use std::time::Instant;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{
    GpuDevice,
    dema_gpu, dema_batch_gpu,
    tema_gpu, tema_batch_gpu,
    hma_gpu,
    kama_gpu, kama_batch_gpu,
    KamaParams,
};

fn generate_price_data(n: usize) -> Array1<f64> {
    let mut prices = Vec::with_capacity(n);
    let mut base = 100.0;
    for i in 0..n {
        let noise = (i as f64 * 0.05).sin() * 0.5 + (i as f64 * 0.13).cos() * 0.2;
        base += noise;
        prices.push(base);
    }
    Array1::from_vec(prices)
}

fn generate_batch_price_data(num_series: usize, series_len: usize) -> Array2<f64> {
    let mut flat = Vec::with_capacity(num_series * series_len);
    for s in 0..num_series {
        let mut base = 100.0 + (s as f64 * 10.0);
        for i in 0..series_len {
            let noise = (i as f64 * 0.05).sin() * 0.5 + (i as f64 * 0.13).cos() * 0.2;
            base += noise;
            flat.push(base);
        }
    }
    Array2::from_shape_vec((num_series, series_len), flat).unwrap()
}

fn main() {
    println!("==============================================================");
    println!("Advanced Moving Averages (DEMA/TEMA/HMA/KAMA) GPU vs CPU Bench");
    println!("==============================================================\n");

    #[cfg(feature = "gpu")]
    let device = match GpuDevice::new() {
        Ok(dev) => {
            println!("✓ GPU Device initialized successfully.");
            Some(dev)
        }
        Err(e) => {
            println!("⚠️ GPU initialization failed (CPU-only run): {:?}", e);
            None
        }
    };

    #[cfg(not(feature = "gpu"))]
    println!("ℹ️ Built without `gpu` feature: running CPU-only benchmark.");

    #[cfg(feature = "gpu")]
    let gpu_enabled = device.is_some();
    #[cfg(not(feature = "gpu"))]
    let gpu_enabled = false;

    // -----------------------------------------------------------------------
    // Benchmark 1: Single-Series Latency comparison (Large series)
    // -----------------------------------------------------------------------
    println!("\n--- Benchmark 1: Single-Series Latency (Size = 100,000) ---");
    let series_len = 100_000;
    let prices = generate_price_data(series_len);
    let period = 20;

    if gpu_enabled {
        println!("{:<10} {:<15} {:<15} {:<15}", "Indicator", "CPU (ms)", "GPU (ms)", "Speedup");
    } else {
        println!("{:<10} {:<15}", "Indicator", "CPU (ms)");
    }
    println!("{:-<60}", "");

    // DEMA
    let cpu_dema = DEMA::new(period).unwrap();
    let start = Instant::now();
    let _ = cpu_dema.calculate(prices.view()).unwrap();
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        let start = Instant::now();
        let _ = dema_gpu(dev, &prices, period, None).unwrap();
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "DEMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "DEMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "DEMA", cpu_time);

    // TEMA
    let cpu_tema = TEMA::new(period).unwrap();
    let start = Instant::now();
    let _ = cpu_tema.calculate(prices.view()).unwrap();
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        let start = Instant::now();
        let _ = tema_gpu(dev, &prices, period, None).unwrap();
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "TEMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "TEMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "TEMA", cpu_time);

    // KAMA
    let cpu_kama = KAMA::new(period, 2, 30).unwrap();
    let start = Instant::now();
    let _ = cpu_kama.calculate(prices.view()).unwrap();
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        let start = Instant::now();
        let _ = kama_gpu(dev, &prices, period, 2, 30, None).unwrap();
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "KAMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "KAMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "KAMA", cpu_time);

    // HMA
    let cpu_hma = HMA::new(period).unwrap();
    let start = Instant::now();
    let _ = cpu_hma.calculate(prices.view()).unwrap();
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        let start = Instant::now();
        let _ = hma_gpu(dev, &prices, period, None).unwrap();
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "HMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "HMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "HMA", cpu_time);


    // -----------------------------------------------------------------------
    // Benchmark 2: Batch Throughput comparison (Parameter Sweeps / Multi-Series)
    // -----------------------------------------------------------------------
    println!("\n--- Benchmark 2: Batch Throughput (100 Series x 10,000 candles) ---");
    let num_series = 100;
    let series_len = 10_000;
    let batch_prices = generate_batch_price_data(num_series, series_len);
    let periods = vec![10, 20, 30, 40, 50, 100]; // 6 parameter variations

    if gpu_enabled {
        println!("{:<10} {:<15} {:<15} {:<15}", "Indicator", "CPU (ms)", "GPU (ms)", "Speedup");
    } else {
        println!("{:<10} {:<15}", "Indicator", "CPU (ms)");
    }
    println!("{:-<60}", "");

    // DEMA Batch
    // CPU requires looping over all series and all periods sequentially
    let start = Instant::now();
    for s in 0..num_series {
        let row = batch_prices.row(s);
        for &p in &periods {
            let cpu_dema = DEMA::new(p).unwrap();
            let _ = cpu_dema.calculate(row).unwrap();
        }
    }
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        // GPU processes all 600 combinations in parallel
        let start = Instant::now();
        let _ = dema_batch_gpu(dev, &batch_prices, &periods, None).unwrap();
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "DEMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "DEMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "DEMA", cpu_time);

    // TEMA Batch
    let start = Instant::now();
    for s in 0..num_series {
        let row = batch_prices.row(s);
        for &p in &periods {
            let cpu_tema = TEMA::new(p).unwrap();
            let _ = cpu_tema.calculate(row).unwrap();
        }
    }
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        let start = Instant::now();
        let _ = tema_batch_gpu(dev, &batch_prices, &periods, None).unwrap();
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "TEMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "TEMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "TEMA", cpu_time);

    // KAMA Batch
    #[cfg(feature = "gpu")]
    let kama_params: Vec<KamaParams> = periods
        .iter()
        .map(|&p| KamaParams {
            er_period: p,
            fast_period: 2,
            slow_period: 30,
        })
        .collect();

    let start = Instant::now();
    for s in 0..num_series {
        let row = batch_prices.row(s);
        for &p in &periods {
            let cpu_kama = KAMA::new(p, 2, 30).unwrap();
            let _ = cpu_kama.calculate(row).unwrap();
        }
    }
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        let start = Instant::now();
        let _ = kama_batch_gpu(dev, &batch_prices, &kama_params, None).unwrap();
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "KAMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "KAMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "KAMA", cpu_time);

    // HMA Sequential GPU Launch (element-parallelized per launch)
    let start = Instant::now();
    for s in 0..num_series {
        let row = batch_prices.row(s);
        for &p in &periods {
            let cpu_hma = HMA::new(p).unwrap();
            let _ = cpu_hma.calculate(row).unwrap();
        }
    }
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "gpu")]
    if let Some(dev) = device.as_ref() {
        let start = Instant::now();
        for s in 0..num_series {
            let row = batch_prices.row(s);
            let row_arr1 = row.to_owned();
            for &p in &periods {
                let _ = hma_gpu(dev, &row_arr1, p, None).unwrap();
            }
        }
        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}x", "HMA", cpu_time, gpu_time, cpu_time / gpu_time);
    } else {
        println!("{:<10} {:<15.2}", "HMA", cpu_time);
    }

    #[cfg(not(feature = "gpu"))]
    println!("{:<10} {:<15.2}", "HMA", cpu_time);

    println!("\n=== Analysis ===");
    if !gpu_enabled {
        println!("- GPU path is unavailable in this run, so only CPU timing is reported.");
        println!("- Rebuild with `--features gpu` on a CUDA-capable machine for speedup data.");
    } else {
        println!("- For a single series (Benchmark 1), host-device transfer and JIT launch");
        println!("  overhead dominates. HMA achieves high performance because of its fused WMA math.");
        println!("- For batch sweeps (Benchmark 2), the GPU excels by parallelizing across");
        println!("  all series and parameters, yielding massive throughput speedups (often >10x).");
    }
}

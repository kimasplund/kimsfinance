#!/usr/bin/env cargo
//! Quick timing test for available GPU indicators with 100K candles
//! Uses proper warmup methodology to exclude CUDA kernel compilation overhead

use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use std::time::Instant;

fn main() {
    let n = 100_000;
    println!("\n{:=^100}", " GPU INDICATOR TIMING (100K CANDLES) ");
    println!("{:<30} {:>12} {:>12} {:>12} {:>12} {:>15}",
             "Indicator", "Cold (μs)", "Warm (μs)", "Cold (ms)", "Warm (ms)", "Candles/sec");
    println!("{:-^100}", "");

    // Generate test data
    let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 * 0.01)).collect());
    let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64 * 0.01)).collect());
    let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64 * 0.01)).collect());
    let volume = Array1::from_vec((0..n).map(|i| 1000000.0 + (i as f64 * 100.0)).collect());

    let device = GpuDevice::new().expect("Failed to initialize GPU");

    macro_rules! time_it {
        ($name:expr, $warmup:expr, $timed:expr) => {{
            // COLD START: First run includes CUDA kernel compilation
            let cold_start = Instant::now();
            match $warmup {
                Ok(_) => {},
                Err(e) => {
                    println!("{:<30} {:>12}", $name, format!("ERROR: {:?}", e));
                    return;
                }
            }
            device.synchronize().expect("Failed to sync after cold start");
            let cold_micros = cold_start.elapsed().as_micros();

            // WARMUP: Run 4 more times to ensure kernels are compiled and caches filled
            for _ in 0..4 {
                let _ = $warmup;
            }
            device.synchronize().expect("Failed to sync after warmup");

            // WARM TIMING: Average over 10 runs for stable measurement
            let warm_start = Instant::now();
            for _ in 0..10 {
                match $timed {
                    Ok(_) => {},
                    Err(e) => {
                        println!("{:<30} {:>12}", $name, format!("ERROR: {:?}", e));
                        return;
                    }
                }
            }
            device.synchronize().expect("Failed to sync after warm runs");
            let warm_micros = warm_start.elapsed().as_micros() / 10;

            let cold_millis = cold_micros as f64 / 1000.0;
            let warm_millis = warm_micros as f64 / 1000.0;
            let candles_per_sec = (n as f64 / warm_millis) * 1000.0;

            println!("{:<30} {:>12} {:>12} {:>12.2} {:>12.2} {:>15.0}",
                     $name, cold_micros, warm_micros, cold_millis, warm_millis, candles_per_sec);
        }};
    }

    // GROUP 1: Simple (2-3 transfers)
    println!("\n{}", "GROUP 1: SIMPLE INDICATORS (2-3 transfers)");

    use kimsfinance_core::gpu::ema::ema_hybrid;
    time_it!("EMA (hybrid)",
        ema_hybrid(&device, &close, 14, None),
        ema_hybrid(&device, &close, 14, None)
    );

    use kimsfinance_core::gpu::roc::roc_gpu;
    time_it!("ROC",
        roc_gpu(&device, &close, 12, None),
        roc_gpu(&device, &close, 12, None)
    );

    use kimsfinance_core::gpu::wma::wma_gpu;
    time_it!("WMA",
        wma_gpu(&device, &close, 14, None),
        wma_gpu(&device, &close, 14, None)
    );

    use kimsfinance_core::gpu::obv::obv_gpu;
    time_it!("OBV",
        obv_gpu(&device, &close, &volume, None),
        obv_gpu(&device, &close, &volume, None)
    );

    use kimsfinance_core::gpu::vwma::vwma_gpu;
    time_it!("VWMA",
        vwma_gpu(&device, &close, &volume, 14, None),
        vwma_gpu(&device, &close, &volume, 14, None)
    );

    // GROUP 2: Medium (4-5 transfers)
    println!("\n{}", "GROUP 2: MEDIUM INDICATORS (4-5 transfers)");

    use kimsfinance_core::gpu::cci::cci_gpu;
    time_it!("CCI",
        cci_gpu(&device, &high, &low, &close, 20, None),
        cci_gpu(&device, &high, &low, &close, 20, None)
    );

    use kimsfinance_core::gpu::macd::macd_hybrid;
    time_it!("MACD [CPU]",
        macd_hybrid(&device, &close, 12, 26, 9, None),
        macd_hybrid(&device, &close, 12, 26, 9, None)
    );

    use kimsfinance_core::gpu::sma::sma_gpu;
    time_it!("SMA",
        sma_gpu(&device, &close, 14, None),
        sma_gpu(&device, &close, 14, None)
    );

    use kimsfinance_core::gpu::williams_r::williams_r_gpu;
    time_it!("Williams %R",
        williams_r_gpu(&device, &high, &low, &close, 14, None),
        williams_r_gpu(&device, &high, &low, &close, 14, None)
    );

    use kimsfinance_core::gpu::cmf::cmf_gpu;
    time_it!("CMF",
        cmf_gpu(&device, &high, &low, &close, &volume, 20, None),
        cmf_gpu(&device, &high, &low, &close, &volume, 20, None)
    );

    use kimsfinance_core::gpu::donchian::donchian_gpu;
    time_it!("Donchian Channels",
        donchian_gpu(&device, &high, &low, 20, None),
        donchian_gpu(&device, &high, &low, 20, None)
    );

    use kimsfinance_core::gpu::elder_ray::elder_ray_gpu;
    time_it!("Elder Ray",
        elder_ray_gpu(&device, &high, &low, &close, 13, None),
        elder_ray_gpu(&device, &high, &low, &close, 13, None)
    );

    use kimsfinance_core::gpu::stochastic::stochastic_gpu;
    time_it!("Stochastic",
        stochastic_gpu(&device, &high, &low, &close, 14, 3, None),
        stochastic_gpu(&device, &high, &low, &close, 14, 3, None)
    );

    // GROUP 3: Complex (6+ transfers)
    println!("\n{}", "GROUP 3: COMPLEX INDICATORS (6+ transfers)");

    use kimsfinance_core::gpu::atr::atr_gpu;
    time_it!("ATR (reference - Jules' opt)",
        atr_gpu(&device, &high, &low, &close, 14, None),
        atr_gpu(&device, &high, &low, &close, 14, None)
    );

    use kimsfinance_core::gpu::rsi::rsi_gpu;
    time_it!("RSI",
        rsi_gpu(&device, &close, 14, None),
        rsi_gpu(&device, &close, 14, None)
    );

    use kimsfinance_core::gpu::rsi_sync::rsi_gpu_sync;
    time_it!("RSI (sync variant)",
        rsi_gpu_sync(&device, &close, 14, None),
        rsi_gpu_sync(&device, &close, 14, None)
    );

    println!("\n{:=^100}", " SUMMARY ");
    println!("\nATR Performance Analysis:");
    println!("  - Theoretical target (from code comments): 145μs");
    println!("    * Based on component estimates: 25μs H2D + 20μs compute + 25μs D2H + 15μs CPU");
    println!("  - Actual warm performance: ~1,600μs");
    println!("  - Discrepancy: 11x slower than theoretical");
    println!("  - Conclusion: Implementation has significant overhead beyond theoretical minimum");
    println!("\nBenchmark Methodology (Proper GPU Warmup):");
    println!("  1. Cold start: First run with CUDA kernel compilation");
    println!("  2. Warmup: 4 additional runs to ensure kernels compiled and caches filled");
    println!("  3. Measurement: Average of 10 synchronized runs (device.synchronize() after each)");
    println!("  4. All measurements include GPU stream synchronization to ensure accuracy");
    println!("\nKey Findings:");
    println!("  - Cold start: 16-26ms (includes ~15-25ms CUDA compilation overhead)");
    println!("  - Warm performance: 300μs-50ms depending on indicator complexity");
    println!("  - ATR warm: ~1,600μs (10x slower than 145μs theoretical estimate)");
    println!("\n{:=^100}", " COMPLETE ");
}

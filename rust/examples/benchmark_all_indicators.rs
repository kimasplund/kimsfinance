#!/usr/bin/env cargo
//! Quick timing test for all GPU indicators with 100K candles

use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use std::time::Instant;

fn main() {
    let n = 100_000;
    println!("\n{:=^80}", " GPU INDICATOR TIMING (100K CANDLES) ");
    println!(
        "{:<30} {:>15} {:>15}",
        "Indicator", "Time (μs)", "Time (ms)"
    );
    println!("{:-^80}", "");

    // Generate test data
    let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 * 0.01)).collect());
    let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64 * 0.01)).collect());
    let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64 * 0.01)).collect());
    let volume = Array1::from_vec((0..n).map(|i| 1000000.0 + (i as f64 * 100.0)).collect());
    let timestamps = Array1::from_vec((0..n).map(|i| i as f64).collect());

    let device = GpuDevice::new().expect("Failed to initialize GPU");

    macro_rules! time_it {
        ($name:expr, $code:expr) => {{
            let start = Instant::now();
            match $code {
                Ok(_) => {
                    let micros = start.elapsed().as_micros();
                    let millis = micros as f64 / 1000.0;
                    println!("{:<30} {:>15} {:>15.2}", $name, micros, millis);
                }
                Err(e) => {
                    println!("{:<30} {:>15}", $name, format!("ERROR: {}", e));
                }
            }
        }};
    }

    // GROUP 1: Simple (2-3 transfers)
    println!("\n{}", "GROUP 1: SIMPLE INDICATORS (2-3 transfers)");

    time_it!("EMA", {
        use kimsfinance_core::gpu::ema::ema_hybrid;
        ema_hybrid(&close, 14, &device, None)
    });

    time_it!("ROC", {
        use kimsfinance_core::gpu::roc::roc_gpu;
        roc_gpu(&device, &close, 12, None)
    });

    time_it!("WMA", {
        use kimsfinance_core::gpu::wma::wma_gpu;
        wma_gpu(&device, &close, 14, None)
    });

    time_it!("OBV", {
        use kimsfinance_core::gpu::obv::obv_gpu;
        obv_gpu(&device, &close, &volume, None)
    });

    time_it!("VWMA", {
        use kimsfinance_core::gpu::vwma::vwma_gpu;
        vwma_gpu(&device, &close, &volume, 14, None)
    });

    // GROUP 2: Medium (4-5 transfers)
    println!("\n{}", "GROUP 2: MEDIUM INDICATORS (4-5 transfers)");

    time_it!("Bollinger Bands", {
        use kimsfinance_core::gpu::bollinger::bollinger_gpu;
        bollinger_gpu(&device, &close, 20, 2.0, None)
    });

    time_it!("CCI", {
        use kimsfinance_core::gpu::cci::cci_gpu;
        cci_gpu(&device, &high, &low, &close, 20, None)
    });

    time_it!("MACD [CPU]", {
        use kimsfinance_core::gpu::macd::macd_hybrid;
        macd_hybrid(&device, &close, 12, 26, 9, None)
    });

    time_it!("SMA", {
        use kimsfinance_core::gpu::sma::sma_gpu;
        sma_gpu(&device, &close, 14, None)
    });

    time_it!("Williams %R", {
        use kimsfinance_core::gpu::williams_r::williams_r_gpu;
        williams_r_gpu(&device, &high, &low, &close, 14, None)
    });

    time_it!("CMF", {
        use kimsfinance_core::gpu::cmf::cmf_gpu;
        cmf_gpu(&device, &high, &low, &close, &volume, 20, None)
    });

    time_it!("Donchian Channels", {
        use kimsfinance_core::gpu::donchian::donchian_gpu;
        donchian_gpu(&device, &high, &low, 20, None)
    });

    time_it!("Elder Ray", {
        use kimsfinance_core::gpu::elder_ray::elder_ray_gpu;
        elder_ray_gpu(&device, &high, &low, &close, 13, None)
    });

    time_it!("Keltner Channels", {
        use kimsfinance_core::gpu::keltner::keltner_gpu;
        keltner_gpu(&device, &high, &low, &close, 20, 2.0, None)
    });

    time_it!("Stochastic", {
        use kimsfinance_core::gpu::stochastic::stochastic_gpu;
        stochastic_gpu(&device, &high, &low, &close, 14, 3, None)
    });

    time_it!("VWAP", {
        use kimsfinance_core::gpu::vwap::vwap_gpu;
        vwap_gpu(&device, &high, &low, &close, &volume, &timestamps, None)
    });

    // GROUP 3: Complex (6+ transfers)
    println!("\n{}", "GROUP 3: COMPLEX INDICATORS (6+ transfers)");

    time_it!("ATR (reference)", {
        use kimsfinance_core::gpu::atr::atr_gpu;
        atr_gpu(&device, &high, &low, &close, 14, None)
    });

    time_it!("RSI", {
        use kimsfinance_core::gpu::rsi::rsi_gpu;
        rsi_gpu(&device, &close, 14, None)
    });

    time_it!("Supertrend", {
        use kimsfinance_core::gpu::supertrend::supertrend_gpu;
        use std::sync::Arc;
        supertrend_gpu(
            Arc::new(device.clone()),
            high.as_slice().unwrap(),
            low.as_slice().unwrap(),
            close.as_slice().unwrap(),
            10,
            3.0,
            None,
        )
    });

    println!("\n{:=^80}", " COMPLETE ");
}

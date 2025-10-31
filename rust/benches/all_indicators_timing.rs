#!/usr/bin/env -S cargo bench --bench
//! Comprehensive GPU Indicator Timing Benchmark
//!
//! Tests all optimized GPU indicators with 100K candles to measure actual performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kimsfinance_core::gpu::device::GpuDevice;
use ndarray::Array1;
use std::time::Instant;

/// Generate test OHLCV data
fn generate_ohlcv(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let high = Array1::from_vec((0..n).map(|i| 100.0 + (i as f64 * 0.01)).collect());
    let low = Array1::from_vec((0..n).map(|i| 95.0 + (i as f64 * 0.01)).collect());
    let close = Array1::from_vec((0..n).map(|i| 98.0 + (i as f64 * 0.01)).collect());
    let open = Array1::from_vec((0..n).map(|i| 97.0 + (i as f64 * 0.01)).collect());
    let volume = Array1::from_vec((0..n).map(|i| 1000000.0 + (i as f64 * 100.0)).collect());
    (high, low, close, open, volume)
}

fn benchmark_all_indicators(c: &mut Criterion) {
    let n = 100_000; // 100K candles
    let (high, low, close, _open, volume) = generate_ohlcv(n);

    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("gpu_indicators_100k");
    group.sample_size(10);

    // ============================================================================
    // GROUP 1: SIMPLE INDICATORS (2-3 transfers)
    // ============================================================================

    // EMA
    group.bench_function("ema", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::ema::ema_hybrid;
            black_box(ema_hybrid(&close, 14, &device, None).unwrap())
        })
    });

    // ROC
    group.bench_function("roc", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::roc::roc_gpu;
            black_box(roc_gpu(&close, 12, &device, None).unwrap())
        })
    });

    // WMA
    group.bench_function("wma", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::wma::wma_gpu;
            black_box(wma_gpu(&close, 14, &device, None).unwrap())
        })
    });

    // OBV
    group.bench_function("obv", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::obv::obv_gpu;
            black_box(obv_gpu(&close, &volume, &device, None).unwrap())
        })
    });

    // VWMA
    group.bench_function("vwma", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::vwma::vwma_gpu;
            black_box(vwma_gpu(&close, &volume, 14, &device, None).unwrap())
        })
    });

    // ============================================================================
    // GROUP 2: MEDIUM INDICATORS (4-5 transfers)
    // ============================================================================

    // Bollinger Bands
    group.bench_function("bollinger", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::bollinger::bollinger_gpu;
            black_box(bollinger_gpu(&close, 20, 2.0, &device, None).unwrap())
        })
    });

    // CCI
    group.bench_function("cci", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::cci::cci_gpu;
            black_box(cci_gpu(&high, &low, &close, 20, &device, None).unwrap())
        })
    });

    // MACD [CPU]
    group.bench_function("macd_cpu", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::macd::macd_hybrid;
            black_box(macd_hybrid(&device, &close, 12, 26, 9, None).unwrap())
        })
    });

    // SMA
    group.bench_function("sma", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::sma::sma_gpu;
            black_box(sma_gpu(&close, 14, &device, None).unwrap())
        })
    });

    // Williams %R
    group.bench_function("williams_r", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::williams_r::williams_r_gpu;
            black_box(williams_r_gpu(&high, &low, &close, 14, &device, None).unwrap())
        })
    });

    // CMF
    group.bench_function("cmf", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::cmf::cmf_gpu;
            black_box(cmf_gpu(&high, &low, &close, &volume, 20, &device, None).unwrap())
        })
    });

    // Donchian Channels
    group.bench_function("donchian", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::donchian::donchian_gpu;
            black_box(donchian_gpu(&high, &low, 20, &device, None).unwrap())
        })
    });

    // Elder Ray
    group.bench_function("elder_ray", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::elder_ray::elder_ray_gpu;
            black_box(elder_ray_gpu(&high, &low, &close, 13, &device, None).unwrap())
        })
    });

    // Keltner Channels
    group.bench_function("keltner", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::keltner::keltner_gpu;
            black_box(keltner_gpu(&high, &low, &close, 20, 2.0, &device, None).unwrap())
        })
    });

    // Stochastic
    group.bench_function("stochastic", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::stochastic::stochastic_gpu;
            black_box(stochastic_gpu(&high, &low, &close, 14, 3, &device, None).unwrap())
        })
    });

    // VWAP
    group.bench_function("vwap", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::vwap::vwap_gpu;
            let timestamps = Array1::from_vec((0..n).map(|i| i as f64).collect());
            black_box(vwap_gpu(&high, &low, &close, &volume, &timestamps, &device, None).unwrap())
        })
    });

    // ============================================================================
    // GROUP 3: COMPLEX INDICATORS (6-10 transfers)
    // ============================================================================

    // ATR (reference - Jules' optimization)
    group.bench_function("atr", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::atr::atr_gpu;
            black_box(atr_gpu(&high, &low, &close, 14, &device, None).unwrap())
        })
    });

    // RSI
    group.bench_function("rsi", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::rsi::rsi_gpu;
            black_box(rsi_gpu(&close, 14, &device, None).unwrap())
        })
    });

    // Supertrend
    group.bench_function("supertrend", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::supertrend::supertrend_gpu;
            black_box(supertrend_gpu(&high, &low, &close, 10, 3.0, &device, None).unwrap())
        })
    });

    group.finish();
}

fn single_run_timing() {
    let n = 100_000;
    let (high, low, close, _open, volume) = generate_ohlcv(n);
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    println!("\n{:=^80}", " GPU INDICATOR TIMING (100K CANDLES) ");
    println!("{:<25} {:>12} {:>15}", "Indicator", "Time (μs)", "Speedup vs CPU");
    println!("{:-^80}", "");

    macro_rules! time_indicator {
        ($name:expr, $code:expr) => {{
            let start = Instant::now();
            let _ = black_box($code);
            let elapsed = start.elapsed().as_micros();
            println!("{:<25} {:>12} {:>15}", $name, elapsed, "N/A");
        }};
    }

    // Group 1: Simple (2-3 transfers)
    println!("\n{}", "GROUP 1: SIMPLE INDICATORS (2-3 transfers)");
    time_indicator!("EMA", {
        use kimsfinance_core::gpu::ema::ema_hybrid;
        ema_hybrid(&close, 14, &device, None).unwrap()
    });
    time_indicator!("ROC", {
        use kimsfinance_core::gpu::roc::roc_gpu;
        roc_gpu(&close, 12, &device, None).unwrap()
    });
    time_indicator!("WMA", {
        use kimsfinance_core::gpu::wma::wma_gpu;
        wma_gpu(&close, 14, &device, None).unwrap()
    });
    time_indicator!("OBV", {
        use kimsfinance_core::gpu::obv::obv_gpu;
        obv_gpu(&close, &volume, &device, None).unwrap()
    });
    time_indicator!("VWMA", {
        use kimsfinance_core::gpu::vwma::vwma_gpu;
        vwma_gpu(&close, &volume, 14, &device, None).unwrap()
    });

    // Group 2: Medium (4-5 transfers)
    println!("\n{}", "GROUP 2: MEDIUM INDICATORS (4-5 transfers)");
    time_indicator!("Bollinger Bands", {
        use kimsfinance_core::gpu::bollinger::bollinger_gpu;
        bollinger_gpu(&close, 20, 2.0, &device, None).unwrap()
    });
    time_indicator!("CCI", {
        use kimsfinance_core::gpu::cci::cci_gpu;
        cci_gpu(&high, &low, &close, 20, &device, None).unwrap()
    });
    time_indicator!("MACD [CPU]", {
        use kimsfinance_core::gpu::macd::macd_hybrid;
        macd_hybrid(&device, &close, 12, 26, 9, None).unwrap()
    });
    time_indicator!("SMA", {
        use kimsfinance_core::gpu::sma::sma_gpu;
        sma_gpu(&close, 14, &device, None).unwrap()
    });
    time_indicator!("Williams %R", {
        use kimsfinance_core::gpu::williams_r::williams_r_gpu;
        williams_r_gpu(&high, &low, &close, 14, &device, None).unwrap()
    });
    time_indicator!("CMF", {
        use kimsfinance_core::gpu::cmf::cmf_gpu;
        cmf_gpu(&high, &low, &close, &volume, 20, &device, None).unwrap()
    });
    time_indicator!("Donchian Channels", {
        use kimsfinance_core::gpu::donchian::donchian_gpu;
        donchian_gpu(&high, &low, 20, &device, None).unwrap()
    });
    time_indicator!("Elder Ray", {
        use kimsfinance_core::gpu::elder_ray::elder_ray_gpu;
        elder_ray_gpu(&high, &low, &close, 13, &device, None).unwrap()
    });
    time_indicator!("Keltner Channels", {
        use kimsfinance_core::gpu::keltner::keltner_gpu;
        keltner_gpu(&high, &low, &close, 20, 2.0, &device, None).unwrap()
    });
    time_indicator!("Stochastic", {
        use kimsfinance_core::gpu::stochastic::stochastic_gpu;
        stochastic_gpu(&high, &low, &close, 14, 3, &device, None).unwrap()
    });
    time_indicator!("VWAP", {
        use kimsfinance_core::gpu::vwap::vwap_gpu;
        let timestamps = Array1::from_vec((0..n).map(|i| i as f64).collect());
        vwap_gpu(&high, &low, &close, &volume, &timestamps, &device, None).unwrap()
    });

    // Group 3: Complex (6+ transfers)
    println!("\n{}", "GROUP 3: COMPLEX INDICATORS (6+ transfers)");
    time_indicator!("ATR (reference)", {
        use kimsfinance_core::gpu::atr::atr_gpu;
        atr_gpu(&high, &low, &close, 14, &device, None).unwrap()
    });
    time_indicator!("RSI", {
        use kimsfinance_core::gpu::rsi::rsi_gpu;
        rsi_gpu(&close, 14, &device, None).unwrap()
    });
    time_indicator!("Supertrend", {
        use kimsfinance_core::gpu::supertrend::supertrend_gpu;
        supertrend_gpu(&high, &low, &close, 10, 3.0, &device, None).unwrap()
    });

    println!("\n{:=^80}", " COMPLETE ");
}

criterion_group!(benches, benchmark_all_indicators);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_single_run() {
        single_run_timing();
    }
}

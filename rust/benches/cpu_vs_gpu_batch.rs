//! CPU vs GPU Batch Processing Benchmark
//!
//! **CRITICAL BENCHMARK**: Answers the key question:
//! "Does GPU show speedup when processing MANY indicators/settings simultaneously?"
//!
//! # Test Matrix
//!
//! | Batch Size | Indicators | Expected GPU Advantage |
//! |------------|------------|------------------------|
//! | Small (9)  | 9 unique   | Marginal (<2x)         |
//! | Medium (25)| 25 configs | Moderate (2-4x)        |
//! | Large (50) | 50 configs | Significant (4-8x)     |
//!
//! # Methodology
//!
//! **CPU Batch**:
//! ```rust
//! for indicator in indicators {
//!     result = calculate_cpu(data, params);
//! }
//! ```
//!
//! **GPU Batch**:
//! ```rust
//! results = calculate_indicators_batch_gpu(device, data, all_indicators);
//! ```
//!
//! # Expected Results
//!
//! **RTX 3500 Ada (12GB) vs Intel i9-13980HX (24 cores)**
//!
//! Based on prior testing:
//! - **Small batch (9 indicators)**: GPU ~260ms, CPU expected ~100-200ms
//! - **Medium batch (25 indicators)**: GPU scales linearly, CPU scales linearly
//! - **Large batch (50 indicators)**: GPU should show advantage if true parallel execution
//!
//! # Usage
//!
//! ```bash
//! # Run all CPU vs GPU batch tests
//! cargo bench --features gpu --bench cpu_vs_gpu_batch
//!
//! # Run specific batch size
//! cargo bench --features gpu --bench cpu_vs_gpu_batch -- small_batch
//! cargo bench --features gpu --bench cpu_vs_gpu_batch -- medium_batch
//! cargo bench --features gpu --bench cpu_vs_gpu_batch -- large_batch
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ndarray::Array1;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{
    batch::{calculate_indicators_batch_gpu, IndicatorRequest},
    GpuDevice,
};

use kimsfinance_core::indicators::{
    momentum::{Aroon, CCI, ROC, RSI, Stochastic, WilliamsR},
    volatility::ATR,
    core::Indicator,
};

/// Generate realistic OHLCV test data
fn generate_ohlcv_data(size: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut high = Vec::with_capacity(size);
    let mut low = Vec::with_capacity(size);
    let mut close = Vec::with_capacity(size);
    let mut open = Vec::with_capacity(size);

    let base_price = 100.0;
    let trend = 0.01; // Slight uptrend
    let volatility = 2.0;

    for i in 0..size {
        let t = i as f64;

        // Price with trend, sine wave oscillation, and noise
        let price = base_price
            + trend * t
            + volatility * (t * 0.01).sin()
            + (t * 0.123).sin() * 0.5;

        // OHLC with realistic spread
        let spread = volatility * 0.5;
        high.push(price + spread * 0.7);
        low.push(price - spread * 0.7);
        close.push(price);
        open.push(price - spread * 0.3);
    }

    (
        Array1::from_vec(open),
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
    )
}

/// CPU batch processing (sequential calculation of multiple indicators)
fn calculate_batch_cpu(
    open: &Array1<f64>,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    num_indicators: usize,
) -> Vec<Array1<f64>> {
    let mut results = Vec::new();

    // Create indicator set based on batch size
    for i in 0..num_indicators {
        match i % 9 {
            0 => {
                // RSI with varying periods
                let period = 14 + (i / 9) * 2;
                let rsi = RSI::new(period).unwrap();
                if let Ok(result) = rsi.calculate(close.view()) {
                    results.push(result);
                }
            }
            1 => {
                // ATR with varying periods
                let period = 14 + (i / 9);
                let atr = ATR::new(period).unwrap();
                if let Ok(result) = atr.calculate_hlc(high.view(), low.view(), close.view()) {
                    results.push(result);
                }
            }
            2 => {
                // Stochastic with varying periods
                let k_period = 14 + (i / 9);
                let d_period = 3;
                let stoch = Stochastic::new(k_period, d_period).unwrap();
                if let Ok(result) = stoch.calculate_hlc(high.view(), low.view(), close.view()) {
                    results.push(result);
                }
            }
            3 => {
                // Williams %R with varying periods
                let period = 14 + (i / 9);
                let williams = WilliamsR::new(period).unwrap();
                if let Ok(result) = williams.calculate_hlc(high.view(), low.view(), close.view()) {
                    results.push(result);
                }
            }
            4 => {
                // ROC with varying periods
                let period = 12 + (i / 9) * 2;
                let roc = ROC::new(period).unwrap();
                if let Ok(result) = roc.calculate(close.view()) {
                    results.push(result);
                }
            }
            5 => {
                // CCI with varying periods
                let period = 20 + (i / 9);
                let cci = CCI::new(period).unwrap();
                if let Ok(result) = cci.calculate_hlc(high.view(), low.view(), close.view()) {
                    results.push(result);
                }
            }
            6 => {
                // Aroon with varying periods
                let period = 25 + (i / 9);
                let aroon = Aroon::new(period).unwrap();
                if let Ok(result) = aroon.calculate_hl(high.view(), low.view()) {
                    results.push(result);
                }
            }
            7 => {
                // Additional ROC with different period
                let period = 6 + (i / 9);
                let roc = ROC::new(period).unwrap();
                if let Ok(result) = roc.calculate(close.view()) {
                    results.push(result);
                }
            }
            8 => {
                // Additional RSI with different period
                let period = 7 + (i / 9);
                let rsi = RSI::new(period).unwrap();
                if let Ok(result) = rsi.calculate(close.view()) {
                    results.push(result);
                }
            }
            _ => unreachable!(),
        }
    }

    results
}

/// GPU batch processing (single batch API call)
#[cfg(feature = "gpu")]
fn calculate_batch_gpu(
    device: &GpuDevice,
    open: &Array1<f64>,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    num_indicators: usize,
) -> Result<std::collections::HashMap<kimsfinance_core::gpu::batch::BatchIndicatorType, Vec<Array1<f64>>>, kimsfinance_core::gpu::GpuError> {
    let mut requests = Vec::new();

    // Create same indicator set as CPU
    for i in 0..num_indicators {
        match i % 9 {
            0 => {
                let period = 14 + (i / 9) * 2;
                requests.push(IndicatorRequest::RSI { period });
            }
            1 => {
                let period = 14 + (i / 9);
                requests.push(IndicatorRequest::ATR { period });
            }
            2 => {
                let k_period = 14 + (i / 9);
                let d_period = 3;
                requests.push(IndicatorRequest::Stochastic { k_period, d_period });
            }
            3 => {
                let period = 14 + (i / 9);
                requests.push(IndicatorRequest::WilliamsR { period });
            }
            4 => {
                let period = 12 + (i / 9) * 2;
                requests.push(IndicatorRequest::ROC { period });
            }
            5 => {
                let period = 20 + (i / 9);
                requests.push(IndicatorRequest::CCI { period });
            }
            6 => {
                let period = 25 + (i / 9);
                requests.push(IndicatorRequest::Aroon { period });
            }
            7 => {
                let period = 6 + (i / 9);
                requests.push(IndicatorRequest::ROC { period });
            }
            8 => {
                let period = 7 + (i / 9);
                requests.push(IndicatorRequest::RSI { period });
            }
            _ => unreachable!(),
        }
    }

    calculate_indicators_batch_gpu(device, high, low, open, close, &requests, None)
}

#[cfg(feature = "gpu")]
fn bench_small_batch_9_indicators(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");
    let size = 44640; // 1 month of 1-minute candles
    let (open, high, low, close) = generate_ohlcv_data(size);

    let mut group = c.benchmark_group("cpu_vs_gpu_small_batch");
    group.throughput(Throughput::Elements(9));
    group.sample_size(50); // Fewer samples for faster benchmarking

    println!("\n=== Small Batch: 9 Indicators (44,640 candles) ===");

    // CPU batch
    group.bench_function("cpu_9_indicators", |b| {
        b.iter(|| {
            black_box(calculate_batch_cpu(
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                9,
            ))
        })
    });

    // GPU batch
    group.bench_function("gpu_9_indicators", |b| {
        b.iter(|| {
            black_box(calculate_batch_gpu(
                black_box(&device),
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                9,
            ).expect("GPU batch failed"))
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_medium_batch_25_indicators(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");
    let size = 44640;
    let (open, high, low, close) = generate_ohlcv_data(size);

    let mut group = c.benchmark_group("cpu_vs_gpu_medium_batch");
    group.throughput(Throughput::Elements(25));
    group.sample_size(30);

    println!("\n=== Medium Batch: 25 Indicators (parameter sweep scenario) ===");

    // CPU batch
    group.bench_function("cpu_25_indicators", |b| {
        b.iter(|| {
            black_box(calculate_batch_cpu(
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                25,
            ))
        })
    });

    // GPU batch
    group.bench_function("gpu_25_indicators", |b| {
        b.iter(|| {
            black_box(calculate_batch_gpu(
                black_box(&device),
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                25,
            ).expect("GPU batch failed"))
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_large_batch_50_indicators(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");
    let size = 44640;
    let (open, high, low, close) = generate_ohlcv_data(size);

    let mut group = c.benchmark_group("cpu_vs_gpu_large_batch");
    group.throughput(Throughput::Elements(50));
    group.sample_size(20);

    println!("\n=== Large Batch: 50 Indicators (multi-indicator backtesting) ===");

    // CPU batch
    group.bench_function("cpu_50_indicators", |b| {
        b.iter(|| {
            black_box(calculate_batch_cpu(
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                50,
            ))
        })
    });

    // GPU batch
    group.bench_function("gpu_50_indicators", |b| {
        b.iter(|| {
            black_box(calculate_batch_gpu(
                black_box(&device),
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                50,
            ).expect("GPU batch failed"))
        })
    });

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_small_batch_9_indicators(_c: &mut Criterion) {
    eprintln!("GPU feature not enabled. Run with: cargo bench --features gpu");
}

#[cfg(not(feature = "gpu"))]
fn bench_medium_batch_25_indicators(_c: &mut Criterion) {
    eprintln!("GPU feature not enabled. Run with: cargo bench --features gpu");
}

#[cfg(not(feature = "gpu"))]
fn bench_large_batch_50_indicators(_c: &mut Criterion) {
    eprintln!("GPU feature not enabled. Run with: cargo bench --features gpu");
}

criterion_group!(
    benches,
    bench_small_batch_9_indicators,
    bench_medium_batch_25_indicators,
    bench_large_batch_50_indicators
);
criterion_main!(benches);

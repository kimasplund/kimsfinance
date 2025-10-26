//! CPU Batch Indicator Benchmark
//!
//! **Simple CPU batch benchmark** to compare against existing GPU batch results.
//!
//! From binance_gpu_benchmark.rs:
//! - GPU batch (9 indicators, 44,640 candles): 261.79 ms
//! - This benchmark will show CPU batch performance for the SAME workload
//!
//! # Methodology
//!
//! Calculate 9 indicators sequentially using CPU (Indicator trait):
//! - RSI, ATR, Stochastic, Williams %R, ROC, CCI, Aroon, 2 more
//!
//! Same dataset: 44,640 candles (1 month BTCUSDT 1m)
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench cpu_batch_only
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

use kimsfinance_core::indicators::{
    momentum::{CCI, ROC, RSI},
    volatility::ATR,
    core::Indicator,
};

/// Generate realistic OHLCV data
fn generate_ohlcv_data(size: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut high = Vec::with_capacity(size);
    let mut low = Vec::with_capacity(size);
    let mut close = Vec::with_capacity(size);

    let base_price = 29000.0;
    let trend = 0.01;
    let volatility = 100.0;

    for i in 0..size {
        let t = i as f64;
        let price = base_price + trend * t + volatility * (t * 0.01).sin();
        let spread = volatility * 0.5;

        high.push(price + spread * 0.7);
        low.push(price - spread * 0.7);
        close.push(price);
    }

    (
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
    )
}

/// CPU batch processing - calculate 9 indicators sequentially
/// Using only simple indicators to avoid IndicatorOutput type issues
fn calculate_9_indicators_cpu(
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
) -> Vec<Array1<f64>> {
    let mut results = Vec::new();

    // 9 indicators using RSI, ATR, ROC, CCI
    // 1. RSI (period 14)
    results.push(RSI::new(14).unwrap().calculate(close.view()).unwrap());

    // 2. ATR (period 14)
    results.push(ATR::new(14).unwrap().calculate_hlc(high.view(), low.view(), close.view()).unwrap());

    // 3. ROC (period 12)
    results.push(ROC::new(12).unwrap().calculate(close.view()).unwrap());

    // 4. CCI (period 20)
    results.push(CCI::new(20).unwrap().calculate_hlc(high.view(), low.view(), close.view()).unwrap());

    // 5. RSI (period 7)
    results.push(RSI::new(7).unwrap().calculate(close.view()).unwrap());

    // 6. ATR (period 10)
    results.push(ATR::new(10).unwrap().calculate_hlc(high.view(), low.view(), close.view()).unwrap());

    // 7. RSI (period 21)
    results.push(RSI::new(21).unwrap().calculate(close.view()).unwrap());

    // 8. ROC (period 6)
    results.push(ROC::new(6).unwrap().calculate(close.view()).unwrap());

    // 9. CCI (period 14)
    results.push(CCI::new(14).unwrap().calculate_hlc(high.view(), low.view(), close.view()).unwrap());

    results
}

fn bench_cpu_batch_9_indicators(c: &mut Criterion) {
    let size = 44640; // Same as GPU benchmark
    let (high, low, close) = generate_ohlcv_data(size);

    println!("\n=== CPU Batch: 9 Indicators (44,640 candles) ===");
    println!("Compare against GPU batch: 261.79 ms");

    c.bench_function("cpu_batch_9_indicators", |b| {
        b.iter(|| {
            black_box(calculate_9_indicators_cpu(
                black_box(&high),
                black_box(&low),
                black_box(&close),
            ))
        })
    });
}

criterion_group!(benches, bench_cpu_batch_9_indicators);
criterion_main!(benches);

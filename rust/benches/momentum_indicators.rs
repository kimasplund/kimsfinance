//! Benchmarks for optimized momentum indicators
//!
//! Tests all 8 momentum indicators with different dataset sizes
//! to validate 3-5x performance improvement over NumPy baseline.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::indicators::*;
use ndarray::{Array1, arr1};
use std::f64::consts::PI;

/// Generate synthetic price data for benchmarking
fn generate_price_data(n: usize) -> Array1<f64> {
    let mut prices = Array1::zeros(n);
    let base_price = 100.0;

    for i in 0..n {
        let t = i as f64;
        // Realistic price movement: trend + oscillation + noise
        let trend = t * 0.01;
        let wave = 5.0 * (t * 2.0 * PI / 100.0).sin();
        let noise = (t * 1234.56).sin() * 0.5;
        prices[i] = base_price + trend + wave + noise;
    }

    prices
}

/// Generate OHLC data for indicators that need it
fn generate_ohlc_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let close = generate_price_data(n);
    let mut high = Array1::zeros(n);
    let mut low = Array1::zeros(n);

    for i in 0..n {
        high[i] = close[i] + ((i as f64 * 789.0).sin().abs() * 2.0);
        low[i] = close[i] - ((i as f64 * 456.0).sin().abs() * 2.0);
    }

    (high, low, close)
}

/// Benchmark RSI calculation
fn bench_rsi(c: &mut Criterion) {
    let mut group = c.benchmark_group("RSI");

    for size in [100, 500, 1000, 5000].iter() {
        let prices = generate_price_data(*size);
        let rsi = RSI::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = rsi.calculate(black_box(prices.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark ROC calculation
fn bench_roc(c: &mut Criterion) {
    let mut group = c.benchmark_group("ROC");

    for size in [100, 500, 1000, 5000].iter() {
        let prices = generate_price_data(*size);
        let roc = ROC::new(12).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = roc.calculate(black_box(prices.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark Williams %R calculation
fn bench_williams_r(c: &mut Criterion) {
    let mut group = c.benchmark_group("Williams_R");

    for size in [100, 500, 1000, 5000].iter() {
        let (high, low, close) = generate_ohlc_data(*size);
        let williams = WilliamsR::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = williams
                    .calculate_hlc(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                    )
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark Stochastic Oscillator calculation
fn bench_stochastic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stochastic");

    for size in [100, 500, 1000, 5000].iter() {
        let (high, low, close) = generate_ohlc_data(*size);
        let stoch = Stochastic::new(14, 3).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = stoch
                    .calculate_hlc(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                    )
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark Aroon Indicator calculation
fn bench_aroon(c: &mut Criterion) {
    let mut group = c.benchmark_group("Aroon");

    for size in [100, 500, 1000, 5000].iter() {
        let (high, low, _) = generate_ohlc_data(*size);
        let aroon = Aroon::new(25).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = aroon
                    .calculate_hl(black_box(high.view()), black_box(low.view()))
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark CCI calculation
fn bench_cci(c: &mut Criterion) {
    let mut group = c.benchmark_group("CCI");

    for size in [100, 500, 1000, 5000].iter() {
        let (high, low, close) = generate_ohlc_data(*size);
        let cci = CCI::new(20).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = cci
                    .calculate_hlc(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                    )
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark MACD calculation
fn bench_macd(c: &mut Criterion) {
    let mut group = c.benchmark_group("MACD");

    for size in [100, 500, 1000, 5000].iter() {
        let prices = generate_price_data(*size);
        let macd = MACD::new(12, 26, 9).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = macd.calculate_multi(black_box(prices.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark TSI calculation
fn bench_tsi(c: &mut Criterion) {
    let mut group = c.benchmark_group("TSI");

    for size in [100, 500, 1000, 5000].iter() {
        let prices = generate_price_data(*size);
        let tsi = TSI::new(25, 13, 13).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = tsi.calculate_multi(black_box(prices.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rsi,
    bench_roc,
    bench_williams_r,
    bench_stochastic,
    bench_aroon,
    bench_cci,
    bench_macd,
    bench_tsi
);

criterion_main!(benches);

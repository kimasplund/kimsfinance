//! Benchmark for candlestick pattern recognition
//!
//! Validates performance target of >1M candles/second

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::indicators::candlestick::{PatternConfig, recognize_patterns};

fn generate_ohlcv(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let mut price = 100.0;
    for _ in 0..n {
        let o = price;
        let c = price + (rand::random::<f64>() - 0.5) * 5.0;
        let h = o.max(c) + rand::random::<f64>() * 2.0;
        let l = o.min(c) - rand::random::<f64>() * 2.0;
        let v = 1000.0 + rand::random::<f64>() * 1000.0;

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(v);

        price = c;
    }

    (open, high, low, close, volume)
}

fn bench_pattern_recognition(c: &mut Criterion) {
    let mut group = c.benchmark_group("candlestick_patterns");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let (open, high, low, close, volume) = generate_ohlcv(*size);
        let config = PatternConfig::default();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let patterns = recognize_patterns(
                    black_box(&open),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(&volume),
                    black_box(&config),
                );
                black_box(patterns);
            });
        });
    }

    group.finish();
}

fn bench_config_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_variations");
    let (open, high, low, close, volume) = generate_ohlcv(10_000);

    group.throughput(Throughput::Elements(10_000));

    // Default config
    let config_default = PatternConfig::default();
    group.bench_function("default", |b| {
        b.iter(|| {
            recognize_patterns(
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                black_box(&volume),
                black_box(&config_default),
            )
        })
    });

    // Strict config
    let config_strict = PatternConfig::strict();
    group.bench_function("strict", |b| {
        b.iter(|| {
            recognize_patterns(
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                black_box(&volume),
                black_box(&config_strict),
            )
        })
    });

    // Relaxed config
    let config_relaxed = PatternConfig::relaxed();
    group.bench_function("relaxed", |b| {
        b.iter(|| {
            recognize_patterns(
                black_box(&open),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                black_box(&volume),
                black_box(&config_relaxed),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pattern_recognition, bench_config_variations);
criterion_main!(benches);

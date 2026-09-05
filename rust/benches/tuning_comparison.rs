//! Benchmark: Dynamic Threshold vs Static Threshold
//!
//! Measures performance improvement from dynamic threshold calculation.

#![cfg(feature = "gpu")]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::backtest::batch::calculate_optimal_threshold;
use kimsfinance_core::gpu::device::GpuDevice;
use std::sync::Arc;

/// Benchmark dynamic threshold calculation overhead
fn bench_threshold_calculation(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("GPU required"));

    c.bench_function("threshold_small_dataset", |b| {
        b.iter(|| {
            let threshold = calculate_optimal_threshold(black_box(10), black_box(1000));
            black_box(threshold);
        })
    });

    c.bench_function("threshold_medium_dataset", |b| {
        b.iter(|| {
            let threshold = calculate_optimal_threshold(black_box(500), black_box(5000));
            black_box(threshold);
        })
    });

    c.bench_function("threshold_large_dataset", |b| {
        b.iter(|| {
            let threshold = calculate_optimal_threshold(black_box(1000), black_box(10000));
            black_box(threshold);
        })
    });
}

criterion_group!(benches, bench_threshold_calculation);
criterion_main!(benches);

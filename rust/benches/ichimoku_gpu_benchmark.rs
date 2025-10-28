//! Benchmark: Ichimoku Cloud GPU vs CPU
//!
//! Validates the 8-20x speedup claim for GPU-accelerated Ichimoku Cloud.
//!
//! # Running
//!
//! ```bash
//! cargo bench --bench ichimoku_gpu_benchmark --features gpu
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, ichimoku_gpu};
use kimsfinance_core::indicators::core::Indicator;
use kimsfinance_core::indicators::trend::IchimokuCloud;
use ndarray::Array1;
use std::sync::Arc;

fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let high: Vec<f64> = (0..n)
        .map(|i| {
            let base = 110.0 + (i as f64 * 0.1);
            let noise = ((i as f64 * 0.05).sin() * 2.0);
            base + noise
        })
        .collect();

    let low: Vec<f64> = (0..n)
        .map(|i| {
            let base = 100.0 + (i as f64 * 0.1);
            let noise = ((i as f64 * 0.05).sin() * 2.0);
            base + noise
        })
        .collect();

    let close: Vec<f64> = (0..n)
        .map(|i| {
            let base = 105.0 + (i as f64 * 0.1);
            let noise = ((i as f64 * 0.05).sin() * 2.0);
            base + noise
        })
        .collect();

    (high, low, close)
}

fn bench_ichimoku_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ichimoku_cpu");

    for size in [1_000, 10_000, 100_000].iter() {
        let (high, low, close) = generate_test_data(*size);
        let high_arr = Array1::from_vec(high);
        let low_arr = Array1::from_vec(low);
        let close_arr = Array1::from_vec(close);

        let ichimoku = IchimokuCloud::new(9, 26, 52, 26).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                ichimoku
                    .calculate_hlc(
                        black_box(high_arr.view()),
                        black_box(low_arr.view()),
                        black_box(close_arr.view()),
                    )
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_ichimoku_gpu(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let mut group = c.benchmark_group("ichimoku_gpu");

    for size in [1_000, 10_000, 100_000].iter() {
        let (high, low, close) = generate_test_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                ichimoku_gpu(
                    Arc::clone(&device),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    None,
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_ichimoku_gpu_with_warmup(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let mut group = c.benchmark_group("ichimoku_gpu_warmup");

    // Warmup: compile kernels once
    let (warmup_high, warmup_low, warmup_close) = generate_test_data(1000);
    ichimoku_gpu(
        Arc::clone(&device),
        &warmup_high,
        &warmup_low,
        &warmup_close,
        None,
    )
    .expect("Warmup failed");

    for size in [1_000, 10_000, 100_000].iter() {
        let (high, low, close) = generate_test_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                ichimoku_gpu(
                    Arc::clone(&device),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    None,
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_ichimoku_comparison(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let mut group = c.benchmark_group("ichimoku_comparison");

    // Warmup GPU
    let (warmup_high, warmup_low, warmup_close) = generate_test_data(1000);
    ichimoku_gpu(
        Arc::clone(&device),
        &warmup_high,
        &warmup_low,
        &warmup_close,
        None,
    )
    .expect("Warmup failed");

    let size = 100_000;
    let (high, low, close) = generate_test_data(size);
    let high_arr = Array1::from_vec(high.clone());
    let low_arr = Array1::from_vec(low.clone());
    let close_arr = Array1::from_vec(close.clone());

    let ichimoku = IchimokuCloud::new(9, 26, 52, 26).unwrap();

    // CPU benchmark
    group.bench_function("cpu_100k", |b| {
        b.iter(|| {
            ichimoku
                .calculate_hlc(
                    black_box(high_arr.view()),
                    black_box(low_arr.view()),
                    black_box(close_arr.view()),
                )
                .unwrap()
        });
    });

    // GPU benchmark
    group.bench_function("gpu_100k", |b| {
        b.iter(|| {
            ichimoku_gpu(
                Arc::clone(&device),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                None,
            )
            .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ichimoku_cpu,
    bench_ichimoku_gpu,
    bench_ichimoku_gpu_with_warmup,
    bench_ichimoku_comparison
);
criterion_main!(benches);

//! Benchmark: Supertrend GPU vs CPU
//!
//! Compares GPU-accelerated vs CPU-only Supertrend implementation.
//!
//! Run with:
//! ```bash
//! cargo bench --bench supertrend_gpu_benchmark --features gpu
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, supertrend_gpu};
use kimsfinance_core::indicators::core::Indicator;
use kimsfinance_core::indicators::trend::Supertrend;
use ndarray::Array1;
use std::sync::Arc;

fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let high: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            110.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let low: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            100.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();
    let close: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            105.0 + 10.0 * (x * 0.1).sin()
        })
        .collect();

    (high, low, close)
}

fn bench_supertrend_gpu_vs_cpu(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes {
        let (high, low, close) = generate_test_data(size);
        let high_arr = Array1::from_vec(high.clone());
        let low_arr = Array1::from_vec(low.clone());
        let close_arr = Array1::from_vec(close.clone());

        let period = 10;
        let multiplier = 3.0;

        let mut group = c.benchmark_group(format!("supertrend_{}", size));
        group.throughput(Throughput::Elements(size as u64));

        // GPU benchmark
        group.bench_function(BenchmarkId::new("gpu", size), |b| {
            b.iter(|| {
                supertrend_gpu(
                    device.clone(),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(period),
                    black_box(multiplier),
                    None,
                )
                .unwrap()
            });
        });

        // CPU benchmark
        group.bench_function(BenchmarkId::new("cpu", size), |b| {
            let supertrend = Supertrend::new(period, multiplier).unwrap();
            b.iter(|| {
                supertrend
                    .calculate_hlc(
                        black_box(high_arr.view()),
                        black_box(low_arr.view()),
                        black_box(close_arr.view()),
                    )
                    .unwrap()
            });
        });

        group.finish();
    }
}

fn bench_supertrend_gpu_parameters(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let size = 100_000;
    let (high, low, close) = generate_test_data(size);

    let mut group = c.benchmark_group("supertrend_parameters");
    group.throughput(Throughput::Elements(size as u64));

    // Test different periods
    for period in [5, 10, 14, 20] {
        group.bench_function(BenchmarkId::new("period", period), |b| {
            b.iter(|| {
                supertrend_gpu(
                    device.clone(),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(period),
                    black_box(3.0),
                    None,
                )
                .unwrap()
            });
        });
    }

    // Test different multipliers
    for multiplier in [1.0, 2.0, 3.0, 4.0] {
        group.bench_function(
            BenchmarkId::new("multiplier", (multiplier * 10.0) as u32),
            |b| {
                b.iter(|| {
                    supertrend_gpu(
                        device.clone(),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(10),
                        black_box(multiplier),
                        None,
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_supertrend_gpu_warmup(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let size = 10_000;
    let (high, low, close) = generate_test_data(size);

    c.bench_function("supertrend_gpu_with_warmup", |b| {
        b.iter(|| {
            // This includes compilation time on first run
            supertrend_gpu(
                device.clone(),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                black_box(10),
                black_box(3.0),
                None,
            )
            .unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_supertrend_gpu_vs_cpu,
    bench_supertrend_gpu_parameters,
    bench_supertrend_gpu_warmup
);
criterion_main!(benches);

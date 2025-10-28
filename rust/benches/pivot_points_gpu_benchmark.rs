//! Benchmark: Pivot Points GPU vs CPU
//!
//! Measures performance of GPU-accelerated Pivot Points indicator
//! against CPU implementation across different dataset sizes.
//!
//! # Expected Results
//!
//! - Small datasets (1K): CPU faster (GPU overhead dominates)
//! - Medium datasets (10K): GPU 5-10x faster
//! - Large datasets (100K): GPU 15-30x faster
//!
//! # Run Benchmark
//!
//! ```bash
//! cargo bench --features gpu --bench pivot_points_gpu_benchmark
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, pivot_points_gpu};
use kimsfinance_core::indicators::trend::PivotPoints;
use ndarray::{Array1, arr1};
use std::sync::Arc;

/// Generate test OHLC data with realistic price movements
fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);

    let mut price = 100.0;

    for i in 0..n {
        // Simulate price movement with trend + noise
        let trend = (i as f64 * 0.01).sin() * 5.0;
        let noise = ((i * 7919) % 100) as f64 * 0.1 - 5.0; // Pseudo-random

        price += trend + noise;

        high.push(price + 2.0);
        low.push(price - 2.0);
        close.push(price);
    }

    (high, low, close)
}

/// CPU implementation for comparison
fn pivot_points_cpu(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let n = high.len();
    let mut pp = Array1::from_elem(n, f64::NAN);
    let mut r1 = Array1::from_elem(n, f64::NAN);
    let mut r2 = Array1::from_elem(n, f64::NAN);
    let mut r3 = Array1::from_elem(n, f64::NAN);
    let mut s1 = Array1::from_elem(n, f64::NAN);
    let mut s2 = Array1::from_elem(n, f64::NAN);
    let mut s3 = Array1::from_elem(n, f64::NAN);

    let pivot_calc = PivotPoints::new();

    // Calculate pivots using previous period's data
    for i in 1..n {
        let levels = pivot_calc.calculate_single(high[i - 1], low[i - 1], close[i - 1]);
        pp[i] = levels[0];
        r1[i] = levels[1];
        r2[i] = levels[2];
        r3[i] = levels[3];
        s1[i] = levels[4];
        s2[i] = levels[5];
        s3[i] = levels[6];
    }

    (pp, r1, r2, r3, s1, s2, s3)
}

/// Benchmark CPU implementation
fn bench_pivot_points_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("pivot_points_cpu");

    for size in [1_000, 10_000, 100_000].iter() {
        let (high, low, close) = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| pivot_points_cpu(black_box(&high), black_box(&low), black_box(&close)));
        });
    }

    group.finish();
}

/// Benchmark GPU implementation
fn bench_pivot_points_gpu(c: &mut Criterion) {
    // Initialize GPU device once
    let device = match GpuDevice::new() {
        Ok(d) => Arc::new(d),
        Err(_) => {
            eprintln!("GPU not available, skipping GPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("pivot_points_gpu");

    for size in [1_000, 10_000, 100_000].iter() {
        let (high, low, close) = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                pivot_points_gpu(
                    device.clone(),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    None,
                )
                .expect("GPU calculation failed")
            });
        });
    }

    group.finish();
}

/// Benchmark GPU vs CPU comparison
fn bench_pivot_points_comparison(c: &mut Criterion) {
    let device = match GpuDevice::new() {
        Ok(d) => Arc::new(d),
        Err(_) => {
            eprintln!("GPU not available, skipping comparison benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("pivot_points_comparison");

    for size in [1_000, 10_000, 100_000].iter() {
        let (high, low, close) = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // CPU benchmark
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |b, &_size| {
            b.iter(|| pivot_points_cpu(black_box(&high), black_box(&low), black_box(&close)));
        });

        // GPU benchmark
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |b, &_size| {
            b.iter(|| {
                pivot_points_gpu(
                    device.clone(),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    None,
                )
                .expect("GPU calculation failed")
            });
        });
    }

    group.finish();
}

/// Benchmark GPU memory transfer overhead
fn bench_pivot_points_gpu_transfers(c: &mut Criterion) {
    let device = match GpuDevice::new() {
        Ok(d) => Arc::new(d),
        Err(_) => {
            eprintln!("GPU not available, skipping transfer benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("pivot_points_gpu_transfers");

    let size = 100_000;
    let (high, low, close) = generate_test_data(size);

    group.throughput(Throughput::Bytes((size * 3 * 8) as u64)); // 3 inputs * 8 bytes

    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            pivot_points_gpu(
                device.clone(),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                None,
            )
            .expect("GPU calculation failed")
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pivot_points_cpu,
    bench_pivot_points_gpu,
    bench_pivot_points_comparison,
    bench_pivot_points_gpu_transfers
);
criterion_main!(benches);

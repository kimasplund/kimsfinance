//! Benchmark: MFI (Money Flow Index) GPU vs CPU
//!
//! Compares performance of GPU-accelerated MFI vs CPU-only implementation.
//!
//! # Expected Results
//!
//! - Small datasets (<1K): CPU faster (GPU overhead dominates)
//! - Medium datasets (1K-10K): GPU ~5-10x faster
//! - Large datasets (>10K): GPU ~10-20x faster
//!
//! # Performance Target
//!
//! GPU should achieve **10-20x speedup** for datasets >10K rows

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::indicators::volume::MFI;
use ndarray::Array1;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{GpuDevice, mfi_gpu};

fn generate_ohlcv(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    // Generate realistic oscillating data
    let high: Array1<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            105.0 + 5.0 * (x * 0.1).sin() + 0.5 * (x * 0.3).cos()
        })
        .collect();
    let low: Array1<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            95.0 + 5.0 * (x * 0.1).sin() + 0.5 * (x * 0.3).cos()
        })
        .collect();
    let close: Array1<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            100.0 + 5.0 * (x * 0.1).sin() + 0.5 * (x * 0.3).cos()
        })
        .collect();
    let volume: Array1<f64> = (0..n)
        .map(|i| {
            let base = 1000.0 + (i % 500) as f64;
            let noise = (i % 100) as f64 * 10.0;
            base + noise
        })
        .collect();

    (high, low, close, volume)
}

/// Benchmark CPU-only MFI implementation
fn bench_mfi_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("MFI_CPU");

    // Test various dataset sizes
    for size in [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000].iter() {
        let (high, low, close, volume) = generate_ohlcv(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let mfi = MFI::new(14).unwrap();
            b.iter(|| {
                black_box(
                    mfi.calculate_hlcv(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                        black_box(volume.view()),
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark GPU-accelerated MFI implementation
#[cfg(feature = "gpu")]
fn bench_mfi_gpu(c: &mut Criterion) {
    // Initialize GPU device once
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("MFI_GPU");

    // Test various dataset sizes
    for size in [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000].iter() {
        let (high, low, close, volume) = generate_ohlcv(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(
                    mfi_gpu(
                        &device,
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                        14,
                        None,
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark comparison: CPU vs GPU side-by-side
#[cfg(feature = "gpu")]
fn bench_mfi_comparison(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("MFI_Comparison");

    // Focus on large datasets where GPU should excel
    for size in [10_000, 50_000, 100_000].iter() {
        let (high, low, close, volume) = generate_ohlcv(*size);

        // CPU benchmark
        group.bench_with_input(BenchmarkId::new("CPU", size), size, |b, _| {
            let mfi = MFI::new(14).unwrap();
            b.iter(|| {
                black_box(
                    mfi.calculate_hlcv(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                        black_box(volume.view()),
                    )
                    .unwrap(),
                )
            });
        });

        // GPU benchmark
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    mfi_gpu(
                        &device,
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                        14,
                        None,
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark different MFI periods
#[cfg(feature = "gpu")]
fn bench_mfi_periods(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("MFI_Periods");

    let size = 100_000;
    let (high, low, close, volume) = generate_ohlcv(size);

    // Test different periods (affects rolling window size)
    for period in [7, 14, 21, 28, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(period), period, |b, &period| {
            b.iter(|| {
                black_box(
                    mfi_gpu(
                        &device,
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                        period,
                        None,
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark throughput (candles/second)
#[cfg(feature = "gpu")]
fn bench_mfi_throughput(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("MFI_Throughput");
    group.sample_size(20); // Reduce sample size for large datasets

    let size = 100_000;
    let (high, low, close, volume) = generate_ohlcv(size);

    group.bench_function("GPU_100K_candles", |b| {
        b.iter(|| {
            black_box(
                mfi_gpu(
                    &device,
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(&volume),
                    14,
                    None,
                )
                .unwrap(),
            )
        });
    });

    group.finish();
}

// Criterion benchmark groups
#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_mfi_cpu,
    bench_mfi_gpu,
    bench_mfi_comparison,
    bench_mfi_periods,
    bench_mfi_throughput
);

#[cfg(not(feature = "gpu"))]
criterion_group!(benches, bench_mfi_cpu);

criterion_main!(benches);

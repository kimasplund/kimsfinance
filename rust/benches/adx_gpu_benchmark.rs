//! Benchmark: ADX GPU vs CPU Performance
//!
//! Compares GPU-accelerated ADX against CPU-only implementation.
//!
//! Expected Results (100K candles):
//! - GPU (hybrid): ~180-200μs
//! - CPU-only: ~1800-2000μs
//! - Speedup: 8-12x
//!
//! Run with: cargo bench --bench adx_gpu_benchmark --features gpu

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, adx_gpu};
use ndarray::Array1;

// CPU-only baseline implementation for comparison
fn adx_cpu_baseline(
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    period: usize,
) -> Array1<f64> {
    use kimsfinance_core::cpu::sequential::wilders_smoothing_cpu;

    let n = high.len();
    let mut plus_dm = Array1::zeros(n);
    let mut minus_dm = Array1::zeros(n);
    let mut true_range = Array1::zeros(n);

    // Calculate DM and TR (sequential)
    for i in 0..n {
        if i == 0 {
            plus_dm[i] = 0.0;
            minus_dm[i] = 0.0;
            true_range[i] = high[i] - low[i];
        } else {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > 0.0 {
                plus_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                minus_dm[i] = down_move;
            }

            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            true_range[i] = hl.max(hc).max(lc);
        }
    }

    // Wilder's smoothing
    let plus_dm_smooth = wilders_smoothing_cpu(&plus_dm, period).unwrap();
    let minus_dm_smooth = wilders_smoothing_cpu(&minus_dm, period).unwrap();
    let tr_smooth = wilders_smoothing_cpu(&true_range, period).unwrap();

    // Calculate DI
    let mut plus_di = Array1::from_elem(n, f64::NAN);
    let mut minus_di = Array1::from_elem(n, f64::NAN);

    for i in period..n {
        if tr_smooth[i] > 1e-10 {
            plus_di[i] = 100.0 * (plus_dm_smooth[i] / tr_smooth[i]);
            minus_di[i] = 100.0 * (minus_dm_smooth[i] / tr_smooth[i]);
        }
    }

    // Calculate DX
    let mut dx = Array1::from_elem(n, f64::NAN);
    for i in period..n {
        if !plus_di[i].is_nan() && !minus_di[i].is_nan() {
            let di_sum = plus_di[i] + minus_di[i];
            if di_sum > 1e-10 {
                let di_diff = (plus_di[i] - minus_di[i]).abs();
                dx[i] = 100.0 * (di_diff / di_sum);
            }
        }
    }

    // ADX = Wilder's smoothing of DX
    wilders_smoothing_cpu(&dx, period).unwrap()
}

fn bench_adx_sizes(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU device initialization failed");

    let mut group = c.benchmark_group("adx_comparison");

    // Test with different dataset sizes
    let sizes = [1_000, 10_000, 100_000];
    let period = 14;

    for &size in &sizes {
        // Generate trending data with noise
        let high = Array1::from_vec(
            (0..size)
                .map(|i| {
                    let trend = 100.0 + (i as f64) * 0.01;
                    let noise = ((i * 7) % 100) as f64 * 0.05;
                    trend + noise + 2.0
                })
                .collect(),
        );
        let low = Array1::from_vec(
            (0..size)
                .map(|i| {
                    let trend = 100.0 + (i as f64) * 0.01;
                    let noise = ((i * 7) % 100) as f64 * 0.05;
                    trend + noise - 2.0
                })
                .collect(),
        );
        let close = Array1::from_vec(
            (0..size)
                .map(|i| {
                    let trend = 100.0 + (i as f64) * 0.01;
                    let noise = ((i * 7) % 100) as f64 * 0.05;
                    trend + noise
                })
                .collect(),
        );

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark GPU implementation
        group.bench_with_input(BenchmarkId::new("GPU", size), &size, |b, _| {
            b.iter(|| {
                adx_gpu(
                    black_box(&device),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(period),
                    None,
                )
                .unwrap()
            });
        });

        // Benchmark CPU baseline
        group.bench_with_input(BenchmarkId::new("CPU", size), &size, |b, _| {
            b.iter(|| {
                adx_cpu_baseline(
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(period),
                )
            });
        });
    }

    group.finish();
}

fn bench_adx_periods(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU device initialization failed");

    let mut group = c.benchmark_group("adx_period_variation");

    let size = 100_000;
    let periods = [7, 14, 21, 28];

    // Generate consistent data
    let high = Array1::from_vec((0..size).map(|i| 100.0 + i as f64 * 0.01).collect());
    let low = Array1::from_vec((0..size).map(|i| 98.0 + i as f64 * 0.01).collect());
    let close = Array1::from_vec((0..size).map(|i| 99.0 + i as f64 * 0.01).collect());

    for &period in &periods {
        group.throughput(Throughput::Elements(size as u64));

        // GPU benchmark
        group.bench_with_input(BenchmarkId::new("GPU", period), &period, |b, _| {
            b.iter(|| {
                adx_gpu(
                    black_box(&device),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(period),
                    None,
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_adx_throughput(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU device initialization failed");

    let size = 100_000;
    let period = 14;

    // Generate data
    let high = Array1::from_vec((0..size).map(|i| 100.0 + i as f64 * 0.01).collect());
    let low = Array1::from_vec((0..size).map(|i| 98.0 + i as f64 * 0.01).collect());
    let close = Array1::from_vec((0..size).map(|i| 99.0 + i as f64 * 0.01).collect());

    c.bench_function("adx_throughput_100k", |b| {
        b.iter(|| {
            adx_gpu(
                black_box(&device),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                black_box(period),
                None,
            )
            .unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_adx_sizes,
    bench_adx_periods,
    bench_adx_throughput
);
criterion_main!(benches);

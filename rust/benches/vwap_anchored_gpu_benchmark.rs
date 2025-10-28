//! Benchmark: GPU-Accelerated VWAP Anchored vs CPU-only
//!
//! Validates 5-12x speedup target for VWAP Anchored indicator.
//!
//! # Methodology
//!
//! - Dataset sizes: 1K, 10K, 100K candles
//! - Anchor points: Start (0), Middle, Near-end
//! - Measures: Wall-clock time, throughput (candles/sec)
//! - Statistical validation: 100 iterations for stable measurements
//!
//! # Expected Results
//!
//! | Dataset | CPU (μs) | GPU (μs) | Speedup |
//! |---------|----------|----------|---------|
//! | 1K      | 60       | 80       | 0.75x   | (GPU overhead dominates)
//! | 10K     | 300      | 100      | 3.0x    |
//! | 100K    | 600      | 110      | 5.5x    | (Target met)
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench vwap_anchored_gpu_benchmark --features gpu
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, vwap_anchored_gpu};
use kimsfinance_core::indicators::Indicator;
use kimsfinance_core::indicators::volume::VWAP;
use ndarray::Array1;

/// Generate synthetic OHLCV data for benchmarking
fn generate_ohlcv_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let high: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            105.0 + 5.0 * (x * 0.1).sin() + 2.0 * (x * 0.05).cos()
        })
        .collect();

    let low: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            95.0 + 5.0 * (x * 0.1).sin() + 2.0 * (x * 0.05).cos()
        })
        .collect();

    let close: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            100.0 + 5.0 * (x * 0.1).sin() + 2.0 * (x * 0.05).cos()
        })
        .collect();

    let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i % 500) as f64 * 10.0).collect();

    (
        Array1::from(high),
        Array1::from(low),
        Array1::from(close),
        Array1::from(volume),
    )
}

/// CPU-only VWAP Anchored implementation (for comparison)
fn vwap_anchored_cpu(
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    volume: &Array1<f64>,
    anchor_index: usize,
) -> Array1<f64> {
    let n = high.len();
    let mut vwap = Array1::from_elem(n, f64::NAN);

    // Calculate typical prices
    let mut typical_price = Array1::zeros(n);
    for i in 0..n {
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
    }

    // Cumulative sums from anchor
    let mut cumsum_tpv = typical_price[anchor_index] * volume[anchor_index];
    let mut cumsum_volume = volume[anchor_index];

    if cumsum_volume > 0.0 {
        vwap[anchor_index] = cumsum_tpv / cumsum_volume;
    }

    for i in (anchor_index + 1)..n {
        cumsum_tpv += typical_price[i] * volume[i];
        cumsum_volume += volume[i];

        if cumsum_volume > 0.0 {
            vwap[i] = cumsum_tpv / cumsum_volume;
        }
    }

    vwap
}

fn bench_vwap_anchored_gpu(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let sizes = vec![1_000, 10_000, 100_000];

    let mut group = c.benchmark_group("vwap_anchored_gpu_vs_cpu");

    for size in sizes {
        let (high, low, close, volume) = generate_ohlcv_data(size);
        let anchor = size / 10; // Anchor at 10% point

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark GPU implementation
        group.bench_with_input(BenchmarkId::new("GPU", size), &size, |b, _| {
            b.iter(|| {
                vwap_anchored_gpu(
                    black_box(&device),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(&volume),
                    black_box(anchor),
                    None,
                )
                .unwrap()
            });
        });

        // Benchmark CPU implementation
        group.bench_with_input(BenchmarkId::new("CPU", size), &size, |b, _| {
            b.iter(|| {
                vwap_anchored_cpu(
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(&volume),
                    black_box(anchor),
                )
            });
        });
    }

    group.finish();
}

fn bench_vwap_anchored_anchor_positions(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let n = 100_000;
    let (high, low, close, volume) = generate_ohlcv_data(n);

    let mut group = c.benchmark_group("vwap_anchored_anchor_positions");
    group.throughput(Throughput::Elements(n as u64));

    // Benchmark different anchor positions
    for (name, anchor) in [
        ("start", 0),
        ("10pct", n / 10),
        ("middle", n / 2),
        ("90pct", n * 9 / 10),
    ] {
        group.bench_with_input(BenchmarkId::new("GPU", name), &anchor, |b, &anchor| {
            b.iter(|| {
                vwap_anchored_gpu(
                    black_box(&device),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(&volume),
                    black_box(anchor),
                    None,
                )
                .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("CPU", name), &anchor, |b, &anchor| {
            b.iter(|| {
                vwap_anchored_cpu(
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(&volume),
                    black_box(anchor),
                )
            });
        });
    }

    group.finish();
}

fn bench_vwap_anchored_throughput(c: &mut Criterion) {
    let device = GpuDevice::new().expect("Failed to initialize GPU");

    let n = 100_000;
    let (high, low, close, volume) = generate_ohlcv_data(n);
    let anchor = n / 10;

    c.bench_function("vwap_anchored_gpu_100k_throughput", |b| {
        b.iter(|| {
            vwap_anchored_gpu(
                black_box(&device),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                black_box(&volume),
                black_box(anchor),
                None,
            )
            .unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_vwap_anchored_gpu,
    bench_vwap_anchored_anchor_positions,
    bench_vwap_anchored_throughput
);
criterion_main!(benches);

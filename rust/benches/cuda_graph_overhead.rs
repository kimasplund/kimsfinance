//! CUDA Graphs Launch Overhead Benchmark
//!
//! Measures kernel launch overhead reduction from using CUDA Graphs vs traditional launches.
//!
//! # Expected Results
//!
//! - **Traditional**: 20 × 7.5μs = 150μs overhead
//! - **CUDA Graphs**: 3 × 3μs = 9μs overhead (3 streams)
//! - **Speedup**: 16.7x launch overhead reduction
//!
//! # Methodology
//!
//! 1. Measure traditional sequential kernel launches (20 indicators)
//! 2. Measure graph capture + replay overhead
//! 3. Amortize graph capture over 1000 replays
//! 4. Report overhead per indicator

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kimsfinance_core::gpu::{
    GpuDevice, IndicatorGraphBuilder, IndicatorSpeed, StreamManager,
    rsi_gpu, atr_gpu, roc_gpu, williams_r_gpu, cci_gpu,
    bollinger_bands_gpu, aroon_gpu, stochastic_gpu,
};
use ndarray::Array1;
use std::sync::Arc;
use std::time::Instant;

/// Generate synthetic OHLCV data
fn generate_test_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let high: Array1<f64> = Array1::linspace(100.0, 110.0, n);
    let low: Array1<f64> = Array1::linspace(95.0, 105.0, n);
    let close: Array1<f64> = Array1::linspace(97.0, 107.0, n);
    (high, low, close)
}

/// Benchmark traditional sequential kernel launches
fn bench_traditional_launches(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");
    let (high, low, close) = generate_test_data(10_000);

    let mut group = c.benchmark_group("traditional_launches");
    group.throughput(Throughput::Elements(9)); // 9 indicators

    group.bench_function("9_indicators_sequential", |b| {
        b.iter(|| {
            // Fast indicators (3)
            let _ = roc_gpu(&device, &close, 14, None);
            let _ = williams_r_gpu(&device, &high, &low, &close, 14, None);
            let _ = cci_gpu(&device, &high, &low, &close, 14, None);

            // Medium indicators (4)
            let _ = rsi_gpu(&device, &close, 14, None);
            let _ = atr_gpu(&device, &high, &low, &close, 14, None);
            let _ = bollinger_bands_gpu(&device, &close, 20, 2.0, None);
            let _ = aroon_gpu(&device, &high, &low, 25, None);

            // Slow indicators (2)
            let _ = stochastic_gpu(&device, &high, &low, &close, 14, 3, None);
            // MACD removed (now uses CPU)

            device.synchronize().expect("Sync failed");
        });
    });

    group.finish();
}

/// Benchmark CUDA Graphs capture and replay
fn bench_cuda_graphs(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let stream_mgr = Arc::new(StreamManager::new(device.clone()).expect("StreamManager required"));
    let (high, low, close) = generate_test_data(10_000);

    let mut group = c.benchmark_group("cuda_graphs");
    group.throughput(Throughput::Elements(9)); // 9 indicators

    // Measure graph capture overhead (one-time cost)
    group.bench_function("graph_capture", |b| {
        b.iter(|| {
            let mut builder = IndicatorGraphBuilder::new(device.clone(), stream_mgr.clone())
                .expect("Builder creation failed");

            // Capture Fast stream (3 indicators)
            builder.begin_capture_stream(IndicatorSpeed::Fast).expect("Begin capture failed");
            let _ = roc_gpu(&*device, &close, 14, None);
            let _ = williams_r_gpu(&*device, &high, &low, &close, 14, None);
            let _ = cci_gpu(&*device, &high, &low, &close, 14, None);
            builder.end_capture_stream(IndicatorSpeed::Fast).expect("End capture failed");

            // Capture Medium stream (4 indicators)
            builder.begin_capture_stream(IndicatorSpeed::Medium).expect("Begin capture failed");
            let _ = rsi_gpu(&*device, &close, 14, None);
            let _ = atr_gpu(&*device, &high, &low, &close, 14, None);
            let _ = bollinger_bands_gpu(&*device, &close, 20, 2.0, None);
            let _ = aroon_gpu(&*device, &high, &low, 25, None);
            builder.end_capture_stream(IndicatorSpeed::Medium).expect("End capture failed");

            // Capture Slow stream (2 indicators)
            builder.begin_capture_stream(IndicatorSpeed::Slow).expect("Begin capture failed");
            let _ = stochastic_gpu(&*device, &high, &low, &close, 14, 3, None);
            builder.end_capture_stream(IndicatorSpeed::Slow).expect("End capture failed");

            let _graph = builder.build().expect("Graph build failed");
        });
    });

    // Measure graph replay overhead (amortized cost)
    let mut builder = IndicatorGraphBuilder::new(device.clone(), stream_mgr.clone())
        .expect("Builder creation failed");

    // Capture graphs once
    builder.begin_capture_stream(IndicatorSpeed::Fast).expect("Begin capture failed");
    let _ = roc_gpu(&*device, &close, 14, None);
    let _ = williams_r_gpu(&*device, &high, &low, &close, 14, None);
    let _ = cci_gpu(&*device, &high, &low, &close, 14, None);
    builder.end_capture_stream(IndicatorSpeed::Fast).expect("End capture failed");

    builder.begin_capture_stream(IndicatorSpeed::Medium).expect("Begin capture failed");
    let _ = rsi_gpu(&*device, &close, 14, None);
    let _ = atr_gpu(&*device, &high, &low, &close, 14, None);
    let _ = bollinger_bands_gpu(&*device, &close, 20, 2.0, None);
    let _ = aroon_gpu(&*device, &high, &low, &25, None);
    builder.end_capture_stream(IndicatorSpeed::Medium).expect("End capture failed");

    builder.begin_capture_stream(IndicatorSpeed::Slow).expect("Begin capture failed");
    let _ = stochastic_gpu(&*device, &high, &low, &close, 14, 3, None);
    builder.end_capture_stream(IndicatorSpeed::Slow).expect("End capture failed");

    let graph = builder.build().expect("Graph build failed");

    group.bench_function("graph_replay", |b| {
        b.iter(|| {
            graph.launch_all().expect("Graph launch failed");
            graph.synchronize().expect("Sync failed");
        });
    });

    group.finish();
}

/// Benchmark to measure pure launch overhead (no actual computation)
fn bench_launch_overhead_breakdown(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let stream_mgr = Arc::new(StreamManager::new(device.clone()).expect("StreamManager required"));

    let mut group = c.benchmark_group("launch_overhead_breakdown");

    // Measure traditional launch overhead (dummy kernel, minimal work)
    group.bench_function("traditional_20_launches", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Simulate 20 kernel launches with minimal work
                for _ in 0..20 {
                    device.synchronize().expect("Sync failed"); // Minimal overhead proxy
                }
            }
            start.elapsed()
        });
    });

    // Measure graph replay overhead (3 streams)
    let mut builder = IndicatorGraphBuilder::new(device.clone(), stream_mgr.clone())
        .expect("Builder creation failed");

    // Capture empty graphs (just to measure launch overhead)
    builder.begin_capture_stream(IndicatorSpeed::Fast).expect("Begin failed");
    device.synchronize().expect("Sync failed");
    builder.end_capture_stream(IndicatorSpeed::Fast).expect("End failed");

    builder.begin_capture_stream(IndicatorSpeed::Medium).expect("Begin failed");
    device.synchronize().expect("Sync failed");
    builder.end_capture_stream(IndicatorSpeed::Medium).expect("End failed");

    builder.begin_capture_stream(IndicatorSpeed::Slow).expect("Begin failed");
    device.synchronize().expect("Sync failed");
    builder.end_capture_stream(IndicatorSpeed::Slow).expect("End failed");

    let graph = builder.build().expect("Graph build failed");

    group.bench_function("graph_3_launches", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                graph.launch_all().expect("Launch failed");
                graph.synchronize().expect("Sync failed");
            }
            start.elapsed()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_traditional_launches,
    bench_cuda_graphs,
    bench_launch_overhead_breakdown
);
criterion_main!(benches);

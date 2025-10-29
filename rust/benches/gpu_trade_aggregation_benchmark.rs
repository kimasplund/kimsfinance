//! GPU Trade Aggregation Benchmark
//!
//! Measures CPU vs GPU performance for OHLCV candle aggregation across
//! various dataset sizes and timeframes.
//!
//! # Benchmark Scenarios
//!
//! 1. **Scalability**: 1K, 10K, 50K, 100K, 500K, 1M trades
//! 2. **Timeframes**: 1m, 5m, 1h, 1d
//! 3. **Candle Distribution**: Single candle vs many candles
//!
//! # Expected Results (RTX 3500 Ada)
//!
//! - **<10K trades**: CPU faster (kernel overhead)
//! - **10-100K**: GPU 2-5x faster
//! - **>100K**: GPU 5-10x faster
//! - **Crossover point**: ~10K-20K trades

#![cfg(feature = "gpu")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kimsfinance_core::binance::{aggregate_trades_to_candles, Timeframe, Trade};
use kimsfinance_core::gpu::GpuAggregator;
use std::time::Duration;

/// Generate realistic test trades
fn generate_trades(n: usize, num_candles: usize, timeframe_minutes: i64) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(n);

    let base_price = 50_000.0;
    let base_time = 1_600_000_000_000i64;
    let timeframe_ms = timeframe_minutes * 60 * 1000;

    for i in 0..n {
        // Distribute trades across candles
        let candle_idx = (i * num_candles) / n;
        let timestamp = base_time + (candle_idx as i64 * timeframe_ms) + (i as i64 * 10);

        // Price variation within candle
        let price_variation = ((i % 100) as f64 / 100.0) * 100.0 - 50.0;
        let price = base_price + price_variation;

        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity: 0.1,
            quote_quantity: price * 0.1,
            timestamp_ms: timestamp,
            is_buyer_maker: i % 2 == 0,
        });
    }

    trades
}

/// Benchmark CPU aggregation
fn bench_cpu_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_aggregation");
    group.measurement_time(Duration::from_secs(10));

    for &size in &[1_000, 10_000, 50_000, 100_000, 500_000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let trades = generate_trades(size, 100, 5); // 100 candles, 5min timeframe
            let timeframe = Timeframe::minutes(5);

            b.iter(|| {
                let candles = aggregate_trades_to_candles(black_box(&trades), timeframe);
                black_box(candles);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU aggregation
fn bench_gpu_aggregation(c: &mut Criterion) {
    let aggregator = match GpuAggregator::new() {
        Ok(agg) => agg,
        Err(_) => {
            println!("GPU not available, skipping GPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_aggregation");
    group.measurement_time(Duration::from_secs(10));

    for &size in &[1_000, 10_000, 50_000, 100_000, 500_000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let trades = generate_trades(size, 100, 5);
            let timeframe = Timeframe::minutes(5);

            b.iter(|| {
                let candles = aggregator
                    .aggregate_trades(black_box(&trades), timeframe)
                    .expect("GPU aggregation failed");
                black_box(candles);
            });
        });
    }

    group.finish();
}

/// Benchmark CPU vs GPU comparison
fn bench_cpu_vs_gpu(c: &mut Criterion) {
    let aggregator = match GpuAggregator::new() {
        Ok(agg) => agg,
        Err(_) => {
            println!("GPU not available, skipping CPU vs GPU comparison");
            return;
        }
    };

    let mut group = c.benchmark_group("cpu_vs_gpu");
    group.measurement_time(Duration::from_secs(15));

    for &size in &[10_000, 50_000, 100_000] {
        let trades = generate_trades(size, 100, 5);
        let timeframe = Timeframe::minutes(5);

        // CPU benchmark
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let candles = aggregate_trades_to_candles(black_box(&trades), timeframe);
                    black_box(candles);
                });
            },
        );

        // GPU benchmark
        group.bench_with_input(
            BenchmarkId::new("gpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let candles = aggregator
                        .aggregate_trades(black_box(&trades), timeframe)
                        .expect("GPU aggregation failed");
                    black_box(candles);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different timeframes
fn bench_timeframes(c: &mut Criterion) {
    let aggregator = match GpuAggregator::new() {
        Ok(agg) => agg,
        Err(_) => {
            println!("GPU not available, skipping timeframe benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("timeframes");
    group.measurement_time(Duration::from_secs(10));

    let size = 100_000;
    let trades = generate_trades(size, 1000, 5);

    for (name, timeframe) in &[
        ("1m", Timeframe::minutes(1)),
        ("5m", Timeframe::minutes(5)),
        ("1h", Timeframe::hours(1)),
        ("1d", Timeframe::days(1)),
    ] {
        // CPU
        group.bench_with_input(
            BenchmarkId::new("cpu", name),
            timeframe,
            |b, &timeframe| {
                b.iter(|| {
                    let candles = aggregate_trades_to_candles(black_box(&trades), timeframe);
                    black_box(candles);
                });
            },
        );

        // GPU
        group.bench_with_input(
            BenchmarkId::new("gpu", name),
            timeframe,
            |b, &timeframe| {
                b.iter(|| {
                    let candles = aggregator
                        .aggregate_trades(black_box(&trades), timeframe)
                        .expect("GPU aggregation failed");
                    black_box(candles);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark candle distribution (few vs many candles)
fn bench_candle_distribution(c: &mut Criterion) {
    let aggregator = match GpuAggregator::new() {
        Ok(agg) => agg,
        Err(_) => {
            println!("GPU not available, skipping candle distribution benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("candle_distribution");
    group.measurement_time(Duration::from_secs(10));

    let size = 100_000;
    let timeframe = Timeframe::minutes(5);

    for &num_candles in &[1, 10, 100, 1000] {
        let trades = generate_trades(size, num_candles, 5);

        // CPU
        group.bench_with_input(
            BenchmarkId::new("cpu", num_candles),
            &num_candles,
            |b, _| {
                b.iter(|| {
                    let candles = aggregate_trades_to_candles(black_box(&trades), timeframe);
                    black_box(candles);
                });
            },
        );

        // GPU
        group.bench_with_input(
            BenchmarkId::new("gpu", num_candles),
            &num_candles,
            |b, _| {
                b.iter(|| {
                    let candles = aggregator
                        .aggregate_trades(black_box(&trades), timeframe)
                        .expect("GPU aggregation failed");
                    black_box(candles);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cpu_aggregation,
    bench_gpu_aggregation,
    bench_cpu_vs_gpu,
    bench_timeframes,
    bench_candle_distribution
);
criterion_main!(benches);

//! Parameter Sweep Batch API Benchmark
//!
//! Validates performance targets for parameter optimization workflows:
//! - 10 parameters: 10-15x speedup vs sequential
//! - 50 parameters: 20-30x speedup vs sequential
//! - 100 parameters: 30-50x speedup vs sequential
//!
//! # Scenarios
//!
//! 1. **RSI Period Sweep**: Find optimal RSI period (10-100)
//! 2. **SMA Period Sweep**: Find optimal SMA period (10-200)
//! 3. **Multiple Indicators**: Compare sweep performance across indicators
//! 4. **Optimization Metrics**: Benchmark metric calculation overhead
//!
//! # Performance Metrics
//!
//! - Sequential execution time (N individual GPU calls)
//! - Batch execution time (sweep API)
//! - Speedup ratio (sequential / batch)
//! - Throughput (parameters/sec)
//! - Metric calculation overhead

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{
    GpuDevice, IndicatorData, IndicatorType, OptimizationMetric, ParameterSweep, rsi_gpu, sma_gpu,
};

/// Generate realistic price data with trend and volatility
fn generate_price_data(n: usize) -> Array1<f64> {
    use std::f64::consts::PI;

    let mut prices = Vec::with_capacity(n);
    let base_price = 100.0;

    for i in 0..n {
        let t = i as f64;
        // Trend + sine wave volatility + noise
        let trend = base_price + t * 0.01;
        let cycle = 5.0 * (2.0 * PI * t / 50.0).sin();
        let noise = (t * 0.1).sin() * 0.5;
        prices.push(trend + cycle + noise);
    }

    Array1::from_vec(prices)
}

/// Generate OHLC data
fn generate_ohlc_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let close = generate_price_data(n);
    let high = close.mapv(|x| x + 1.0);
    let low = close.mapv(|x| x - 1.0);
    let open = close.clone();
    (open, high, low, close)
}

#[cfg(feature = "gpu")]
fn benchmark_rsi_sweep_10_params(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let close = generate_price_data(10_000);

    let mut group = c.benchmark_group("rsi_sweep_10_params");

    // Sequential: 10 individual GPU calls
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for period in 10..=19 {
                let _result = rsi_gpu(&device, black_box(&close), period, None)
                    .expect("RSI calculation failed");
            }
        })
    });

    // Batch: Parameter sweep API
    group.bench_function("batch_sweep", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=19)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_rsi_sweep_50_params(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let close = generate_price_data(10_000);

    let mut group = c.benchmark_group("rsi_sweep_50_params");

    // Sequential: 50 individual GPU calls
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for period in 10..=59 {
                let _result = rsi_gpu(&device, black_box(&close), period, None)
                    .expect("RSI calculation failed");
            }
        })
    });

    // Batch: Parameter sweep API
    group.bench_function("batch_sweep", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=59)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_rsi_sweep_100_params(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let close = generate_price_data(10_000);

    let mut group = c.benchmark_group("rsi_sweep_100_params");
    group.sample_size(10); // Reduce sample size for long benchmark

    // Sequential: 100 individual GPU calls
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for period in 10..=109 {
                let _result = rsi_gpu(&device, black_box(&close), period, None)
                    .expect("RSI calculation failed");
            }
        })
    });

    // Batch: Parameter sweep API
    group.bench_function("batch_sweep", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=109)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_sma_sweep_50_params(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let close = generate_price_data(10_000);

    let mut group = c.benchmark_group("sma_sweep_50_params");

    // Sequential: 50 individual GPU calls
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for period in 10..=59 {
                let _result = sma_gpu(&device, black_box(&close), period, None)
                    .expect("SMA calculation failed");
            }
        })
    });

    // Batch: Parameter sweep API
    group.bench_function("batch_sweep", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::SMA)
                .parameter_range(10..=59)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_sweep_with_metrics(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let close = generate_price_data(10_000);

    let mut group = c.benchmark_group("sweep_with_metrics");

    // Sweep without metrics
    group.bench_function("no_metric", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=30)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    // Sweep with Sharpe ratio
    group.bench_function("sharpe_metric", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=30)
                .data_close(black_box(&close))
                .metric(OptimizationMetric::Sharpe)
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    // Sweep with max drawdown
    group.bench_function("drawdown_metric", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=30)
                .data_close(black_box(&close))
                .metric(OptimizationMetric::MaxDrawdown)
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    // Sweep with win rate
    group.bench_function("winrate_metric", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=30)
                .data_close(black_box(&close))
                .metric(OptimizationMetric::WinRate)
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    // Sweep with profit factor
    group.bench_function("profit_factor_metric", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=30)
                .data_close(black_box(&close))
                .metric(OptimizationMetric::ProfitFactor)
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_sweep_scalability(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let mut group = c.benchmark_group("sweep_scalability");

    for data_size in [1_000, 5_000, 10_000, 50_000].iter() {
        let close = generate_price_data(*data_size);

        group.bench_with_input(
            BenchmarkId::new("rsi_20_params", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _sweep = ParameterSweep::new(device.clone())
                        .indicator(IndicatorType::RSI)
                        .parameter_range(10..=29)
                        .data_close(black_box(&close))
                        .execute()
                        .expect("Parameter sweep failed");
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_multi_indicator_sweep(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let close = generate_price_data(10_000);

    let mut group = c.benchmark_group("multi_indicator_sweep");

    // RSI sweep
    group.bench_function("rsi_20_params", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=29)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    // SMA sweep
    group.bench_function("sma_20_params", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::SMA)
                .parameter_range(10..=29)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    // EMA sweep
    group.bench_function("ema_20_params", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::EMA)
                .parameter_range(10..=29)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    // WMA sweep
    group.bench_function("wma_20_params", |b| {
        b.iter(|| {
            let _sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::WMA)
                .parameter_range(10..=29)
                .data_close(black_box(&close))
                .execute()
                .expect("Parameter sweep failed");
        })
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_optimization_workflow(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let close = generate_price_data(10_000);

    c.bench_function("complete_optimization_workflow", |b| {
        b.iter(|| {
            // Full workflow: Sweep 50 RSI periods, calculate Sharpe, find best
            let sweep = ParameterSweep::new(device.clone())
                .indicator(IndicatorType::RSI)
                .parameter_range(10..=59)
                .data_close(black_box(&close))
                .metric(OptimizationMetric::Sharpe)
                .execute()
                .expect("Parameter sweep failed");

            let _best = sweep.find_optimal().expect("No optimal parameter");
        })
    });
}

#[cfg(not(feature = "gpu"))]
fn no_gpu_benchmark(c: &mut Criterion) {
    c.bench_function("no_gpu_feature", |b| {
        b.iter(|| {
            // Dummy benchmark when GPU feature is disabled
            println!("GPU feature not enabled");
        })
    });
}

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    benchmark_rsi_sweep_10_params,
    benchmark_rsi_sweep_50_params,
    benchmark_rsi_sweep_100_params,
    benchmark_sma_sweep_50_params,
    benchmark_sweep_with_metrics,
    benchmark_sweep_scalability,
    benchmark_multi_indicator_sweep,
    benchmark_optimization_workflow,
);

#[cfg(not(feature = "gpu"))]
criterion_group!(benches, no_gpu_benchmark);

criterion_main!(benches);

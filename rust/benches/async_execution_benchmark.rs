//! Benchmark: Async vs Fused execution mode
//!
//! Validates 1.2-1.4x speedup claim for async triple-buffered execution

#[cfg(feature = "gpu")]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::batch::{BatchBacktestSweep, ExecutionMode, StrategyType};
#[cfg(feature = "gpu")]
use kimsfinance_core::backtest::engine::BacktestConfig;
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::device::GpuDevice;
#[cfg(feature = "gpu")]
use ndarray::Array1;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
fn generate_test_data(
    n_candles: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let timestamps: Vec<i64> = (0..n_candles).map(|i| i as i64 * 86400).collect();

    let mut open = vec![100.0; n_candles];
    let mut high = vec![105.0; n_candles];
    let mut low = vec![95.0; n_candles];
    let mut close = vec![102.0; n_candles];
    let volume = vec![1_000_000.0; n_candles];

    // Add some trend
    for i in 0..n_candles {
        let trend = (i as f64 / n_candles as f64) * 50.0;
        open[i] += trend;
        high[i] += trend;
        low[i] += trend;
        close[i] += trend;
    }

    (
        timestamps,
        Array1::from(open),
        Array1::from(high),
        Array1::from(low),
        Array1::from(close),
        Array1::from(volume),
    )
}

#[cfg(feature = "gpu")]
fn bench_async_vs_fused(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("GPU required for benchmarks"));
    let (timestamps, open, high, low, close, volume) = generate_test_data(10_000);

    let mut group = c.benchmark_group("async_execution");
    group.sample_size(10); // Reduce sample size for long-running GPU benchmarks

    // Test different batch sizes
    for num_strategies in [500, 1000, 2000] {
        // Generate parameters
        let mut params = vec![];
        for i in 0..num_strategies {
            let rsi_period = 14.0 + (i % 10) as f64;
            let buy = 20.0 + (i % 10) as f64;
            let sell = 70.0 + (i % 10) as f64;
            params.push(vec![rsi_period, buy, sell]);
        }

        let config = BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            use_gpu: true,
            force_cpu: false,
            execution_latency_ms: 0,
        };

        // Benchmark Fused mode (baseline)
        group.bench_with_input(
            BenchmarkId::new("fused", num_strategies),
            &num_strategies,
            |b, _| {
                b.iter(|| {
                    let _result = BatchBacktestSweep::new(device.clone())
                        .strategy_type(StrategyType::RsiCrossover)
                        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                        .parameters_batch(&params)
                        .execution_mode(ExecutionMode::Fused)
                        .config(config.clone())
                        .execute()
                        .expect("Fused execution failed");
                    black_box(_result)
                });
            },
        );

        // Benchmark Async mode (target: 1.2-1.4x faster)
        group.bench_with_input(
            BenchmarkId::new("async", num_strategies),
            &num_strategies,
            |b, _| {
                b.iter(|| {
                    let _result = BatchBacktestSweep::new(device.clone())
                        .strategy_type(StrategyType::RsiCrossover)
                        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                        .parameters_batch(&params)
                        .execution_mode(ExecutionMode::Async)
                        .config(config.clone())
                        .execute()
                        .expect("Async execution failed");
                    black_box(_result)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(benches, bench_async_vs_fused);

#[cfg(feature = "gpu")]
criterion_main!(benches);

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU feature not enabled. Run with: cargo bench --features gpu");
}

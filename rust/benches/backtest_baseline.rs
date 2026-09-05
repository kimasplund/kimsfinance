//! Baseline benchmarks for backtest engine optimization
//!
//! This benchmark establishes performance baselines for:
//! - Per-candle processing time in engine hot path
//! - Sharpe ratio calculation in metrics
//! - Full backtest execution (100, 1K, 10K candles)
//! - Genetic optimizer generation time (10 individuals)
//!
//! Run with: `cargo bench --bench backtest_baseline -- --save-baseline before`
//!
//! After optimizations:
//! - Agent 2: Zero-allocation hot path + static errors
//! - Agent 3: Parallel evaluation
//! - Agent 4: SIMD metrics + early exit
//! - Agent 5: Cache-friendly data layout
//!
//! Compare with: `cargo bench --bench backtest_baseline --baseline before`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::backtest::core::{
    IndicatorConfig, OHLCVBar, ParameterGrid, ParameterRange, Signal, Strategy,
};
use kimsfinance_core::backtest::engine::{BacktestConfig, BacktestEngine};
use kimsfinance_core::backtest::metrics::{calculate_max_drawdown, calculate_sharpe_ratio};
use kimsfinance_core::backtest::optimizer::GeneticOptimizer;
use ndarray::Array1;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Generate realistic synthetic OHLCV data for benchmarking
fn generate_ohlcv_data(
    n: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Array1::zeros(n);
    let mut high = Array1::zeros(n);
    let mut low = Array1::zeros(n);
    let mut close = Array1::zeros(n);
    let mut volume = Array1::zeros(n);

    let base_price = 100.0;
    let base_time = 1_640_000_000i64; // Start time

    for i in 0..n {
        let t = i as f64;
        timestamps.push(base_time + (i as i64 * 60)); // 1-minute bars

        // Realistic price movement: trend + oscillation + noise
        let trend = t * 0.01;
        let wave = 5.0 * (t * 2.0 * PI / 100.0).sin();
        let noise = (t * 1234.56).sin() * 0.5;
        let close_price = base_price + trend + wave + noise;

        close[i] = close_price;
        open[i] = close_price + ((i as f64 * 789.0).sin() * 0.3);
        high[i] = close_price + ((i as f64 * 456.0).sin().abs() * 2.0);
        low[i] = close_price - ((i as f64 * 123.0).sin().abs() * 2.0);
        volume[i] = 1000.0 + ((i as f64 * 321.0).sin().abs() * 500.0);
    }

    (timestamps, open, high, low, close, volume)
}

/// Simple RSI-based strategy for benchmarking
#[derive(Clone)]
struct SimpleRSIStrategy {
    rsi_period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
    initial_capital: f64,
}

impl SimpleRSIStrategy {
    fn new(rsi_period: usize, buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            rsi_period,
            buy_threshold,
            sell_threshold,
            initial_capital: 10_000.0,
        }
    }
}

impl Strategy for SimpleRSIStrategy {
    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::RSI {
            period: self.rsi_period,
        }]
    }

    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &HashMap<String, f64>) -> Signal {
        let rsi = indicators
            .get(&format!("RSI_{}", self.rsi_period))
            .unwrap_or(&50.0);

        if *rsi < self.buy_threshold {
            Signal::Buy
        } else if *rsi > self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn position_size(&self, equity: f64, _signal: Signal) -> f64 {
        equity / 100.0 // Simple position sizing
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

/// Benchmark Sharpe ratio calculation (metrics hot path)
fn bench_sharpe_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics/sharpe_ratio");

    for size in [100, 1000, 10000].iter() {
        let equity_curve: Vec<f64> = (0..*size).map(|i| 10000.0 + (i as f64 * 1.5)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = calculate_sharpe_ratio(black_box(&equity_curve));
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark max drawdown calculation (metrics hot path)
fn bench_max_drawdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics/max_drawdown");

    for size in [100, 1000, 10000].iter() {
        let equity_curve: Vec<f64> = (0..*size)
            .map(|i| 10000.0 + ((i as f64 * 2.0 * PI / 100.0).sin() * 500.0))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = calculate_max_drawdown(black_box(&equity_curve));
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark full backtest execution (engine hot path)
fn bench_backtest_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/full_backtest");

    // Test with different dataset sizes
    for size in [100, 1000, 10000].iter() {
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(*size);
        let engine = BacktestEngine::with_config(BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            use_gpu: false, // CPU-only for consistent baseline
            force_cpu: true,
            execution_latency_ms: 0,
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut strategy = SimpleRSIStrategy::new(14, 30.0, 70.0);
                let result = engine
                    .run(
                        &mut strategy,
                        black_box(&timestamps),
                        black_box(&open),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                    )
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark per-candle processing time (critical hot path)
fn bench_per_candle_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/per_candle_time");

    // Single iteration to measure per-candle overhead
    for size in [1000, 10000].iter() {
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(*size);
        let engine = BacktestEngine::with_config(BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            use_gpu: false,
            force_cpu: true,
            execution_latency_ms: 0,
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut strategy = SimpleRSIStrategy::new(14, 30.0, 70.0);
                let result = engine
                    .run(
                        &mut strategy,
                        black_box(&timestamps),
                        black_box(&open),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                    )
                    .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark genetic optimizer generation time
fn bench_genetic_optimizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer/genetic_generation");

    let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(1000);
    let engine = BacktestEngine::with_config(BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
        execution_latency_ms: 0,
    });

    // Create parameter grid
    let mut grid = ParameterGrid::new();
    grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 2,
        },
    );
    grid.add_range(
        "buy_threshold",
        ParameterRange::Float {
            min: 20.0,
            max: 40.0,
            step: 5.0,
        },
    );
    grid.add_range(
        "sell_threshold",
        ParameterRange::Float {
            min: 60.0,
            max: 80.0,
            step: 5.0,
        },
    );

    // Benchmark with 10 individuals (single generation)
    group.bench_function("10_individuals", |b| {
        b.iter(|| {
            let optimizer = GeneticOptimizer::new().population_size(10).generations(1); // Single generation for baseline

            let mut strategy = SimpleRSIStrategy::new(14, 30.0, 70.0);
            let result = optimizer
                .optimize(
                    black_box(&engine),
                    black_box(&mut strategy),
                    black_box(&timestamps),
                    black_box(&open),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    black_box(&volume),
                    black_box(&grid),
                )
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sharpe_ratio,
    bench_max_drawdown,
    bench_backtest_execution,
    bench_per_candle_processing,
    bench_genetic_optimizer
);

criterion_main!(benches);

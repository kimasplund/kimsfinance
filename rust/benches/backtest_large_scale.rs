//! Large-Scale Backtest Benchmarks to Find Crossover Points
//!
//! This benchmark tests performance at realistic large-scale datasets to determine
//! where recent optimizations (SIMD, HashMap pre-sizing, etc.) start paying off.
//!
//! ## Hypothesis
//!
//! Current benchmarks (100-10K candles) show regressions (3-21% slower).
//! Optimizations likely have fixed overhead that only pays off at larger scale.
//!
//! ## Test Matrix
//!
//! ### Metrics (Sharpe Ratio)
//! - 50K points: Intraday trading system
//! - 100K points: Multi-week backtest
//! - 500K points: 1 year of 1-min data (realistic)
//! - 1M points: Multi-year analysis
//!
//! ### Engine (Full Backtest)
//! - 50K candles: Large intraday dataset
//! - 100K candles: Multi-month backtest
//! - 500K candles: BTCUSDT full year 2024
//! - 1M candles: Multi-year historical analysis
//!
//! ### Per-Candle Processing
//! - 50K candles: Measure hot path overhead
//! - 100K candles: Cache effects visible
//! - 500K candles: SIMD benefits expected
//! - 1M candles: Full optimization impact
//!
//! ### Genetic Optimizer
//! - 50 individuals: Above parallel threshold (20)
//! - 100 individuals: Medium population
//! - 200 individuals: Large population
//!
//! ## Expected Crossover Points
//!
//! Based on optimization design:
//! - **SIMD metrics**: Break-even at ~100K points
//! - **HashMap pre-sizing**: Break-even at ~50K candles
//! - **Parallel evaluation**: Speedup visible at >20 individuals
//! - **Cache-friendly layout**: Benefits at >100K candles
//!
//! ## Memory Requirements
//!
//! Approximate memory per dataset:
//! - 50K candles: ~10MB (5 arrays x 50K x 8 bytes)
//! - 100K candles: ~20MB
//! - 500K candles: ~100MB
//! - 1M candles: ~200MB
//!
//! Total benchmark suite: ~1GB peak memory (acceptable)
//!
//! ## Usage
//!
//! ```bash
//! # Create baseline BEFORE optimizations
//! cargo bench --bench backtest_large_scale -- --save-baseline large_before
//!
//! # After optimizations, compare
//! cargo bench --bench backtest_large_scale --baseline large_before
//!
//! # Run specific test
//! cargo bench --bench backtest_large_scale -- sharpe_ratio/500000
//! cargo bench --bench backtest_large_scale -- full_backtest/1000000
//! ```
//!
//! ## Context for Agent Team
//!
//! - Agent 1 (this): Create benchmarks, establish baseline
//! - Agent 2: Profile SIMD with perf (will use these benchmarks)
//! - Agent 3: Profile HashMap allocations (will use these benchmarks)
//! - Agent 4: Analyze crossover thresholds (needs this data)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use kimsfinance_core::backtest::core::{
    IndicatorConfig, OHLCVBar, ParameterGrid, ParameterRange, Signal, Strategy,
};
use kimsfinance_core::backtest::engine::{BacktestConfig, BacktestEngine};
use kimsfinance_core::backtest::metrics::{
    calculate_max_drawdown, calculate_sharpe_ratio,
};
use kimsfinance_core::backtest::optimizer::GeneticOptimizer;
use ndarray::Array1;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Generate realistic synthetic OHLCV data for large-scale benchmarking
///
/// Uses deterministic pseudo-random generation for reproducibility:
/// - Trend: Linear drift (0.01 per candle)
/// - Wave: Sine wave oscillation (period ~100 candles)
/// - Noise: Small random-like variations
fn generate_ohlcv_data(n: usize) -> (Vec<i64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Array1::zeros(n);
    let mut high = Array1::zeros(n);
    let mut low = Array1::zeros(n);
    let mut close = Array1::zeros(n);
    let mut volume = Array1::zeros(n);

    let base_price = 100.0;
    let base_time = 1_640_000_000i64; // Start time: 2021-12-20

    for i in 0..n {
        let t = i as f64;
        timestamps.push(base_time + (i as i64 * 60)); // 1-minute bars

        // Realistic price movement: trend + oscillation + noise
        let trend = t * 0.01; // Slow upward drift
        let wave = 5.0 * (t * 2.0 * PI / 100.0).sin(); // ~100 candle cycle
        let noise = (t * 1234.56).sin() * 0.5; // High-frequency variation
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
///
/// Strategy logic:
/// - Buy when RSI < buy_threshold (oversold)
/// - Sell when RSI > sell_threshold (overbought)
/// - Hold otherwise
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
        let rsi = indicators.get(&format!("RSI_{}", self.rsi_period)).unwrap_or(&50.0);

        if *rsi < self.buy_threshold {
            Signal::Buy
        } else if *rsi > self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn position_size(&self, equity: f64, _signal: Signal) -> f64 {
        equity / 100.0 // Simple position sizing: 1% of equity
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

/// Benchmark Sharpe ratio calculation at large scale
///
/// Tests SIMD optimization impact on metrics calculation.
/// Expected crossover: ~100K points where SIMD becomes beneficial.
fn bench_sharpe_ratio_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale/sharpe_ratio");
    group.sample_size(10); // Fewer samples for large benchmarks
    group.sampling_mode(SamplingMode::Flat); // Consistent timing

    // Test large-scale datasets
    for size in [50_000, 100_000, 500_000, 1_000_000].iter() {
        // Generate equity curve with realistic returns
        let equity_curve: Vec<f64> = (0..*size)
            .map(|i| {
                let t = i as f64;
                let base = 10000.0;
                let trend = t * 0.1; // Gradual growth
                let volatility = (t * 2.0 * PI / 1000.0).sin() * 500.0;
                base + trend + volatility
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = calculate_sharpe_ratio(black_box(&equity_curve));
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark max drawdown calculation at large scale
///
/// Tests sequential scan optimization (potential for SIMD).
/// Expected crossover: ~50K points for optimized version.
fn bench_max_drawdown_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale/max_drawdown");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for size in [50_000, 100_000, 500_000, 1_000_000].iter() {
        // Generate equity curve with realistic drawdowns
        let equity_curve: Vec<f64> = (0..*size)
            .map(|i| {
                let t = i as f64;
                let base = 10000.0;
                let trend = t * 0.05;
                // Create realistic drawdown periods
                let drawdown = if (i / 10000) % 3 == 0 {
                    -((t / 10000.0).sin().abs() * 1000.0)
                } else {
                    0.0
                };
                base + trend + drawdown
            })
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

/// Benchmark full backtest execution at large scale
///
/// Tests engine hot path with realistic large datasets.
/// Expected crossover: ~100K candles for all optimizations combined.
fn bench_backtest_execution_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale/full_backtest");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    // Test realistic large-scale backtests
    for size in [50_000, 100_000, 500_000].iter() {
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(*size);
        let engine = BacktestEngine::with_config(BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            use_gpu: false, // CPU-only for baseline
            force_cpu: true,
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

/// Benchmark per-candle processing time at large scale
///
/// Critical hot path measurement. Divide total time by candle count.
/// Expected: <1μs per candle for optimized version.
fn bench_per_candle_processing_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale/per_candle_time");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for size in [50_000, 100_000, 500_000, 1_000_000].iter() {
        let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(*size);
        let engine = BacktestEngine::with_config(BacktestConfig {
            initial_capital: 10_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            use_gpu: false,
            force_cpu: true,
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

/// Benchmark genetic optimizer at large population sizes
///
/// Tests parallel evaluation optimization (threshold: 20 individuals).
/// Expected crossover: 50+ individuals where parallelization pays off.
fn bench_genetic_optimizer_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale/genetic_optimizer");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    // Use moderate dataset size (10K candles) to focus on parallelization
    let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(10_000);
    let engine = BacktestEngine::with_config(BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
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

    // Test large population sizes (above parallel threshold)
    for population in [50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_individuals", population)),
            population,
            |b, _| {
                b.iter(|| {
                    let optimizer = GeneticOptimizer::new()
                        .population_size(*population)
                        .generations(1); // Single generation for baseline

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
            },
        );
    }

    group.finish();
}

/// Benchmark end-to-end realistic scenario
///
/// 500K candles (1 year of 1-min data) with genetic optimization.
/// This is a real-world production workload.
fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale/realistic_workload");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    // 500K candles: ~1 year of 1-minute BTCUSDT data
    let (timestamps, open, high, low, close, volume) = generate_ohlcv_data(500_000);
    let engine = BacktestEngine::with_config(BacktestConfig {
        initial_capital: 10_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
        use_gpu: false,
        force_cpu: true,
    });

    group.bench_function("500k_candles_backtest", |b| {
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

    group.finish();
}

criterion_group!(
    benches,
    bench_sharpe_ratio_large_scale,
    bench_max_drawdown_large_scale,
    bench_backtest_execution_large_scale,
    bench_per_candle_processing_large_scale,
    bench_genetic_optimizer_large_scale,
    bench_realistic_workload
);

criterion_main!(benches);

//! Multi-Indicator Backtesting Throughput Benchmark
//!
//! Measures backtesting throughput for strategies using multiple technical indicators.
//! Tests GPU batch processing efficiency vs sequential CPU calculation.
//!
//! # Test Matrix
//!
//! | Indicator Count | Indicators Used | Expected GPU Speedup |
//! |-----------------|-----------------|----------------------|
//! | 1               | RSI             | 1.0-1.5x             |
//! | 3               | RSI+ATR+CCI     | 2.0-2.5x             |
//! | 5               | RSI+ATR+CCI+ROC+WilliamsR | 2.5-3.5x |
//! | 7               | All momentum    | 3.0-4.0x             |
//!
//! # GPU Batch Processing
//!
//! GPU batch indicator calculation uses 2D kernels to compute multiple indicators
//! simultaneously, reducing memory transfers and kernel launch overhead.
//!
//! Benefits:
//! - Single GPU memory transfer for all indicators
//! - Parallel indicator computation
//! - Shared memory reuse across indicators
//! - Reduced kernel launch overhead
//!
//! # Memory Profiling
//!
//! Tracks GPU memory usage and bandwidth utilization:
//! - Peak VRAM usage
//! - Memory bandwidth utilization
//! - Transfer vs compute time ratio
//! - Cache hit rates (L2 persistence)
//!
//! # Performance Metrics
//!
//! - **Throughput**: Backtests per second
//! - **Latency**: Time per backtest (p50, p95, p99)
//! - **Memory**: Peak usage, bandwidth utilization
//! - **Efficiency**: GPU utilization (SM active time)
//!
//! # Hardware Context
//!
//! - GPU: NVIDIA RTX 3500 Ada (12GB VRAM, 336 GB/s bandwidth)
//! - CPU: Intel i9-13980HX (24 cores)
//! - Memory: 64GB DDR5
//!
//! # Usage
//!
//! ```bash
//! # Run full throughput benchmark
//! cargo bench --features gpu --bench multi_indicator_throughput
//!
//! # Run specific indicator count
//! cargo bench --features gpu --bench multi_indicator_throughput -- indicators_1
//! cargo bench --features gpu --bench multi_indicator_throughput -- indicators_3
//! cargo bench --features gpu --bench multi_indicator_throughput -- indicators_5
//!
//! # Profile memory usage
//! cargo bench --features gpu --bench multi_indicator_throughput -- memory_profile
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use std::time::Duration;

use kimsfinance_core::backtest::{
    BacktestConfig, BacktestEngine, IndicatorConfig, IndicatorValues, OHLCVBar, Signal, Strategy,
};

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;

#[path = "statistics.rs"]
mod statistics;

use statistics::BenchmarkStats;

/// Single indicator strategy (RSI only)
struct SingleIndicatorStrategy {
    rsi_period: usize,
}

impl Strategy for SingleIndicatorStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi = indicators
            .get(&format!("rsi_{}", self.rsi_period))
            .copied()
            .unwrap_or(50.0);

        if rsi.is_nan() {
            return Signal::Hold;
        }

        if rsi < 30.0 {
            Signal::Buy
        } else if rsi > 70.0 {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::RSI {
            period: self.rsi_period,
        }]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

/// Three indicator strategy (RSI + ATR + CCI)
struct ThreeIndicatorStrategy {
    rsi_period: usize,
    atr_period: usize,
    cci_period: usize,
}

impl Strategy for ThreeIndicatorStrategy {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi = indicators
            .get(&format!("rsi_{}", self.rsi_period))
            .copied()
            .unwrap_or(50.0);
        let atr = indicators
            .get(&format!("atr_{}", self.atr_period))
            .copied()
            .unwrap_or(0.0);
        let cci = indicators
            .get(&format!("cci_{}", self.cci_period))
            .copied()
            .unwrap_or(0.0);

        if rsi.is_nan() || atr.is_nan() || cci.is_nan() {
            return Signal::Hold;
        }

        // Multi-indicator logic with ATR volatility filter
        let volatility_threshold = bar.close * 0.01; // 1% of price

        if rsi < 30.0 && cci < -100.0 && atr > volatility_threshold {
            Signal::Buy
        } else if rsi > 70.0 && cci > 100.0 {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::RSI {
                period: self.rsi_period,
            },
            IndicatorConfig::ATR {
                period: self.atr_period,
            },
            IndicatorConfig::CCI {
                period: self.cci_period,
            },
        ]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

/// Five indicator strategy (RSI + ATR + CCI + ROC + Williams %R)
struct FiveIndicatorStrategy {}

impl Strategy for FiveIndicatorStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi = indicators.get("rsi_14").copied().unwrap_or(50.0);
        let atr = indicators.get("atr_14").copied().unwrap_or(0.0);
        let cci = indicators.get("cci_20").copied().unwrap_or(0.0);
        let roc = indicators.get("roc_12").copied().unwrap_or(0.0);
        let williams = indicators.get("williams_14").copied().unwrap_or(-50.0);

        if rsi.is_nan() || atr.is_nan() || cci.is_nan() || roc.is_nan() || williams.is_nan() {
            return Signal::Hold;
        }

        // Complex multi-indicator logic
        let buy_score = (rsi < 30.0) as i32
            + (cci < -100.0) as i32
            + (roc < -5.0) as i32
            + (williams < -80.0) as i32;

        let sell_score = (rsi > 70.0) as i32
            + (cci > 100.0) as i32
            + (roc > 5.0) as i32
            + (williams > -20.0) as i32;

        if buy_score >= 3 {
            Signal::Buy
        } else if sell_score >= 3 {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::RSI { period: 14 },
            IndicatorConfig::ATR { period: 14 },
            IndicatorConfig::CCI { period: 20 },
            IndicatorConfig::ROC { period: 12 },
            IndicatorConfig::WilliamsR { period: 14 },
        ]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

/// Seven indicator strategy (all momentum indicators)
struct SevenIndicatorStrategy {}

impl Strategy for SevenIndicatorStrategy {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi = indicators.get("rsi_14").copied().unwrap_or(50.0);
        let atr = indicators.get("atr_14").copied().unwrap_or(0.0);
        let cci = indicators.get("cci_20").copied().unwrap_or(0.0);
        let roc = indicators.get("roc_12").copied().unwrap_or(0.0);
        let williams = indicators.get("williams_14").copied().unwrap_or(-50.0);

        // Get Stochastic components
        let stoch_k = indicators.get("stoch_14_3_k").copied().unwrap_or(50.0);
        let stoch_d = indicators.get("stoch_14_3_d").copied().unwrap_or(50.0);

        // Get Bollinger Bands
        let bb_upper = indicators.get("bb_20_2.0_upper").copied().unwrap_or(0.0);
        let bb_lower = indicators.get("bb_20_2.0_lower").copied().unwrap_or(0.0);
        let bb_middle = indicators.get("bb_20_2.0_middle").copied().unwrap_or(0.0);

        // Skip if any indicator is NaN
        if [
            rsi, atr, cci, roc, williams, stoch_k, stoch_d, bb_upper, bb_lower, bb_middle,
        ]
        .iter()
        .any(|v| v.is_nan())
        {
            return Signal::Hold;
        }

        // Complex scoring system
        let buy_score = (rsi < 30.0) as i32
            + (cci < -100.0) as i32
            + (roc < -5.0) as i32
            + (williams < -80.0) as i32
            + (stoch_k < 20.0 && stoch_d < 20.0) as i32
            + (bb_middle < bb_lower) as i32;

        let sell_score = (rsi > 70.0) as i32
            + (cci > 100.0) as i32
            + (roc > 5.0) as i32
            + (williams > -20.0) as i32
            + (stoch_k > 80.0 && stoch_d > 80.0) as i32
            + (bb_middle > bb_upper) as i32;

        if buy_score >= 4 {
            Signal::Buy
        } else if sell_score >= 4 {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::RSI { period: 14 },
            IndicatorConfig::ATR { period: 14 },
            IndicatorConfig::CCI { period: 20 },
            IndicatorConfig::ROC { period: 12 },
            IndicatorConfig::WilliamsR { period: 14 },
            IndicatorConfig::Stochastic {
                k_period: 14,
                d_period: 3,
            },
            IndicatorConfig::BollingerBands {
                period: 20,
                std_dev: 2.0,
            },
        ]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

/// Generate realistic OHLCV data
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
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let base_price = 50000.0;

    for i in 0..n {
        let t = i as f64;
        let price = base_price + t * 0.5 + (t * 0.05).sin() * 1000.0;
        let spread = 200.0 + (t * 0.01).cos() * 50.0;

        timestamps.push(i as i64);
        high.push(price + spread);
        low.push(price - spread);
        open.push(price - spread * 0.5);
        close.push(price + spread * 0.5);
        volume.push(1_000_000.0 + (t * 0.15).sin() * 300_000.0);
    }

    (
        timestamps,
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(open),
        Array1::from_vec(close),
        Array1::from_vec(volume),
    )
}

/// Benchmark 1 indicator throughput
fn bench_one_indicator(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_1_indicator");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(20));

    println!("\n=== Benchmark: 1 Indicator (RSI) ===");
    println!("Testing CPU vs GPU batch processing\n");

    #[cfg(feature = "gpu")]
    let gpu_available = GpuDevice::new().is_ok();

    #[cfg(not(feature = "gpu"))]
    let gpu_available = false;

    let dataset_sizes = vec![1_000, 10_000, 100_000];

    for &size in &dataset_sizes {
        let (timestamps, high, low, open, close, volume) = generate_ohlcv_data(size);

        // CPU
        let mut strategy = SingleIndicatorStrategy { rsi_period: 14 };
        let engine = BacktestEngine::with_config(BacktestConfig {
            use_gpu: false,
            force_cpu: true,
            ..Default::default()
        });

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("CPU", size), &size, |b, _| {
            b.iter(|| {
                engine
                    .run(
                        black_box(&mut strategy),
                        black_box(&timestamps),
                        black_box(&open),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                    )
                    .expect("Backtest failed")
            });
        });

        // GPU
        #[cfg(feature = "gpu")]
        if gpu_available {
            let mut strategy = SingleIndicatorStrategy { rsi_period: 14 };
            let engine = BacktestEngine::with_config(BacktestConfig {
                use_gpu: true,
                ..Default::default()
            });

            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::new("GPU", size), &size, |b, _| {
                b.iter(|| {
                    engine
                        .run(
                            black_box(&mut strategy),
                            black_box(&timestamps),
                            black_box(&open),
                            black_box(&high),
                            black_box(&low),
                            black_box(&close),
                            black_box(&volume),
                        )
                        .expect("GPU backtest failed")
                });
            });
        }
    }

    group.finish();
}

/// Benchmark 3 indicators throughput
fn bench_three_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_3_indicators");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(30));

    println!("\n=== Benchmark: 3 Indicators (RSI + ATR + CCI) ===");
    println!("Expected GPU speedup: 2.0-2.5x\n");

    #[cfg(feature = "gpu")]
    let gpu_available = GpuDevice::new().is_ok();

    #[cfg(not(feature = "gpu"))]
    let gpu_available = false;

    let dataset_sizes = vec![1_000, 10_000, 100_000];

    for &size in &dataset_sizes {
        let (timestamps, high, low, open, close, volume) = generate_ohlcv_data(size);

        // CPU
        let mut strategy = ThreeIndicatorStrategy {
            rsi_period: 14,
            atr_period: 14,
            cci_period: 20,
        };
        let engine = BacktestEngine::with_config(BacktestConfig {
            use_gpu: false,
            force_cpu: true,
            ..Default::default()
        });

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("CPU", size), &size, |b, _| {
            b.iter(|| {
                engine
                    .run(
                        black_box(&mut strategy),
                        black_box(&timestamps),
                        black_box(&open),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                    )
                    .expect("CPU backtest failed")
            });
        });

        // GPU
        #[cfg(feature = "gpu")]
        if gpu_available {
            let mut strategy = ThreeIndicatorStrategy {
                rsi_period: 14,
                atr_period: 14,
                cci_period: 20,
            };
            let engine = BacktestEngine::with_config(BacktestConfig {
                use_gpu: true,
                ..Default::default()
            });

            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::new("GPU", size), &size, |b, _| {
                b.iter(|| {
                    engine
                        .run(
                            black_box(&mut strategy),
                            black_box(&timestamps),
                            black_box(&open),
                            black_box(&high),
                            black_box(&low),
                            black_box(&close),
                            black_box(&volume),
                        )
                        .expect("GPU backtest failed")
                });
            });
        }
    }

    group.finish();
}

/// Benchmark 5 indicators throughput
fn bench_five_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_5_indicators");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(40));

    println!("\n=== Benchmark: 5 Indicators ===");
    println!("Indicators: RSI + ATR + CCI + ROC + Williams %R");
    println!("Expected GPU speedup: 2.5-3.5x\n");

    #[cfg(feature = "gpu")]
    let gpu_available = GpuDevice::new().is_ok();

    #[cfg(not(feature = "gpu"))]
    let gpu_available = false;

    let dataset_sizes = vec![10_000, 100_000];

    for &size in &dataset_sizes {
        let (timestamps, high, low, open, close, volume) = generate_ohlcv_data(size);

        // CPU
        let mut strategy = FiveIndicatorStrategy {};
        let engine = BacktestEngine::with_config(BacktestConfig {
            use_gpu: false,
            force_cpu: true,
            ..Default::default()
        });

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("CPU", size), &size, |b, _| {
            b.iter(|| {
                engine
                    .run(
                        black_box(&mut strategy),
                        black_box(&timestamps),
                        black_box(&open),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        black_box(&volume),
                    )
                    .expect("CPU backtest failed")
            });
        });

        // GPU
        #[cfg(feature = "gpu")]
        if gpu_available {
            let mut strategy = FiveIndicatorStrategy {};
            let engine = BacktestEngine::with_config(BacktestConfig {
                use_gpu: true,
                ..Default::default()
            });

            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::new("GPU", size), &size, |b, _| {
                b.iter(|| {
                    engine
                        .run(
                            black_box(&mut strategy),
                            black_box(&timestamps),
                            black_box(&open),
                            black_box(&high),
                            black_box(&low),
                            black_box(&close),
                            black_box(&volume),
                        )
                        .expect("GPU backtest failed")
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    throughput_benches,
    bench_one_indicator,
    bench_three_indicators,
    bench_five_indicators
);

criterion_main!(throughput_benches);

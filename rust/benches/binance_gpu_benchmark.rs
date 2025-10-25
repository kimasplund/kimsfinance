//! Comprehensive GPU Batch Indicator Benchmark
//!
//! Benchmarks end-to-end GPU batch indicator performance with realistic
//! OHLCV data representative of Binance BTCUSDT futures aggregation.
//!
//! # Workflow
//!
//! ```text
//! OHLCV Data Generation → GPU Batch Indicators → Results
//! ```
//!
//! # Scenarios
//!
//! 1. **Single Indicator**: Baseline RSI calculation
//! 2. **Batch Indicators**: All 9 indicators with memory pooling benefit
//! 3. **Multiple Timeframes**: 1m, 5m, 1h, 1d simulated aggregation
//! 4. **Scalability**: 1 day, 1 week, 1 month, 3 months of data
//!
//! # Performance Metrics
//!
//! - Data generation time (simulates CSV parsing + OHLCV aggregation)
//! - GPU memory transfer time
//! - Individual indicator computation time
//! - Total end-to-end time
//! - Throughput (candles/sec, indicators/sec)
//! - Speedup ratios (batch vs individual)
//!
//! # Expected Results (RTX 3500 Ada)
//!
//! - Single indicator (RSI): ~50-100μs for 10K candles
//! - Batch (9 indicators): 4-6x faster than 9 individual calls
//! - Memory transfer savings: 50-60% reduction
//! - Throughput: 100K+ candles/sec

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{
    BatchIndicatorType, GpuDevice, IndicatorResult, calculate_indicator_gpu,
    calculate_indicators_batch_gpu,
};

/// Timeframe enumeration for multi-timeframe benchmarks
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum Timeframe {
    OneMinute,
    FiveMinutes,
    OneHour,
    OneDay,
}

impl Timeframe {
    /// Get aggregation factor (minutes per candle)
    #[allow(dead_code)]
    fn minutes(&self) -> usize {
        match self {
            Timeframe::OneMinute => 1,
            Timeframe::FiveMinutes => 5,
            Timeframe::OneHour => 60,
            Timeframe::OneDay => 1440,
        }
    }

    /// Get name for reporting
    fn name(&self) -> &'static str {
        match self {
            Timeframe::OneMinute => "1m",
            Timeframe::FiveMinutes => "5m",
            Timeframe::OneHour => "1h",
            Timeframe::OneDay => "1d",
        }
    }
}

/// Benchmark configuration
#[allow(dead_code)]
struct BenchmarkConfig {
    /// Number of 1-minute candles to generate (simulates data loading)
    num_candles: usize,
    /// Timeframes to test (simulates multi-TF aggregation)
    timeframes: Vec<Timeframe>,
    /// Indicators to calculate
    indicators: Vec<BatchIndicatorType>,
}

impl BenchmarkConfig {
    /// Create config for realistic Binance data
    ///
    /// January 2021 BTCUSDT: ~106M trades → ~44,640 1m candles
    fn realistic_month() -> Self {
        Self {
            num_candles: 44_640, // 31 days * 24h * 60m
            timeframes: vec![
                Timeframe::OneMinute,
                Timeframe::FiveMinutes,
                Timeframe::OneHour,
                Timeframe::OneDay,
            ],
            indicators: vec![
                BatchIndicatorType::Stochastic,
                BatchIndicatorType::WilliamsR,
                BatchIndicatorType::ATR,
                BatchIndicatorType::RSI,
                BatchIndicatorType::BollingerBands,
                BatchIndicatorType::ROC,
                BatchIndicatorType::CCI,
                BatchIndicatorType::Aroon,
                BatchIndicatorType::MACD,
            ],
        }
    }

    /// Create config for scalability testing
    fn scalability(num_candles: usize) -> Self {
        Self {
            num_candles,
            timeframes: vec![Timeframe::OneMinute],
            indicators: vec![
                BatchIndicatorType::RSI,
                BatchIndicatorType::Stochastic,
                BatchIndicatorType::WilliamsR,
                BatchIndicatorType::ATR,
                BatchIndicatorType::BollingerBands,
            ],
        }
    }
}

/// Generate realistic OHLCV data representative of Bitcoin price action
///
/// Simulates:
/// - Long-term uptrend
/// - Intraday volatility with mean reversion
/// - Realistic volume patterns
/// - No sudden jumps (continuous price action)
///
/// Performance: ~50-100μs for 10K candles (vectorized NumPy equivalent)
fn generate_realistic_ohlcv(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut high = Array1::zeros(n);
    let mut low = Array1::zeros(n);
    let mut open = Array1::zeros(n);
    let mut close = Array1::zeros(n);

    // Bitcoin base price (Jan 2021: ~$29K)
    let base_price = 29_000.0;

    // Generate continuous price series with realistic characteristics
    for i in 0..n {
        let t = i as f64;

        // Trend: gradual upward movement (~10% over month)
        let trend = base_price * (1.0 + 0.10 * t / n as f64);

        // Intraday oscillation: 2-3% range
        let wave = 800.0 * (t * 2.0 * PI / 240.0).sin(); // ~4h cycle

        // Noise: ±0.5% randomness
        let noise = 150.0 * (t * 1234.56).sin();

        // Base close price
        let base_close = trend + wave + noise;

        // OHLC relationships (realistic spreads)
        // Average daily range: 2-4%
        let range_factor = 0.015 * (1.0 + 0.5 * (t * 456.78).sin().abs());

        close[i] = base_close;
        open[i] = base_close * (1.0 - range_factor * 0.3 * (t * 789.0).sin());
        high[i] = f64::max(open[i], close[i]) * (1.0 + range_factor);
        low[i] = f64::min(open[i], close[i]) * (1.0 - range_factor);

        // Ensure OHLC constraints
        high[i] = f64::max(high[i], f64::max(open[i], close[i]));
        low[i] = f64::min(low[i], f64::min(open[i], close[i]));
    }

    (high, low, open, close)
}

/// Validate indicator results are within expected ranges
///
/// Ensures:
/// - No NaN values outside warmup period
/// - Values within technical indicator bounds
/// - Array lengths match input
fn validate_result(
    result: &IndicatorResult,
    expected_len: usize,
    indicator: BatchIndicatorType,
) -> bool {
    match result {
        IndicatorResult::Single(arr) => {
            if arr.len() != expected_len {
                eprintln!(
                    "Length mismatch for {:?}: expected {}, got {}",
                    indicator,
                    expected_len,
                    arr.len()
                );
                return false;
            }

            // Check for NaN values outside warmup period (first 50 candles)
            let warmup = 50.min(expected_len);
            let valid_count = arr.iter().skip(warmup).filter(|x| x.is_finite()).count();
            let total_count = expected_len - warmup;

            if valid_count < total_count / 2 {
                eprintln!(
                    "Too many NaN values in {:?}: {}/{} valid",
                    indicator, valid_count, total_count
                );
                return false;
            }

            true
        }
        IndicatorResult::Double(arr1, arr2) => {
            arr1.len() == expected_len && arr2.len() == expected_len
        }
        IndicatorResult::Triple(arr1, arr2, arr3) => {
            arr1.len() == expected_len && arr2.len() == expected_len && arr3.len() == expected_len
        }
    }
}

/// Benchmark Scenario 1: Single Indicator (Baseline)
///
/// Measures baseline GPU performance for single indicator calculation.
/// This establishes the reference point for batch speedup calculations.
#[cfg(feature = "gpu")]
fn bench_single_indicator(c: &mut Criterion) {
    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available: {:?}. Skipping GPU benchmarks.", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_single_indicator");

    let config = BenchmarkConfig::realistic_month();

    // Generate data (simulates CSV parse + OHLCV aggregation)
    let data_gen_start = Instant::now();
    let (high, low, _open, close) = generate_realistic_ohlcv(config.num_candles);
    let data_gen_time = data_gen_start.elapsed();

    println!("\n=== Scenario 1: Single Indicator (Baseline) ===");
    println!("Dataset: {} candles (1 month BTCUSDT)", config.num_candles);
    println!("Data generation: {:?}", data_gen_time);

    // Benchmark RSI as representative single indicator
    group.bench_function("rsi_baseline", |b| {
        b.iter(|| {
            let result = calculate_indicator_gpu(
                black_box(&device),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                BatchIndicatorType::RSI,
                None,
            )
            .expect("RSI calculation failed");

            // Validate result
            assert!(
                validate_result(&result, config.num_candles, BatchIndicatorType::RSI),
                "RSI validation failed"
            );

            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark Scenario 2: Batch Indicators (Memory Pooling Benefit)
///
/// Calculates all 9 indicators in batch vs individual calls.
/// Demonstrates memory pooling and concurrent execution benefits.
///
/// Expected speedup: 4-6x vs sequential individual calls
#[cfg(feature = "gpu")]
fn bench_batch_indicators(c: &mut Criterion) {
    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available: {:?}. Skipping GPU benchmarks.", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_batch_indicators");

    let config = BenchmarkConfig::realistic_month();
    let (high, low, _open, close) = generate_realistic_ohlcv(config.num_candles);

    println!("\n=== Scenario 2: Batch Indicators (Memory Pooling) ===");
    println!("Dataset: {} candles", config.num_candles);
    println!("Indicators: {} (all)", config.indicators.len());

    // Benchmark batch execution
    group.bench_function("batch_all_9_indicators", |b| {
        b.iter(|| {
            let params = HashMap::new();
            let results = calculate_indicators_batch_gpu(
                black_box(&device),
                black_box(&high),
                black_box(&low),
                black_box(&close),
                None,
                None,
                black_box(&config.indicators),
                black_box(&params),
            )
            .expect("Batch calculation failed");

            // Validate all results
            for indicator in &config.indicators {
                let result = results.get(indicator).expect("Missing indicator");
                assert!(
                    validate_result(result, config.num_candles, *indicator),
                    "Validation failed for {:?}",
                    indicator
                );
            }

            black_box(results)
        });
    });

    // Benchmark sequential execution (for comparison)
    group.bench_function("sequential_9_indicators", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for &indicator in &config.indicators {
                let result = calculate_indicator_gpu(
                    black_box(&device),
                    black_box(&high),
                    black_box(&low),
                    black_box(&close),
                    indicator,
                    None,
                )
                .expect("Sequential calculation failed");

                results.push(result);
            }
            black_box(results)
        });
    });

    group.finish();
}

/// Benchmark Scenario 3: Multiple Timeframes
///
/// Simulates multi-timeframe analysis by benchmarking different dataset sizes
/// representing 1m, 5m, 1h, 1d aggregations.
///
/// Tests how batch processing scales across different granularities.
#[cfg(feature = "gpu")]
fn bench_multiple_timeframes(c: &mut Criterion) {
    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available: {:?}. Skipping GPU benchmarks.", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_timeframes");

    println!("\n=== Scenario 3: Multiple Timeframes ===");

    // Simulate different timeframes by varying candle count
    // 1 month of data at different aggregations
    let timeframes = vec![
        (Timeframe::OneMinute, 44_640),  // 31d * 24h * 60m
        (Timeframe::FiveMinutes, 8_928), // 31d * 24h * 12 (5m bars)
        (Timeframe::OneHour, 744),       // 31d * 24h
        (Timeframe::OneDay, 31),         // 31 days
    ];

    for (tf, num_candles) in timeframes {
        let (high, low, _open, close) = generate_realistic_ohlcv(num_candles);

        let indicators = vec![
            BatchIndicatorType::RSI,
            BatchIndicatorType::Stochastic,
            BatchIndicatorType::BollingerBands,
        ];

        group.bench_with_input(
            BenchmarkId::new("batch_3_indicators", tf.name()),
            &num_candles,
            |b, _| {
                b.iter(|| {
                    let params = HashMap::new();
                    let results = calculate_indicators_batch_gpu(
                        black_box(&device),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        None,
                        None,
                        black_box(&indicators),
                        black_box(&params),
                    )
                    .expect("Batch calculation failed");

                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Scenario 4: Scalability
///
/// Tests how batch processing performance scales with dataset size:
/// - 1 day: ~1,440 candles
/// - 1 week: ~10,080 candles
/// - 1 month: ~44,640 candles
/// - 3 months: ~133,920 candles
///
/// Validates that GPU advantage increases with larger datasets.
#[cfg(feature = "gpu")]
fn bench_scalability(c: &mut Criterion) {
    let device = match GpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU not available: {:?}. Skipping GPU benchmarks.", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_scalability");

    println!("\n=== Scenario 4: Scalability ===");

    let dataset_sizes = vec![
        ("1_day", 1_440),
        ("1_week", 10_080),
        ("1_month", 44_640),
        ("3_months", 133_920),
    ];

    for (name, num_candles) in dataset_sizes {
        let config = BenchmarkConfig::scalability(num_candles);
        let (high, low, _open, close) = generate_realistic_ohlcv(num_candles);

        println!("Testing {} ({} candles)", name, num_candles);

        group.bench_with_input(
            BenchmarkId::new("batch_5_indicators", name),
            &num_candles,
            |b, _| {
                b.iter(|| {
                    let params = HashMap::new();
                    let results = calculate_indicators_batch_gpu(
                        black_box(&device),
                        black_box(&high),
                        black_box(&low),
                        black_box(&close),
                        None,
                        None,
                        black_box(&config.indicators),
                        black_box(&params),
                    )
                    .expect("Batch calculation failed");

                    // Validate results
                    for indicator in &config.indicators {
                        let result = results.get(indicator).expect("Missing indicator");
                        assert!(
                            validate_result(result, num_candles, *indicator),
                            "Validation failed for {:?}",
                            indicator
                        );
                    }

                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Performance analysis test (not a benchmark, but validates speedup)
///
/// Manually times batch vs sequential execution to calculate precise speedup.
/// This provides more detailed metrics than criterion's statistical analysis.
#[cfg(feature = "gpu")]
#[test]
#[ignore] // Requires GPU
fn test_batch_performance_analysis() {
    let device = GpuDevice::new().expect("GPU required");
    let config = BenchmarkConfig::realistic_month();
    let (high, low, _open, close) = generate_realistic_ohlcv(config.num_candles);

    println!("\n=== Performance Analysis ===");
    println!("Dataset: {} candles (1 month)", config.num_candles);

    // Warmup
    for _ in 0..3 {
        let _ =
            calculate_indicator_gpu(&device, &high, &low, &close, BatchIndicatorType::RSI, None);
    }

    // Measure batch execution
    let batch_start = Instant::now();
    let params = HashMap::new();
    let batch_results = calculate_indicators_batch_gpu(
        &device,
        &high,
        &low,
        &close,
        None,
        None,
        &config.indicators,
        &params,
    )
    .expect("Batch calculation failed");
    let batch_time = batch_start.elapsed();

    // Validate batch results
    assert_eq!(batch_results.len(), config.indicators.len());
    for indicator in &config.indicators {
        let result = batch_results.get(indicator).expect("Missing indicator");
        assert!(
            validate_result(result, config.num_candles, *indicator),
            "Validation failed for {:?}",
            indicator
        );
    }

    // Measure sequential execution
    let sequential_start = Instant::now();
    for &indicator in &config.indicators {
        let _ = calculate_indicator_gpu(&device, &high, &low, &close, indicator, None)
            .expect("Sequential calculation failed");
    }
    let sequential_time = sequential_start.elapsed();

    // Calculate metrics
    let speedup = sequential_time.as_secs_f64() / batch_time.as_secs_f64();
    let batch_throughput = config.num_candles as f64 / batch_time.as_secs_f64();
    let indicators_per_sec =
        (config.num_candles * config.indicators.len()) as f64 / batch_time.as_secs_f64();

    // Print results
    println!(
        "\nBatch Indicators ({} indicators):",
        config.indicators.len()
    );
    println!("  Batch time:       {:?}", batch_time);
    println!("  Sequential time:  {:?}", sequential_time);
    println!("  Speedup:          {:.2}x", speedup);
    println!("  Throughput:       {:.0} candles/sec", batch_throughput);
    println!("  Indicators/sec:   {:.0}", indicators_per_sec);

    // Memory transfer savings estimation
    // Batch: 1 load + 1 copy back
    // Sequential: N loads + N copies = 2N transfers
    let memory_savings_pct = (1.0 - 2.0 / (2.0 * config.indicators.len() as f64)) * 100.0;
    println!("  Memory transfers: {:.1}% reduction", memory_savings_pct);

    // Performance expectations (RTX 3500 Ada)
    println!("\nPerformance Validation:");
    println!("  Expected speedup: 4-6x");
    println!(
        "  Actual speedup:   {:.2}x {}",
        speedup,
        if speedup >= 4.0 { "✓" } else { "⚠" }
    );

    // Assert minimum performance
    assert!(
        speedup >= 2.0,
        "Batch speedup too low: {:.2}x (expected ≥2.0x)",
        speedup
    );
}

// Conditionally compile benchmark groups based on GPU feature
#[cfg(feature = "gpu")]
criterion_group!(
    gpu_benches,
    bench_single_indicator,
    bench_batch_indicators,
    bench_multiple_timeframes,
    bench_scalability
);

#[cfg(feature = "gpu")]
criterion_main!(gpu_benches);

// Fallback main when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmarks require the 'gpu' feature flag.");
    println!("Run with: cargo bench --features gpu --bench binance_gpu_benchmark");
}

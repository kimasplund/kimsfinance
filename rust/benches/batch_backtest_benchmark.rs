//! Comprehensive Benchmark Suite for GPU Batch Backtesting
//!
//! **Performance Targets** (RTX 3500 Ada, 12GB VRAM):
//! - Sequential CPU baseline: ~10,000ms for 1000 strategies × 10K candles
//! - GPU batch target: ~250ms for same workload (40x speedup)
//! - VRAM usage: <1GB for 1000 strategies × 10K candles
//!
//! **Benchmark Configurations**:
//! - Small: 10 strategies × 1K candles (sanity check)
//! - Medium: 100 strategies × 10K candles (typical use case)
//! - Large: 1000 strategies × 10K candles (genetic optimization)
//! - Stress: 1000 strategies × 50K candles (VRAM limit test)
//!
//! **Usage**:
//! ```bash
//! # Run all benchmarks
//! cargo bench --bench batch_backtest_benchmark
//!
//! # Run only GPU benchmarks
//! cargo bench --bench batch_backtest_benchmark -- gpu
//!
//! # Run only CPU baseline (slower)
//! cargo bench --bench batch_backtest_benchmark -- cpu_baseline
//! ```
//!
//! **Dependencies**: CUDA kernels must be implemented first (Task 1)
//! **Estimated Run Time**: 15-30 minutes (full suite)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

mod test_data_generator;
use test_data_generator::{DataGeneratorConfig, OHLCVData, generate_realistic_ohlcv};

// NOTE: These imports will be available after Task 1-2 complete
// Commenting out for now so file compiles in prep stage
// use kimsfinance_core::backtest::batch::{BatchBacktestSweep, StrategyType};
// use kimsfinance_core::gpu::device::GpuDevice;

/// Benchmark configuration: (N_strategies, N_candles)
fn benchmark_configurations() -> Vec<(&'static str, usize, usize)> {
    vec![
        // Small configs (quick validation)
        ("small_10x1k", 10, 1000),
        ("small_10x10k", 10, 10000),
        // Medium configs (typical use case)
        ("medium_100x1k", 100, 1000),
        ("medium_100x10k", 100, 10000),
        // Large configs (genetic optimization)
        ("large_500x10k", 500, 10000),
        ("large_1000x1k", 1000, 1000),
        ("large_1000x10k", 1000, 10000),
        // Stress test (VRAM limits)
        ("stress_1000x50k", 1000, 50000),
        ("stress_2000x10k", 2000, 10000),
    ]
}

/// Generate random strategy parameters for RSI crossover
fn generate_rsi_parameters(n_strategies: usize, seed: u64) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    use rand::prelude::*;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..n_strategies)
        .map(|_| {
            vec![
                rng.gen_range(10.0..20.0), // RSI period (10-20)
                rng.gen_range(20.0..40.0), // Buy threshold (oversold)
                rng.gen_range(60.0..80.0), // Sell threshold (overbought)
            ]
        })
        .collect()
}

/// Generate random strategy parameters for MA crossover
fn generate_ma_parameters(n_strategies: usize, seed: u64) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    use rand::prelude::*;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..n_strategies)
        .map(|_| {
            vec![
                rng.gen_range(5.0..20.0),  // Fast MA period
                rng.gen_range(20.0..50.0), // Slow MA period
            ]
        })
        .collect()
}

// ============================================================================
// GPU Batch Backtesting Benchmarks
// ============================================================================

/// Benchmark: GPU batch backtesting (RSI strategy)
///
/// **Target Performance** (1000 strategies × 10K candles):
/// - GPU batch: ~250ms (40x speedup vs sequential)
/// - VRAM usage: <1GB
/// - GPU utilization: >70%
fn bench_batch_backtest_gpu_rsi(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_backtest_gpu/rsi");
    group.sample_size(20); // Reduce samples for long-running benchmarks

    for (name, n_strategies, n_candles) in benchmark_configurations() {
        let config = DataGeneratorConfig::bull_market(n_candles, 12345);
        let data = generate_realistic_ohlcv(&config);
        let params = generate_rsi_parameters(n_strategies, 67890);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(n_strategies, n_candles),
            |b, _| {
                b.iter(|| {
                    // TODO: Uncomment after Task 1-2 complete
                    // let device = GpuDevice::new().expect("GPU not available");
                    // let sweep = BatchBacktestSweep::new(&device)
                    //     .strategy_type(StrategyType::RsiCrossover)
                    //     .data_ohlcv(&data.open, &data.high, &data.low, &data.close, &data.volume)
                    //     .parameters_batch(&params)
                    //     .execute()
                    //     .expect("GPU batch backtest failed");
                    //
                    // black_box(sweep.results);

                    // Placeholder: Simulate work
                    black_box(data.close.iter().sum::<f64>());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: GPU batch backtesting (MA crossover strategy)
fn bench_batch_backtest_gpu_ma(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_backtest_gpu/ma_crossover");
    group.sample_size(20);

    // Only benchmark medium/large configs for MA (it's slower)
    let configs = vec![
        ("medium_100x10k", 100, 10000),
        ("large_1000x10k", 1000, 10000),
    ];

    for (name, n_strategies, n_candles) in configs {
        let config = DataGeneratorConfig::sideways_market(n_candles, 11111);
        let data = generate_realistic_ohlcv(&config);
        let params = generate_ma_parameters(n_strategies, 22222);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(n_strategies, n_candles),
            |b, _| {
                b.iter(|| {
                    // TODO: Uncomment after Task 1-2 complete
                    // let device = GpuDevice::new().expect("GPU not available");
                    // let sweep = BatchBacktestSweep::new(&device)
                    //     .strategy_type(StrategyType::MaCrossover)
                    //     .data_ohlcv(&data.open, &data.high, &data.low, &data.close, &data.volume)
                    //     .parameters_batch(&params)
                    //     .execute()
                    //     .expect("GPU batch backtest failed");
                    //
                    // black_box(sweep.results);

                    // Placeholder
                    black_box(data.close.iter().sum::<f64>());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// CPU Sequential Baseline Benchmarks
// ============================================================================

/// Benchmark: CPU sequential backtesting (baseline for comparison)
///
/// **Expected Performance** (1000 strategies × 10K candles):
/// - CPU sequential: ~10,000ms
/// - Per-strategy overhead: ~10ms
///
/// **Note**: Only small/medium configs tested (CPU is too slow for large)
fn bench_batch_backtest_cpu_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_backtest_cpu_baseline");
    group.sample_size(10); // Fewer samples (CPU is slow!)

    // Only benchmark small configs for CPU
    let small_configs = vec![
        ("small_10x1k", 10, 1000),
        ("small_10x10k", 10, 10000),
        ("medium_100x1k", 100, 1000),
        ("medium_100x10k", 100, 10000),
    ];

    for (name, n_strategies, n_candles) in small_configs {
        let config = DataGeneratorConfig::bull_market(n_candles, 12345);
        let data = generate_realistic_ohlcv(&config);
        let params = generate_rsi_parameters(n_strategies, 67890);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(n_strategies, n_candles),
            |b, _| {
                b.iter(|| {
                    // TODO: Uncomment after existing backtest engine is available
                    // Simulate sequential backtesting (one strategy at a time)
                    // let results: Vec<_> = params.iter().map(|p| {
                    //     backtest_sequential_cpu(&data, p)
                    // }).collect();
                    // black_box(results);

                    // Placeholder: Simulate O(N) work
                    let _result: Vec<f64> = params
                        .iter()
                        .map(|p| data.close.iter().sum::<f64>() * p[0])
                        .collect();
                    black_box(_result);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// VRAM Usage Analysis
// ============================================================================

/// Benchmark: VRAM usage scaling (measure only, not timed)
///
/// **Target**: Document VRAM usage for different configurations
/// - 1000 × 10K: <1GB
/// - 1000 × 50K: <2.5GB
/// - 5000 × 10K: <2.5GB
///
/// **Usage**: Run with `nvidia-smi dmon` in parallel to track VRAM
fn bench_vram_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("vram_scaling");
    group.sample_size(5); // Only measure a few times

    let vram_configs = vec![
        ("1000x10k", 1000, 10000),
        ("1000x50k", 1000, 50000),
        ("2000x10k", 2000, 10000),
        ("5000x10k", 5000, 10000),
    ];

    for (name, n_strategies, n_candles) in vram_configs {
        let config = DataGeneratorConfig::default();
        let data = generate_realistic_ohlcv(&config);
        let params = generate_rsi_parameters(n_strategies, 99999);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(n_strategies, n_candles),
            |b, _| {
                b.iter(|| {
                    // TODO: Uncomment after Task 1-2 complete
                    // Allocate buffers on GPU (measure VRAM usage)
                    // let device = GpuDevice::new().expect("GPU not available");
                    // let sweep = BatchBacktestSweep::new(&device)
                    //     .strategy_type(StrategyType::RsiCrossover)
                    //     .data_ohlcv(&data.open, &data.high, &data.low, &data.close, &data.volume)
                    //     .parameters_batch(&params)
                    //     .execute()
                    //     .expect("GPU batch backtest failed");
                    //
                    // black_box(sweep.results);

                    // Placeholder
                    black_box(params.len());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Throughput Analysis (strategies/second)
// ============================================================================

/// Benchmark: Throughput measurement (strategies/second)
///
/// **Target**: >4000 strategies/second (1000 strategies in 250ms)
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.sample_size(20);

    // Fixed dataset size, vary number of strategies
    let throughput_configs = vec![
        ("10_strategies", 10),
        ("100_strategies", 100),
        ("500_strategies", 500),
        ("1000_strategies", 1000),
        ("2000_strategies", 2000),
    ];

    let n_candles = 10000;
    let config = DataGeneratorConfig::bull_market(n_candles, 12345);
    let data = generate_realistic_ohlcv(&config);

    for (name, n_strategies) in throughput_configs {
        let params = generate_rsi_parameters(n_strategies, 77777);

        group.bench_with_input(BenchmarkId::from_parameter(name), &n_strategies, |b, _| {
            b.iter(|| {
                // TODO: Uncomment after Task 1-2 complete
                // let device = GpuDevice::new().expect("GPU not available");
                // let sweep = BatchBacktestSweep::new(&device)
                //     .strategy_type(StrategyType::RsiCrossover)
                //     .data_ohlcv(&data.open, &data.high, &data.low, &data.close, &data.volume)
                //     .parameters_batch(&params)
                //     .execute()
                //     .expect("GPU batch backtest failed");
                //
                // black_box(sweep.results);

                // Placeholder
                black_box(data.close.iter().sum::<f64>());
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_batch_backtest_gpu_rsi,
    bench_batch_backtest_gpu_ma,
    bench_batch_backtest_cpu_baseline,
    bench_vram_scaling,
    bench_throughput,
);

criterion_main!(benches);

// ============================================================================
// Post-Benchmark Analysis Instructions
// ============================================================================

// **After running benchmarks**:
//
// 1. **Generate HTML report**:
//    ```bash
//    cargo bench --bench batch_backtest_benchmark
//    firefox target/criterion/batch_backtest_gpu/report/index.html
//    ```
//
// 2. **Monitor GPU utilization** (run in parallel terminal):
//    ```bash
//    nvidia-smi dmon -s u
//    ```
//
// 3. **Measure VRAM usage** (run during bench_vram_scaling):
//    ```bash
//    watch -n 0.5 nvidia-smi
//    ```
//
// 4. **Calculate speedup**:
//    - GPU time: Read from criterion report (e.g., 250ms)
//    - CPU time: Read from cpu_baseline report (e.g., 10,000ms)
//    - Speedup: CPU time / GPU time (e.g., 40x)
//
// 5. **Update performance report**:
//    - Fill in `benchmarks/BATCH_BACKTEST_RESULTS.md`
//    - Include confidence intervals from criterion
//    - Document VRAM usage per configuration
//
// 6. **Statistical validation**:
//    ```bash
//    python scripts/validate_batch_accuracy.py
//    ```

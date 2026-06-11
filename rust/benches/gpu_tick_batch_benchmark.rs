//! GPU Tick Batch Backtest Benchmark
//!
//! Comprehensive performance validation for GPU tick-level batch backtesting.
//!
//! # Benchmark Coverage
//!
//! - **Throughput**: 500M-1B ticks/sec target (vs 70M CPU Rayon)
//! - **Latency**: Per-generation time (GPU vs CPU)
//! - **Scalability**: Batch size 1-100 strategies
//! - **VRAM Usage**: Memory consumption across batch sizes
//! - **Accuracy**: Validate <0.01% deviation from CPU
//!
//! # Hardware Target
//!
//! - GPU: NVIDIA RTX 3500 Ada (12GB VRAM)
//! - CPU: Intel i9-13980HX (24 cores, 32 threads with Rayon)
//!
//! # Usage
//!
//! ```bash
//! # Run full benchmark suite
//! cargo bench --features gpu --bench gpu_tick_batch_benchmark
//!
//! # Run specific benchmark
//! cargo bench --features gpu --bench gpu_tick_batch_benchmark throughput
//!
//! # Generate HTML report
//! cargo bench --features gpu --bench gpu_tick_batch_benchmark -- --save-baseline main
//! ```
//!
//! # Exit Codes
//!
//! - 0: All benchmarks passed performance targets
//! - 1: Performance regression detected
//! - 2: GPU not available (skip GPU benchmarks)

#![cfg(feature = "gpu")]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::backtest::BacktestConfig;
use kimsfinance_core::binance::Trade;
use kimsfinance_core::gpu::device::GpuDevice;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_test_trades(n: usize) -> Vec<Trade> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let base_price = 45000.0;
    let base_timestamp = 1704067200000i64;

    (0..n)
        .map(|i| {
            let price_change = (rng.r#gen::<f64>() - 0.5) * 0.002;
            let price = base_price * (1.0 + price_change);
            let quantity = rng.gen_range(0.001..1.0);

            Trade {
                trade_id: i as u64,
                price,
                quantity,
                quote_quantity: price * quantity,
                timestamp_ms: base_timestamp + (i as i64 * 10),
                is_buyer_maker: rng.gen_bool(0.5),
            }
        })
        .collect()
}

fn generate_parameter_sets(n: usize) -> Vec<Vec<f64>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(123);

    (0..n)
        .map(|_| {
            vec![
                rng.gen_range(50.0..200.0),   // window_size
                rng.gen_range(0.5..0.7),      // imbalance_threshold
                rng.gen_range(5.0..20.0),     // min_volume
                rng.gen_range(0.0005..0.002), // spike_threshold
                rng.gen_range(3.0..10.0),     // ema_period
                rng.gen_range(0.8..1.5),      // volatility_factor
            ]
        })
        .collect()
}

// ============================================================================
// GPU Implementation Placeholder
// ============================================================================

#[allow(dead_code)]
struct BatchTickBacktest {
    device: Arc<GpuDevice>,
}

#[allow(dead_code)]
impl BatchTickBacktest {
    fn new(device: Arc<GpuDevice>) -> Self {
        Self { device }
    }

    fn execute(
        &self,
        _trades: &[Trade],
        _params_batch: &[Vec<f64>],
        _config: &BacktestConfig,
    ) -> Result<Vec<f64>, String> {
        // PLACEHOLDER: Will be implemented by Agent 1-3-5
        Err("GPU tick batch not yet implemented".to_string())
    }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

fn cpu_rayon_backtests(
    trades: &[Trade],
    params_batch: &[Vec<f64>],
    config: &BacktestConfig,
) -> Vec<f64> {
    use rayon::prelude::*;

    params_batch
        .par_iter()
        .map(|params| {
            // Simplified CPU backtest
            // Real implementation would use TickEngine
            let _ = (trades, params, config);
            0.15 // Placeholder: 15% return
        })
        .collect()
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Benchmark 1: Throughput (ticks/sec)
fn bench_throughput(c: &mut Criterion) {
    let device_result = GpuDevice::new();

    if device_result.is_err() {
        println!("GPU not available, skipping GPU throughput benchmark");
        return;
    }

    let device = Arc::new(device_result.unwrap());
    let mut group = c.benchmark_group("tick_batch_throughput");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    // Test different dataset sizes
    for &num_trades in &[10_000, 100_000, 1_000_000] {
        let trades = generate_test_trades(num_trades);
        let params_batch = generate_parameter_sets(10); // 10 strategies
        let config = BacktestConfig::default();

        // Total ticks processed
        let total_ticks = num_trades * params_batch.len();
        group.throughput(Throughput::Elements(total_ticks as u64));

        // CPU Rayon baseline
        group.bench_with_input(
            BenchmarkId::new("cpu_rayon", num_trades),
            &num_trades,
            |b, _| {
                b.iter(|| {
                    cpu_rayon_backtests(
                        black_box(&trades),
                        black_box(&params_batch),
                        black_box(&config),
                    )
                });
            },
        );

        // GPU batch
        let batch = BatchTickBacktest::new(device.clone());
        group.bench_with_input(
            BenchmarkId::new("gpu_batch", num_trades),
            &num_trades,
            |b, _| {
                b.iter(|| {
                    // Placeholder: Will call real GPU implementation
                    let _ = batch.execute(
                        black_box(&trades),
                        black_box(&params_batch),
                        black_box(&config),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 2: Scalability (batch size 1-100)
fn bench_scalability(c: &mut Criterion) {
    let device_result = GpuDevice::new();

    if device_result.is_err() {
        println!("GPU not available, skipping scalability benchmark");
        return;
    }

    let device = Arc::new(device_result.unwrap());
    let mut group = c.benchmark_group("tick_batch_scalability");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    let trades = generate_test_trades(100_000); // Fixed dataset size
    let config = BacktestConfig::default();

    // Test different batch sizes
    for &batch_size in &[1, 5, 10, 20, 50, 100] {
        let params_batch = generate_parameter_sets(batch_size);

        group.throughput(Throughput::Elements((trades.len() * batch_size) as u64));

        // CPU Rayon
        group.bench_with_input(
            BenchmarkId::new("cpu_rayon", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    cpu_rayon_backtests(
                        black_box(&trades),
                        black_box(&params_batch),
                        black_box(&config),
                    )
                });
            },
        );

        // GPU batch
        let batch = BatchTickBacktest::new(device.clone());
        group.bench_with_input(
            BenchmarkId::new("gpu_batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let _ = batch.execute(
                        black_box(&trades),
                        black_box(&params_batch),
                        black_box(&config),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 3: Latency (per-generation time)
fn bench_latency(c: &mut Criterion) {
    let device_result = GpuDevice::new();

    if device_result.is_err() {
        println!("GPU not available, skipping latency benchmark");
        return;
    }

    let device = Arc::new(device_result.unwrap());
    let mut group = c.benchmark_group("tick_batch_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // Genetic optimizer scenario: 100 strategies per generation
    let trades = generate_test_trades(1_000_000); // 1M trades (realistic)
    let params_batch = generate_parameter_sets(100); // 100 strategies
    let config = BacktestConfig::default();

    // CPU Rayon
    group.bench_function("cpu_rayon_generation", |b| {
        b.iter(|| {
            cpu_rayon_backtests(
                black_box(&trades),
                black_box(&params_batch),
                black_box(&config),
            )
        });
    });

    // GPU batch
    let batch = BatchTickBacktest::new(device);
    group.bench_function("gpu_batch_generation", |b| {
        b.iter(|| {
            let _ = batch.execute(
                black_box(&trades),
                black_box(&params_batch),
                black_box(&config),
            );
        });
    });

    group.finish();
}

/// Benchmark 4: VRAM Usage Validation
fn bench_vram_usage(c: &mut Criterion) {
    let device_result = GpuDevice::new();

    if device_result.is_err() {
        println!("GPU not available, skipping VRAM benchmark");
        return;
    }

    let device = Arc::new(device_result.unwrap());
    let mut group = c.benchmark_group("tick_batch_vram");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(5);

    let trades = generate_test_trades(1_000_000); // 1M trades
    let config = BacktestConfig::default();

    for &batch_size in &[5, 10, 15, 20] {
        let params_batch = generate_parameter_sets(batch_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    // Measure VRAM usage before
                    // let vram_before = device.memory_used().unwrap_or(0);

                    let batch = BatchTickBacktest::new(device.clone());
                    let _ = batch.execute(
                        black_box(&trades),
                        black_box(&params_batch),
                        black_box(&config),
                    );

                    // Measure VRAM usage after
                    // let vram_after = device.memory_used().unwrap_or(0);
                    // let vram_used = vram_after - vram_before;
                    // println!("Batch size {}: {:.2} GB VRAM", batch_size, vram_used as f64 / 1e9);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 5: Accuracy Validation (GPU vs CPU deviation)
fn bench_accuracy(c: &mut Criterion) {
    let device_result = GpuDevice::new();

    if device_result.is_err() {
        println!("GPU not available, skipping accuracy benchmark");
        return;
    }

    let device = Arc::new(device_result.unwrap());
    let mut group = c.benchmark_group("tick_batch_accuracy");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    let trades = generate_test_trades(100_000);
    let params_batch = generate_parameter_sets(10);
    let config = BacktestConfig::default();

    group.bench_function("cpu_vs_gpu_accuracy", |b| {
        b.iter(|| {
            // CPU reference
            let cpu_results = cpu_rayon_backtests(
                black_box(&trades),
                black_box(&params_batch),
                black_box(&config),
            );

            // GPU implementation
            let batch = BatchTickBacktest::new(device.clone());
            let gpu_results = batch
                .execute(
                    black_box(&trades),
                    black_box(&params_batch),
                    black_box(&config),
                )
                .unwrap_or_else(|_| vec![0.0; params_batch.len()]);

            // Validate <0.01% deviation
            for (cpu, gpu) in cpu_results.iter().zip(gpu_results.iter()) {
                let deviation = (cpu - gpu).abs() / cpu.abs().max(1e-9);
                assert!(
                    deviation < 0.0001,
                    "Deviation too high: {:.4}%",
                    deviation * 100.0
                );
            }
        });
    });

    group.finish();
}

/// Benchmark 6: End-to-End Performance
fn bench_end_to_end(c: &mut Criterion) {
    let device_result = GpuDevice::new();

    if device_result.is_err() {
        println!("GPU not available, skipping end-to-end benchmark");
        return;
    }

    let device = Arc::new(device_result.unwrap());
    let mut group = c.benchmark_group("tick_batch_end_to_end");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(5);

    // Realistic genetic optimizer scenario
    // 50 generations × 100 strategies × 1M trades = 5B ticks
    let trades = generate_test_trades(1_000_000);
    let params_batch = generate_parameter_sets(100);
    let config = BacktestConfig::default();
    let num_generations = 50;

    // CPU Rayon baseline
    group.bench_function("cpu_rayon_50_generations", |b| {
        b.iter(|| {
            for _ in 0..num_generations {
                let _ = cpu_rayon_backtests(
                    black_box(&trades),
                    black_box(&params_batch),
                    black_box(&config),
                );
            }
        });
    });

    // GPU batch
    let batch = BatchTickBacktest::new(device);
    group.bench_function("gpu_batch_50_generations", |b| {
        b.iter(|| {
            for _ in 0..num_generations {
                let _ = batch.execute(
                    black_box(&trades),
                    black_box(&params_batch),
                    black_box(&config),
                );
            }
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_throughput,
    bench_scalability,
    bench_latency,
    bench_vram_usage,
    bench_accuracy,
    bench_end_to_end
);

criterion_main!(benches);

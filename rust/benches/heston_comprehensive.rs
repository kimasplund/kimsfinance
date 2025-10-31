//! Comprehensive Benchmark Suite for Heston Calibrator
//!
//! Benchmarks all major components:
//! - GPU option pricing (batch sizes: 10-1000)
//! - Calibration engine (30-100 iterations)
//! - Greeks calculation (single and batch)
//! - Kernel compilation overhead
//! - Memory transfer (pinned vs pageable)
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench heston_comprehensive --features gpu,heston
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{
    HestonCalibrator, HestonGreeksCalculator, HestonParams, OptionQuote, OptionType,
};
use parking_lot::Mutex;
use std::sync::Arc;

/// Generate test options with various strikes
fn generate_test_options(n: usize, base_strike: f64) -> Vec<OptionQuote> {
    let now = chrono::Utc::now().timestamp();
    let expiry_3months = now + (90 * 24 * 3600);

    (0..n)
        .map(|i| {
            let strike = base_strike + (i as f64 * 500.0);
            OptionQuote {
                underlying: "BTC".to_string(),
                strike,
                expiration: expiry_3months,
                option_type: if i % 2 == 0 {
                    OptionType::Call
                } else {
                    OptionType::Put
                },
                spot_price: 50000.0,
                risk_free_rate: 0.05,
                bid: Some(2000.0),
                ask: Some(2200.0),
                last: Some(2100.0),
                implied_vol: Some(0.8),
                volume: 100.0,
                open_interest: 500.0,
                greeks: None,
            }
        })
        .collect()
}

/// Benchmark GPU option pricing across different batch sizes
fn bench_gpu_pricing(c: &mut Criterion) {
    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).expect("Invalid parameters");
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let mut group = c.benchmark_group("heston_gpu_pricing");

    for n in [10, 50, 100, 500, 1000].iter() {
        let options = generate_test_options(*n, 48000.0);
        let mut pricer =
            HestonGpuPricer::new(device.clone(), 4096, *n).expect("Failed to create pricer");

        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                pricer
                    .price_options(black_box(&params), black_box(&options))
                    .expect("GPU pricing failed")
            })
        });
    }

    group.finish();
}

/// Benchmark calibration with different option counts and iteration limits
fn bench_calibration(c: &mut Criterion) {
    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).expect("Invalid parameters");
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let mut group = c.benchmark_group("heston_calibration");
    group.sample_size(10); // Fewer samples for slow benchmarks

    // Test case 1: 30 options, 20 iterations
    {
        let mut pricer_for_gen =
            HestonGpuPricer::new(device.clone(), 4096, 100).expect("Failed to create pricer");
        let mut options = generate_test_options(30, 48000.0);

        let prices = pricer_for_gen
            .price_options(&params, &options)
            .expect("Failed to price");
        for (i, opt) in options.iter_mut().enumerate() {
            opt.bid = Some(prices[i] * 0.98);
            opt.ask = Some(prices[i] * 1.02);
        }

        group.bench_function("30_options_20_iters", |b| {
            b.iter(|| {
                let gpu_pricer = Arc::new(
                    HestonGpuPricer::new(device.clone(), 4096, 100)
                        .expect("Failed to create pricer"),
                );
                let calibrator = HestonCalibrator::new(gpu_pricer, options.clone(), params)
                    .expect("Failed to create calibrator")
                    .with_max_iterations(20);

                calibrator.calibrate().expect("Calibration failed")
            })
        });
    }

    // Test case 2: 50 options, 30 iterations
    {
        let mut pricer_for_gen =
            HestonGpuPricer::new(device.clone(), 4096, 100).expect("Failed to create pricer");
        let mut options = generate_test_options(50, 48000.0);

        let prices = pricer_for_gen
            .price_options(&params, &options)
            .expect("Failed to price");
        for (i, opt) in options.iter_mut().enumerate() {
            opt.bid = Some(prices[i] * 0.98);
            opt.ask = Some(prices[i] * 1.02);
        }

        group.bench_function("50_options_30_iters", |b| {
            b.iter(|| {
                let gpu_pricer = Arc::new(
                    HestonGpuPricer::new(device.clone(), 4096, 100)
                        .expect("Failed to create pricer"),
                );
                let calibrator = HestonCalibrator::new(gpu_pricer, options.clone(), params)
                    .expect("Failed to create calibrator")
                    .with_max_iterations(30);

                calibrator.calibrate().expect("Calibration failed")
            })
        });
    }

    group.finish();
}

/// Benchmark Greeks calculation (single and batch)
fn bench_greeks(c: &mut Criterion) {
    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).expect("Invalid parameters");
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let pricer = HestonGpuPricer::new(device, 4096, 100).expect("Failed to create pricer");
    let calculator = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer)));

    let mut group = c.benchmark_group("heston_greeks");

    // Single option Greeks
    {
        let option = generate_test_options(1, 50000.0).pop().unwrap();

        group.bench_function("single_option", |b| {
            b.iter(|| {
                calculator
                    .calculate_greeks(black_box(&params), black_box(&option))
                    .expect("Greeks calculation failed")
            })
        });
    }

    // Batch Greeks
    for n in [10, 50, 100].iter() {
        let options = generate_test_options(*n, 48000.0);

        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::new("batch", n), n, |b, _| {
            b.iter(|| {
                calculator
                    .calculate_greeks_batch(black_box(&params), black_box(&options))
                    .expect("Batch Greeks calculation failed")
            })
        });
    }

    group.finish();
}

/// Benchmark kernel compilation (cold vs warm start)
fn bench_kernel_compilation(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let mut group = c.benchmark_group("heston_compilation");
    group.sample_size(10);

    // Note: First call will compile, subsequent calls will use cache
    // This benchmark measures warm start performance (cached)
    group.bench_function("warm_start", |b| {
        b.iter(|| HestonGpuPricer::new(device.clone(), 4096, 100).expect("Failed to create pricer"))
    });

    group.finish();
}

/// Benchmark memory transfer overhead (pinned vs pageable)
fn bench_memory_transfer(c: &mut Criterion) {
    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).expect("Invalid parameters");
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));

    let mut group = c.benchmark_group("heston_memory_transfer");

    // Test with different batch sizes to see pinned memory benefit
    for n in [100, 500, 1000].iter() {
        let options = generate_test_options(*n, 48000.0);

        // With pinned memory (default)
        {
            let mut pricer_pinned =
                HestonGpuPricer::new(device.clone(), 4096, *n).expect("Failed to create pricer");

            group.throughput(Throughput::Elements(*n as u64));
            group.bench_with_input(BenchmarkId::new("pinned", n), n, |b, _| {
                b.iter(|| {
                    pricer_pinned
                        .price_options(black_box(&params), black_box(&options))
                        .expect("Pricing failed")
                })
            });
        }
    }

    group.finish();
}

/// Benchmark FFT size impact on performance
fn bench_fft_size(c: &mut Criterion) {
    let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).expect("Invalid parameters");
    let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
    let options = generate_test_options(100, 48000.0);

    let mut group = c.benchmark_group("heston_fft_size");

    for fft_size in [2048, 4096, 8192].iter() {
        let mut pricer =
            HestonGpuPricer::new(device.clone(), *fft_size, 100).expect("Failed to create pricer");

        group.bench_with_input(BenchmarkId::from_parameter(fft_size), fft_size, |b, _| {
            b.iter(|| {
                pricer
                    .price_options(black_box(&params), black_box(&options))
                    .expect("Pricing failed")
            })
        });
    }

    group.finish();
}

/// Benchmark parameter validation overhead
fn bench_parameter_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("heston_parameter_validation");

    // Valid parameters
    group.bench_function("valid_params", |b| {
        b.iter(|| {
            HestonParams::new(
                black_box(2.0),
                black_box(0.04),
                black_box(0.3),
                black_box(-0.7),
                black_box(0.04),
            )
        })
    });

    // Invalid parameters (Feller violation)
    group.bench_function("invalid_params", |b| {
        b.iter(|| {
            HestonParams::new(
                black_box(1.0),
                black_box(0.01),
                black_box(1.5), // Violates Feller
                black_box(-0.7),
                black_box(0.04),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gpu_pricing,
    bench_calibration,
    bench_greeks,
    bench_kernel_compilation,
    bench_memory_transfer,
    bench_fft_size,
    bench_parameter_validation,
);
criterion_main!(benches);
